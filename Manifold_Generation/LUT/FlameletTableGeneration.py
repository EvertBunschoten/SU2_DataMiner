###############################################################################################
#       #      _____ __  _____      ____        __        __  ____                   #        #
#       #     / ___// / / /__ \    / __ \____ _/ /_____ _/  |/  (_)___  ___  _____   #        #
#       #     \__ \/ / / /__/ /   / / / / __ `/ __/ __ `/ /|_/ / / __ \/ _ \/ ___/   #        #
#       #    ___/ / /_/ // __/   / /_/ / /_/ / /_/ /_/ / /  / / / / / /  __/ /       #        #
#       #   /____/\____//____/  /_____/\__,_/\__/\__,_/_/  /_/_/_/ /_/\___/_/        #        #
#       #                                                                            #        #
###############################################################################################

######################### FILE NAME: FlameletTableGenerator.py ################################
#=============================================================================================#
# author: Evert Bunschoten                                                                    |
#    :PhD Candidate ,                                                                         |
#    :Flight Power and Propulsion                                                             |
#    :TU Delft,                                                                               |
#    :The Netherlands                                                                         |
#                                                                                             |
#                                                                                             |
# Description:                                                                                |
#   Table generator class for generating SU2-supported tables of flamelet data.               |
# Version: 1.0.0                                                                              |
#                                                                                             |
#=============================================================================================#

import numpy as np 
from scipy.spatial import ConvexHull, Delaunay
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt 
from tqdm import tqdm
import sys,os
from Common.DataDrivenConfig import FlameletAIConfig, Config
from Common.CommonMethods import GetReferenceData 
from Common.Properties import DefaultSettings_FGM
import cantera as ct
import gmsh 
import pickle
from multiprocessing import Pool 
from sklearn.metrics import mean_squared_error
from Common.Interpolators import Invdisttree 
from random import sample 

class SU2TableGenerator_Base:
    _Config = None 
    _savedir:str 
    _table_variables:list[str] = None 
    _manifold_variables:list[str]
    _controlling_variables:list[str] = None 
    _manifold_data:np.ndarray[float] = None 
    _manifold_data_interpolator:Invdisttree = None 
    _base_cell_size:float = 1e-2#3.7e-3      # Table level base cell size.

    _refined_cell_size:float = 1e-3#2.5e-3#1.5e-3   # Table level refined cell size.
    _refinement_radius:float = 5e-3#5e-2     # Table level radius within which refinement is applied.
    _curvature_threshold:float = 0.3    # Curvature threshold above which refinement is applied.
    _n_near:int = 4     # Number of nearest neighbors from which to evaluate flamelet data.
    _p_fac:int = 5      # Power by which to weigh distances from query point.
    _control_var_scaler:MinMaxScaler =None 
    _table_nodes = []       # Progress variable, total enthalpy, and mixture fraction node values for each table level.
    _table_nodes_norm = []  # Normalized table nodes for each level.
    _table_connectivity = []    # Table node connectivity per table level.
    _table_hullnodes = []   # Hull node indices per table level.
    
    def __init__(self, Config_in):
        self._Config = Config_in
        self._savedir = self._Config.GetOutputDir()
        return 
    
    def SetSaveDir(self, save_dir:str):
        if not os.path.isdir(save_dir):
            raise Exception("Output directory %s not present on current hardware." % save_dir)
        self._savedir = save_dir 
        return 
    
    def SetBaseCellSize(self, cell_size:float):
        """
        Define the base cell size for the table levels.

        :param cell_size: Normalized coarse cell size for each 2D table mesh.
        :type cell_size: float
        :raise: Exception: if cell size is lower or equal to zero.
        """
        if cell_size > 0:
            self._base_cell_size = cell_size 
        else:
            raise Exception("Proviced cell size should be higher than zero.")
        return
    
    def SetRefinedCellSize(self, cell_size:float):
        """
        Define the refinement cell size for the table levels.

        :param cell_size: Normalized fine cell size for each 2D table mesh.
        :type cell_size: float
        :raise: Exception: if cell size is lower or equal to zero.
        """
        if cell_size > 0:
            self._refined_cell_size = cell_size 
        else:
            raise Exception("Proviced cell size should be higher than zero.")
        return 
    
    def SetRefinementThreshold(self, val_threshold:float):
        """
        Define normalized curvature threshold beyond which refinement should be applied to each table level.

        :param val_threshold: Normalized curvature threshold value. All locations in the mesh with a higher curvature receive refinement.
        :type val_threshold: float
        :raises: Exception: If the threshold value is lower than zero.
        """       

        if val_threshold > 0:
            self._curvature_threshold = val_threshold
        else:
            raise Exception("Curvature threshold value should be higher than zero.")
        return 
    
    
    def DefineFlameletDataInterpolator(self):

        print("Configuring KD-tree for most accurate lookups")

        print("Loading flamelet data...")
        # Define scaler for FGM controlling variables.
        full_data_file = self._Config.GetOutputDir()+"/LUT_data_full.csv"
        with open(full_data_file,'r') as fid:
            self._manifold_variables = fid.readline().strip().split(',')
        #D_full = np.loadtxt(full_data_file,delimiter=',',skiprows=1)
        self._control_var_scaler = MinMaxScaler()
        CV_full, D_full = GetReferenceData(full_data_file, self._controlling_variables, self._manifold_variables)
        data_scaler = MinMaxScaler()
        data_scaler.fit_transform(D_full)

        CV_full_scaled = self._control_var_scaler.fit_transform(CV_full)

        # Exctract train and test data
        train_data_file = self._Config.GetOutputDir()+"/"+self._Config.GetConcatenationFileHeader()+"_train.csv"
        test_data_file = self._Config.GetOutputDir()+"/"+self._Config.GetConcatenationFileHeader()+"_test.csv"
        
        CV_train, D_train = GetReferenceData(train_data_file, self._controlling_variables, self._manifold_variables)
        CV_test, D_test = GetReferenceData(test_data_file, self._controlling_variables, self._manifold_variables)
        
        CV_train_scaled = self._control_var_scaler.transform(CV_train)
        CV_test_scaled = self._control_var_scaler.transform(CV_test)
        D_train_scaled = data_scaler.transform(D_train)
        D_test_scaled = data_scaler.transform(D_test)
        
        print("Done!")
        print("Setting up KD-tree...")
        self._lookup_tree = Invdisttree(X=CV_train_scaled,z=D_train_scaled)
        print("Done!")
        
        print("Search for best tree parameters...")
        # Do brute-force search to get the optimum number of nearest neighbors and distance power.
        n_near_range = range(1, 20)
        p_range = range(1, 6)
        RMS_ppv = np.zeros([len(n_near_range), len(p_range)])
        for i in tqdm(range(len(n_near_range))):
            for j in range(len(p_range)):
                PPV_predicted = self._lookup_tree(q=CV_test_scaled, nnear=n_near_range[i], p=p_range[j])
                rms_local = mean_squared_error(y_true=D_test_scaled, y_pred=PPV_predicted)
                RMS_ppv[i,j] = rms_local 
        [imin,jmin] = divmod(RMS_ppv.argmin(), RMS_ppv.shape[1])
        self._n_near = n_near_range[imin]
        self._p_fac = p_range[jmin]
        print("Done!")
        print("Best found number of nearest neighbors: "+str(self._n_near))
        print("Best found distance power: "+str(self._p_fac))
        print("Setting up KD-tree...")
        self._lookup_tree = Invdisttree(X=CV_full_scaled,z=D_full)
        print("Done!")
        
        return
    
    def EvaluateManifoldInterpolator(self, CV_unscaled:np.ndarray):
        CV_scaled = self._control_var_scaler.transform(CV_unscaled)
        data_interp = self._lookup_tree(q=CV_scaled,nnear=self._n_near,p=self._p_fac)
        return data_interp

    def Compute2DTable(self, CV_1:str, CV_2:str):
        Np_grid = 300
        
        return
    
class SU2TableGenerator:

    _Config:FlameletAIConfig = None # FlameletAIConfig class from which to read settings.
    _savedir:str

    _mixfrac_min:float = None     # Minimum mixture fraction value of the flamelet data.
    _mixfrac_max:float = None     # Maximum mixture fraction value of the flamelet data.

    _pv_full_norm:np.ndarray[float] = None        # Normalized progress variable values of the flamelet data.
    _enth_full_norm:np.ndarray[float] = None      # Normalized total enthalpy values of the flamelet data.
    _mixfrac_full_norm:np.ndarray[float] = None   # Normalized mixture fraction values of the flamelet data.

    _Flamelet_Variables:list[str] = None  # Variable names in the concatenated flamelet data file.
    _Flamelet_Data:np.ndarray[float] = None     # Concatenated flamelet data.

    _custom_table_limits_set:bool = False 
    _mixfrac_min_table:float = None     # Lower mixture fraction limit of the table.
    _mixfrac_max_table:float = None     # Upper mixture fraction limit of the table.

    __run_parallel:bool = False 
    __Np_cores:int = 1 

    _N_table_levels:int = 100   # Number of table levels.
    _mixfrac_range_table:np.ndarray[float] = None   # Mixture fraction values of the table levels.
    _base_cell_size:float = 1e-2#3.7e-3      # Table level base cell size.

    _refined_cell_size:float = 1e-3#2.5e-3#1.5e-3   # Table level refined cell size.
    _refinement_radius:float = 5e-3#5e-2     # Table level radius within which refinement is applied.
    _curvature_threshold:float = 0.3    # Curvature threshold above which refinement is applied.

    _table_nodes = []       # Progress variable, total enthalpy, and mixture fraction node values for each table level.
    _table_nodes_norm = []  # Normalized table nodes for each level.
    _table_connectivity = []    # Table node connectivity per table level.
    _table_hullnodes = []   # Hull node indices per table level.
    __table_insert_levels:list[float] = []

    _controlling_variables:list[str]=[DefaultSettings_FGM.name_pv,\
                                      DefaultSettings_FGM.name_enth,\
                                      DefaultSettings_FGM.name_mixfrac]  # FGM controlling variables
    _lookup_tree:Invdisttree = None     # KD tree with inverse distance weighted interpolation for flamelet data interpolation.
    _flamelet_data_scaler:MinMaxScaler = None   # Scaler for flamelet data controlling variables.
    _n_near:int = 14     # Number of nearest neighbors from which to evaluate flamelet data.
    _p_fac:int = 3      # Power by which to weigh distances from query point.
    _custom_KDtreeparams:bool = False 

    _preprocessed:bool = False 

    def __init__(self, Config:FlameletAIConfig, load_file:str=None, n_near:int=None, p_fac:int=None):
        """
        Initiate table generator class.

        :param Config: FlameletAIConfig object.
        :type Config: FlameletAIConfig
        """

        if n_near and p_fac:
            self._custom_KDtreeparams = True

        if load_file:
            # Load an existing TableGenerator object.
            with open(load_file, "rb") as fid:
                loaded_table_generator = pickle.load(fid)
            self.__dict__ = loaded_table_generator.__dict__.copy()
        else:
            # Create new TableGenerator object.
            self._Config = Config 

            self.__DefineFlameletDataInterpolator()

        self._savedir = self._Config.GetOutputDir()
        return 
    
    def SetSaveDir(self, save_dir:str):
        if not os.path.isdir(save_dir):
            raise Exception("Output directory %s not present on current hardware." % save_dir)
        self._savedir = save_dir 

    def SetNTableLevels(self, N_levels:int):
        """
        Define the number of table levels in the mixture fraction direction.

        :param N_levels: number of table levels.
        :type N_levels: int
        :raise: Exception: if number of levels is lower than 2
        """
        if N_levels >= 2:
            self._N_table_levels = N_levels
        else:
            raise Exception("Number of table levels should be higher than 2.")
        return 
    
    def SetBaseCellSize(self, cell_size:float):
        """
        Define the base cell size for the table levels.

        :param cell_size: Normalized coarse cell size for each 2D table mesh.
        :type cell_size: float
        :raise: Exception: if cell size is lower or equal to zero.
        """
        if cell_size > 0:
            self._base_cell_size = cell_size 
        else:
            raise Exception("Proviced cell size should be higher than zero.")
        return 
    
    def SetRefinedCellSize(self, cell_size:float):
        """
        Define the refinement cell size for the table levels.

        :param cell_size: Normalized fine cell size for each 2D table mesh.
        :type cell_size: float
        :raise: Exception: if cell size is lower or equal to zero.
        """
        if cell_size > 0:
            self._refined_cell_size = cell_size 
        else:
            raise Exception("Proviced cell size should be higher than zero.")
        return
    
    def SetRefinementThreshold(self, val_threshold:float):
        """
        Define normalized curvature threshold beyond which refinement should be applied to each table level.

        :param val_threshold: Normalized curvature threshold value. All locations in the mesh with a higher curvature receive refinement.
        :type val_threshold: float
        :raises: Exception: If the threshold value is lower than zero.
        """       

        if val_threshold > 0:
            self._curvature_threshold = val_threshold
        else:
            raise Exception("Curvature threshold value should be higher than zero.")
        return
    
    def SetMixtureFractionLimits(self, mix_frac_min:float, mix_frac_max:float):
        """
        Define the mixture fraction limits of the table.

        :param mix_frac_min: Lower mixture fraction limit.
        :type mix_frac_min: float
        :param mix_frac_max: Upper mixture fraction limit.
        :type mix_frac_max: float
        :raise: Exception: If the upper mixture fraction limit is below the lower mixture fraction limit.
        """   
        
        self._mixfrac_min_table = mix_frac_min 
        self._mixfrac_max_table = mix_frac_max
        self.__PrepareTableLevels()
        return
    
    def InsertMixtureFractionLevel(self, val_mixfrac_level:float):
        self.__table_insert_levels.append(val_mixfrac_level)
        self.__PrepareTableLevels()

    def __PrepareTableLevels(self):
        self._mixfrac_range_table = np.linspace(self._mixfrac_min_table, self._mixfrac_max_table, self._N_table_levels-len(self.__table_insert_levels))
        for z in self.__table_insert_levels:
            self._mixfrac_range_table = np.append(self._mixfrac_range_table, z)
        self._mixfrac_range_table = np.unique(np.sort(self._mixfrac_range_table))
        self._N_table_levels = len(self._mixfrac_range_table)
        return
    
    def SetNCores(self, n_cores:int):
        """Set the number of cores and enable parallel computing of the table level connectivity generation.

        :param n_cores: number of cores to distribute tasks over.
        :type n_cores: int
        :raises Exception: if the number of cores is lower than one.
        """
        if n_cores < 1:
            raise Exception("Number of cores should be at least one.")
        self.__Np_cores = n_cores 
        self.__run_parallel = True 
        return 
    
    def __DefineFlameletDataInterpolator(self):

        print("Configuring KD-tree for most accurate lookups")

        print("Loading flamelet data...")
        # Define scaler for FGM controlling variables.
        full_data_file = self._Config.GetOutputDir()+"/LUT_data_full.csv"
        with open(full_data_file,'r') as fid:
            self._Flamelet_Variables = fid.readline().strip().split(',')
        D_full = np.loadtxt(full_data_file,delimiter=',',skiprows=1)
        self._scaler = MinMaxScaler()
        CV_full = D_full[:,:3]
        self.__min_CV, self.__max_CV = np.min(CV_full,axis=0), np.max(CV_full,axis=0)

        min_mixfrac_dataset = self.__min_CV[2]
        max_mixfrac_dataset = self.__max_CV[2]
        
        self._mixfrac_min_table = min_mixfrac_dataset + 0.1*(max_mixfrac_dataset - min_mixfrac_dataset)
        self._mixfrac_max_table = max_mixfrac_dataset - 0.1*(max_mixfrac_dataset - min_mixfrac_dataset)
        
        CV_full_scaled = self._scaler.fit_transform(CV_full)

        # Exctract train and test data
        train_data_file = self._Config.GetOutputDir()+"/"+self._Config.GetConcatenationFileHeader()+"_train.csv"
        test_data_file = self._Config.GetOutputDir()+"/"+self._Config.GetConcatenationFileHeader()+"_test.csv"
        
        var_to_test_for = "ProdRateTot_PV"
        
        D_train = np.loadtxt(train_data_file,delimiter=',',skiprows=1)
        D_test = np.loadtxt(test_data_file,delimiter=',',skiprows=1)
        
        CV_train = np.vstack(tuple(D_train[:, self._Flamelet_Variables.index(c)] for c in self._controlling_variables)).T 
        CV_test = np.vstack(tuple(D_test[:, self._Flamelet_Variables.index(c)] for c in self._controlling_variables)).T 

        CV_train_scaled = self._scaler.transform(CV_train)
        CV_test_scaled = self._scaler.transform(CV_test)

        PPV_test = D_test[:, self._Flamelet_Variables.index(var_to_test_for)]
        print("Done!")
        print("Setting up KD-tree...")
        self._lookup_tree = Invdisttree(X=CV_train_scaled,z=D_train)
        print("Done!")
        
        if not self._custom_KDtreeparams:
            print("Search for best tree parameters...")
            # Do brute-force search to get the optimum number of nearest neighbors and distance power.
            n_near_range = range(1, 20)
            p_range = range(1, 6)
            RMS_ppv = np.zeros([len(n_near_range), len(p_range)])
            for i in tqdm(range(len(n_near_range))):
                for j in range(len(p_range)):
                    PPV_predicted = self._lookup_tree(q=CV_test_scaled, nnear=n_near_range[i], p=p_range[j])[:, self._Flamelet_Variables.index(var_to_test_for)]
                    rms_local = mean_squared_error(y_true=PPV_test, y_pred=PPV_predicted)
                    RMS_ppv[i,j] = rms_local 
            [imin,jmin] = divmod(RMS_ppv.argmin(), RMS_ppv.shape[1])
            self._n_near = n_near_range[imin]
            self._p_fac = p_range[jmin]
            print("Done!")
        print("Best found number of nearest neighbors: "+str(self._n_near))
        print("Best found distance power: "+str(self._p_fac))
        print("Setting up KD-tree...")
        self._lookup_tree = Invdisttree(X=CV_full_scaled,z=D_full)
        print("Done!")
        return

    def __EvaluateFlameletInterpolator(self, CV_unscaled:np.ndarray):
        CV_scaled = self._scaler.transform(CV_unscaled)
        data_interp = self._lookup_tree(q=CV_scaled,nnear=self._n_near,p=self._p_fac)
        return data_interp 
    
    def VisualizeTableLevel(self, val_mix_frac:float, var_to_plot:str=None):
        """Compute and visualize the table connectivity for a certain mixture fraction value.

        :param val_mix_frac: mixture fraction value for which to compute the table connectivity.
        :type val_mix_frac: float
        :raises Exception: if the mixture fraction value lies outside the flamelet data range.
        """
        
        Tria, Nodes, HullIdx,level_data = self.ComputeTableLevelMesh(val_mix_frac)

        if var_to_plot == None:
            _ = plt.figure(figsize=[10,10])
            ax = plt.axes()
            ax.triplot(Nodes[:, 0], Nodes[:, 1], Tria)
            ax.plot(Nodes[HullIdx, 0], Nodes[HullIdx, 1], 'ko', label=r"Hull nodes")
            ax.set_xlabel(r"Progress Variable $(\mathcal{Y})[-]$", fontsize=20)
            ax.set_ylabel(r"Total Enthalpy $(h)[J kg^{-1}]$", fontsize=20)
            ax.legend(fontsize=20)
            ax.set_title(r"2D table mesh at Z="+str(val_mix_frac))
            plt.show()
        else:
            _ = plt.figure(figsize=[10,10])
            ax = plt.axes(projection='3d')
            ax.plot3D(Nodes[:, 0], Nodes[:, 1], level_data[:, self._Flamelet_Variables.index(var_to_plot)],'k.')
            ax.set_xlabel(r"Progress Variable $(\mathcal{Y})[-]$", fontsize=20)
            ax.set_ylabel(r"Total Enthalpy $(h)[J kg^{-1}]$", fontsize=20)
            ax.legend(fontsize=20)
            ax.set_title(r"Table data at Z="+str(val_mix_frac))
            plt.show()
        return 
    
    def GenerateTableNodes(self):
        """
        Generate the table nodes and connectivity. 
        """
        
        self.__PrepareTableLevels()
        
        self._table_nodes = [None] * self._N_table_levels
        self._table_nodes_norm = [None] * self._N_table_levels
        self._table_connectivity = [None] * self._N_table_levels
        self._table_hullnodes = [None] * self._N_table_levels
        self.table_data = [None] * self._N_table_levels
    
        flamelet_vars = []
        for var in self._Flamelet_Variables:
            flamelet_vars.append(var)
        flamelet_vars.remove("ProgressVariable")
        flamelet_vars.remove("EnthalpyTot")
        flamelet_vars.remove("MixtureFraction")
        if "FlameletID" in self._Flamelet_Variables:
            flamelet_vars.remove("FlameletID")

        self.table_vars = flamelet_vars 
        nVars = len(self.table_vars)

        # Generate the table cells for each table level.
        NTria = 0
        NHull = 0
        NNodes = 0
        if self.__run_parallel:
            pool = Pool(self.__Np_cores)
            results = pool.map(self.ComputeTableNodes, [i for i in range(self._N_table_levels)])
            pool.close()
            for iLevel in range(self._N_table_levels):
                self._table_nodes[iLevel] = results[iLevel][0]
                self._table_connectivity[iLevel] = results[iLevel][1]
                self._table_hullnodes[iLevel] = results[iLevel][2]
                data_interp = results[iLevel][3]
                table_data_level = [None] * nVars
                for iVar in range(nVars):
                    var = self.table_vars[iVar]
                    table_data_level[iVar] = data_interp[:, self._Flamelet_Variables.index(var)]
                self.table_data[iLevel] = table_data_level
                NTria += np.shape(self._table_connectivity[iLevel])[0]
                NHull += np.shape(self._table_hullnodes[iLevel])[0]
                NNodes += np.shape(self._table_nodes[iLevel])[0]
        else:
            for iLevel in range(self._N_table_levels):
                table_data_level = [None] * nVars
                result = self.ComputeTableNodes(iLevel)
                self._table_nodes[iLevel] = result[0]
                self._table_connectivity[iLevel] = result[1]
                self._table_hullnodes[iLevel] = result[2]
                data_interp = result[3]
                for iVar in range(nVars):
                    var = self.table_vars[iVar]
                    table_data_level[iVar] = data_interp[:, self._Flamelet_Variables.index(var)]
                self.table_data[iLevel] = table_data_level

                NTria += np.shape(self._table_connectivity[iLevel])[0]
                NHull += np.shape(self._table_hullnodes[iLevel])[0]
                NNodes += np.shape(self._table_nodes[iLevel])[0]

        NTria_average = int(NTria / self._N_table_levels)
        NHull_average = int(NHull / self._N_table_levels)
        NNodes_average = int(NNodes / self._N_table_levels)
        print("Average number of nodes: %i" % NNodes_average)
        print("Average number of elements: %i" % NTria_average)
        print("Average number of hull nodes: %i" % NHull_average)
        return 
    
    def ComputeTableNodes(self, iLevel:int):
        """Compute the table connectivity for a specific table level.

        :param iLevel: table level index.
        :type iLevel: int
        :raises Exception: if the table level index is not between 0 and the number of table levels.
        :return: table nodes(dimensional), table nodes(normalized), connectivity, hull node indices
        :rtype: list[np.ndarray]
        """
        if iLevel < 0 or iLevel > self._N_table_levels:
            raise Exception("Specified table level out of bounds.")
        
        # Compute the connectivity, normalized node values, and hull indices for the table level at the respective
        #   mixture fraction value.
        Z_Level = self._mixfrac_range_table[iLevel]
        Tria, Nodes_dim, HullIdx, TableDataLevel = self.ComputeTableLevelMesh(Z_Level)

        print("Computed triagulation on level %i out of %i with %i nodes." % (iLevel+1, self._N_table_levels, len(Nodes_dim)))

        return [Nodes_dim, Tria, HullIdx, TableDataLevel]
    
    
    def WriteTableFile(self, output_filepath:str=None):
        """
        Save the table data and connectivity as a Dragon library file. If no file name is provided, the table file will be named according to the FlameletAIConfig class name.

        :param output_filepath: optional output filepath for table file.
        :type output_filepath: str
        """

        if output_filepath:
            file_out = output_filepath
        else:
            file_out = self._savedir + "/LUT_"+self._Config.GetConfigName()+".drg"

        print("Writing LUT file with name " + file_out)
        fid = open(file_out, "w+")
        fid.write("Dragon library\n\n")
        fid.write("<Header>\n\n")
        fid.write("[Version]\n1.1.0\n\n")
        fid.write("[Progress variable definition]\n")
        fid.write("+".join(("%+.4e * %s" % (w, s)) for w, s in zip(self._Config.GetProgressVariableWeights(), self._Config.GetProgressVariableSpecies())) + "\n\n")
        
        fid.write("[Number of table levels]\n%i\n\n" % self._N_table_levels)
        fid.write("[Table levels]\n")
        for z in self._mixfrac_range_table:
            fid.write("%+.16e\n" % z)
        fid.write("\n")

        fid.write("[Number of points]\n")
        for Nodes in self._table_nodes:
            fid.write("%i\n" % np.shape(Nodes)[0])
        fid.write("\n")

        fid.write("[Number of triangles]\n")
        for Elements in self._table_connectivity:
            fid.write("%i\n" % np.shape(Elements)[0])
        fid.write("\n")

        fid.write("[Number of hull points]\n")
        for HullNodes in self._table_hullnodes:
            fid.write("%i\n" % np.shape(HullNodes)[0])
        fid.write("\n")

        fid.write("[Number of variables]\n%i\n\n" % len(self._Flamelet_Variables))
        fid.write("[Variable names]\n")
        for iVar, Var in enumerate(self._Flamelet_Variables):
            fid.write(str(iVar + 1)+":"+Var+"\n")
        fid.write("\n")

        fid.write("</Header>\n\n")

        print("Writing table data...")
        fid.write("<Data>\n")
        for iLevel in tqdm(range(len(self._table_nodes))):
            fid.write("<Level>\n")
            Np = np.shape(self._table_nodes[iLevel])[0]
            for iNode in range(Np):
                fid.write("\t".join("%+.14e" % cv for cv in self._table_nodes[iLevel][iNode, :]))
                for iVar in range(len(self.table_vars)):
                    fid.write("\t%+.14e" % self.table_data[iLevel][iVar][iNode])
                fid.write("\n")
            fid.write("</Level>\n")
        fid.write("</Data>\n\n")
        print("Done!")

        print("Writing table connectivity...")
        fid.write("<Connectivity>\n")
        for iLevel in tqdm(range(len(self._table_connectivity))):
            fid.write("<Level>\n")
            for iCell in range(len(self._table_connectivity[iLevel])):
                fid.write("\t".join("%i" % c for c in self._table_connectivity[iLevel][iCell, :]+1) + "\n")
            fid.write("</Level>\n")
        fid.write("</Connectivity>\n\n")
        print("Done!")

        print("Writing hull nodes...")
        fid.write("<Hull>\n")
        for iLevel in tqdm(range(len(self._table_hullnodes))):
            fid.write("<Level>\n")
            for iCell in range(len(self._table_hullnodes[iLevel])):
                fid.write(("%i" % (self._table_hullnodes[iLevel][iCell]+1)) + "\n")
            fid.write("</Level>\n")
        fid.write("</Hull>\n\n")
        print("Done!")

        fid.close()
        return 
    
    def ComputeTableLevelMesh(self, val_mix_frac:float):
        """
        Compute the table nodes, connectivity, and convex hull node indices of a 2D table level for a given mixture fraction value.

        :param val_mix_frac: Mixture fraction value for which to generate a 2D table.
        :type val_mix_frac: float
        :return Connectivity: Delaunay triangulation connectivity
        :rtype Connecivity: NDarray
        :return MeshNodes: 
        """
        Coord_refinement, Coord_hull, hull_area,z_norm, CV_mesh, table_level_data  = self.__ComputeCurvature(val_mix_frac)
        MeshNodes_Norm, table_level_data = self.__Compute2DMesh(XY_hull=Coord_hull, XY_refinement=Coord_refinement,val_mixfrac_norm=z_norm, level_area=hull_area)
        
        Tria = Delaunay(MeshNodes_Norm[:, :2])
        HullNodes = Tria.convex_hull[:, 0]
        MeshNodes_dim = self._scaler.inverse_transform(MeshNodes_Norm)
        return Tria.simplices, MeshNodes_dim, HullNodes, table_level_data
    
    def __ComputeCurvature(self, val_mix_frac:float):
        """
        Compute the curvature of the reaction rate surface at a constant mixture fraction level. Identify the locations of high curvature where table refinement is required.

        :param val_mix_frac: mixture fraction of current table level.
        :type val_mix_frac: float 
        :return XY_refinement: normalized pv and enth coordinates where refinement should be applied.
        :rtype XY_refinement: array
        :return XY_hull: normalized pv and enth coordinates of the convex hull of the current table level.
        :rtype XY_hull: array
        """

        # 1: Generate initial pv-enth grid.
        self._Config.gas.set_mixture_fraction(val_mix_frac, self._Config.GetFuelString(),self._Config.GetOxidizerString())
        self._Config.gas.TP=self._Config.GetUnbTempBounds()[0],DefaultSettings_FGM.pressure
        h_min_unb = self._Config.gas.enthalpy_mass 

        # Compute reactant progress variable for the current mixture fraction.
        pv_unb = self._Config.ComputeProgressVariable(variables=None, flamelet_data=None, Y_flamelet=self._Config.gas.Y[:,np.newaxis])[0]
        
        # Define maximum enthalpy as the reactant enthalpy at the maximum reactant temperature.
        self._Config.gas.TP=self._Config.GetUnbTempBounds()[1],DefaultSettings_FGM.pressure
        h_max = self._Config.gas.enthalpy_mass

        # Equilibrate at constant enthalpy to get product progress variable value.
        self._Config.gas.equilibrate("HP")
        pv_b = self._Config.ComputeProgressVariable(variables=None, flamelet_data=None, Y_flamelet=self._Config.gas.Y[:,np.newaxis])[0]
        
        # Define minimum enthalpy as the product enthalpy cooled to minimum reactant temperature.
        self._Config.gas.TP=self._Config.GetUnbTempBounds()[0],DefaultSettings_FGM.pressure
        h_min = self._Config.gas.enthalpy_mass 

        # Define 2D grid between minimum and maximum progress variable and total enthalpy
        pv_range = np.linspace(pv_unb, pv_b, 100)
        h_range = np.linspace(h_min, h_max, 100)
        xgrid, ygrid = np.meshgrid(pv_range, h_range)
        zgrid = val_mix_frac*np.ones(np.shape(xgrid))

        # 2: Locate nodes that are above the burner-stabilized enthalpy line
        CV_grid_init = np.vstack((xgrid.flatten(), ygrid.flatten(), zgrid.flatten())).transpose()
        pv_grid = CV_grid_init[:,0]
        h_grid = CV_grid_init[:,1]

        h_limit = ((h_min_unb - h_min) * pv_grid + (h_min*pv_unb - h_min_unb*pv_b))/(pv_unb - pv_b)
        idx_keep = h_grid >= h_limit 

        CV_grid = CV_grid_init[idx_keep, :]
        
        CV_grid_norm_init = self._scaler.transform(CV_grid_init)
        CV_grid_norm = self._scaler.transform(CV_grid)

        # 3: Generate convex hull on initial pv-h grid
        hull = ConvexHull(CV_grid_norm[:, :2])
        x_hull = CV_grid_norm[hull.vertices, 0]
        y_hull = CV_grid_norm[hull.vertices, 1]

        # 4: Locate refinement locations based on pv source term curvature
        Q_interp = self.__EvaluateFlameletInterpolator(CV_unscaled=CV_grid_init)
        ppv_grid = Q_interp[:, self._Flamelet_Variables.index("ProdRateTot_PV")]
        ppv_grid = np.reshape(ppv_grid, np.shape(xgrid))
        idx_ref = self.__ComputeSourceTermCurvature(ppv_grid)

        x_refinement = CV_grid_norm_init[idx_ref, 0]
        y_refinement = CV_grid_norm_init[idx_ref, 1]
    

        # 5: Generate refinement locations at reactant and product progress variable
        h_unb_range = np.linspace(h_min_unb, h_max, self._Config.GetNpTemp())
        pv_unb_range = pv_unb * np.ones(np.shape(h_unb_range))
        z_unb_range = val_mix_frac*np.ones(np.shape(h_unb_range))
        CV_unb = np.hstack((pv_unb_range[:,np.newaxis], h_unb_range[:,np.newaxis],z_unb_range[:,np.newaxis]))
        CV_unb_norm = self._scaler.transform(CV_unb)

        h_b_range = np.linspace(h_min, h_max, self._Config.GetNpTemp())
        pv_b_range = pv_b * np.ones(np.shape(h_b_range))
        z_b_range = val_mix_frac*np.ones(np.shape(h_b_range))
        CV_b = np.hstack((pv_b_range[:,np.newaxis], h_b_range[:,np.newaxis],z_b_range[:,np.newaxis]))
        CV_b_norm = self._scaler.transform(CV_b)

        x_refinement = np.append(x_refinement, CV_unb_norm[:,0])
        x_refinement = np.append(x_refinement, CV_b_norm[:,0])
        y_refinement = np.append(y_refinement, CV_unb_norm[:,1])
        y_refinement = np.append(y_refinement, CV_b_norm[:,1])

        XY_refinement = np.vstack((x_refinement, y_refinement)).T
        XY_hull = np.vstack((x_hull, y_hull)).T

        val_mix_frac_norm = CV_grid_norm[0, -1]
        

        return XY_refinement, XY_hull, hull.area, val_mix_frac_norm, CV_grid, Q_interp
    
    def __ComputeSourceTermCurvature(self, PPV_interp:np.ndarray[float]):
        Q_norm = (PPV_interp - np.min(PPV_interp))/(np.max(PPV_interp) - np.min(PPV_interp))
        dQdy, dQdx = np.gradient(Q_norm)
        dQ_mag = np.sqrt(np.power(dQdy, 2) + np.power(dQdx, 2))
        dQ_norm = dQ_mag / np.max(dQ_mag)
        d2Qdy2, d2Qdx2 = np.gradient(dQ_norm)
        d2Q_mag = np.sqrt(np.power(d2Qdy2, 2) + np.power(d2Qdx2, 2))
        d2Q_norm = d2Q_mag / np.max(d2Q_mag)
        d2Q_norm = d2Q_norm.flatten()
        idx_ref = np.where(d2Q_norm > self._curvature_threshold)
        return idx_ref 
    
    def __Compute2DMesh(self, XY_hull:np.ndarray, XY_refinement:np.ndarray, val_mixfrac_norm:float, level_area:float):
        """
        Generate a 2D mesh for the current table level.

        :param XY_hull: Array containing normalized pv and enth coordinates of the outline of the table level.
        :type XY_hull: NDArray
        :param XY_refinement: Array containing normalized pv and enth coordinates where refinement should be applied.
        :type XY_refinement: NDArray
        :return: mesh nodes of the 2D table mesh.
        :rtype: NDArray
        """
        gmsh.initialize() 

        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.Verbosity", 1)
        gmsh.model.add("table_level")
        factory = gmsh.model.geo

        base_cell_size = self._base_cell_size * level_area
        refined_cell_size = self._refined_cell_size * level_area 
        refinement_radius = self._refinement_radius * np.sqrt(level_area)

        hull_pts = []
        for i in range(int(len(XY_hull)/2)):
            hull_pts.append(factory.addPoint(XY_hull[i, 0], XY_hull[i, 1], 0, base_cell_size))
        hull_pts_2 = [hull_pts[-1]]
        for i in range(int(len(XY_hull)/2), len(XY_hull)):
            hull_pts_2.append(factory.addPoint(XY_hull[i, 0], XY_hull[i, 1], 0, base_cell_size))
        hull_pts_2.append(hull_pts[0])
        embed_pts = []
        for i in range(len(XY_refinement)):
            pt_idx = factory.addPoint(XY_refinement[i, 0], XY_refinement[i, 1], 0, refined_cell_size)
            embed_pts.append(pt_idx)


        hull_curve_1 = factory.addPolyline(hull_pts)
        hull_curve_2 = factory.addPolyline(hull_pts_2)
        
        CL = factory.addCurveLoop([hull_curve_1, hull_curve_2])
        
        surf = factory.addPlaneSurface([CL])
        gmsh.model.addPhysicalGroup(1, [hull_curve_1], name="hull_curve_1")
        gmsh.model.addPhysicalGroup(1, [hull_curve_2], name="hull_curve_2")
        gmsh.model.addPhysicalGroup(2, [surf], name="table_level")
        gmsh.model.geo.synchronize()

        gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(1, "PointsList", embed_pts)
        gmsh.model.mesh.field.setNumber(1, "Sampling", 100)
        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "InField", 1)
        gmsh.model.mesh.field.setNumber(2, "SizeMin", refined_cell_size)
        gmsh.model.mesh.field.setNumber(2, "SizeMax", base_cell_size)
        gmsh.model.mesh.field.setNumber(2, "DistMin", refinement_radius)
        gmsh.model.mesh.field.setNumber(2, "DistMax", 1.5*refinement_radius)

        gmsh.model.mesh.field.add("Min", 7)
        gmsh.model.mesh.field.setNumbers(7, "FieldsList", [2])
        gmsh.model.mesh.field.setAsBackgroundMesh(7)
        
        lc = base_cell_size
        def meshSizeCallback(dim,tag,x,y,z,lc):
            return lc
        
        gmsh.model.mesh.setSizeCallback(meshSizeCallback)
        gmsh.option.setNumber("Mesh.Algorithm", 5)
        gmsh.model.mesh.generate(2)
        nodes = gmsh.model.mesh.getNodes(dim=2, tag=-1, includeBoundary=True, returnParametricCoord=False)[1]
        MeshPoints = np.array([nodes[::3], nodes[1::3]]).T
        gmsh.finalize()

        # Remove mesh nodes that are out of bounds.
        pv_norm, enth_norm = MeshPoints[:, 0], MeshPoints[:, 1]

        mixfrac_norm = val_mixfrac_norm*np.ones(np.shape(pv_norm))
        CV_level_norm = np.vstack((pv_norm, enth_norm, mixfrac_norm)).T 
        CV_level_dim = self._scaler.inverse_transform(CV_level_norm)

        MeshPoints = np.zeros([np.shape(MeshPoints)[0], 3])
        MeshPoints[:, 0] = pv_norm 
        MeshPoints[:, 1] = enth_norm 
        MeshPoints[:, 2] = mixfrac_norm

        table_level_data = self.__EvaluateFlameletInterpolator(CV_level_dim)

        return MeshPoints, table_level_data
    
    def __GetStochMixtureFraction(self):
        fuel_definition = self._Config.GetFuelDefinition()
        fuel_weights = self._Config.GetFuelWeights()
        fuel_string = ",".join(fuel_definition[i] + ":" + str(fuel_weights[i]) for i in range(len(fuel_definition)))

        ox_definition = self._Config.GetOxidizerDefinition()
        ox_weights = self._Config.GetOxidizerWeights()
        ox_string = ",".join(ox_definition[i] + ":" + str(ox_weights[i]) for i in range(len(ox_definition)))

        self._Config.gas.set_equivalence_ratio(1.0, fuel_string, ox_string)
        mixfrac_stoch = self._Config.gas.mixture_fraction(fuel_string, ox_string)
        return mixfrac_stoch
    
    def SaveTableGenerator(self, file_name:str):
        """Save the current TableGenerator object settings such that subsequent tables can be
        generated faster.

        :param file_name: file path and name to which to save the current TableGenerator.
        :type file_name: str
        """
        file = open(self._savedir + "/"+file_name +".tgen", "wb")
        pickle.dump(self, file)
        file.close()

    def Inverse_LookUp_T(self, val_pv, val_mixfrac, val_T, val_h_start=2000):
        CV_array = np.array([[val_pv, val_h_start, val_mixfrac]])
        delta = 1e32
        while np.abs(delta) > 1e-2:
            Q_interp = self.__EvaluateFlameletInterpolator(CV_array)
            val_T_interp = Q_interp[0, self._Flamelet_Variables.index("Temperature")]
            val_cp_interp  = Q_interp[0, self._Flamelet_Variables.index("Cp")]
            delta = val_T - val_T_interp
            delta_h = val_cp_interp * delta 
            CV_array[0,1] += delta_h
        return CV_array[0,1]
    
if __name__ == "__main__":
    config_input_file = sys.argv[-2]
    N_cores = int(sys.argv[-1])
    Config = FlameletAIConfig(config_input_file)
    T = SU2TableGenerator(Config)
    if N_cores > 1:
        T.SetNCores(N_cores)
    T.SetMixtureFractionLimits(mix_frac_min=0.009, mix_frac_max=0.022)
    T.InsertMixtureFractionLevel(0.01446751783896619)
    T.InsertMixtureFractionLevel(0.01447)
    T.InsertMixtureFractionLevel(0.01445)
    T.VisualizeTableLevel(0.01446751783896619)
    T.SetNTableLevels(200)
    T.GenerateTableNodes()
    T.WriteTableFile()
    # #T.InterpolateTableData()
    # T.WriteTableFile()
    # T.SaveTableGenerator("LUT_"+Config.GetConfigName())
