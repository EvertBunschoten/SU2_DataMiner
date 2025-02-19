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
# Version: 2.0.0                                                                              |
#                                                                                             |
#=============================================================================================#

import numpy as np 
from scipy.spatial import ConvexHull, Delaunay
from sklearn.preprocessing import MinMaxScaler,RobustScaler,StandardScaler, QuantileTransformer
import matplotlib.pyplot as plt 
from tqdm import tqdm
import sys,os
from Common.DataDrivenConfig import Config_NICFD
import cantera as ct
import gmsh 
import pickle
from multiprocessing import Pool 
from sklearn.metrics import mean_squared_error
from Common.Interpolators import Invdisttree 
from random import sample 

class SU2TableGenerator:

    _Config:Config_NICFD = None # Config_FGM class from which to read settings.
    _savedir:str

    _mixfrac_min:float = None     # Minimum mixture fraction value of the flamelet data.
    _mixfrac_max:float = None     # Maximum mixture fraction value of the flamelet data.

    _pv_full_norm:np.ndarray[float] = None        # Normalized progress variable values of the flamelet data.
    _enth_full_norm:np.ndarray[float] = None      # Normalized total enthalpy values of the flamelet data.
    _mixfrac_full_norm:np.ndarray[float] = None   # Normalized mixture fraction values of the flamelet data.

    _Fluid_Variables:list[str] = None  # Variable names in the concatenated flamelet data file.
    _Flamelet_Data:np.ndarray[float] = None     # Concatenated flamelet data.

    _custom_table_limits_set:bool = False 
    _mixfrac_min_table:float = None     # Lower mixture fraction limit of the table.
    _mixfrac_max_table:float = None     # Upper mixture fraction limit of the table.

    __run_parallel:bool = False 
    __Np_cores:int = 1 

    _N_table_levels:int = 100   # Number of table levels.
    _mixfrac_range_table:np.ndarray[float] = None   # Mixture fraction values of the table levels.
    _base_cell_size:float = 1e-2#3.7e-3      # Table level base cell size.

    _refined_cell_size:float = 3e-4#2.5e-3#1.5e-3   # Table level refined cell size.
    _refinement_radius:float = 5e-3#5e-2     # Table level radius within which refinement is applied.
    _curvature_threshold:float = 0.15    # Curvature threshold above which refinement is applied.

    _table_nodes = []       # Progress variable, total enthalpy, and mixture fraction node values for each table level.
    _table_nodes_norm = []  # Normalized table nodes for each level.
    _table_connectivity = []    # Table node connectivity per table level.
    _table_hullnodes = []   # Hull node indices per table level.
    __table_insert_levels:list[float] = []

    _controlling_variables:list[str]=["Density",\
                                      "Energy"]  # FGM controlling variables
    _lookup_tree:Invdisttree = None     # KD tree with inverse distance weighted interpolation for flamelet data interpolation.
    _flamelet_data_scaler:MinMaxScaler = None   # Scaler for flamelet data controlling variables.
    _n_near:int = 9     # Number of nearest neighbors from which to evaluate flamelet data.
    _p_fac:int = 3      # Power by which to weigh distances from query point.

    def __init__(self, Config:Config_NICFD, load_file:str=None):
        """
        Initiate table generator class.

        :param Config: Config_FGM object.
        :type Config: Config_FGM
        """

        if load_file:
            # Load an existing TableGenerator object.
            with open(load_file, "rb") as fid:
                loaded_table_generator = pickle.load(fid)
            self.__dict__ = loaded_table_generator.__dict__.copy()
        else:
            # Create new TableGenerator object.
            self._Config = Config 

            self.__DefineFluidDataInterpolator()

        self._savedir = self._Config.GetOutputDir()

        self._table_nodes, self.table_data, self._table_connectivity, self._table_hullnodes = self.__ComputeCurvature()

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot3D(self._table_nodes[:,0],self._table_nodes[:,1], self.table_data[:, self._Fluid_Variables.index("d2sdrho2")],'k.')
        plt.show()
        self.table_vars = ['s','dsdrho_e','dsde_rho','d2sdrho2','d2sdedrho','d2sde2']

        self.WriteTableFile(self._Config.GetOutputDir()+"/LUT_"+self._Config.GetConfigName()+".drg")

    def __DefineFluidDataInterpolator(self):
        print("Configuring KD-tree for most accurate lookups")

        print("Loading fluid data...")
        # Define scaler for FGM controlling variables.
        full_data_file = self._Config.GetOutputDir()+"/"+self._Config.GetConcatenationFileHeader()+"_full.csv"
        with open(full_data_file,'r') as fid:
            self._Fluid_Variables = fid.readline().strip().split(',')
        D_full = np.loadtxt(full_data_file,delimiter=',',skiprows=1)
        self._scaler = MinMaxScaler()
        self.CV_full = np.vstack(tuple(D_full[:, self._Fluid_Variables.index(c)] for c in self._controlling_variables)).T
        self.__min_CV, self.__max_CV = np.min(self.CV_full,axis=0), np.max(self.CV_full,axis=0)

        CV_full_scaled = self._scaler.fit_transform(self.CV_full)

        # Exctract train and test data
        train_data_file = self._Config.GetOutputDir()+"/"+self._Config.GetConcatenationFileHeader()+"_full.csv"
        test_data_file = self._Config.GetOutputDir()+"/"+self._Config.GetConcatenationFileHeader()+"_test.csv"
        
        var_to_test_for = "d2sdrho2"
        
        D_train = np.loadtxt(train_data_file,delimiter=',',skiprows=1)
        D_test = np.loadtxt(test_data_file,delimiter=',',skiprows=1)
        
        CV_train = np.vstack(tuple(D_train[:, self._Fluid_Variables.index(c)] for c in self._controlling_variables)).T 
        CV_test = np.vstack(tuple(D_test[:, self._Fluid_Variables.index(c)] for c in self._controlling_variables)).T 

        CV_train_scaled = self._scaler.transform(CV_train)
        CV_test_scaled = self._scaler.transform(CV_test)

        PPV_test = D_test[:, self._Fluid_Variables.index(var_to_test_for)]
        print("Done!")

        self._lookup_tree = Invdisttree(X=CV_train_scaled,z=D_train)
        
        print("Search for best tree parameters...")
        # Do brute-force search to get the optimum number of nearest neighbors and distance power.
        n_near_range = range(1, 20)
        p_range = range(2, 6)
        RMS_ppv = np.zeros([len(n_near_range), len(p_range)])
        for i in tqdm(range(len(n_near_range))):
            for j in range(len(p_range)):
                PPV_predicted = self._lookup_tree(q=CV_test_scaled, nnear=n_near_range[i], p=p_range[j])[:, self._Fluid_Variables.index(var_to_test_for)]
                rms_local = mean_squared_error(y_true=PPV_test, y_pred=PPV_predicted)
                RMS_ppv[i,j] = rms_local 
        [imin,jmin] = divmod(RMS_ppv.argmin(), RMS_ppv.shape[1])
        self._n_near = n_near_range[imin]
        self._p_fac = p_range[jmin]
        print("Done!")
        print("Best found number of nearest neighbors: "+str(self._n_near))
        print("Best found distance power: "+str(self._p_fac))

    def __EvaluateFluidInterpolator(self, CV_unscaled:np.ndarray):
        CV_scaled = self._scaler.transform(CV_unscaled)
        data_interp = self._lookup_tree(q=CV_scaled,nnear=self._n_near,p=self._p_fac)
        return data_interp
    
    def ComputeTableMesh(self):
        return
    
    def __ComputeCurvature(self):
        rho_min = self.__min_CV[self._controlling_variables.index("Density")]
        rho_max = self.__max_CV[self._controlling_variables.index("Density")]
        e_min = self.__min_CV[self._controlling_variables.index("Energy")]
        e_max = self.__max_CV[self._controlling_variables.index("Energy")]

        rho_range = (rho_min - rho_max)* (np.cos(np.linspace(0, 0.5*np.pi, 800))) + rho_max
        e_range = np.linspace(e_min, e_max, 100)
        xgrid, ygrid = np.meshgrid(rho_range, e_range)

        CV_probe = self.CV_full
        probe_data = self.__EvaluateFluidInterpolator(CV_probe)

        DT = Delaunay(self.CV_full)
        Tria = DT.simplices 
        HullNodes = DT.convex_hull[:,0]
        
        return CV_probe, probe_data, Tria, HullNodes

        # fig = plt.figure()
        # ax = plt.axes(projection='3d')
        # ax.plot3D(CV_probe[:,0],CV_probe[:,1],probe_data[:,self._Fluid_Variables.index("d2sdrho2")],'k.')
        # ax.plot3D(CV_probe[HullNodes, 0], CV_probe[HullNodes, 1], probe_data[HullNodes,self._Fluid_Variables.index("d2sdrho2")],'r.')
        # plt.show()
        # plt.triplot(CV_probe[:,0],CV_probe[:,1],DT.simplices)
        # plt.show()
        

        # rho_probe = probe_data[:, self._Fluid_Variables.index("Density")]
        # e_probe = probe_data[:, self._Fluid_Variables.index("Energy")]

        # test_points = np.hstack((rho_probe[:,np.newaxis], e_probe[:,np.newaxis]))
        # CV_unique, idx_unique = np.unique(test_points,axis=0,return_index=True)

        # CV_vals_norm = self._scaler.transform(CV_probe)

        # hull = ConvexHull(CV_vals_norm)
        # s_grid = probe_data[:, self._Fluid_Variables.index("s")]
        # s_grid = np.reshape(s_grid, np.shape(xgrid))

        # idx_ref = self.__ComputeEntropyCurvature(s_grid)
        # x_refinement = CV_vals_norm[idx_ref, 0]
        # y_refinement = CV_vals_norm[idx_ref, 1]
        # x_hull = CV_vals_norm[hull.vertices, 0]
        # y_hull = CV_vals_norm[hull.vertices, 1]
        
        # XY_refinement = np.vstack((x_refinement, y_refinement)).T
        # XY_hull = np.vstack((x_hull, y_hull)).T

        # return XY_refinement, XY_hull, hull.area
    
    def __ComputeEntropyCurvature(self, s_interp:np.ndarray[float]):
        Q_norm = (s_interp - np.min(s_interp))/(np.max(s_interp) - np.min(s_interp))
        dQdy, dQdx = np.gradient(Q_norm)
        dQ_mag = np.sqrt(np.power(dQdy, 2) + np.power(dQdx, 2))
        dQ_norm = dQ_mag / np.max(dQ_mag)
        d2Qdy2, d2Qdx2 = np.gradient(dQ_norm)
        d2Q_mag = np.sqrt(np.power(d2Qdy2, 2) + np.power(d2Qdx2, 2))
        d2Q_norm = d2Q_mag / np.max(d2Q_mag)
        d2Q_norm = d2Q_norm.flatten()
        idx_ref = np.where(d2Q_norm > self._curvature_threshold/10)
        return idx_ref 
    
    def __Compute2DMesh(self, XY_hull:np.ndarray, XY_refinement:np.ndarray, level_area:float):
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

        CV_level_norm = np.vstack((pv_norm, enth_norm)).T 
        CV_level_dim = self._scaler.inverse_transform(CV_level_norm)

        MeshPoints = np.zeros([np.shape(MeshPoints)[0], 2])
        MeshPoints[:, 0] = pv_norm 
        MeshPoints[:, 1] = enth_norm 

        table_level_data = self.__EvaluateFluidInterpolator(CV_level_dim)

        return MeshPoints, table_level_data
    
    def WriteTableFile(self, output_filepath:str=None):
        """
        Save the table data and connectivity as a Dragon library file. If no file name is provided, the table file will be named according to the Config_FGM class name.

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
        fid.write("[Version]\n1.0.1\n\n")

        fid.write("[Number of points]\n")
        fid.write("%i\n" % np.shape(self._table_nodes)[0])
        fid.write("\n")

        fid.write("[Number of triangles]\n")
        fid.write("%i\n" % np.shape(self._table_connectivity)[0])
        fid.write("\n")

        fid.write("[Number of hull points]\n")
        fid.write("%i\n" % np.shape(self._table_hullnodes)[0])
        fid.write("\n")

        fid.write("[Number of variables]\n%i\n\n" % (len(self.table_vars)+2))
        fid.write("[Variable names]\n")
        for iVar, Var in enumerate(self._controlling_variables + self.table_vars):
            fid.write(str(iVar + 1)+":"+Var+"\n")
        fid.write("\n")

        fid.write("</Header>\n\n")

        print("Writing table data...")
        fid.write("<Data>\n")
        for iNode in range(np.shape(self._table_nodes)[0]):
            fid.write("\t".join("%+.14e" % cv for cv in self._table_nodes[iNode, :]))
            for var in self.table_vars:
                fid.write("\t%+.14e" % self.table_data[:, self._Fluid_Variables.index(var)][iNode])
            fid.write("\n")
        fid.write("</Data>\n\n")
        print("Done!")

        print("Writing table connectivity...")
        fid.write("<Connectivity>\n")
        for iCell in range(len(self._table_connectivity)):
            fid.write("\t".join("%i" % c for c in self._table_connectivity[iCell, :]+1) + "\n")
        fid.write("</Connectivity>\n\n")
        print("Done!")

        print("Writing hull nodes...")
        fid.write("<Hull>\n")
        for iCell in range(len(self._table_hullnodes)):
            fid.write(("%i" % (self._table_hullnodes[iCell]+1)) + "\n")
        fid.write("</Hull>\n\n")
        print("Done!")

        fid.close()
