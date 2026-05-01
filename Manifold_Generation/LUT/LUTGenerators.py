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
# Version: 3.1.0                                                                              |
#                                                                                             |
#=============================================================================================#

import numpy as np 
from Common.Properties import EntropicVars,DefaultSettings_NICFD
from su2dataminer.generate_data import DataGenerator_CoolProp
from scipy.spatial import Delaunay
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from Common.DataDrivenConfig import Config_NICFD
import gmsh 
from concave_hull import concave_hull, concave_hull_indexes
import meshio
import matplotlib.pyplot as plt 

def shoelace(XY:np.ndarray[float]):
    """Shoelace algorithm for area computations

    :param XY: hull node coordinates
    :type XY: np.ndarray[float]
    :return: area of concave hull
    :rtype: float
    """
    x = XY[:,0]
    y = XY[:,1]
    S1 = np.sum(x*np.roll(y,-1))
    S2 = np.sum(y*np.roll(x,-1))

    area = .5*np.absolute(S1 - S2)
    return area

def FiniteDifferenceDerivative(y:np.ndarray[float], x:np.ndarray[float]):
    """Calculate second-order accurate, one-dimensional finite-difference derivatives of y with respect to x.

    :param y: data to calculate the finite-differences for.
    :type y: np.ndarray[float]
    :param x: axial coordinates.
    :type x: np.ndarray[float]
    :return: finite-difference derivatives of y with respect to x.
    :rtype: np.ndarray[float]
    """
    Np = len(x)
    dydx = np.zeros(Np)
    for i in range(1, Np-1):
        y_m = y[i-1]
        y_p = y[i+1]
        y_0 = y[i]
        x_m = x[i-1]
        x_p = x[i+1]
        x_0 = x[i]
        dx_1 = x_p - x_0 
        dx_2 = x_0 - x_m 
        dx2_1 = dx_1*dx_1 
        dx2_2 = dx_2*dx_2
        if (dx_1==0) or (dx_2==0):
            dydx[i] = 0.0
        else:
            dydx[i] = (dx2_2 * y_p + (dx2_1 - dx2_2)*y_0 - dx2_1*y_m)/(dx_1*dx_2*(dx_1+dx_2))
    dx_1 = x[1] - x[0]
    dx_2 = x[2] - x[0]
    dx2_1 = dx_1*dx_1 
    dx2_2 = dx_2*dx_2 
    y_0 = y[0]
    y_p = y[1]
    y_pp = y[2]
    if (dx_1==0) or (dx_2==0):
        dydx[0] = 0.0
    else:
        dydx[0] = (dx2_1 * y_pp + (dx2_2 - dx2_1)*y_0 - dx2_2*y_p)/(dx_1*dx_2*(dx_1 - dx_2))

    dx_1 = x[-2] - x[-1]
    dx_2 = x[-3] - x[-1]
    dx2_1 = dx_1*dx_1 
    dx2_2 = dx_2*dx_2 
    y_0 = y[-1]
    y_p = y[-2]
    y_pp = y[-3]
    if (dx_1==0) or (dx_2==0):
        dydx[-1] = 0.0
    else:
        dydx[-1] = (dx2_1 * y_pp + (dx2_2 - dx2_1)*y_0 - dx2_2*y_p)/(dx_1*dx_2*(dx_1 - dx_2))
    return dydx 

class SU2TableGenerator_NICFD:
    _Config:Config_NICFD = None # Config_FGM class from which to read settings.
    _DataGenerator:DataGenerator_CoolProp = None 
    _savedir:str

    _base_cell_size:float = 2e-2      # Table level base cell size.

    _refined_cell_size:float = 5e-3#2.5e-3#1.5e-3   # Table level refined cell size.
    _refinement_radius:float = 1e-2#5e-2     # Table level radius within which refinement is applied.
    __refinement_vars:list[str] = []
    __refinement_norm_min:list[float] = []
    __refinement_norm_max:list[float] = []
    _table_vars:list[str] = [s.name for s in EntropicVars][:-1]
    _table_nodes = []       # Progress variable, total enthalpy, and mixture fraction node values for each table level.
    _table_nodes_norm = []  # Normalized table nodes for each level.
    _table_connectivity = []    # Table node connectivity per table level.
    _table_hullnodes = []   # Hull node indices per table level.

    _controlling_variables:list[str]=["Density",\
                                      "Energy"]  # FGM controlling variables
    _fluid_data_scaler:MinMaxScaler= MinMaxScaler()  # Scaler for flamelet data controlling variables.

    def __init__(self, Config:Config_NICFD):
        """
        Initiate table generator class. Settings regarding the fluid data generation and table resolution are automatically retrieved from the configuration object.

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

        print("Searching for best tree parameters...")
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

    def SetDensityBounds(self, Rho_lower:float=DefaultSettings_NICFD.Rho_min, Rho_upper:float=DefaultSettings_NICFD.Rho_max):
        """Define the density bounds of the density-energy based fluid data grid.

        :param Rho_lower: lower limit density value, defaults to DefaultSettings_NICFD.Rho_min
        :type Rho_lower: float, optional
        :param Rho_upper: upper limit for density, defaults to DefaultSettings_NICFD.Rho_max
        :type Rho_upper: float, optional
        :raises Exception: if lower value for density exceeds upper value.
        """
        self._DataGenerator.UseAutoRange(False)
        self._DataGenerator.SetDensityBounds(Rho_lower, Rho_upper)
        return 
    
    def SetEnergyBounds(self, E_lower:float=DefaultSettings_NICFD.Energy_min, E_upper:float=DefaultSettings_NICFD.Energy_max):
        """Define the internal energy bounds of the density-energy based fluid data grid.

        :param E_lower: lower limit internal energy value, defaults to DefaultSettings_NICFD.Energy_min
        :type E_lower: float, optional
        :param E_upper: upper limit for internal energy, defaults to DefaultSettings_NICFD.Energy_max
        :type E_upper: float, optional
        :raises Exception: if lower value for internal energy exceeds upper value.
        """
        self._DataGenerator.UseAutoRange(False)
        self._DataGenerator.SetEnergyBounds(E_lower, E_upper)
        return 
    def SetCellSize_Coarse(self, cell_size_coarse:float=1e-2):
        """Specify the coarse level cell size of the table

        :param cell_size_coarse: coarse cell size, defaults to 1e-2
        :type cell_size_coarse: float, optional
        :raises Exception: if specified cell size is negative or zero
        """
        if cell_size_coarse <= 0:
            raise Exception("Cell size value should be positive")
        self._base_cell_size = cell_size_coarse 
        return 
    
    def SetCellSize_Refined(self, cell_size_ref:float=5e-3):
        """Specify the refined level cell size of the table

        :param cell_size_ref: refined cell size, defaults to 1e-2
        :type cell_size_ref: float, optional
        :raises Exception: if specified cell size is negative or zero
        """
        if cell_size_ref <= 0:
            raise Exception("Cell size value should be positive")
        self._refined_cell_size = cell_size_ref 
        return 
    
    def SetRefinement_Radius(self, refinement_radius:float=1e-2):
        """Specify the radius around each refinement point within which the refined cell size is applied

        :param refinement_radius: refinement radius, defaults to 1e-2
        :type refinement_radius: float, optional
        :raises Exception: if specified value is negative or zero
        """
        if refinement_radius <= 0:
            raise Exception("Refinement radius should be positive")
        self._refinement_radius = refinement_radius
        return 
    
    def SetTableDiscretization(self, method:str=DefaultSettings_NICFD.tabulation_method):
        """Overwrite the thermodynamic state space discretization method from the configuration.

        :param method: discratization method, defaults to 'cartesian'
        :type method: str, optional
        """
        self._Config.SetTableDiscretization(method)
        return 
    
    def SetTableVars(self, table_vars_in:list[str]):
        """Specify the thermophysical variables to be included in the table file. All quantities are included by default. The list shoud at least contain "Density" and "Energy".
        
        :param table_vars_in: list with thermophysical variables to be included in the table.
        :type table_vars_in: list[str]
        :raises Exception: if any of the specified variables are not supported by SU2 DataMiner.
        """
        self._table_vars = []
        if EntropicVars.Density.name not in table_vars_in:
            print("Density should always be included in table variables")
            self._table_vars.append(EntropicVars.Density.name)

        if EntropicVars.Energy.name not in table_vars_in:
            print("Energy should always be included in table variables")
            self._table_vars.append(EntropicVars.Energy.name)
        
        if self._Config.EnableTwophase() and EntropicVars.VaporQuality.name in table_vars_in:
            print("Table generator not configured for two-phase, ignoring vapor quality from table data.")
            table_vars_in.remove(EntropicVars.VaporQuality.name)
        
        if not self._Config.CalcTransportProperties():
            if EntropicVars.Conductivity.name in table_vars_in:
                print("Table generator not configured for transport properties, ignoring conductivity data")
            if EntropicVars.ViscosityDyn.name in table_vars_in:
                print("Table generator not configured for transport properties, ignoring viscosity data")
            
        valid_vars = True 
        for v in table_vars_in:
            found_var = False
            for q in EntropicVars:
                if v.lower() == q.name.lower():
                    found_var = True
                    self._table_vars.append(q.name)
            if not found_var:
                print("Error, \"%s\" is not supported by SU2 DataMiner" % v)
                valid_vars = False 
        if not valid_vars:
            raise Exception("Some specified thermophysical variables are not supported.")
        return 
    
    def __Compute2DMesh(self, points:np.ndarray[float], ref_pts:np.ndarray[float]=[],show:bool=False,sat_curve_pts:np.ndarray[float]=[]):
        """Populate two-dimensional thermodynamic state space with table nodes according to refinement settings.

        :param points: initial point cloud.
        :type points: np.ndarray[float]
        :param ref_pts: locations to apply refinement to, defaults to []
        :type ref_pts: np.ndarray[float], optional
        :param show: show the discretization generated by Gmesh, defaults to False
        :type show: bool, optional
        :param sat_curve_pts: saturation curve points, defaults to []
        :type sat_curve_pts: np.ndarray[float], optional
        :return: table nodes, table connectivity.
        :rtype: tuple
        """
        # Create concave hull of normalized table coordinates.
        add_sat_curve = (len(sat_curve_pts) > 0)
        XY_hull = concave_hull(np.unique(points,axis=0), length_threshold=self._base_cell_size)
        nans = np.isnan(XY_hull).all(1)
        XY_hull = XY_hull[np.invert(nans), :]

        i = 0
        hull_indices = [i]
        while i < (len(XY_hull)-1):
            i_next = i+1
            found_next_pt = False 
            while not found_next_pt:
                dist = np.sqrt(np.sum(np.power(XY_hull[i_next, :] - XY_hull[i, :], 2)))
                if (dist >= self._base_cell_size) or (i_next == len(XY_hull)-1):
                    found_next_pt = True 
                else:
                    i_next += 1
            i = i_next
            hull_indices.append(i_next)
        XY_hull = XY_hull[hull_indices, :]
        
        # Initiate gmsh
        gmsh.initialize() 
        gmsh.model.add("table_level")
        gmsh.option.setNumber("General.Verbosity", 0)
        factory = gmsh.model.geo

        # Create hull points
        hull_pts = []
        for i in range(int(len(XY_hull))):
            hull_pts.append(factory.addPoint(XY_hull[i, 0], XY_hull[i, 1], 0, self._base_cell_size))
        
        # Connect hull points to a closed multi-component curve
        hull_lines = []
        for i in range(len(hull_pts)-1):
            hull_lines.append(factory.addLine(hull_pts[i], hull_pts[i+1]))
        hull_lines.append(factory.addLine(hull_pts[-1], hull_pts[0]))

        # Create a 2D plane of the enclosed space
        curvloop = factory.addCurveLoop(hull_lines)

        # Apply refinement points
        area_ref = shoelace(concave_hull(XY_hull, length_threshold=self._base_cell_size))
        ref_pt_ids = []
        if len(ref_pts)>0:
            for i in range(len(ref_pts)):
                XY_with_pt = np.vstack((XY_hull, ref_pts[i,:]))
                hull_n = concave_hull(XY_with_pt, length_threshold=self._base_cell_size)
                area_n = shoelace(hull_n)
                within_hull = (area_n <= area_ref)
                if within_hull:
                    ref_pt_ids.append(factory.addPoint(ref_pts[i,0], ref_pts[i, 1], 0.0))

        factory.synchronize()
        
        if add_sat_curve:

            # Create normal vector to saturation curve.
            dedrho_sat_norm = FiniteDifferenceDerivative(sat_curve_pts[:,0], sat_curve_pts[:,1])
            norm_vector = np.column_stack((-1.0 / dedrho_sat_norm, np.ones(len(dedrho_sat_norm))))
            norm_vector = norm_vector / np.sqrt(np.sum(np.power(norm_vector, 2), axis=1))[:,np.newaxis]

            # Create offset curves to ensure that no nodes are generated on the saturation curve itself.
            sat_curve_upper_pts = []
            sat_curve_lower_pts = []
            dx = 1e-3*self._refined_cell_size

            sat_curve_rhoe_upper = sat_curve_pts + dx * norm_vector
            sat_curve_rhoe_lower = sat_curve_pts - dx * norm_vector
            inside_upper = np.logical_and(sat_curve_rhoe_upper > 0, sat_curve_rhoe_upper < 1).all(1)
            inside_lower = np.logical_and(sat_curve_rhoe_lower > 0, sat_curve_rhoe_lower< 1).all(1)
            valid_sat_curve_pts = np.logical_and(inside_upper, inside_lower)
            nans_lower = np.isnan(sat_curve_rhoe_lower).all(1)
            nans_upper = np.isnan(sat_curve_rhoe_upper).all(1)
            valid_nans = np.logical_and(np.invert(nans_lower), np.invert(nans_upper))
            valid_pts = np.logical_and(valid_sat_curve_pts, valid_nans)

            # Clip the saturation curve points to the bounds of the thermodynamic mesh
            
            within_hull = np.zeros(len(sat_curve_pts),dtype=np.bool)
            for i in range(len(sat_curve_pts)):
                XY_with_pt = np.vstack((XY_hull, sat_curve_rhoe_upper[i,:]))
                hull_n = concave_hull(XY_with_pt, length_threshold=self._base_cell_size)
                area_n = shoelace(hull_n)
                within_hull_upper = (area_n <= area_ref)
                XY_with_pt = np.vstack((XY_hull, sat_curve_rhoe_lower[i,:]))
                hull_n = concave_hull(XY_with_pt, length_threshold=self._base_cell_size)
                area_n = shoelace(hull_n)
                within_hull_lower = (area_n <= area_ref)
                within_hull[i] = (within_hull_upper and within_hull_lower)
            valid_pts = np.logical_and(valid_pts, within_hull)
            
            sat_curve_pts = sat_curve_pts[valid_pts, :]
            norm_vector = norm_vector[valid_pts, :]

            i = 0
            j = 1
            sat_curve_upper_pts.append(factory.addPoint(sat_curve_pts[i,0] + dx*norm_vector[i, 0],\
                                                        sat_curve_pts[i,1] + dx*norm_vector[i, 1],0))
            sat_curve_lower_pts.append(factory.addPoint(sat_curve_pts[i,0] - dx*norm_vector[i, 0],\
                                                        sat_curve_pts[i,1] - dx*norm_vector[i, 1],0))
            dists = []
            while j < len(sat_curve_pts):
                dist = np.sqrt(np.sum(np.power(sat_curve_pts[j,:] - sat_curve_pts[i,:],2)))
                if dist < 0.6*self._refined_cell_size:
                    j += 1 
                else:
                    i = j 
                    j += 1 
                    dists.append(dist)
                    sat_curve_upper_pts.append(factory.addPoint(sat_curve_pts[i,0] + dx*norm_vector[i, 0],\
                                                                sat_curve_pts[i,1] + dx*norm_vector[i, 1],0))
                    sat_curve_lower_pts.append(factory.addPoint(sat_curve_pts[i,0] - dx*norm_vector[i, 0],\
                                                                sat_curve_pts[i,1] - dx*norm_vector[i, 1],0))
            sat_curve_connecting_lines = []
            for i in range(len(sat_curve_upper_pts)):
                sat_curve_connecting_lines.append(factory.addLine(sat_curve_lower_pts[i], sat_curve_upper_pts[i]))
            sat_curve_upper_lines = []
            sat_curve_lower_lines = []

            sat_curve_crvloops = []
            sat_curve_cornertags = []
            for i in range(len(sat_curve_lower_pts)-1):
                if dists[i] < self._base_cell_size:
                    c_upper=factory.addLine(sat_curve_upper_pts[i],sat_curve_upper_pts[i+1])
                    c_lower=factory.addLine(sat_curve_lower_pts[i],sat_curve_lower_pts[i+1])
                    sat_curve_upper_lines.append(c_upper)
                    sat_curve_lower_lines.append(c_lower)
                    sat_curve_crvloops.append(factory.addCurveLoop([c_upper, -sat_curve_connecting_lines[i+1], -c_lower, sat_curve_connecting_lines[i]]))
                    sat_curve_cornertags.append([sat_curve_lower_pts[i], sat_curve_lower_pts[i+1], sat_curve_upper_pts[i+1], sat_curve_upper_pts[i]])
            factory.synchronize()
            fluid_plane_crvloops = [curvloop] + [-c for c in sat_curve_crvloops]
            fluid_surf = factory.addPlaneSurface(fluid_plane_crvloops)
            sat_surfs = [factory.addPlaneSurface([c]) for c in sat_curve_crvloops]
            factory.addPhysicalGroup(2, [fluid_surf] + sat_surfs)
        else:
            factory.synchronize()
            fluid_surf = factory.addPlaneSurface([curvloop])
            factory.addPhysicalGroup(2, [fluid_surf])
        # Apply conditional refinement, where the refined cell size is applied in proximity to the refinement points
        factory.synchronize()
        threshold_fields = []
        dist_field_ref_pt = gmsh.model.mesh.field.add("Attractor")
        gmsh.model.mesh.field.setNumbers(dist_field_ref_pt, "NodesList", ref_pt_ids)
        t_field_ref_pt = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(t_field_ref_pt, "InField", dist_field_ref_pt)
        gmsh.model.mesh.field.setNumber(t_field_ref_pt, "SizeMin", self._refined_cell_size)
        gmsh.model.mesh.field.setNumber(t_field_ref_pt, "SizeMax", self._base_cell_size)
        gmsh.model.mesh.field.setNumber(t_field_ref_pt, "DistMin", 0.5*self._refinement_radius)
        gmsh.model.mesh.field.setNumber(t_field_ref_pt, "DistMax", 1.5*self._refinement_radius)
        threshold_fields.append(t_field_ref_pt)

        if add_sat_curve:
            dist_field_sat_crv = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(dist_field_sat_crv, \
                                             "PointsList", sat_curve_upper_pts + sat_curve_lower_pts)
            gmsh.model.mesh.field.setNumber(dist_field_sat_crv, "Sampling", 10)
            t_field_sat_crv = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(t_field_sat_crv, "InField", dist_field_sat_crv)
            gmsh.model.mesh.field.setNumber(t_field_sat_crv, "SizeMin", self._refined_cell_size)
            gmsh.model.mesh.field.setNumber(t_field_sat_crv, "SizeMax", self._base_cell_size)
            gmsh.model.mesh.field.setNumber(t_field_sat_crv, "DistMin", 0.5*self._refinement_radius)
            gmsh.model.mesh.field.setNumber(t_field_sat_crv, "DistMax", 1.5*self._refinement_radius)
            threshold_fields.append(t_field_sat_crv)

        back_field = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(back_field, "FieldsList", threshold_fields)
        gmsh.model.mesh.field.setAsBackgroundMesh(back_field)

        factory.synchronize()
        if add_sat_curve:
            for i in range(len(sat_curve_connecting_lines)):
                gmsh.model.mesh.setTransfiniteCurve(sat_curve_connecting_lines[i], 2)
            for i in range(len(sat_curve_lower_lines)):
                gmsh.model.mesh.setTransfiniteCurve(sat_curve_lower_lines[i], 2)
                gmsh.model.mesh.setTransfiniteCurve(sat_curve_upper_lines[i], 2)
                
            for s, c in zip(sat_surfs, sat_curve_cornertags):
                gmsh.model.mesh.setTransfiniteSurface(s, cornerTags=c)
                gmsh.model.mesh.setRecombine(2, s)
        # Generate 2D mesh and extract table nodes
        gmsh.model.mesh.generate(2)

        if show:
            gmsh.fltk.run()
        # Global nodes
        nodeTags, coords, _ = gmsh.model.mesh.getNodes()  
        nodeTags = np.asarray(nodeTags, dtype=np.int64)
        MeshPoints = np.asarray(coords, dtype=float).reshape(-1, 3)[:, :2]

        order = np.argsort(nodeTags)
        nodeTags_sorted = nodeTags[order]

        # 2) 2D elements
        if fluid_surf is None:
            elemTypes, _, elemNodeTags = gmsh.model.mesh.getElements(2)
        else:
            elemTypes, _, elemNodeTags = gmsh.model.mesh.getElements(2, fluid_surf)

        tris = []
        quads = []

        for et, nodes_flat in zip(elemTypes, elemNodeTags):
            if et == 2:  # triangles with 3 nodes
                tri_tags = np.asarray(nodes_flat, dtype=np.int64).reshape(-1, 3)
                tris.append(self.__map_tags(tri_tags, nodeTags_sorted,order).reshape(-1, 3))
            elif et == 3:  # quad with 4 nodes
                quad_tags = np.asarray(nodes_flat, dtype=np.int64).reshape(-1, 4)
                quads.append(self.__map_tags(quad_tags, nodeTags_sorted,order).reshape(-1, 4))

        tris = np.vstack(tris) if tris else np.zeros((0, 3), dtype=np.int64)

        if quads:
            quads = np.vstack(quads)
            # split quad -> 2 tri: (0,1,2) + (0,2,3)
            tris = np.vstack([
                tris,
                quads[:, [0, 1, 2]],
                quads[:, [0, 2, 3]],
            ])
        gmsh.finalize()
        
        return MeshPoints, tris
    
    def __map_tags(self,tags,nodeTags_sorted,order):
        tags = np.asarray(tags, dtype=np.int64).ravel()
        pos = np.searchsorted(nodeTags_sorted, tags)
        ok = (pos < len(nodeTags_sorted)) & (nodeTags_sorted[pos] == tags)
        if not np.all(ok):
            missing = np.unique(tags[~ok])
            raise RuntimeError(f"Node tags non trovati in getNodes(): {missing[:20]} (tot missing={len(missing)})")
        return order[pos]
    
    def __CalcMeshData(self, fluid_data_mesh:np.ndarray[float]):
        """Calculate the fluid thermodynamic state variables for the table nodes

        :param fluid_data_mesh: table mesh nodes of density and static energy
        :type fluid_data_mesh: np.ndarray[float]
        :return: filtered thermodynamic state data at the table nodes
        :rtype: np.ndarray[float]
        """
        fluid_data_out = fluid_data_mesh.copy()
        self.valid_mask = np.zeros(len(fluid_data_mesh),dtype=np.bool)
        for i in tqdm(range(len(fluid_data_mesh)),desc="Evaluating fluid properties on table nodes..."):
            try:
                self._DataGenerator.UpdateFluid(fluid_data_mesh[i, EntropicVars.Density.value], fluid_data_mesh[i, EntropicVars.Energy.value])
                state_vector, correct_phase = self._DataGenerator.GetStateVector()
                if correct_phase:
                    fluid_data_out[i, :] = state_vector
                    self.valid_mask[i] = True
                else:
                    fluid_data_out[i, :] = None
            except:
                fluid_data_out[i, :] = None
        fluid_data_out = fluid_data_out[self.valid_mask,:]
        return fluid_data_out
    
    def __CartesianTableData(self):
        print("Generating table on Cartesian grid")
        Np_rho = self._Config.GetNpDensity()
        Np_e = self._Config.GetNpEnergy()
        self._DataGenerator.PreprocessData()
        if self._Config.GetAutoRange():
            rho_min, rho_max = self._DataGenerator.GetDensityBounds()
            e_min, e_max = self._DataGenerator.GetEnergyBounds()
        else:
            rho_minmax = self._Config.GetDensityBounds()
            rho_min = rho_minmax[0]
            rho_max = rho_minmax[1]
            e_minmax = self._Config.GetEnergyBounds()
            e_min = e_minmax[0]
            e_max = e_minmax[1]
        rho_range = np.linspace(rho_min, rho_max, Np_rho)
        e_range = np.linspace(e_min, e_max, Np_e)
        self.rho_grid, self.e_grid = np.meshgrid(rho_range, e_range)

        print(f"Grid Configuration:")
        print(f"  Density: [{rho_min:.2f}, {rho_max:.2f}] kg/m3 ({Np_rho} points)")
        print(f"  Energy:  [{e_min:.0f}, {e_max:.0f}] J/kg ({Np_e} points)")
        print(f"  Total grid points: {Np_rho * Np_e:,}")
        print()

        shape = self.rho_grid.shape
        n_points = shape[0] * shape[1]

        # Initialize storage arrays
        self.state_data = np.zeros([shape[0], shape[1], EntropicVars.N_STATE_VARS.value])

        # Validity mask
        self.valid_mask = np.zeros(shape, dtype=bool)

        # Flatten for iteration
        rho_flat = self.rho_grid.flatten()
        e_flat = self.e_grid.flatten()

        success_count = 0
        for i in tqdm(range(n_points), desc="Evaluating"):
            rho = rho_flat[i]
            e = e_flat[i]
            idx_2d = np.unravel_index(i, shape)
            try:
                self._DataGenerator.UpdateFluid(rho, e)
                state_data, correct_phase = self._DataGenerator.GetStateVector()
                if correct_phase:
                    self.state_data[idx_2d[0], idx_2d[1], :] = state_data 
                    success_count += 1
                    self.valid_mask[idx_2d] = True
                else:
                    self.state_data[idx_2d[0], idx_2d[1], :] = None
            except:
                self.state_data[idx_2d[0], idx_2d[1], :] = None
        return 
    
    def __CartesianTriangulation(self):
        """
        Create Delaunay triangulation of valid grid points.
        """
        print("Creating Delaunay triangulation...")

        # Extract valid points
        rho_table = self.state_data[:,:,EntropicVars.Density.value]
        e_table = self.state_data[:,:,EntropicVars.Energy.value]
        rho_valid = rho_table[self.valid_mask].flatten()
        e_valid = e_table[self.valid_mask].flatten()
        
        # Stack as (N, 2) array
        cv_table = np.column_stack([rho_valid, e_valid])

        #self._table_nodes = np.column_stack(tuple(self.state_data[:,:,EntropicVars[v].value][self.valid_mask].flatten() for v in self._table_vars))
        self._table_nodes = np.column_stack(tuple(self.state_data[:,:,i][self.valid_mask].flatten() for i in range(EntropicVars.N_STATE_VARS.value)))
            
        # Create Delaunay triangulation
        tri = Delaunay(cv_table)
        self._table_connectivity = tri.simplices

        # Identify hull nodes
        edges = np.vstack([tri.simplices[:, [0, 1]],
                           tri.simplices[:, [1, 2]],
                           tri.simplices[:, [2, 0]]])
        edges = np.sort(edges, axis=1)
        unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
        boundary_edges = unique_edges[counts == 1]
        self._table_hullnodes= np.unique(boundary_edges.flatten())

        print(f"  Triangulation nodes: {len(self._table_nodes):,}")
        print(f"  Triangles: {len(self._table_connectivity):,}")
        print(f"  Hull nodes: {len(self._table_hullnodes):,}")
        print()
        return 

    def __CreateSaturationCurve(self):
        rhoe_sat_curve = self._DataGenerator.ComputeSaturationCurve()
        rho_min, rho_max = self._DataGenerator.GetDensityBounds()
        e_min, e_max = self._DataGenerator.GetEnergyBounds()
        within_bounds_density = np.logical_and(rhoe_sat_curve[:,0] > rho_min, rhoe_sat_curve[:,0] < rho_max)
        within_bounds_energy = np.logical_and(rhoe_sat_curve[:,1] > e_min, rhoe_sat_curve[:,1] < e_max)
        within_bounds = np.logical_and(within_bounds_density, within_bounds_energy)

        state_sat_curve = np.zeros([len(rhoe_sat_curve), EntropicVars.N_STATE_VARS.value])
        state_sat_curve[:, EntropicVars.Density.value] = rhoe_sat_curve[:,0]
        state_sat_curve[:, EntropicVars.Energy.value] = rhoe_sat_curve[:,1]

        state_sat_curve_norm = self._fluid_data_scaler.transform(state_sat_curve[within_bounds, :])

        sat_curve_pts_norm = state_sat_curve_norm[:, [EntropicVars.Density.value,EntropicVars.Energy.value]]
        return sat_curve_pts_norm
    
    def __GenerateMeshAndData(self, rhoe_norm:np.ndarray[float], sat_curve_pts:np.ndarray[float],ix_ref=[]):
        
        rhoe_norm_mesh_nodes,tria = self.__Compute2DMesh(rhoe_norm, ref_pts=rhoe_norm[ix_ref,:], show=False, sat_curve_pts=sat_curve_pts)
        
        # Calculate thermodynamic state variables of initial table nodes
        fluid_data_norm = np.zeros([len(rhoe_norm_mesh_nodes), EntropicVars.N_STATE_VARS.value])
        fluid_data_norm[:, EntropicVars.Density.value] = rhoe_norm_mesh_nodes[:,0]
        fluid_data_norm[:, EntropicVars.Energy.value] = rhoe_norm_mesh_nodes[:,1]
        fluid_data_mesh = self._fluid_data_scaler.inverse_transform(fluid_data_norm)
        fluid_data_mesh = self.__CalcMeshData(fluid_data_mesh)
        return fluid_data_mesh, tria
    
    def GenerateTable(self):
        """Initiate table generation process
        """

        self.__CartesianTableData()
        self._table_nodes = np.column_stack(tuple(self.state_data[:,:,i][self.valid_mask].flatten() for i in range(EntropicVars.N_STATE_VARS.value)))
        self._fluid_data_scaler = MinMaxScaler()
        self._fluid_data_scaler.fit(self._table_nodes)

        # Load initial fluid data and scale it
        if self._Config.GetTableDiscretization()=="cartesian":
            self.__CartesianTriangulation()
        else:
            print("Generating table with adaptive refinement")

            self._table_nodes = np.column_stack(tuple(self.state_data[:,:,i][self.valid_mask].flatten() for i in range(EntropicVars.N_STATE_VARS.value)))
            
            fluid_data_norm = self._fluid_data_scaler.fit_transform(self._table_nodes)
            
            sat_curve_pts_norm = []
            if self._Config.TwoPhase():
                sat_curve_pts_norm = self.__CreateSaturationCurve()

            # Generate initial coarse table of fluid data
            rhoe_norm = fluid_data_norm[:, [EntropicVars.Density.value, EntropicVars.Energy.value]]
            print("Generating coarse thermodynamic mesh...")
            fluid_data_coarse, _ = self.__GenerateMeshAndData(rhoe_norm, sat_curve_pts_norm)
            print("Done!")
            # Identify refinement locations
            fluid_data_norm = self._fluid_data_scaler.transform(fluid_data_coarse)
            ix_ref = self.__ApplyRefinement(fluid_data_norm)

            # # Regenerate table including refinement locations
            rhoe_norm_mesh = fluid_data_norm[:, [EntropicVars.Density.value, EntropicVars.Energy.value]]
        
            # Create triangulation of filtered thermodynamic state data
            print("Generating refined thermodynamic mesh...")
            fluid_data_ref, _ = self.__GenerateMeshAndData(rhoe_norm_mesh, sat_curve_pts_norm, ix_ref=ix_ref)
            print("Done!")
            fluid_data_norm_ref = self._fluid_data_scaler.transform(fluid_data_ref)

            DT = Delaunay(fluid_data_norm_ref[:, [EntropicVars.Density.value,EntropicVars.Energy.value]])

            # Extract triangulation, hull nodes, and table data
            Tria = DT.simplices 
            HullNodes = concave_hull_indexes(fluid_data_norm_ref[:, [EntropicVars.Density.value,EntropicVars.Energy.value]])
            #plt.triplot(DT.points[:,0],DT.points[:,1], Tria)
            #plt.show()
            self._table_nodes = fluid_data_ref 
            self._table_connectivity = Tria 
            self._table_hullnodes = HullNodes

        return
    
    def __remove_invalid_nodes_from_mesh(self, connectivity, valid_mask, rhoe_mesh_norm):

        conn = np.asarray(connectivity, dtype=np.int64)

        tri_keep = np.all(valid_mask[conn], axis=1)
        conn2 = conn[tri_keep]

        keep_nodes = np.flatnonzero(valid_mask)
        old_to_new = -np.ones(len(rhoe_mesh_norm), dtype=np.int64)
        old_to_new[keep_nodes] = np.arange(len(keep_nodes), dtype=np.int64)

        conn2 = old_to_new[conn2]

        return conn2
    def WriteOutParaview(self,file_name_out:str="vtk_table"):
        """
        write a file containing all the LuT data that can be opened with Paraview
        
        :param file_name_out: string indicating the name and extension of the saved file
        """

        #x, y = self._table_nodes[:, EntropicVars.Density.value], self._table_nodes[:, EntropicVars.Energy.value]
        table_data_norm = self._fluid_data_scaler.transform(self._table_nodes)
        x, y = table_data_norm[:, EntropicVars.Density.value], table_data_norm[:, EntropicVars.Energy.value]
        # scale_x= self._fluid_data_scaler.data_max_[EntropicVars.Density.value] - self._fluid_data_scaler.data_min_[EntropicVars.Density.value]
        # scale_y= self._fluid_data_scaler.data_max_[EntropicVars.Energy.value] - self._fluid_data_scaler.data_min_[EntropicVars.Energy.value]
        
        pts = np.column_stack([x, y, np.zeros_like(x)])  # z=0

        conn = np.asarray(self._table_connectivity, dtype=np.int64)
        if conn.min() == 1:
            conn = conn - 1

        point_data = {}
        for var in self._table_vars:
            ivar = EntropicVars[var].value 
            point_data[var] = np.asarray(self._table_nodes[:, ivar])

        mesh = meshio.Mesh(
            points=pts,
            cells=[("triangle", conn)],
            point_data=point_data
        )
        mesh.write("%s.vtk" % file_name_out)

        return

    def AddRefinementCriterion(self, TD_variable:str, norm_val_min:float=np.inf, norm_val_max:float=-np.inf):
        """Apply refinement in the table where the normalized value of the thermodynamic variable lies between the specified bounds.

        :param TD_variable: name of the thermodynamic variable for which to apply refinement
        :type TD_variable: str
        :param norm_val_min: lower bound of the normalized thermodynamic variable, defaults to np.inf
        :type norm_val_min: float, optional
        :param norm_val_max: upper bound of the normalized thermodynamic variable, defaults to -np.inf
        :type norm_val_max: float, optional
        :raises Exception: if thermodynamic state variable is unknown to SU2 DataMiner
        """
        if TD_variable not in self._table_vars:
            raise Exception("%s is not present in fluid data" % TD_variable)
        
        self.__refinement_vars.append(TD_variable)
        self.__refinement_norm_min.append(norm_val_min)
        self.__refinement_norm_max.append(norm_val_max)
        return 
    
    def __ApplyRefinement(self, fluid_data_norm_ref:np.ndarray[float]):
        ix_ref = np.array([],dtype=np.int64)
        fluid_vars = [a.name for a in EntropicVars][:-1]
        fluid_data_inv = self._fluid_data_scaler.inverse_transform(fluid_data_norm_ref)
        for TD_var, val_min, val_max in zip(self.__refinement_vars, self.__refinement_norm_min, self.__refinement_norm_max):
            norm_data_var = fluid_data_inv[:, fluid_vars.index(TD_var)]

            ix = np.argwhere(np.logical_and(norm_data_var>=val_min, norm_data_var<=val_max))[:,0]
            ix_ref = np.append(ix_ref, ix)
        if len(ix_ref) > 0:
            return np.unique(ix_ref)
        else:
            return []

            
    def WriteTableFile(self, output_filepath:str=None):
        """
        Save the table data and connectivity as a Dragon library file. If no file name is provided, the table file will be named according to the Config_FGM class name.

        :param output_filepath: optional output filepath for table file.
        :type output_filepath: str
        """

        if output_filepath:
            file_out = output_filepath + ".drg"
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

        fid.write("[Number of variables]\n%i\n\n" % (len(self._table_vars)))
        fid.write("[Variable names]\n")
        for iVar, Var in enumerate(self._table_vars):
            fid.write(str(iVar + 1)+":"+Var+"\n")
        fid.write("\n")

        fid.write("</Header>\n\n")

        print("Writing table data...")
        fid.write("<Data>\n")
        for iNode in range(len(self._table_nodes)):
            for var in self._table_vars:
                ivar = EntropicVars[var].value 
                fid.write("\t%+.14e" % self._table_nodes[iNode, ivar])
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

        return
    