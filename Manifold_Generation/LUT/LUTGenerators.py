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
# Version: 3.0.0                                                                              |
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

    _table_vars:list[str] = [s.name for s in EntropicVars][:-1]
    _table_nodes = []       # Progress variable, total enthalpy, and mixture fraction node values for each table level.
    _table_nodes_norm = []  # Normalized table nodes for each level.
    _table_connectivity = []    # Table node connectivity per table level.
    _table_hullnodes = []   # Hull node indices per table level.

    _controlling_variables:list[str]=["Density",\
                                      "Energy"]  # FGM controlling variables
    _fluid_data_scaler:MinMaxScaler = None   # Scaler for flamelet data controlling variables.

    # TODO: option for adaptive mesh/Cartesian mesh 

    def __init__(self, Config:Config_NICFD, load_file:str=None):
        """
        Initiate table generator class.

        :param Config: Config_FGM object.
        :type Config: Config_FGM
        """
        self._Config = Config 
        self._controlling_variables= [c for c in self._Config.GetControllingVariables()]

        self._DataGenerator = DataGenerator_CoolProp(self._Config)

        entropic_vars = [a.name for a in EntropicVars][:-1]
        self._table_vars = entropic_vars.copy()
        if not self._Config.TwoPhase():
            self._table_vars.remove(EntropicVars.VaporQuality.name)
        if not self._Config.CalcTransportProperties():
            self._table_vars.remove(EntropicVars.ViscosityDyn.name)
            self._table_vars.remove(EntropicVars.Conductivity.name)
            
        self.__LoadFluidData()
        return 
    
    # TODO: setters for Cartesian table options 

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
        self._Config.SetTableDiscretization(method)
        return 
    
    def __LoadFluidData(self):
        # TODO: generate coarse data grid from data generator
        fluid_data_file = self._Config.GetOutputDir() + "/" + self._Config.GetConcatenationFileHeader() + "_full.csv"
        with open(fluid_data_file, 'r') as fid:
            vars = fid.readline().strip().split(',')
        D = np.loadtxt(fluid_data_file,delimiter=',',skiprows=1)
        fluid_data_out = np.zeros([len(D), EntropicVars.N_STATE_VARS.value])
        for ivar, x in enumerate(vars):
            fluid_data_out[:, EntropicVars[x].value] = D[:, ivar]
        self._fluid_data_scaler = MinMaxScaler()
        fluid_data_norm = self._fluid_data_scaler.fit_transform(fluid_data_out)
   
        
        return fluid_data_norm
    
    def SetTableVars(self, table_vars_in:list[str]):
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
            
            
        for v in table_vars_in:
            found_var = False
            for q in EntropicVars:
                if v.lower() == q.name.lower():
                    found_var = True
                    self._table_vars.append(q.name)
            if not found_var:
                print("Error, \"%s\" is not supported by SU2 DataMiner" % v)
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
        XY_hull = concave_hull(np.unique(points,axis=0), length_threshold=1e-1)
        
        # Filter concave hull to remove nodes that are too close together.
        hull_pts = []
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
        factory = gmsh.model.geo

        # Create hull points
        for i in range(int(len(XY_hull))):
            hull_pts.append(factory.addPoint(XY_hull[i, 0], XY_hull[i, 1], 0, self._base_cell_size))
        
        # Connect hull points to a closed multi-component curve
        hull_lines = []
        for i in range(len(hull_pts)-1):
            hull_lines.append(factory.addLine(hull_pts[i], hull_pts[i+1]))
        hull_lines.append(factory.addLine(hull_pts[-1], hull_pts[0]))

        # Create a 2D plane of the enclosed space
        curvloop = factory.addCurveLoop(hull_lines)
        fluid_surf = factory.addPlaneSurface([curvloop])

        # Apply refinement points
        ref_pt_ids = []
        if len(ref_pts)>0:
            for i in range(len(ref_pts)):
                ref_pt_ids.append(factory.addPoint(ref_pts[i,0], ref_pts[i, 1], 0.0))

        # TODO: points with increased refinement 

        factory.addPhysicalGroup(2, [fluid_surf])
        
        add_sat_curve = (len(sat_curve_pts) > 0)
        if add_sat_curve:

            # Create normal vector to saturation curve.
            dedrho_sat_norm = FiniteDifferenceDerivative(sat_curve_pts[:,0], sat_curve_pts[:,1])
            norm_vector = np.column_stack((-1.0 / dedrho_sat_norm, np.ones(len(dedrho_sat_norm))))
            norm_vector = norm_vector / np.sqrt(np.sum(np.power(norm_vector, 2), axis=1))[:,np.newaxis]

            # Create offset curves to ensure that no nodes are generated on the saturation curve itself.
            sat_curve_upper_pts = []
            sat_curve_lower_pts = []
            i = 0
            j = 1
            dx = 0.5*self._refined_cell_size
            sat_curve_upper_pts.append(factory.addPoint(sat_curve_pts[i,0] + dx*norm_vector[i, 0],\
                                                        sat_curve_pts[i,1] + dx*norm_vector[i, 1],0, self._refined_cell_size))
            sat_curve_lower_pts.append(factory.addPoint(sat_curve_pts[i,0] - dx*norm_vector[i, 0],\
                                                        sat_curve_pts[i,1] - dx*norm_vector[i, 1],0, self._refined_cell_size))
            while j < len(sat_curve_pts):
                dist = np.sqrt(np.sum(np.power(sat_curve_pts[j,:] - sat_curve_pts[i,:],2)))
                if dist < dx:
                    j += 1 
                else:
                    i = j 
                    j += 1 
                    sat_curve_upper_pts.append(factory.addPoint(sat_curve_pts[i,0] + dx*norm_vector[i, 0],\
                                                                sat_curve_pts[i,1] + dx*norm_vector[i, 1],0, self._refined_cell_size))
                    sat_curve_lower_pts.append(factory.addPoint(sat_curve_pts[i,0] - dx*norm_vector[i, 0],\
                                                                sat_curve_pts[i,1] - dx*norm_vector[i, 1],0, self._refined_cell_size))
            sat_curve_connecting_lines = []
            for i in range(len(sat_curve_upper_pts)):
                sat_curve_connecting_lines.append(factory.addLine(sat_curve_lower_pts[i], sat_curve_upper_pts[i]))
            sat_curve_upper_lines = []
            sat_curve_lower_lines = []
            for i in range(len(sat_curve_lower_pts)-1):
                sat_curve_upper_lines.append(factory.addLine(sat_curve_upper_pts[i],sat_curve_upper_pts[i+1]))
                sat_curve_lower_lines.append(factory.addLine(sat_curve_lower_pts[i],sat_curve_lower_pts[i+1]))
                
            factory.synchronize()
            gmsh.model.mesh.embed(1, sat_curve_upper_lines, 2, fluid_surf)
            gmsh.model.mesh.embed(1, sat_curve_lower_lines, 2, fluid_surf)
            gmsh.model.mesh.embed(1, sat_curve_connecting_lines, 2, fluid_surf)
        # Apply conditional refinement, where the refined cell size is applied in proximity to the refinement points
        threshold_fields = []
        dist_field_ref_pt = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(dist_field_ref_pt, "PointsList", ref_pt_ids)
        gmsh.model.mesh.field.setNumber(dist_field_ref_pt, "Sampling", 100)
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
                                             "CurvesList", sat_curve_lower_lines + sat_curve_upper_lines + sat_curve_connecting_lines)
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
        gmsh.model.mesh.setRecombine(2, fluid_surf)
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
        MeshPoints = np.array([nodes[::3], nodes[1::3]]).T
        return MeshPoints
    
    
    def __CalcMeshData(self, fluid_data_mesh:np.ndarray[float]):
        """Calculate the fluid thermodynamic state variables for the table nodes

        :param fluid_data_mesh: table mesh nodes of density and static energy
        :type fluid_data_mesh: np.ndarray[float]
        :return: filtered thermodynamic state data at the table nodes
        :rtype: np.ndarray[float]
        """
        fluid_data_out = fluid_data_mesh.copy()
        self.valid_mask = np.zeros(len(fluid_data_mesh),dtype=np.bool)
        for i in tqdm(range(len(fluid_data_mesh)),desc="Evaluating fluid properties..."):
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
    
    # TODO: include derivative and transport validation methods
    def __CartesianTableData(self):
        print("Generating table on Cartesian grid")
        Np_rho = self._Config.GetNpDensity()
        Np_e = self._Config.GetNpEnergy()
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
        twophase_count = 0
        fd_fallback_count = 0
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

        self._table_nodes = np.column_stack(tuple(self.state_data[:,:,EntropicVars[v].value][self.valid_mask].flatten() for v in self._table_vars))
        
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

    def GenerateTable(self):
        """Initiate table generation process
        """

        # Load initial fluid data and scale it
        # TODO: use adaptive refinement or Cartesian refinement based on settings.
        if self._Config.GetTableDiscretization()=="cartesian":
            
            self.__CartesianTableData()

            self.__CartesianTriangulation()
        else:
            print("Generating table with adaptive refinement")

            fluid_data_norm = self.__LoadFluidData()
            rhoe_norm = fluid_data_norm[:, [EntropicVars.Density.value, EntropicVars.Energy.value]]

            # Generate initial coarse table of fluid data
            rhoe_mesh_norm_coarse = self.__Compute2DMesh(rhoe_norm)

            # Calculate thermodynamic state variables of initial table nodes
            fluid_data_norm_coarse = np.zeros([len(rhoe_mesh_norm_coarse), EntropicVars.N_STATE_VARS.value])
            fluid_data_norm_coarse[:, EntropicVars.Density.value] = rhoe_mesh_norm_coarse[:,0]
            fluid_data_norm_coarse[:, EntropicVars.Energy.value] = rhoe_mesh_norm_coarse[:,1]
            fluid_data_coarse = self._fluid_data_scaler.inverse_transform(fluid_data_norm_coarse)
            fluid_data_coarse = self.__CalcMeshData(fluid_data_coarse)

            # Identify refinement locations
            fluid_data_norm = self._fluid_data_scaler.transform(fluid_data_coarse)
            ix_ref = self.__ApplyRefinement(fluid_data_norm)

            # Regenerate table including refinement locations
            rhoe_norm_mesh = fluid_data_norm[:, [EntropicVars.Density.value, EntropicVars.Energy.value]]
            rhoe_norm_ref = rhoe_norm_mesh[ix_ref, :]
            rhoe_mesh_norm = self.__Compute2DMesh(rhoe_norm, ref_pts=rhoe_norm_ref,show=True)

            # Extract thermodynamic state variables of refined table
            fluid_data_norm_ref = np.zeros([len(rhoe_mesh_norm), EntropicVars.N_STATE_VARS.value])
            fluid_data_norm_ref[:, EntropicVars.Density.value] = rhoe_mesh_norm[:,0]
            fluid_data_norm_ref[:, EntropicVars.Energy.value] = rhoe_mesh_norm[:,1]
            fluid_data_ref = self._fluid_data_scaler.inverse_transform(fluid_data_norm_ref)
            fluid_data_ref = self.__CalcMeshData(fluid_data_ref)

            # Create triangulation of filtered thermodynamic state data
            fluid_data_norm_ref = self._fluid_data_scaler.transform(fluid_data_ref)
            DT = Delaunay(fluid_data_norm_ref[:, [EntropicVars.Density.value,EntropicVars.Energy.value]])

            # Extract triangulation, hull nodes, and table data
            Tria = DT.simplices 
            HullNodes = concave_hull_indexes(fluid_data_norm_ref[:, [EntropicVars.Density.value,EntropicVars.Energy.value]])

            self._table_nodes = fluid_data_ref 
            self._table_connectivity = Tria 
            self._table_hullnodes = HullNodes
            
            # Add static enthalpy and the specific heat at constant volume
            # self.table_vars.append("Enthalpy")
            # h = self._table_nodes[:, EntropicVars.Energy.value] + self._table_nodes[:, EntropicVars.p.value] / self._table_nodes[:, EntropicVars.Density.value]
            # self._table_nodes = np.hstack((self._table_nodes, h[:,np.newaxis]))

            # self.table_vars.append("cv")
            # cv = 1 /self._table_nodes[:, EntropicVars.dTde_rho.value]
            # self._table_nodes = np.hstack((self._table_nodes, cv[:,np.newaxis]))

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
        
        self.refinement_vars.append(TD_variable)
        self.refinement_norm_min.append(norm_val_min)
        self.refinement_norm_max.append(norm_val_max)
        return 
    
    def __ApplyRefinement(self, fluid_data_norm_ref:np.ndarray[float]):
        ix_ref = np.array([],dtype=np.int64)
        fluid_vars = [a.name for a in EntropicVars][:-1]
        for TD_var, val_min, val_max in zip(self.refinement_vars, self.refinement_norm_min, self.refinement_norm_max):
            norm_data_var = fluid_data_norm_ref[:, fluid_vars.index(TD_var)]

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

        fid.write("[Number of variables]\n%i\n\n" % (len(self._table_vars)))
        fid.write("[Variable names]\n")
        for iVar, Var in enumerate(self._table_vars):
            fid.write(str(iVar + 1)+":"+Var+"\n")
        fid.write("\n")

        fid.write("</Header>\n\n")

        print("Writing table data...")
        fid.write("<Data>\n")
        for iNode in range(len(self._table_nodes)):
            for ivar in range(len(self._table_vars)):
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
    
    # TODO: update configuration function