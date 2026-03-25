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
import CoolProp.CoolProp as CP
from Common.Properties import EntropicVars
from su2dataminer.generate_data import DataGenerator_CoolProp
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
from concave_hull import concave_hull, concave_hull_indexes

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
    
    def __LoadFluidData(self):
        # TODO: generate coarse data grid from data generator
        fluid_data_file = self._Config.GetOutputDir() + "/" + self._Config.GetConcatenationFileHeader() + "_full.csv"
        with open(fluid_data_file, 'r') as fid:
            vars = fid.readline().strip().split(',')
        D = np.loadtxt(fluid_data_file,delimiter=',',skiprows=1)
        entropic_vars = [a.name for a in EntropicVars][:-1]
        self.table_vars = entropic_vars.copy()
        fluid_data_out = np.zeros([len(D), EntropicVars.N_STATE_VARS.value])
        for ivar, x in enumerate(vars):
            fluid_data_out[:, entropic_vars.index(x)] = D[:, ivar]
        self._fluid_data_scaler = MinMaxScaler()
        fluid_data_norm = self._fluid_data_scaler.fit_transform(fluid_data_out)
   
        
        return fluid_data_norm
        
    def __Compute2DMesh(self, points:np.ndarray[float], ref_pts:np.ndarray[float]=[],show:bool=False):
        
        
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
        factory.addPlaneSurface([curvloop])

        # Apply refinement points
        ref_pt_ids = []
        if len(ref_pts)>0:
            for i in range(len(ref_pts)):
                ref_pt_ids.append(factory.addPoint(ref_pts[i,0], ref_pts[i, 1], 0.0))

        # Apply conditional refinement, where the refined cell size is applied in proximity to the refinement points
        gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(1, "PointsList", ref_pt_ids)
        gmsh.model.mesh.field.setNumber(1, "Sampling", 100)
        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "InField", 1)
        gmsh.model.mesh.field.setNumber(2, "SizeMin", self._refined_cell_size)
        gmsh.model.mesh.field.setNumber(2, "SizeMax", self._base_cell_size)
        gmsh.model.mesh.field.setNumber(2, "DistMin", 0.5*self._refinement_radius)
        gmsh.model.mesh.field.setNumber(2, "DistMax", 1.5*self._refinement_radius)

        gmsh.model.mesh.field.add("Min", 7)
        gmsh.model.mesh.field.setNumbers(7, "FieldsList", [2])
        gmsh.model.mesh.field.setAsBackgroundMesh(7)

        factory.synchronize()

        # Generate 2D mesh and extract table nodes
        gmsh.model.mesh.generate(2)
        if show:
            gmsh.fltk.run()
        nodes = gmsh.model.mesh.getNodes(dim=2, tag=-1, includeBoundary=True, returnParametricCoord=False)[1]
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
        for i in range(len(fluid_data_mesh)):
            try:
                self._DataGenerator.UpdateFluid(fluid_data_mesh[i, EntropicVars.Density.value], fluid_data_mesh[i, EntropicVars.Energy.value])
                state_vector, correct_phase = self._DataGenerator.GetStateVector()
                if correct_phase:
                    fluid_data_out[i, :] = state_vector
                else:
                    fluid_data_out[i, :] = None
            except:
                fluid_data_out[i, :] = None
        fluid_data_out = fluid_data_out[~np.isnan(fluid_data_out[:,0]),:]
        return fluid_data_out
    
    # TODO: include derivative and transport validation methods

    def GenerateTable(self):
        """Initiate table generation process
        """

        # Load initial fluid data and scale it
        # TODO: use adaptive refinement or Cartesian refinement based on settings.

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
        self.table_vars.append("Enthalpy")
        h = self._table_nodes[:, EntropicVars.Energy.value] + self._table_nodes[:, EntropicVars.p.value] / self._table_nodes[:, EntropicVars.Density.value]
        self._table_nodes = np.hstack((self._table_nodes, h[:,np.newaxis]))

        self.table_vars.append("cv")
        cv = 1 /self._table_nodes[:, EntropicVars.dTde_rho.value]
        self._table_nodes = np.hstack((self._table_nodes, cv[:,np.newaxis]))

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
        if TD_variable not in self.table_vars:
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

        fid.write("[Number of variables]\n%i\n\n" % (len(self.table_vars)))
        fid.write("[Variable names]\n")
        for iVar, Var in enumerate(self.table_vars):
            fid.write(str(iVar + 1)+":"+Var+"\n")
        fid.write("\n")

        fid.write("</Header>\n\n")

        print("Writing table data...")
        fid.write("<Data>\n")
        for iNode in range(len(self._table_nodes)):
            for ivar in range(len(self.table_vars)):
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