###############################################################################################
#       #      _____ __  _____      ____        __        __  ____                   #        #  
#       #     / ___// / / /__ \    / __ \____ _/ /_____ _/  |/  (_)___  ___  _____   #        #  
#       #     \__ \/ / / /__/ /   / / / / __ `/ __/ __ `/ /|_/ / / __ \/ _ \/ ___/   #        #      
#       #    ___/ / /_/ // __/   / /_/ / /_/ / /_/ /_/ / /  / / / / / /  __/ /       #        #  
#       #   /____/\____//____/  /_____/\__,_/\__/\__,_/_/  /_/_/_/ /_/\___/_/        #        #
#       #                                                                            #        #
###############################################################################################

############################## FILE NAME: DataDrivenConfig.py #################################
#=============================================================================================#
# author: Evert Bunschoten                                                                    |
#    :PhD Candidate ,                                                                         |
#    :Flight Power and Propulsion                                                             |
#    :TU Delft,                                                                               |
#    :The Netherlands                                                                         |
#                                                                                             |
#                                                                                             |
# Description:                                                                                |
#  Derived DataMiner configuration classes for flamelet-generated manifold and NI-CFD         |
#  applications.                                                                              |
#                                                                                             |  
# Version: 1.0.0                                                                              |
#                                                                                             |
#=============================================================================================#

#---------------------------------------------------------------------------------------------#
# Importing general packages
#---------------------------------------------------------------------------------------------#
import numpy as np 
import cantera as ct 
import pickle 
import CoolProp
import cantera as ct 

#---------------------------------------------------------------------------------------------#
# Importing DataMiner classes and functions
#---------------------------------------------------------------------------------------------#
from Common.Properties import DefaultProperties 
from Config_base import Config 
from CommonMethods import *

#---------------------------------------------------------------------------------------------#
# NI-CFD DataMiner configuration class
#---------------------------------------------------------------------------------------------#
class EntropicAIConfig(Config):
    """
    Define EntropicAIConfig class or load existing configuration. If `load_file` is set, the settings from an existing
    is loaded. If no file name is provided, a new `EntropicAIConfig` class is created.
    
    :param load_file: path to file of configuration to be loaded.
    :type load_file: str
    """

    # Fluid definition settings
    __fluid_names:list[str] = ["MM"]
    __fluid_string:str="MM"
    __fluid_mole_fractions:list[float] = []
    __use_PT:bool = True 


    __T_lower:float = DefaultProperties.T_min   # Lower temperature bound.
    __T_upper:float = DefaultProperties.T_max   # Upper temperature bound.
    __Np_T:int = DefaultProperties.Np_temp      # Number of temperature samples between bounds.

    __P_lower:float = DefaultProperties.P_min
    __P_upper:float = DefaultProperties.P_max
    __Np_P:int = DefaultProperties.Np_p

    __Rho_lower:float = DefaultProperties.Rho_min
    __Rho_upper:float = DefaultProperties.Rho_max 
    __Energy_lower:float = DefaultProperties.Energy_min
    __Energy_upper:float = DefaultProperties.Energy_max
    
    # MLP training settings
    __init_learning_rate_expo:float = DefaultProperties.init_learning_rate_expo
    __learning_rate_decay:float =  DefaultProperties.learning_rate_decay
    __batch_size_exponent:int = DefaultProperties.batch_size_exponent
    __NN_hidden:int = DefaultProperties.NN_hidden
    __N_epochs:int = DefaultProperties.N_epochs 

    # Table Generation Settings

    __Table_base_cell_size:float = None     # Table base cell size per table level.
    __Table_ref_cell_size:float = None      # Refined cell size per table level.
    __Table_ref_radius:float = None         # Refinement radius within which refined cell size is applied.
    __Table_curv_threshold:float = None     # Curvature threshold beyond which refinement is applied.


    def __init__(self, load_file:str=None):
        Config.__init__(self)
        """Class constructor
        """
        self._config_name = "EntropicAIConfig" # Configuration name.
        if load_file:
            print("Loading configuration for entropic model generation...")
            with open(load_file, "rb") as fid:
                loaded_config = pickle.load(fid)
            self.__dict__ = loaded_config.__dict__.copy()
            print("Loaded configuration file with name " + loaded_config._config_name)
        else:
            print("Generating empty EntropicAI config")

        return 
    
    def PrintBanner(self):
        """Print banner visualizing EntropicAI configuration settings."""

        print("EntropicAIConfiguration: " + self._config_name)
        print("")
        print("Fluid data generation settings:")
        print("Fluid data output directory: " + self.GetOutputDir())
        print("Fluid name(s): " + ",".join(self.__fluid_names))
        print("")
        if self.__use_PT:
            print("Temperature range: %.2f K -> %.2f K (%i steps)" % (self.__T_lower, self.__T_upper, self.__Np_T))
            print("Pressure range: %.3e Pa -> %.3e Pa (%i steps)" % (self.__P_lower, self.__P_upper, self.__Np_P))
        else:
            print("Energy range: %.2e J/kg -> %.2e J/kg (%i steps)" % (self.__Energy_lower, self.__Energy_upper, self.__Np_T))
            print("Density range: %.2f kg/m3 -> %.2f kg/m3 (%i steps)" % (self.__Rho_lower, self.__Rho_upper, self.__Np_P))
        if self.__use_PT:
            print("Data generation grid: pressure-based")
        else:
            print("Data generation grid: density-based")
        
        return 
    
    def GetConfigName(self):
        """
        Return configuration name.

        :return: EntropicAI configuration name.
        :rtype: str

        """
        return self._config_name
    

    def SetFluid(self, fluid_name):
        """
        Define the fluid name used for entropic data generation. By default, \"MM\" is used.

        :param fluid_name: CoolProp fluid name or list of names.
        :type fluid_name: str or list[str]
        :raise: Exception: If the fluid could not be defined in CoolProp.

        """

        # Check if one or multiple fluids are provided.
        if type(fluid_name) == list:
            if len(fluid_name) > 2:
                raise Exception("Only two fluids can be used for mixtures")
            
            self.__fluid_names = []
            fluid_mixing = []
            for f in fluid_name:
                self.__fluid_names.append(f)
                fluid_mixing.append(CoolProp.CoolProp.get_fluid_param_string(f,"CAS"))
            CoolProp.CoolProp.apply_simple_mixing_rule(fluid_mixing[0], fluid_mixing[1],"linear")
            if len(self.__fluid_mole_fractions) == 0:
                self.__fluid_mole_fractions = [0.5, 0.5]

        elif type(fluid_name) == str:
            self.__fluid_names = [fluid_name]
        
        fluid_string = "&".join(f for f in self.__fluid_names)
        self.__fluid_string=fluid_string
        try:
            CoolProp.AbstractState("HEOS", fluid_string)
        except:
            raise Exception("Specified fluid name not found.")

    def SetFluidMoleFractions(self, mole_fraction_1:float=0.5, mole_fraction_2:float=0.5):
        """Set fluid mole fractions for mixture.

        :param mole_fraction_1: _description_, defaults to 0.5
        :type mole_fraction_1: float, optional
        :param mole_fraction_2: _description_, defaults to 0.5
        :type mole_fraction_2: float, optional
        :raises Exception: if either mole fraction value is negative.
        """
        if (mole_fraction_1 < 0) or (mole_fraction_2 < 0):
            raise Exception("Mole fractions should be positive")
        
        # Normalize molar fractions
        self.__fluid_mole_fractions = []
        mole_fraction_1_norm = mole_fraction_1 / (mole_fraction_1 + mole_fraction_2)
        mole_fraction_2_norm = mole_fraction_2 / (mole_fraction_1 + mole_fraction_2)
        
        self.__fluid_mole_fractions.append(mole_fraction_1_norm)
        self.__fluid_mole_fractions.append(mole_fraction_2_norm)
        return 
        
    def GetFluidName(self):
        """
        Get the fluid used for entropic data generation.
        :return: fluid name
        :rtype: str

        """
        return self.__fluid_string
    
    def GetFluidNames(self):
        return self.__fluid_names 
    
    def GetMoleFractions(self):
        return self.__fluid_mole_fractions
    
    def UsePTGrid(self, PT_grid:bool=True):
        """Define fluid data grid in the pressure-temperature space. If not, the fluid data grid is defined in the density-energy space.

        :param PT_grid: use pressure-temperature based grid, defaults to True
        :type PT_grid: bool, optional
        """
        self.__use_PT = PT_grid 
        return 
    
    def GetPTGrid(self):
        return self.__use_PT 
    
    def SetTemperatureBounds(self, T_lower:float=DefaultProperties.T_min, T_upper:float=DefaultProperties.T_max):
        """Set the upper and lower temperature limits for the fluid data grid.

        :param T_lower: lower temperature limit in Kelvin.
        :type T_lower: float
        :param T_upper: upper temperature limit in Kelvin.
        :type T_upper: float
        :raises Exception: if lower temperature limit exceeds upper temperature limit.
        """
        if (T_lower >= T_upper):
            raise Exception("Lower temperature should be lower than upper temperature.")
        else:
            self.__T_lower = T_lower
            self.__T_upper = T_upper
        return
    
    def SetEnergyBounds(self, E_lower:float=DefaultProperties.Energy_min, E_upper:float=DefaultProperties.Energy_max):
        self.__Energy_lower=E_lower
        self.__Energy_upper=E_upper
        return 
    
    def GetEnergyBounds(self):
        return [self.__Energy_lower, self.__Energy_upper]
    
    def SetNpEnergy(self, Np_Energy:int=DefaultProperties.Np_temp):
        self.__Np_T = Np_Energy
        return 
    
    def SetDensityBounds(self, Rho_lower:float=DefaultProperties.Rho_min, Rho_upper:float=DefaultProperties.Rho_max):
        self.__Rho_lower=Rho_lower
        self.__Rho_upper=Rho_upper
        return 
    
    def SetNpDensity(self, Np_rho:int=DefaultProperties.Np_p):
        self.__Np_P = Np_rho
        return 
    
    def GetDensityBounds(self):
        return [self.__Rho_lower, self.__Rho_upper]
    
    def GetTemperatureBounds(self):
        return [self.__T_lower, self.__T_upper]
    
    
    def SetNpTemp(self, Np_Temp:int=DefaultProperties.Np_temp):
        """
        Set number of divisions for the temperature grid.

        :param Np_Temp: Number of divisions for the temperature range.
        :type Np_Temp: int
        :rase: Exception: If the number of divisions is lower than one.

        """
        if (Np_Temp <= 0):
            raise Exception("Number of unburnt temperature samples should be higher than one.")
        else:
            self.__Np_T = Np_Temp
        return 
    
    def GetNpTemp(self):
        """
        Get the number of divisions for the fluid temperature range.

        :return: Number of divisions for the fluid temperature range.
        :rtype: int

        """
        return self.__Np_T
    

    def SetPressureBounds(self, P_lower:float=DefaultProperties.P_min, P_upper:float=DefaultProperties.P_max):
        """Set the upper and lower limits for the fluid pressure.

        :param P_lower: lower pressure limit in Pa.
        :type P_lower: float
        :param P_upper: upper pressure limit in Pa.
        :type P_upper: float
        :raises Exception: if lower pressure limit exceeds upper pressure limit.
        """
        if (P_lower >= P_upper):
            raise Exception("Lower temperature should be lower than upper temperature.")
        else:
            self.__P_lower = P_lower
            self.__P_upper = P_upper
        return 
    
    def GetPressureBounds(self):
        return [self.__P_lower, self.__P_upper]
    
    
    def SetNpPressure(self, Np_P:int=DefaultProperties.Np_p):
        """
        Set number of divisions for the fluid pressure grid.

        :param Np_Temp: Number of divisions for the fluid pressure.
        :type Np_Temp: int
        :rase: Exception: If the number of divisions is lower than one.

        """
        if (Np_P <= 0):
            raise Exception("Number of unburnt temperature samples should be higher than one.")
        else:
            self.__Np_P = Np_P 
        return 
    
    def GetNpPressure(self):
        return self.__Np_P
    
    def SetTableCellSize(self, base_cell_size:float, refined_cell_size:float=None):
        """Define the base and optional refined 2D table cell sizes.

        :param base_cell_size: Normalized base cell size applied to each table level.
        :type base_cell_size: float
        :param refined_cell_size: Optional refined cell size to be applied in high-curvature regions, defaults to None
        :type refined_cell_size: float, optional
        :raises Exception: If base cell size is lower or equal to zero.
        :raises Exception: If supplied refined cell size is larger than base cell size.
        """
        if base_cell_size <= 0:
            raise Exception("Normalized base cell size should be larger than zero.")
        if refined_cell_size != None:
            if refined_cell_size > base_cell_size:
                raise Exception("Refined cell size should be smaller than base cell size.")
            self.__Table_ref_cell_size = refined_cell_size
        else:
            self.__Table_ref_cell_size = base_cell_size
        self.__Table_base_cell_size = base_cell_size

    def GetTableCellSize(self):
        """Returns the base and refined table 2D cell size values.

        :return: base cell size, refined cell size
        :rtype: float, float
        """
        return self.__Table_base_cell_size, self.__Table_ref_cell_size
    
    def SetTableRefinement(self, refinement_radius:float, refinement_threshold:float):
        """Define the table refinement occurance parameters.

        :param refinement_radius: Normalized radius around refinement location within which refinement is applied.
        :type refinement_radius: float
        :param refinement_threshold: Normalized reaction rate curvature threshold above which refinement is applied.
        :type refinement_threshold: float
        :raises Exception: If refinement radius is not between zero and one.
        :raises Exception: If curvature threshold is not between zero and one.
        """
        if refinement_radius <= 0 or refinement_radius >= 1:
            raise Exception("Refinement radius should be between zero and one.")
        if refinement_threshold <= 0 or refinement_threshold >= 1:
            raise Exception("Refinement threshold should be between zero and one.")
        self.__Table_ref_radius = refinement_radius 
        self.__Table_curv_threshold = refinement_threshold 

    def GetTableRefinement(self):
        """Returns the table refinement radius and refinement threshold values.

        :return: Normalized table refinement radius, normalized refinement curvature threshold.
        :rtype: float, float
        """
        return self.__Table_ref_radius, self.__Table_curv_threshold
    
    def SetAlphaExpo(self, alpha_expo:float=DefaultProperties.init_learning_rate_expo):
        """Set initial learning rate decay parameter.

        :param alpha_expo: initial learning rate exponent (base 10), defaults to -2.6
        :type alpha_expo: float, optional
        :raises Exception: if initial learning rate exponent is positive.
        """

        if alpha_expo >= 0:
            raise Exception("Initial learning rate exponent should be negative.")
        self.__init_learning_rate_expo = alpha_expo 

        return 
    
    def GetAlphaExpo(self):
        """Get initial learning rate exponent (base 10).

        :return: log10 of initial learning rate exponent.
        :rtype: float
        """
        return self.__init_learning_rate_expo 
    
    def SetLearningRateDecay(self, lr_decay:float=DefaultProperties.learning_rate_decay):
        """Set the learning rate decay parameter.

        :param lr_decay: learning rate decay parameter for exponential learning rate scheduler, defaults to 0.9985
        :type lr_decay: float, optional
        :raises Exception: if decay parameter is not between 0.0 and 1.0.
        """

        if lr_decay <= 0.0 or lr_decay >= 1.0:
            raise Exception("Learning rate decay parameter should be between 0 and 1.")
        self.__learning_rate_decay = lr_decay

        return 
    
    def GetLearningRateDecay(self):
        """Get learning rate decay parameter.

        :return: learning rate decay parameter.
        :rtype: float
        """
        
        return self.__learning_rate_decay 
    
    def SetMLPArchitecture(self, group_index:int, architecture_in:list[int]):
        """Save an MLP architecture for a given group index.

        :param group_index: MLP output group index.
        :type group_index: int
        :param architecture_in: number of neurons per hidden layer.
        :type architecture_in: list[int]
        :raises Exception: if the provided list of neuron counts is empty.
        :raises Exception: if the provided group index exceeds the number of MLPs.
        """
        if len(architecture_in) == 0:
            raise Exception("MLP should have at least one layer.")
        if group_index >= len(self.__MLP_architectures):
            raise Exception("Group index should be below "+str(len(self.__MLP_architectures)))
        
        if self.__MLP_architectures == None:
            self.__MLP_architectures = []
            for iGroup in range(len(self.__MLP_output_groups)):
                self.__MLP_architectures[iGroup] = [] 
        
        self.__MLP_architectures[group_index] = []
        for NN in architecture_in:
            self.__MLP_architectures[group_index].append(NN)

    def GetMLPArchitecture(self, group_index:int):
        """Get the saved MLP architecture for a given group.

        :param group_index: MLP output group index.
        :type group_index: int
        :return: neurons per hidden layer in the network.
        :rtype: list[int]
        """
        return self.__MLP_architectures[group_index]
    
    def SetTrainParams(self, group_index:int, alpha_expo:float, lr_decay:float, batch_expo:int, activation_index:int):
        """Save the training parameters for a set MLP.

        :param group_index: MLP output group index.
        :type group_index: int
        :param alpha_expo: alpha exponent value for the learning rate.
        :type alpha_expo: float
        :param lr_decay: learning rate decay parameter.
        :type lr_decay: float
        :param batch_expo: mini-batch size exponent.
        :type batch_expo: int
        :param activation_index: activation function index for hidden layers.
        :type activation_index: int
        :raises Exception: if the provided group index exceeds the number of MLPs.
        """
        if group_index >= len(self.__MLP_architectures):
            raise Exception("Group index should be below "+str(len(self.__MLP_architectures)))
        
        if self.__MLP_trainparams == None:
            self.__MLP_trainparams = []
            for iGroup in range(len(self.__MLP_output_groups)):
                self.__MLP_trainparams[iGroup] = [] 
        
        self.__MLP_trainparams[iGroup] = []
        self.__MLP_trainparams[iGroup].append(alpha_expo)
        self.__MLP_trainparams[iGroup].append(lr_decay)
        self.__MLP_trainparams[iGroup].append(batch_expo)
        self.__MLP_trainparams[iGroup].append(activation_index)

    def GetTrainParams(self, group_index:int):
        """Get the saved training parameters for a given MLP

        :param group_index: MLP output group index.
        :type group_index: int
        :return: learning rate alpha exponent, learning rate decay parameter, batch size exponent, and activation function index.
        :rtype: float, float, int, int
        """

        alpha_expo = self.__MLP_architectures[group_index][0]
        lr_decay = self.__MLP_architectures[group_index][1]
        batch_expo = self.__MLP_architectures[group_index][2]
        activation_index = self.__MLP_architectures[group_index][3]

        return alpha_expo, lr_decay, batch_expo, activation_index
        
    def AddOutputGroup(self, variable_names_in:list[str]):
        """Add an MLP output group of fluid variables.

        :param variable_names_in: list of output variables to include in the outputs of a single MLP.
        :type variable_names_in: list[str]
        :raises Exception: if the number of variables is equal to zero.
        """
        if len(variable_names_in) == 0:
            raise Exception("An MLP output group should be made up of at least one variable.")
        
        if self.__MLP_output_groups == None:
            self.__MLP_output_groups = []
        self.__MLP_output_groups.append([])
        for var in variable_names_in:
            self.__MLP_output_groups[-1].append(var)

    def DefineOutputGroup(self, i_group:int, variable_names_in:list[str]):
        """Re-define the variables in a specific MLP output group.

        :param i_group: MLP output group index to adapt.
        :type i_group: int
        :param variable_names_in: list of output variables to include in the outputs of a single MLP.
        :type variable_names_in: list[str]
        :raises Exception: if group index exceeds number of output groups.
        :raises Exception: if the number of variables is equal to zero.
        """
        if i_group >= len(self.__MLP_output_groups):
            raise Exception("Group not present in MLP outputs.")
        if len(variable_names_in) == 0:
            raise Exception("An MLP output group should be made up of at least one variable.")
        
        self.__MLP_output_groups[i_group] = []
        for var in variable_names_in:
            self.__MLP_output_groups[i_group].append(var)

    def RemoveOutputGroup(self, i_group:int):
        if i_group > len(self.__MLP_output_groups):
            raise Exception("Group not present in MLP outputs.")
        print("Removing output group %i: %s" % (i_group, ",".join(s for s in self.__MLP_output_groups[i_group-1])))
        self.__MLP_output_groups.remove(self.__MLP_output_groups[i_group-1])

    def ClearOutputGroups(self):
        self.__MLP_output_groups = []
        
    def DisplayOutputGroups(self):
        """Print the MLP output variables grouping arrangement.
        """
        for i_group, group in enumerate(self.__MLP_output_groups):
            print("Group %i: " % (i_group+1) +",".join(var for var in group) )
    
    def GetNMLPOutputGroups(self):
        """Get the number of MLP output groups.

        :return: number of MLP output groups
        :rtype: int
        """
        return len(self.__MLP_output_groups)
    
    def GetMLPOutputGroup(self, i_group:int):
        """Get the list of variables in a specific MLP group.

        :param i_group: MLP output group index.
        :type i_group: int
        :return: list of variables for this group.
        :rtype: list[str]
        """
        return self.__MLP_output_groups[i_group]
    
    def SaveConfig(self, file_name:str):
        """
        Save the current EntropicAI configuration.

        :param file_name: configuration file name.
        :type file_name: str
        """

        self._config_name = file_name
        file = open(file_name+'.cfg','wb')
        pickle.dump(self, file)
        file.close()


#---------------------------------------------------------------------------------------------#
# Flamelet-Generated Manifold DataMiner configuration class
#---------------------------------------------------------------------------------------------#
class FlameletAIConfig(Config):
    """
    Define FlameletAIConfig class or load existing configuration. If `load_file` is set, the settings from an existing
    is loaded. If no file name is provided, a new `FlameletAIConfig` class is created.
    
    :param load_file: path to file of configuration to be loaded.
    :type load_file: str
    """
    # FlameletAI configuration class containing all relevant info on flamelet manifold settings.

    # Flamelet Generation Settings           
    __reaction_mechanism:str = 'gri30.yaml'   # Reaction mechanism name.

    __fuel_species:list[str] = ['CH4'] # Fuel species composition.
    __fuel_weights:list[float] = [1.0]  # Fuel species weights.

    __oxidizer_species:list[str] = ['O2', 'N2']   # Oxidizer species composition.
    __oxidizer_weights:list[float] = [1.0, 3.76]    # Oxidizer species weights.

    __carrier_specie:str = 'N2' # Carrier specie definition.

    __run_mixture_fraction:bool = False    # Define premixed status as mixture fraction (True) or as equivalence ratio (False)
    __preferential_diffusion:bool = False  # Include preferential diffusion effects. 

    __T_unb_lower:float = 280   # Lower bound of unburnt reactants temperature.
    __T_unb_upper:float = 500   # Upper bound of unburnt reactants temperature.
    __Np_T_unb:int = 100      # Number of unburnt temperature samples between bounds.

    __mix_status_lower:float = 0.1  # Lower bound of premixed status
    __mix_status_upper:float = 20.0 # Upper bound of premixed status
    __Np_mix_unb:int = 100    # Number of mixture samples between bounds.

    __generate_freeflames:bool = True      # Generate adiabatic flamelets
    __generate_burnerflames:bool = True    # Generate burner-stabilized flamelets
    __generate_equilibrium:bool = True     # Generate chemical equilibrium data
    __generate_counterflames:bool = True   # Generate counter-flow diffusion flamelets.

    __write_MATLAB_files:bool = False  # Write TableGenerator compatible flamelet files.

    gas:ct.Solution = None  # Cantera solution object.
    __species_in_mixture:list[str] = None # Species names in mixture. 

    # Flamelet Data Concatination Settings

    __pv_definition:list[str] = ['H2O', 'CO2'] # Progress variable species.
    __pv_weights:list[float] = [1.0, 1.0]      # Progress variable mass fraction weights.

    __passive_species:list[str] = [] # Passive species for which to generate source terms.

    __lookup_variables:list[str] = ["Heat_Release"] # Extra look-up variables to read from flamelet data 

    __Np_per_flamelet:int = 60    # Number of data points to interpolate from flamelet data.

    # MLP output groups and architecture information.
    __MLP_output_groups:list[list[str]] = None  # Output variables for each MLP.
    __MLP_architectures:list[list[int]] = None  # Hidden layer architecture for each MLP.
    __MLP_trainparams:list[list[float]] = None  # Training parameters and activation function information for each MLP.

    # Table Generation Settings

    __Table_base_cell_size:float = None     # Table base cell size per table level.
    __Table_ref_cell_size:float = None      # Refined cell size per table level.
    __Table_ref_radius:float = None         # Refinement radius within which refined cell size is applied.
    __Table_curv_threshold:float = None     # Reaction rate curvature threshold beyond which refinement is applied.
    __Table_level_count:int = None          # Number of table levels.
    __Table_mixfrac_lower:float = None      # Lower mixture fraction limit of the table.
    __Table_mixfrac_upper:float = None      # Upper mixture fraction limit of the table.

    # Mixture fraction definition and preferential diffusion settings.
    __mixfrac_coefficients:np.ndarray[float] = None 
    __mixfrac_constant:float = None 
    __mixfrac_coeff_carrier:float = None 

    def __init__(self, load_file:str=None):
        """Class constructor
        """
        Config.__init__(self)

        if load_file:
            print("Loading configuration for flamelet generation")
            with open(load_file, "rb") as fid:
                loaded_config = pickle.load(fid)
            self.__dict__ = loaded_config.__dict__.copy()
            print("Loaded configuration file with name " + loaded_config._config_name)
        else:
            print("Generating empty flameletAI config")
        
        self.gas = ct.Solution(self.__reaction_mechanism)
        return 
    
    def PrintBanner(self):
        """Print banner visualizing FlameletAI configuration settings."""

        print("flameletAIConfiguration: " + self._config_name)
        print("")
        print("Flamelet generation settings:")
        print("Flamelet data output directory: " + self._output_dir)
        print("Reaction mechanism: " + self.__reaction_mechanism)
        print("Fuel definition: " + ",".join("%s: %.2e" % (self.__fuel_species[i], self.__fuel_weights[i]) for i in range(len(self.__fuel_species))))
        print("Oxidizer definition: " + ",".join("%s: %.2e" % (self.__oxidizer_species[i], self.__oxidizer_weights[i]) for i in range(len(self.__oxidizer_species))))
        print("")
        print("Reactant temperature range: %.2f K -> %.2f K (%i steps)" % (self.__T_unb_lower, self.__T_unb_upper, self.__Np_T_unb))
        
        if self.__run_mixture_fraction:
            print("Mixture status defined as mixture fraction")
        else: 
            print("Mixture status defined as equivalence ratio")
        print("Reactant mixture status range: %.2e -> %.2e  (%i steps)" % (self.__mix_status_lower, self.__mix_status_upper, self.__Np_mix_unb))
        print("")
        print("Flamelet types included in manifold:")
        if self.__generate_freeflames:
            print("-Adiabatic free-flamelet data")
        if self.__generate_burnerflames:
            print("-Burner-stabilized flamelet data")
        if self.__generate_equilibrium:
            print("-Chemical equilibrium data")
        if self.__generate_counterflames:
            print("-Counter-flow diffusion flamelet data")
        print("")

        print("Flamelet manifold data characteristics: ")
        print("Progress variable definition: " + ", ".join("%+.2e %s" % (self.__pv_weights[i], self.__pv_definition[i]) for i in range(len(self.__pv_weights))))
        if len(self.__passive_species) > 0:
            print("Passive species in manifold: " + ", ".join(s for s in self.__passive_species))
        print("")

        if self.__Table_level_count is not None:
            print("Table generation settings:")
            print("Table mixture fraction range: %.2e -> %.2e" % (self.__Table_mixfrac_lower, self.__Table_mixfrac_upper))
            print("Number of table levels: %i" % self.__Table_level_count)
            print("Table level base cell size: %.2e" % self.__Table_base_cell_size)
            print("Table level refined cell size: %.2e" % self.__Table_ref_cell_size)
            print("Table level refinement radius: %.2e" % self.__Table_ref_radius)
            print("Table level refinement threshold: %.2e" % self.__Table_curv_threshold)
        print("")

        if self.__MLP_output_groups is not None:
            print("MLP Output Groups:")
            self.DisplayOutputGroups()
        print("")
        
    def ComputeMixFracConstants(self):
        """
        
        Compute the species mass fraction coefficients according to the Bilger mixture fraction definition.
        
        """

        # Number of species in fuel and oxidizer definition.
        n_fuel = len(self.__fuel_species)
        n_ox = len(self.__oxidizer_species)

        # Joining fuel and oxidizer definitions into a single string
        fuel_string = ','.join([self.__fuel_species[i] + ':'+str(self.__fuel_weights[i]) for i in range(n_fuel)])
        oxidizer_string = ','.join([self.__oxidizer_species[i] + ':'+str(self.__oxidizer_weights[i]) for i in range(n_ox)])

        
        # Getting the carrier specie index
        idx_c = self.gas.species_index(self.__carrier_specie)

        #--- Computing mixture fraction coefficients ---#
        # setting up mixture in stochiometric condition
        self.gas.TP = 300, ct.one_atm
        self.gas.set_equivalence_ratio(1.0, fuel_string, oxidizer_string)
        self.gas.equilibrate('TP')


        # number of atoms occurrances in fuel
        atoms_in_fuel = np.zeros(self.gas.n_elements)
        for i_e in range(self.gas.n_elements):
            for i_f in range(n_fuel):
                if self.gas.n_atoms(self.__fuel_species[i_f], self.gas.element_names[i_e]) > 0:
                    atoms_in_fuel[i_e] += self.__fuel_weights[i_f]

        # Computing the element mass fractions in the equilibrated mixture
        Z_elements = np.zeros(self.gas.n_elements)
        for i_e in range(self.gas.n_elements):
            for i_s in range(self.gas.n_species):
                Z_elements[i_e] += self.gas.n_atoms(self.gas.species_name(i_s), self.gas.element_name(i_e)) * self.gas.atomic_weights[i_e] * self.gas.Y[i_s]/self.gas.molecular_weights[i_s]

        # Getting element index of oxygen
        idx_O = self.gas.element_index('O')

        # Computing the elemental mass fractions in the fuel
        Z_fuel_elements = 0
        for i_e in range(self.gas.n_elements):
            if i_e != idx_O:
                    Z_fuel_elements += atoms_in_fuel[i_e] * Z_elements[i_e]/self.gas.atomic_weights[i_e]

        # Computing the oxygen stochimetric coefficient
        nu_O = Z_fuel_elements * self.gas.atomic_weights[idx_O]/Z_elements[idx_O]

        # Filling in fuel specie mass fraction array
        __fuel_weights_s = np.zeros(self.gas.n_species)
        for i_fuel in range(n_fuel):
            idx_sp = self.gas.species_index(self.__fuel_species[i_fuel])
            __fuel_weights_s[idx_sp] = self.__fuel_weights[i_fuel]
        Y_fuel_s = __fuel_weights_s * self.gas.molecular_weights/np.sum(__fuel_weights_s * self.gas.molecular_weights)

        # Filling in oxidizer specie mass fraction array
        __oxidizer_weights_s = np.zeros(self.gas.n_species)
        for i_oxidizer in range(n_ox):
            idx_sp = self.gas.species_index(self.__oxidizer_species[i_oxidizer])
            __oxidizer_weights_s[idx_sp] = self.__oxidizer_weights[i_oxidizer]
        Y_oxidizer_s = __oxidizer_weights_s * self.gas.molecular_weights/np.sum(__oxidizer_weights_s * self.gas.molecular_weights)

        # Computing elemental mass fractions of pure fuel stream
        Z_elements_1 = np.zeros(self.gas.n_elements)
        for i_e in range(self.gas.n_elements):
            for i_s in range(self.gas.n_species):
                Z_elements_1[i_e] += self.gas.n_atoms(self.gas.species_name(i_s), self.gas.element_name(i_e)) * self.gas.atomic_weights[i_e] * Y_fuel_s[i_s] / self.gas.molecular_weights[i_s]

        # Computing elemental mass fractions of pure oxidizer stream
        Z_elements_2 = np.zeros(self.gas.n_elements)
        for i_e in range(self.gas.n_elements):
            for i_s in range(self.gas.n_species):
                Z_elements_2[i_e] += self.gas.n_atoms(self.gas.species_name(i_s), self.gas.element_name(i_e)) * self.gas.atomic_weights[i_e] * Y_oxidizer_s[i_s] / self.gas.molecular_weights[i_s]

        # Computing stochimetric coefficient of pure fuel stream
        beta_1 = 0
        for i_e in range(self.gas.n_elements):
            beta_1 += atoms_in_fuel[i_e]*Z_elements_1[i_e]/self.gas.atomic_weights[i_e]
        beta_1 -= nu_O * Z_elements_1[idx_O]/self.gas.atomic_weights[idx_O]

        # Computing stochimetric coefficient of pure oxidizer stream
        beta_2 = 0
        for i_e in range(self.gas.n_elements):
            beta_2 += atoms_in_fuel[i_e] * Z_elements_2[i_e]/self.gas.atomic_weights[i_e]
        beta_2 -= nu_O * Z_elements_2[idx_O]/self.gas.atomic_weights[idx_O]

        # Computing mixture fraction coefficient
        self.__mixfrac_coefficients = np.zeros(self.gas.n_species)
        for i_s in range(self.gas.n_species):
            z_fuel = 0
            for i_e in range(self.gas.n_elements):
                z_fuel += atoms_in_fuel[i_e] * self.gas.n_atoms(self.gas.species_name(i_s), self.gas.element_name(i_e))/self.gas.molecular_weights[i_s]
            z_ox = -nu_O * self.gas.n_atoms(self.gas.species_name(i_s), 'O')/self.gas.molecular_weights[i_s]

            self.__mixfrac_coefficients[i_s] = (1/(beta_1 - beta_2)) * (z_fuel + z_ox)

        # Constant term in mixture fraction equation
        self.__mixfrac_constant = -beta_2 / (beta_1 - beta_2)

        # Mixture fraction weight of the carrier specie
        self.__mixfrac_coeff_carrier = self.__mixfrac_coefficients[idx_c]
        return 
    
    def GetMixtureFractionCoefficients(self):
        """
        Get the species mass fraction coefficients for computation of the mixture fraction according to Bilger's definition.
        
        :return: array of coefficients for mixture fraction computation.
        :rtype: array[float]

        """
        return self.__mixfrac_coefficients
    
    def GetMixtureFractionConstant(self):
        """
        Get the mixture fraction offset value according to Bilger's definition.
        
        :return: mixture fraction offset value.
        :rtype: float
        """
        return self.__mixfrac_constant
    
    def GetMixtureFractionCoeff_Carrier(self):
        """
        Get the mixture frraction coefficient of the carrier specie.

        :return: mixture fraction coefficient of the carrier specie.
        :rtype: float
        """
        return self.__mixfrac_coeff_carrier
    
    def GetFuelDefinition(self):
        """
        Get a list of species comprising the fuel reactants.

        :return: list of fuel reactant specie names.
        :rtype: list[str]
        """
        return self.__fuel_species
    def GetFuelWeights(self):
        """
        Get a list of the molar fractions of the fuel species.

        :return: list of fuel molar fractions of species within the fuel mixture.
        :rtype: list[float]
        """
        return self.__fuel_weights
    
    def GetOxidizerDefinition(self):
        """
        Get a list of species comprising the oxidizer reactants.

        :return: list of oxidizer reactant specie names.
        :rtype: list[str]
        """
        return self.__oxidizer_species
    
    def GetOxidizerWeights(self):
        """
        Get a list of the molar fractions of the oxidizer species.

        :return: list of oxidizer molar fractions of species within the oxidizer mixture.
        :rtype: list[float]
        """
        return self.__oxidizer_weights

    def GetMixtureSpecies(self):
        """
        Get the list of passive species.

        :return: list of passive species names.
        :rtype: list[str]
        """
        return self.__species_in_mixture

    def SetFuelDefinition(self, fuel_species:list[str], fuel_weights:list[float]):
        """
        Define fuel species and weights. By default the fuel is set to pure methane.
        
        :param fuel_species: List containing fuel species names.
        :type fuel_species: list[str]
        :param fuel_weights: List containing fuel molar fractions
        :type fuel_weights: list[float]
        :raise: Exception: If no reactants are provided.
        :raise: Exception: If the number of species and weights are unequal.
        
        """
        if len(fuel_species) == 0:
            raise Exception("Fuel definition should contain at least one species name.")
        
        if (len(fuel_species) != len(fuel_weights)):
            raise Exception("Number of species and weigths for fuel definition should be equal.")
        self.__fuel_species = []
        self.__fuel_weights = []
        for f in fuel_species:
            self.__fuel_species.append(f)
        for w in fuel_weights:
            self.__fuel_weights.append(w)

    def SetOxidizerDefinition(self, oxidizer_species:list[str]=["O2", "N2"], \
                              oxidizer_weights:list[float]=[1.0, 3.76]):
        """
        Define oxidizer species and weights. Default arguments are for air.

        :param oxidizer_species: List containing oxidizer species names.
        :type oxidizer_species: list[str]
        :param oxidizer_weights: List containing oxidizer molar fractions.
        :type oxidizer_weights: list[float]
        :raise: Exception: If no reactants are provided.
        :raise: Exception: If the number of species and weights are unequal.

        """

        if len(oxidizer_species) == 0:
            raise Exception("Oxidizer definition should contain at least one species name.")
        if (len(oxidizer_species) != len(oxidizer_weights)):
            raise Exception("Number of species and weigths for oxidizer definition should be equal.")
        self.__oxidizer_species = []
        self.__oxidizer_weights = []
        for o in oxidizer_species:
            self.__oxidizer_species.append(o)
        for w in oxidizer_weights:
            self.__oxidizer_weights.append(w)

    def SetReactionMechanism(self, mechanism_input:str="gri30.yaml"):
        """
        Define reaction mechanism used for flamelet data generation. The default setting is the gri30 mechanism for methane/hydrogen.

        :param mechanism_input: Reaction mechanism name.
        :type mechanism_input: str
        :raise: Exception: If the reaction mechanism could not be loaded from path.

        """
        
        self.__reaction_mechanism = mechanism_input
        try:
            self.gas = ct.Solution(mechanism_input)
        except:
            raise Exception("Specified reaction mechanism not found.")

    def GetReactionMechanism(self):
        """
        Get the reaction mechanism used for flamelet generation.
        :return: reaction mechanism name.
        :rtype: str

        """
        return self.__reaction_mechanism
    
    def SetMixtureBounds(self, mix_lower:float, mix_upper:float):
        """
        Set upper and lower bounds of mixture status for flamelet data manifold.

        :param mix_lower: Leanest mixture status value.
        :type mix_lower: float
        :param mix_upper: Richest mixture status value.
        :type mix_upper: float
        :raise: If lower mixture status value exceeds the upper mixture status value.

        """

        if (mix_lower >= mix_upper):
            raise Exception("Lower mixture status should be lower than upper mixture status.")
        else:
            self.__mix_status_lower = mix_lower
            self.__mix_status_upper = mix_upper
    
    def GetMixtureBounds(self):
        """
        Get the mixture status bounds.
        :return: List containing lower and upper mixture status values.
        :rtype: list[float]

        """
        return [self.__mix_status_lower, self.__mix_status_upper]
    
    def SetNpMix(self, input:int):
        """
        Set number of divisions between lean and rich mixture status for flamelet generation.

        :param input: Number of divisions between leanest and richest pre-mixed solution. 
        :type input: int
        :raise: Exception: If the number of divisions is lower than one.

        """
        if (input <= 0):
            raise Exception("Flamelets should be generated for at least one mixture status value.")
        else:
            self.__Np_mix_unb = input 

    def GetNpMix(self):
        """
        Get the number of divisions between the lean and rich mixture status for flamelet generation.

        :return: number of divisions between rich and lean.
        :rtype: int

        """
        return self.__Np_mix_unb

    def SetUnbTempBounds(self, T_unb_lower:float, T_unb_upper:float):
        """
        Define lower and upper reactant temperature for flamelet data generation.

        :param T_unb_lower: Lower reactant temperature in Kelvin.
        :type T_unb_lower: float
        :param T_unb_upper: Upper reactant temperature in Kelvin.
        :type T_unb_upper: float
        :raise: Exception: if lower temperature value exceeds upper temperature value.

        """

        if (T_unb_lower >= T_unb_upper):
            raise Exception("Lower unburnt temperature bound should be below upper bound.")
        else:
            self.__T_unb_upper = T_unb_upper
            self.__T_unb_lower = T_unb_lower

    def GetUnbTempBounds(self):
        """
        Get the reactant temperature bounds for flamelet generation.
        
        :return: lower and upper reactant temperature.
        :rtype: list[float]

        """
        return [self.__T_unb_lower, self.__T_unb_upper]
    
    def SetNpTemp(self, Np_Temp:int):
        """
        Set number of divisions for the reactant temperature at each mixture fraction/equivalence ratio.

        :param Np_Temp: Number of divisions for the reactant temperature range.
        :type Np_Temp: int
        :rase: Exception: If the number of divisions is lower than one.

        """
        if (Np_Temp <= 0):
            raise Exception("Number of unburnt temperature samples should be higher than one.")
        else:
            self.__Np_T_unb = Np_Temp
    
    def GetNpTemp(self):
        """
        Get the number of divisions for the reactant temperature range.

        :return: Number of divisions for the reactant temperature range.
        :rtype: int

        """
        return self.__Np_T_unb 

    def DefineMixStatus(self, run_as_mixture_fraction:bool):
        """
        Define the reactant mixture status as mixture fraction or equivalence ratio.

        :param run_as_mixture_fraction: If set to `True`, the mixture status is defined as mixture fraction, while `False` defines it as equivalence ratio.

        """
        self.__run_mixture_fraction = run_as_mixture_fraction
    
    def GetMixtureStatus(self):
        """
        Get the mixture status definition of the current FlameletAIConfig class.

        :return: mixture status definition (`True` for mixture fraction, `False` for equivalence ratio)
        :rtype: bool

        """
        return self.__run_mixture_fraction
    
    def RunFreeFlames(self, input:bool):
        """
        Include adiabatic free flame data in the manifold.

        :param input: enable generation of adabatic free flame data.
        :type input: bool

        """
        self.__generate_freeflames = input 

    def RunBurnerFlames(self, input:bool):
        """
        Include burner-stabilized flame data in the manifold.

        :param input: enable generation of burner-stabilized flame data.
        :type input: bool

        """
        self.__generate_burnerflames = input 

    def RunEquilibrium(self, input:bool):
        """
        Include chemical equilibrium (reactants and products) data in the manifold.

        :param input: enable generation of chemical equilibrium data.
        :type input: bool

        """
        self.__generate_equilibrium = input

    def RunCounterFlames(self, input:bool):
        """
        Include counter-flow diffusion flame data in the manifold.

        :param input: enable generation of counter-flow diffusion flamelets.
        :type input: bool

        """
        self.__generate_counterflames = input 
        
    def GenerateFreeFlames(self):
        """
        Whether the manifold data contains adiabatic free-flame data.

        :return: adiabatic free flamelets are generated.
        :rtype: bool
        """
        return self.__generate_freeflames
    
    def GenerateBurnerFlames(self):
        """
        Whether the manifold data contains burner-stabilized flame data.

        :return: burner-stabilized flamelets are generated.
        :rtype: bool
        """
        return self.__generate_burnerflames
    
    def GenerateEquilibrium(self):
        """
        Whether the manifold data contains chemical equilibrium data.

        :return: chemical equilibrium data are generated.
        :rtype: bool
        """
        return self.__generate_equilibrium
    
    def GenerateCounterFlames(self):
        """
        Whether the manifold data contains counter-flow diffusion flame data.

        :return: counter-flow diffusion flames are generated.
        :rtype: bool
        """
        return self.__generate_counterflames
    
    def TranslateToMatlab(self, input:bool):
        """
        Save a copy of flamelet data as MATLAB TableMaster format.

        :param input: save a MATLAB TableMaster copy of flamelet data.
        """
        self.__write_MATLAB_files = input 

    def WriteMatlabFiles(self):
        """
        Save a copy of flamelet data as MATLAB TableMaster format.

        :return: save a MATLAB TableMaster copy of flamelet data.
        :rtype: bool
        """
        return self.__write_MATLAB_files
    
    def SetProgressVariableDefinition(self, pv_species:list[str], pv_weights:list[float]):
        """
        Set the progress variable species and mass-fraction weights.

        :param pv_species: list of species names defining the progress variable.
        :type pv_species: list[str]
        :param pv_weights: list of progress variable weights for respective species mass fractions.
        :type pv_weights: list[float]
        :raise: Exception: if the provided species and weights have unequal length.
        """
        if (len(pv_species) != len(pv_weights)):
            raise Exception("Number of species and weights of the progress variable definition should be equal.")
        else:
            self.__pv_definition = pv_species
            self.__pv_weights = pv_weights
        return 
    
    def GetProgressVariableSpecies(self):
        """
        Get the progress variable species names.

        :return: list of species names defining the progress variable.
        :rtype: list[str]
        """
        return self.__pv_definition
    
    def GetProgressVariableWeights(self):
        """
        Get the progress variable species weights.

        :return: list of species mass fraction weights defining the progress variable.
        :rtype: list[float]
        """
        return self.__pv_weights
    
    def SetPassiveSpecies(self, __passive_species:list[str]):
        """
        Set the passive transported species for which source terms should be saved in the manifold.

        :param __passive_species: list of species names.
        :type __passive_species: list[str]
        """
        self.__passive_species = __passive_species
        return 
    
    def GetPassiveSpecies(self):
        """
        Get the list of passive species included in the manifold data.

        :return: list of passive species names.
        :rtype: list[str]
        """
        return self.__passive_species
    
    def SetLookUpVariables(self, lookup_vars:list[str]):
        """
        Define passive look-up terms to be included in the manifold.

        :param lookup_vars: list of passive variable names.
        :type lookup_vars: list[str]
        """
        self.__lookup_variables = lookup_vars

    def GetLookUpVariables(self):
        """
        Get the variable names of the passive look-up variables in the manifold.

        :return: __lookup_variables: list of passive look-up variables.
        :rtype: list[str]
        """
        return self.__lookup_variables
    
    def SetNpConcatenation(self, Np_input:int):
        """
        Set the number of query points per flamelet used to define the manifold.

        :param Np_input: number of data points per flamelet.
        :type Np_input: int
        :raise: Exception: if the number of data points is lower than 2
        """
        if (Np_input < 2):
            raise Exception("Number of interpolation points per flamelet should be higher than two.")
        else:
            self.__Np_per_flamelet = Np_input
    
    def GetNpConcatenation(self):
        """
        Get the number of data points per flamelet used to define the manifold.

        :return: number of data points per flamelet.
        :rtype: int
        """
        return self.__Np_per_flamelet
    
    def ComputeProgressVariable(self, variables:list[str], flamelet_data:np.ndarray, Y_flamelet:np.ndarray=None):
        """
        Compute the progress variable based on the corresponding progress variable definition for an array of provided flamelet data.

        :param variables: list of variable names in the flamelet data.
        :type variables: list[str]
        :param flamelet_data: flamelet data array.
        :type flamelet_data: np.ndarray
        :raise: Exception: if number of variables does not match number of columns in flamelet data.
        :return: Progress variable array.
        :rtype: np.array
        """
        if Y_flamelet is not None:
            if np.shape(Y_flamelet)[0] != self.gas.n_species:
                raise Exception("Number of species does not match mass fraction array content.")
            pv = np.zeros(np.shape(Y_flamelet)[1])
            for pv_w, pv_sp in zip(self.__pv_weights, self.__pv_definition):
                pv += pv_w * Y_flamelet[self.gas.species_index(pv_sp), :]
            return pv 
        else:
            if len(variables) != np.shape(flamelet_data)[1]:
                raise Exception("Number of variables does not match data array.")
            
            pv = np.zeros(np.shape(flamelet_data)[0])
            for iPv, pvSp in enumerate(self.__pv_definition):
                pv += self.__pv_weights[iPv] * flamelet_data[:, variables.index("Y-"+pvSp)]
            return pv 

    
    def ComputeProgressVariable_Source(self, variables:list[str], flamelet_data:np.ndarray,net_production_rate_flamelet:np.ndarray=None):
        """
        Compute the progress variable source term based on the corresponding progress variable definition for an array of provided flamelet data.

        :param variables: list of variable names in the flamelet data.
        :type variables: list[str]
        :param flamelet_data: flamelet data array.
        :type flamelet_data: np.ndarray
        :raise: Exception: if number of variables does not match number of columns in flamelet data.
        :return: Progress variable source terms.
        :rtype: np.array
        """

        if net_production_rate_flamelet is not None:
            if np.shape(net_production_rate_flamelet)[0] != self.gas.n_species:
                raise Exception("Number of species does not match mass fraction array content.")
            ppv = np.zeros(np.shape(net_production_rate_flamelet)[1])
            for pv_w, pv_sp in zip(self.__pv_weights, self.__pv_definition):
                ppv += pv_w * net_production_rate_flamelet[self.gas.species_index(pv_sp), :]\
                    * self.gas.molecular_weights[self.gas.species_index(pv_sp)]
            return ppv
        else:
            if len(variables) != np.shape(flamelet_data)[1]:
                raise Exception("Number of variables does not match data array.")
            ppv = np.zeros(np.shape(flamelet_data)[0])
            for iPv, pvSp in enumerate(self.__pv_definition):
                prodrate_pos = flamelet_data[:, variables.index('Y_dot_pos-'+pvSp)]
                prodrate_neg = flamelet_data[:, variables.index('Y_dot_neg-'+pvSp)]
                mass_fraction = flamelet_data[:, variables.index('Y-'+pvSp)]
                ppv += self.__pv_weights[iPv] * (prodrate_pos + prodrate_neg * mass_fraction)
            return ppv 
    
    def EnablePreferentialDiffusion(self, use_PD:bool=True):
        self.__preferential_diffusion = use_PD 
        
    def PreferentialDiffusion(self):
        return self.__preferential_diffusion
    
    def ComputeBetaTerms(self, variables:list[str], flamelet_data:np.ndarray):
        """
        Compute the differential diffusion scalars for a flamelet.

        
        :param variables: list of variable names in the flamelet data.
        :type variables: list[str]
        :param flamelet_data: flamelet data array.
        :type flamelet_data: np.ndarray
        :raise: Exception: if number of variables does not match number of columns in flamelet data.
        :return: Differential diffusion coefficients
        :rtype: float
        """
        if len(variables) != np.shape(flamelet_data)[1]:
            raise Exception("Number of variables does not match data array.")
        
        beta_z = np.zeros(len(flamelet_data))
        beta_h1 = flamelet_data[:, variables.index("Cp")] * np.ones(len(flamelet_data))
        cp_c = flamelet_data[:, variables.index("Cp-"+self.__carrier_specie)]
        h_c = flamelet_data[:, variables.index("h-"+self.__carrier_specie)]
        beta_h2 = np.zeros(len(flamelet_data))

        for iSp in range(len(self.__species_in_mixture)):
            # Get flamelet species Lewis number trend.
            Le_sp = flamelet_data[:, variables.index("Le-"+self.__species_in_mixture[iSp])]
            Le_av = np.average(Le_sp)
            # Get species mass fraction
            Y_sp = flamelet_data[:, variables.index("Y-"+self.__species_in_mixture[iSp])]
            beta_z += (self.__mixfrac_coefficients[iSp] - self.__mixfrac_coeff_carrier) * Y_sp / Le_av

            cp_i = flamelet_data[:, variables.index("Cp-"+self.__species_in_mixture[iSp])]
            beta_h1 -= (cp_i - cp_c) * Y_sp / Le_av

            h_i = flamelet_data[:, variables.index("h-"+self.__species_in_mixture[iSp])]
            beta_h2 += (h_i - h_c) * Y_sp / Le_av
        
        beta_pv = np.zeros(len(flamelet_data))
        for iPv in range(len(self.__pv_definition)):
            Le_sp = flamelet_data[:, variables.index("Le-"+self.__pv_definition[iPv])]
            Le_av = np.average(Le_sp)
            beta_pv += self.__pv_weights[iPv] * flamelet_data[:, variables.index("Y-"+self.__pv_definition[iPv])] / Le_av

        return beta_pv, beta_h1, beta_h2, beta_z
    
    def GetUnburntScalars(self, equivalence_ratio:float, temperature:float):
        """
        Compute the reactant progress variable, total enthalpy, and mixture fraction for a given equivalence ratio and temperature.

        :param equivalence_ratio: reactants equivalence ratio.
        :type equivalence_ratio: float
        :param temperature: reactants temperature in Kelvin.
        :type temperature: float
        :raise: Warning: If temperature or equivalence ratio exceed flamelet data range.
        :return: pv_unb: reactants progress variable value.
        :rtype: pv_unb: float
        :return enth_unb: reactants total enthalpy value.
        :rtype enth_unb: float
        :return mixfrac_unb: reactants mixture fraction value.
        :rtype mixfrac_unb: float
        """

        if equivalence_ratio < 0:
            raise Exception("Equivalence ratio should be positive.")
        if temperature < 200:
            raise Exception("Temperature should be above 200 degrees Kelvin.")
        
        fuel_string = ",".join(self.__fuel_species[i]+":"+str(self.__fuel_weights[i]) for i in range(len(self.__fuel_species)))
        oxidizer_string = ",".join(self.__oxidizer_species[i]+":"+str(self.__oxidizer_weights[i]) for i in range(len(self.__oxidizer_species)))
        self.gas.set_equivalence_ratio(equivalence_ratio, fuel_string, oxidizer_string)
        self.gas.TP = temperature, ct.one_atm

        pv_unb = 0
        for iPV, pvSp in enumerate(self.__pv_definition):
            pv_unb += self.__pv_weights[iPV] * self.gas.Y[self.gas.species_index(pvSp)]
        
        enth_unb = self.gas.enthalpy_mass

        mixfrac_unb = self.gas.mixture_fraction(fuel_string, oxidizer_string)

        if (temperature > self.__T_unb_upper) or (temperature < self.__T_unb_lower):
            raise Warning("Provided temperature is outside flamelet data bounds.")
        if self.__run_mixture_fraction:
            if (mixfrac_unb < self.__mix_status_lower) or (mixfrac_unb > self.__mix_status_upper):
                raise Warning("Provided equivalence ratio exceeds flamelet data range.")
        else:
            if (equivalence_ratio < self.__mix_status_lower) or (equivalence_ratio > self.__mix_status_upper):
                raise Warning("Provided equivalence ratio exceeds flamelet data range.")
            
        return pv_unb, enth_unb, mixfrac_unb
    
    def GetBurntScalars(self, equivalence_ratio:float, temperature:float=300):
        """
        Compute the reaction products (chemical equilibrium) progress variable, total enthalpy, and mixture fraction for a given equivalence ratio and temperature.

        :param equivalence_ratio: reactants equivalence ratio.
        :type equivalence_ratio: float
        :param temperature: products temperature in Kelvin.
        :type temperature: float
        :raise: Warning: If temperature or equivalence ratio exceed flamelet data range.
        :return pv_b: products progress variable value.
        :rtype pv_b: float
        :return enth_b: products total enthalpy value.
        :rtype enth_b: float
        :return mixfrac_b: products mixture fraction value.
        :rtype mixfrac_b: float
        """
        fuel_string = ",".join(self.__fuel_species[i]+":"+str(self.__fuel_weights[i]) for i in range(len(self.__fuel_species)))
        oxidizer_string = ",".join(self.__oxidizer_species[i]+":"+str(self.__oxidizer_weights[i]) for i in range(len(self.__oxidizer_species)))
        self.gas.set_equivalence_ratio(equivalence_ratio, fuel_string, oxidizer_string)
        self.gas.TP = temperature, ct.one_atm
        self.gas.equilibrate('TP')
        pv_b = 0
        for iPV, pvSp in enumerate(self.__pv_definition):
            pv_b += self.__pv_weights[iPV] * self.gas.Y[self.gas.species_index(pvSp)]
        
        enth_b = self.gas.enthalpy_mass

        mixfrac_b = self.gas.mixture_fraction(fuel_string, oxidizer_string)
        
        if (temperature > self.__T_unb_upper) or (temperature < self.__T_unb_lower):
            raise Warning("Provided temperature is outside flamelet data bounds.")
        if self.__run_mixture_fraction:
            if (mixfrac_b < self.__mix_status_lower) or (mixfrac_b > self.__mix_status_upper):
                raise Warning("Provided equivalence ratio exceeds flamelet data range.")
        else:
            if (equivalence_ratio < self.__mix_status_lower) or (equivalence_ratio > self.__mix_status_upper):
                raise Warning("Provided equivalence ratio exceeds flamelet data range.")
            
        return pv_b, enth_b, mixfrac_b
    
    def SetTableCellSize(self, base_cell_size:float, refined_cell_size:float=None):
        """Define the base and optional refined 2D table cell sizes.

        :param base_cell_size: Normalized base cell size applied to each table level.
        :type base_cell_size: float
        :param refined_cell_size: Optional refined cell size to be applied in high-curvature regions, defaults to None
        :type refined_cell_size: float, optional
        :raises Exception: If base cell size is lower or equal to zero.
        :raises Exception: If supplied refined cell size is larger than base cell size.
        """
        if base_cell_size <= 0:
            raise Exception("Normalized base cell size should be larger than zero.")
        if refined_cell_size != None:
            if refined_cell_size > base_cell_size:
                raise Exception("Refined cell size should be smaller than base cell size.")
            self.__Table_ref_cell_size = refined_cell_size
        else:
            self.__Table_ref_cell_size = base_cell_size
        self.__Table_base_cell_size = base_cell_size

    def GetTableCellSize(self):
        """Returns the base and refined table 2D cell size values.

        :return: base cell size, refined cell size
        :rtype: float, float
        """
        return self.__Table_base_cell_size, self.__Table_ref_cell_size
    
    def SetTableRefinement(self, refinement_radius:float, refinement_threshold:float):
        """Define the table refinement occurance parameters.

        :param refinement_radius: Normalized radius around refinement location within which refinement is applied.
        :type refinement_radius: float
        :param refinement_threshold: Normalized reaction rate curvature threshold above which refinement is applied.
        :type refinement_threshold: float
        :raises Exception: If refinement radius is not between zero and one.
        :raises Exception: If curvature threshold is not between zero and one.
        """
        if refinement_radius <= 0 or refinement_radius >= 1:
            raise Exception("Refinement radius should be between zero and one.")
        if refinement_threshold <= 0 or refinement_threshold >= 1:
            raise Exception("Refinement threshold should be between zero and one.")
        self.__Table_ref_radius = refinement_radius 
        self.__Table_curv_threshold = refinement_threshold 

    def GetTableRefinement(self):
        """Returns the table refinement radius and refinement threshold values.

        :return: Normalized table refinement radius, normalized refinement curvature threshold.
        :rtype: float, float
        """
        return self.__Table_ref_radius, self.__Table_curv_threshold
    
    def SetTableLevelCount(self, level_count:int):
        """Define the number of 2D table levels along the mixture fraction direction.

        :param level_count: Number of table levels.
        :type level_count: int
        :raises Exception: If the number of levels is lower than 2.
        """
        if level_count < 2:
            raise Exception("Number of table levels should be higher than 2.")
        self.__Table_level_count = level_count 

    def GetTableLevelCount(self):
        """Returns the number of levels in the table.

        :return: Number of table levels.
        :rtype: int
        """
        return self.__Table_level_count
    
    def SetTableMixtureFractionLimits(self, mixfrac_lower:float, mixfrac_upper:float):
        """Define the mixture fraction values between which table levels are generated.

        :param mixfrac_lower: lower limit mixture fraction value.
        :type mixfrac_lower: float
        :param mixfrac_upper: upper limit mixture fraction value.
        :type mixfrac_upper: float
        :raises Exception: if the lower mixture fraction value is lower than 0 or the upper mixture fraction value exceeds 1.
        :raises Exception: if the lower mixture fraction value equals or exceeds the upper mixture fraction value.
        """
        if mixfrac_lower < 0 or mixfrac_upper > 1:
            raise Exception("Lower and upper mixture fraction values should be between zero and one.")
        if mixfrac_lower >= mixfrac_upper:
            raise Exception("Upper mixture fraction value should be higher than lower mixture fraction value.")
        self.__Table_mixfrac_lower = mixfrac_lower
        self.__Table_mixfrac_upper = mixfrac_upper

    def GetTableMixtureFractionLimits(self):
        """Returns the lower and upper mixture fraction limits of the table between which table levels are generated.

        :return: lower mixture fraction value, upper mixture fraction value
        :rtype: float, float
        """
        return self.__Table_mixfrac_lower, self.__Table_mixfrac_upper
    
    def SetMLPArchitecture(self, group_index:int, architecture_in:list[int]):
        """Save an MLP architecture for a given group index.

        :param group_index: MLP output group index.
        :type group_index: int
        :param architecture_in: number of neurons per hidden layer.
        :type architecture_in: list[int]
        :raises Exception: if the provided list of neuron counts is empty.
        :raises Exception: if the provided group index exceeds the number of MLPs.
        """
        if len(architecture_in) == 0:
            raise Exception("MLP should have at least one layer.")
        if group_index >= len(self.__MLP_architectures):
            raise Exception("Group index should be below "+str(len(self.__MLP_architectures)))
        
        if self.__MLP_architectures == None:
            self.__MLP_architectures = []
            for iGroup in range(len(self.__MLP_output_groups)):
                self.__MLP_architectures[iGroup] = [] 
        
        self.__MLP_architectures[group_index] = []
        for NN in architecture_in:
            self.__MLP_architectures[group_index].append(NN)

    def GetMLPArchitecture(self, group_index:int):
        """Get the saved MLP architecture for a given group.

        :param group_index: MLP output group index.
        :type group_index: int
        :return: neurons per hidden layer in the network.
        :rtype: list[int]
        """
        return self.__MLP_architectures[group_index]
    
    def SetTrainParams(self, group_index:int, alpha_expo:float, lr_decay:float, batch_expo:int, activation_index:int):
        """Save the training parameters for a set MLP.

        :param group_index: MLP output group index.
        :type group_index: int
        :param alpha_expo: alpha exponent value for the learning rate.
        :type alpha_expo: float
        :param lr_decay: learning rate decay parameter.
        :type lr_decay: float
        :param batch_expo: mini-batch size exponent.
        :type batch_expo: int
        :param activation_index: activation function index for hidden layers.
        :type activation_index: int
        :raises Exception: if the provided group index exceeds the number of MLPs.
        """
        if group_index >= len(self.__MLP_architectures):
            raise Exception("Group index should be below "+str(len(self.__MLP_architectures)))
        
        if self.__MLP_trainparams == None:
            self.__MLP_trainparams = []
            for iGroup in range(len(self.__MLP_output_groups)):
                self.__MLP_trainparams[iGroup] = [] 
        
        self.__MLP_trainparams[iGroup] = []
        self.__MLP_trainparams[iGroup].append(alpha_expo)
        self.__MLP_trainparams[iGroup].append(lr_decay)
        self.__MLP_trainparams[iGroup].append(batch_expo)
        self.__MLP_trainparams[iGroup].append(activation_index)

    def GetTrainParams(self, group_index:int):
        """Get the saved training parameters for a given MLP

        :param group_index: MLP output group index.
        :type group_index: int
        :return: learning rate alpha exponent, learning rate decay parameter, batch size exponent, and activation function index.
        :rtype: float, float, int, int
        """

        alpha_expo = self.__MLP_architectures[group_index][0]
        lr_decay = self.__MLP_architectures[group_index][1]
        batch_expo = self.__MLP_architectures[group_index][2]
        activation_index = self.__MLP_architectures[group_index][3]

        return alpha_expo, lr_decay, batch_expo, activation_index
        
    def AddOutputGroup(self, variable_names_in:list[str]):
        """Add an MLP output group of flamelet variables.

        :param variable_names_in: list of output variables to include in the outputs of a single MLP.
        :type variable_names_in: list[str]
        :raises Exception: if the number of variables is equal to zero.
        """
        if len(variable_names_in) == 0:
            raise Exception("An MLP output group should be made up of at least one variable.")
        
        if self.__MLP_output_groups == None:
            self.__MLP_output_groups = []
        self.__MLP_output_groups.append([])
        for var in variable_names_in:
            self.__MLP_output_groups[-1].append(var)

    def DefineOutputGroup(self, i_group:int, variable_names_in:list[str]):
        """Re-define the variables in a specific MLP output group.

        :param i_group: MLP output group index to adapt.
        :type i_group: int
        :param variable_names_in: list of output variables to include in the outputs of a single MLP.
        :type variable_names_in: list[str]
        :raises Exception: if group index exceeds number of output groups.
        :raises Exception: if the number of variables is equal to zero.
        """
        if i_group >= len(self.__MLP_output_groups):
            raise Exception("Group not present in MLP outputs.")
        if len(variable_names_in) == 0:
            raise Exception("An MLP output group should be made up of at least one variable.")
        
        self.__MLP_output_groups[i_group] = []
        for var in variable_names_in:
            self.__MLP_output_groups[i_group].append(var)

    def RemoveOutputGroup(self, i_group:int):
        if i_group > len(self.__MLP_output_groups):
            raise Exception("Group not present in MLP outputs.")
        print("Removing output group %i: %s" % (i_group, ",".join(s for s in self.__MLP_output_groups[i_group-1])))
        self.__MLP_output_groups.remove(self.__MLP_output_groups[i_group-1])

    def ClearOutputGroups(self):
        self.__MLP_output_groups = []
        return 
    
    def DisplayOutputGroups(self):
        """Print the MLP output variables grouping arrangement.
        """
        for i_group, group in enumerate(self.__MLP_output_groups):
            print("Group %i: " % (i_group+1) +",".join(var for var in group) )
    
    def GetNMLPOutputGroups(self):
        """Get the number of MLP output groups.

        :return: number of MLP output groups
        :rtype: int
        """
        return len(self.__MLP_output_groups)
    
    def GetMLPOutputGroup(self, i_group:int):
        """Get the list of variables in a specific MLP group.

        :param i_group: MLP output group index.
        :type i_group: int
        :return: list of variables for this group.
        :rtype: list[str]
        """
        return self.__MLP_output_groups[i_group]
    
    def SaveConfig(self, file_name:str):
        """
        Save the current FlameletAI configuration.

        :param file_name: configuration file name.
        :type file_name: str
        """
        self.ComputeMixFracConstants()
        

        self.__species_in_mixture = self.gas.species_names

        self._config_name = file_name
        file = open(file_name+'.cfg','wb')
        pickle.dump(self, file)
        file.close()

