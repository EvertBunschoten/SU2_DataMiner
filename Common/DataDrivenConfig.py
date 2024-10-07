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
from Common.Properties import DefaultSettings_NICFD, DefaultSettings_FGM
from Common.Config_base import Config 
from Common.CommonMethods import *

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
    __fluid_names:list[str] = ["MM"]                    # List of fluid names used for data generation.
    __fluid_string:str="MM"                             # Fluid string for defining the abstract state in CoolProp
    __EOS_type:str=DefaultSettings_NICFD.EOS_type       # Equation of state used by CoolProp
    __fluid_mole_fractions:list[float] = [1.0]          # Mole fractions for components in fluid mixture.
    __use_PT:bool = DefaultSettings_NICFD.use_PT_grid   # Use a pressure-temperature based grid for fluid training data.


    __T_lower:float = DefaultSettings_NICFD.T_min   # Lower temperature bound.
    __T_upper:float = DefaultSettings_NICFD.T_max   # Upper temperature bound.
    __Np_T:int = DefaultSettings_NICFD.Np_temp      # Number of temperature/energy samples between bounds.

    __P_lower:float = DefaultSettings_NICFD.P_min   # Lower pressure bound.
    __P_upper:float = DefaultSettings_NICFD.P_max   # Upper pressure bound.
    __Np_P:int = DefaultSettings_NICFD.Np_p         # Number of pressure/density samples between bounds.

    __Rho_lower:float = DefaultSettings_NICFD.Rho_min       # Lower density bound.
    __Rho_upper:float = DefaultSettings_NICFD.Rho_max       # Upper density bound.
    __Energy_lower:float = DefaultSettings_NICFD.Energy_min # Lower energy bound.
    __Energy_upper:float = DefaultSettings_NICFD.Energy_max # Upper energy bound.
    
    _state_vars:list[str] = ["T","p","c2"]  # State variable names for which the physics-informed MLP is trained.

    # Table Generation Settings

    __Table_base_cell_size:float = None     # Table base cell size per table level.
    __Table_ref_cell_size:float = None      # Refined cell size per table level.
    __Table_ref_radius:float = None         # Refinement radius within which refined cell size is applied.
    __Table_curv_threshold:float = None     # Curvature threshold beyond which refinement is applied.


    def __init__(self, load_file:str=None):
        """EntropicAI SU2 DataMiner configuration class.

        :param load_file: configuration file name to load, defaults to None
        :type load_file: str, optional
        :raises Exception: if loaded configuration is incompatible with the EntropicAIConfig class.
        """

        Config.__init__(self)
        self._config_type = DefaultSettings_NICFD.config_type

        self._controlling_variables = [DefaultSettings_NICFD.name_density, DefaultSettings_NICFD.name_energy]
        
        # Set default settings.
        self.SetAlphaExpo(DefaultSettings_NICFD.init_learning_rate_expo)
        self.SetLRDecay(DefaultSettings_NICFD.learning_rate_decay)
        self.SetBatchExpo(DefaultSettings_NICFD.batch_size_exponent)
        self.SetHiddenLayerArchitecture(DefaultSettings_NICFD.hidden_layer_architecture)
        self.SetActivationFunction(DefaultSettings_NICFD.activation_function)
        self._config_name = DefaultSettings_NICFD.config_name

        if load_file:
            print("Loading configuration for entropic model generation...")
            with open(load_file, "rb") as fid:
                loaded_config = pickle.load(fid)
            if loaded_config._config_type != self._config_type:
                raise Exception("Improper configuration file for EntropicAI configuration.")
            
            self.__dict__ = loaded_config.__dict__.copy()
            print("Loaded configuration file with name " + loaded_config._config_name)
        else:
            print("Generating empty EntropicAI config")

        return 
    
    def PrintBanner(self):
        """Print banner visualizing EntropicAI configuration settings."""
        super().PrintBanner()

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
        print("")
        print("State variables considered during physics-informed learning: "+", ".join((v for v in self._state_vars)))
        return 
    

    def SetFluid(self, fluid_name):
        """
        Define the fluid name used for entropic data generation. By default, \"MM\" is used.

        :param fluid_name: CoolProp fluid name or list of names.
        :type fluid_name: str or list[str]
        :raise: Exception: If the fluid could not be defined in CoolProp.

        """

        # Check if one or multiple fluids are provided.
        if type(fluid_name) == list:
            # if len(fluid_name) > 2:
            #     raise Exception("Only two fluids can be used for mixtures")
            
            self.__fluid_names = []
            fluid_mixing = []
            for f in fluid_name:
                self.__fluid_names.append(f)
            if len(self.__fluid_mole_fractions) == 0:
                self.__fluid_mole_fractions = np.ones(len(self.__fluid_names))/len(self.__fluid_names)

        elif type(fluid_name) == str:
            self.__fluid_names = [fluid_name]
        
        fluid_string = "&".join(f for f in self.__fluid_names)
        self.__fluid_string=fluid_string
        try:
            CoolProp.AbstractState("HEOS", fluid_string)
        except:
            raise Exception("Specified fluid name not found or mixture is not supported.")
        return 
    
    def SetEquationOfState(self, EOS_type_in:str=DefaultSettings_NICFD.EOS_type):
        self.__EOS_type=EOS_type_in 
        return
    
    def GetEquationOfState(self):
        return self.__EOS_type 
    
    def SetFluidMoleFractions(self, mole_fractions:list[float]):
        """Set fluid mole fractions for mixture.

        :param mole_fraction_1: _description_, defaults to 0.5
        :type mole_fraction_1: float, optional
        :param mole_fraction_2: _description_, defaults to 0.5
        :type mole_fraction_2: float, optional
        :raises Exception: if either mole fraction value is negative.
        """
        if len(mole_fractions) != len(self.__fluid_names):
            raise Exception("Number of mole fractions should match the number of species")
        
        m_sum = 0
        for m in mole_fractions:
            if m < 0:
                raise Exception("Mole fractions should be positive.")
            m_sum += m 
        mole_fractions_norm = np.array(mole_fractions)/m_sum
        # Normalize molar fractions
        self.__fluid_mole_fractions = mole_fractions_norm
        return 
        
    def GetFluid(self):
        """
        Get the fluid used for entropic data generation.
        :return: fluid name
        :rtype: str

        """
        return self.__fluid_string
    
    def GetFluidNames(self):
        return self.__fluid_names.copy()
    
    def GetMoleFractions(self):
        return self.__fluid_mole_fractions.copy()
    
    def UsePTGrid(self, PT_grid:bool=DefaultSettings_NICFD.use_PT_grid):
        """Define fluid data grid in the pressure-temperature space. If not, the fluid data grid is defined in the density-energy space.

        :param PT_grid: use pressure-temperature based grid, defaults to DefaultSettings_NICFD.use_PT_grid
        :type PT_grid: bool, optional
        """
        self.__use_PT = PT_grid 
        return 
    
    def GetPTGrid(self):
        """Get the fluid data grid definition.

        :return: Fluid data grid definition  (pressure-temperature based = True, density-energy-based = False).
        :rtype: bool
        """

        return self.__use_PT 
    
    def SetTemperatureBounds(self, T_lower:float=DefaultSettings_NICFD.T_min, T_upper:float=DefaultSettings_NICFD.T_max):
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
    
    def SetEnergyBounds(self, E_lower:float=DefaultSettings_NICFD.Energy_min, E_upper:float=DefaultSettings_NICFD.Energy_max):
        """Define the internal energy bounds of the density-energy based fluid data grid.

        :param E_lower: lower limit internal energy value, defaults to DefaultSettings_NICFD.Energy_min
        :type E_lower: float, optional
        :param E_upper: upper limit for internal energy, defaults to DefaultSettings_NICFD.Energy_max
        :type E_upper: float, optional
        :raises Exception: if lower value for internal energy exceeds upper value.
        """
        if (E_lower >= E_upper):
            raise Exception("Lower energy level should be below upper energy level.")
        else:
            self.__Energy_lower=E_lower
            self.__Energy_upper=E_upper
        return 
    
    def GetEnergyBounds(self):
        """Get the interal energy bounds for the density-energy based fluid data grid.

        :return: lower and upper internal energy values.
        :rtype: list[float]
        """

        return [self.__Energy_lower, self.__Energy_upper]
    
    def SetNpEnergy(self, Np_Energy:int=DefaultSettings_NICFD.Np_temp):
        """Set the number of data points along the energy axis of the fluid data grid.

        :param Np_Energy: number of divisions between the lower and upper internal energy values, defaults to DefaultSettings_NICFD.Np_temp
        :type Np_Energy: int, optional
        :raises Exception: if fewer than two nodes are set.
        """

        if Np_Energy <= 2:
            raise Exception("Number of divisions should be higher than 2.")
        else:
            self.__Np_T = Np_Energy

        return 
    
    def GetNpEnergy(self):
        """
        Get the number of divisions for the fluid static energy range.

        :return: Number of divisions for the fluid static energy range.
        :rtype: int

        """
        return self.__Np_T
    
    def SetDensityBounds(self, Rho_lower:float=DefaultSettings_NICFD.Rho_min, Rho_upper:float=DefaultSettings_NICFD.Rho_max):
        """Define the density bounds of the density-energy based fluid data grid.

        :param Rho_lower: lower limit density value, defaults to DefaultSettings_NICFD.Rho_min
        :type Rho_lower: float, optional
        :param Rho_upper: upper limit for density, defaults to DefaultSettings_NICFD.Rho_max
        :type Rho_upper: float, optional
        :raises Exception: if lower value for density exceeds upper value.
        """
        if (Rho_lower >= Rho_upper):
            raise Exception("Lower density value should not exceed upper value.")
        else:
            self.__Rho_lower=Rho_lower
            self.__Rho_upper=Rho_upper
        return 
    
    def SetNpDensity(self, Np_rho:int=DefaultSettings_NICFD.Np_p):
        """Set the number of data points along the density axis of the fluid data grid.

        :param Np_rho: number of divisions between the lower and upper density values, defaults to DefaultSettings_NICFD.Np_p
        :type Np_rho: int, optional
        :raises Exception: if fewer than two nodes are set.
        """

        if Np_rho <= 2:
            raise Exception("Number of divisions should be higher than two.")
        else:
            self.__Np_P = Np_rho

        return 
    
    def GetNpDensity(self):
        """
        Get the number of divisions for the fluid density range.

        :return: Number of divisions for the fluid density range.
        :rtype: int

        """
        return self.__Np_P
    
    def GetDensityBounds(self):
        """Get the density bounds for the density-energy based fluid data grid.

        :return: lower and upper density values.
        :rtype: list[float]
        """
        return [self.__Rho_lower, self.__Rho_upper]
    
    def GetTemperatureBounds(self):
        """Get the temperature bounds for the pressure-temperature based fluid data grid.

        :return: lower and upper temperature values.
        :rtype: list[float]
        """
        return [self.__T_lower, self.__T_upper]
    
    
    def SetNpTemp(self, Np_Temp:int=DefaultSettings_NICFD.Np_temp):
        """
        Set number of divisions for the temperature grid.

        :param Np_Temp: Number of divisions for the temperature range.
        :type Np_Temp: int
        :rase: Exception: If the number of divisions is lower than 2.

        """
        if (Np_Temp <= 0):
            raise Exception("Number of divisions should be higher than two.")
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
    

    def SetPressureBounds(self, P_lower:float=DefaultSettings_NICFD.P_min, P_upper:float=DefaultSettings_NICFD.P_max):
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
        """Get the pressure bounds for the pressure-temperature based fluid data grid.

        :return: lower and upper pressure values.
        :rtype: list[float]
        """
        return [self.__P_lower, self.__P_upper]
    
    
    def SetNpPressure(self, Np_P:int=DefaultSettings_NICFD.Np_p):
        """
        Set number of divisions for the fluid pressure grid.

        :param Np_Temp: Number of divisions for the fluid pressure.
        :type Np_Temp: int
        :rase: Exception: If the number of divisions is lower than two.

        """
        if (Np_P <= 2):
            raise Exception("At least two divisions should be provided.")
        else:
            self.__Np_P = Np_P 
        return 
    
    def GetNpPressure(self):
        """
        Get the number of divisions for the fluid pressure range.

        :return: Number of divisions for the fluid pressure range.
        :rtype: int

        """
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
    
        
    def SetStateVars(self, state_vars_in:list[str]):
        """Set the state variables for which the physics-informed neural network is trained.

        :param state_vars_in: list with state variable names.
        :type state_vars_in: list[str]
        :raises Exception: if any of the state variables is not supported.
        """

        if any((v not in DefaultSettings_NICFD.supported_state_vars) for v in state_vars_in):
            raise Exception("Only the following state variables are supported: "+ ",".join((v for v in DefaultSettings_NICFD.supported_state_vars)))
        self._state_vars = state_vars_in.copy()

        return 
    
    def GetStateVars(self):
        """Return the list of state variable names for which the physics-informed MLP is trained.

        :return: list of state variable names.
        :rtype: list[str]
        """

        return self._state_vars
    
    


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

    # Flamelet Generation Settings           
    __reaction_mechanism:str = DefaultSettings_FGM.reaction_mechanism   # Reaction mechanism name.
    __transport_model:str = DefaultSettings_FGM.transport_model 

    __fuel_species:list[str] = DefaultSettings_FGM.fuel_definition # Fuel species composition.
    __fuel_weights:list[float] = DefaultSettings_FGM.fuel_weights  # Fuel species weights.
    __fuel_string:str 

    __oxidizer_species:list[str] = DefaultSettings_FGM.oxidizer_definition   # Oxidizer species composition.
    __oxidizer_weights:list[float] = DefaultSettings_FGM.oxidizer_weights    # Oxidizer species weights.
    __oxidizer_string:str

    __carrier_specie:str = DefaultSettings_FGM.carrier_specie # Carrier specie definition.

    __run_mixture_fraction:bool = DefaultSettings_FGM.run_mixture_fraction    # Define premixed status as mixture fraction (True) or as equivalence ratio (False)
    __preferential_diffusion:bool = DefaultSettings_FGM.preferential_diffusion  # Include preferential diffusion effects. 

    __T_unb_lower:float = DefaultSettings_FGM.T_min   # Lower bound of unburnt reactants temperature.
    __T_unb_upper:float = DefaultSettings_FGM.T_max   # Upper bound of unburnt reactants temperature.
    __Np_T_unb:int = DefaultSettings_FGM.Np_temp      # Number of unburnt temperature samples between bounds.

    __mix_status_lower:float = DefaultSettings_FGM.eq_ratio_min  # Lower bound of premixed status
    __mix_status_upper:float = DefaultSettings_FGM.eq_ratio_max # Upper bound of premixed status
    __Np_mix_unb:int = DefaultSettings_FGM.Np_eq    # Number of mixture samples between bounds.

    __generate_freeflames:bool = DefaultSettings_FGM.include_freeflames      # Generate adiabatic flamelets
    __generate_burnerflames:bool = DefaultSettings_FGM.include_burnerflames   # Generate burner-stabilized flamelets
    __generate_equilibrium:bool = DefaultSettings_FGM.include_equilibrium     # Generate chemical equilibrium data
    __generate_counterflames:bool = DefaultSettings_FGM.include_counterflames   # Generate counter-flow diffusion flamelets.

    __write_MATLAB_files:bool = False  # Write TableGenerator compatible flamelet files.

    gas:ct.Solution = None  # Cantera solution object.
    __species_in_mixture:list[str] = None # Species names in mixture. 

    # Flamelet Data Concatination Settings
    __pv_definition:list[str] = DefaultSettings_FGM.pv_species # Progress variable species.
    __pv_weights:list[float] = DefaultSettings_FGM.pv_weights      # Progress variable mass fraction weights.
    __custom_pv_set:bool = False    # User-defined progress variable 

    __passive_species:list[str] = [] # Passive species for which to generate source terms.

    __lookup_variables:list[str] = ["Heat_Release"] # Extra look-up variables to read from flamelet data 

    __Np_per_flamelet:int = 2**DefaultSettings_FGM.batch_size_exponent    # Number of data points to interpolate from flamelet data.

    # MLP output groups and architecture information.
    __MLP_output_groups:list[list[str]] = None  # Output variables for each MLP.
    __MLP_architectures:list[list[int]] = None  # Hidden layer architecture for each MLP.
    __MLP_trainparams:list[list[float]] = None  # Training parameters and activation function information for each MLP.

    __alpha_expo:list[float] = [DefaultSettings_FGM.init_learning_rate_expo]
    __lr_decay:list[float] = [DefaultSettings_FGM.learning_rate_decay]
    __batch_expo:list[float] = [DefaultSettings_FGM.batch_size_exponent]
    __NN:list[list[int]] = [DefaultSettings_FGM.hidden_layer_architecture]
    __activation_function:list[str] = [DefaultSettings_FGM.activation_function]
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

    __Le_avg_method = avg_Le_arythmic
    __Le_const_sp:np.ndarray[float] = None 
    __custom_Le_av_set:bool = False 

    def __init__(self, load_file:str=None):
        """Class constructor
        """
        Config.__init__(self)

        self._config_type = DefaultSettings_FGM.config_type
        self._config_name = DefaultSettings_FGM.config_name
        
        if load_file:
            print("Loading configuration for flamelet generation")
            with open(load_file, "rb") as fid:
                loaded_config:FlameletAIConfig = pickle.load(fid)
            if loaded_config._config_type != self._config_type:
                raise Exception("Improper configuration file for FlameletAI configuration.")
            self.__dict__ = loaded_config.__dict__.copy()
            print("Loaded configuration file with name " + loaded_config.GetConfigName())
        else:
            self.SetAlphaExpo(DefaultSettings_FGM.init_learning_rate_expo)
            self.SetLRDecay(DefaultSettings_FGM.learning_rate_decay)
            self.SetBatchExpo(DefaultSettings_FGM.batch_size_exponent)
            self.SetHiddenLayerArchitecture(DefaultSettings_FGM.hidden_layer_architecture)
            self.SetControllingVariables(DefaultSettings_FGM.controlling_variables)

            print("Generating empty flameletAI config")
        
        self.__SynchronizeSettings()

        return 
    
    def SetControllingVariables(self, controlling_variables:list[str]=DefaultSettings_FGM.controlling_variables):
        """Define the set of controlling variables for the current manifold.

        :param controlling_variables: list with controlling variable names, defaults to DefaultSettings_FGM.controlling_variables
        :type controlling_variables: list[str], optional
        :raises Exception: if "ProgressVariable" is not included in the list of controlling variables.
        """
        if DefaultSettings_FGM.name_pv not in controlling_variables:
            raise Exception(DefaultSettings_FGM.name_pv + " should be in the controlling variables")
        super().SetControllingVariables(controlling_variables)

        return
    
    def __SynchronizeSettings(self):

        # Re-set cantera solution.
        self.gas = ct.Solution(self.__reaction_mechanism)
        self.__species_in_mixture = self.gas.species_names

        n_fuel = len(self.__fuel_species)
        n_ox = len(self.__oxidizer_species)

        # Re-set fuel and oxidizer string.
        self.__fuel_string = ','.join([self.__fuel_species[i] + ':'+str(self.__fuel_weights[i]) for i in range(n_fuel)])
        self.__oxidizer_string = ','.join([self.__oxidizer_species[i] + ':'+str(self.__oxidizer_weights[i]) for i in range(n_ox)])

        # Compute mixture fraction mass fraction weights.
        self.ComputeMixFracConstants()

        # Check if current progress variable is compatible with reaction mechanism.
        if self.__custom_pv_set:
            # Re-set progress variable definition if any of the components is not present in the reaction mechanism.
            if any([sp not in self.__species_in_mixture for sp in self.__pv_definition]):
                self.ResetProgressVariableDefinition()
                self.__custom_pv_set = False
        else:
            # Set default progress variable definition.
            pv_sp_default, pv_w_default = self.SetDefaultProgressVariable()
            self.SetProgressVariableDefinition(pv_sp_default, pv_w_default)
            self.__custom_pv_set = False
        #print(self.__pv_definition, self.__pv_weights)
        if not self.__custom_Le_av_set:
            self.SetAverageLewisNumbers()

        return 
    
    # def SetConstSpecieLewisNumber(self, sp_name:str, Lewis_number:float):
    #     self.__Le_const_sp[self.gas.species_index(sp_name)] = Lewis_number 
    #     return 
    
    def GetConstSpecieLewisNumbers(self):
        return self.__Le_const_sp 
    
    def PrintBanner(self):
        """Print banner visualizing FlameletAI configuration settings."""
        super().PrintBanner()
        
        print("flameletAIConfiguration: " + self._config_name)
        print("")
        print("Flamelet generation settings:")
        print("Flamelet data output directory: " + self._output_dir)
        print("Reaction mechanism: " + self.__reaction_mechanism)
        print("Transport model: "+self.__transport_model)
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
        print("Controlling variable names: " + ", ".join(c for c in self._controlling_variables))
        print("Progress variable definition: " + ", ".join("%+.2e %s" % (self.__pv_weights[i], self.__pv_definition[i]) for i in range(len(self.__pv_weights))))
        if len(self.__passive_species) > 0:
            print("Passive species in manifold: " + ", ".join(s for s in self.__passive_species))
        print("")

        if self.__preferential_diffusion:
            print("Average specie Lewis numbers:")
            print(", ".join(("%s:%.4e" % (sp, Le) for sp, Le in zip(self.__species_in_mixture, self.__Le_const_sp))))

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
        return 
    
    def ComputeMixFracConstants(self):
        """
        
        Compute the species mass fraction coefficients according to the Bilger mixture fraction definition.
        
        """

        # Number of species in fuel and oxidizer definition.
        n_fuel = len(self.__fuel_species)
        n_ox = len(self.__oxidizer_species)

        # Joining fuel and oxidizer definitions into a single string
        self.__fuel_string = ','.join([self.__fuel_species[i] + ':'+str(self.__fuel_weights[i]) for i in range(n_fuel)])
        self.__oxidizer_string = ','.join([self.__oxidizer_species[i] + ':'+str(self.__oxidizer_weights[i]) for i in range(n_ox)])

        
        # Getting the carrier specie index
        idx_c = self.gas.species_index(self.__carrier_specie)

        #--- Computing mixture fraction coefficients ---#
        # setting up mixture in stochiometric condition
        self.gas.TP = 300, DefaultSettings_FGM.pressure
        self.gas.set_equivalence_ratio(1.0, self.__fuel_string, self.__oxidizer_string)
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
    
    def GetFuelString(self):
        return self.__fuel_string 
    
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

    def GetOxidizerString(self):
        return self.__oxidizer_string
    
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
        self.__fuel_species = fuel_species.copy()
        self.__fuel_weights = fuel_weights.copy()

        self.__SynchronizeSettings()    
        return 
    
    def SetOxidizerDefinition(self, oxidizer_species:list[str]=DefaultSettings_FGM.oxidizer_definition, \
                              oxidizer_weights:list[float]=DefaultSettings_FGM.oxidizer_weights):
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
        self.__oxidizer_species = oxidizer_species.copy()
        self.__oxidizer_weights = oxidizer_weights.copy()

        self.__SynchronizeSettings()
        return 
    
    def SetReactionMechanism(self, mechanism_input:str=DefaultSettings_FGM.reaction_mechanism):
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
        self.__SynchronizeSettings()
        return 
    
    def GetReactionMechanism(self):
        """
        Get the reaction mechanism used for flamelet generation.
        :return: reaction mechanism name.
        :rtype: str

        """
        return self.__reaction_mechanism
    
    def SetTransportModel(self, transport_model:str=DefaultSettings_FGM.transport_model):
        """Define the transport model for the 1D flamelet computations: "unity-Lewis" or "multicomponent".

        :param transport_model: transport model name, defaults to DefaultSettings_FGM.transport_model
        :type transport_model: str, optional
        """
        if transport_model == "multicomponent" or transport_model == "mixture-averaged":
            self.__preferential_diffusion = True 
        elif transport_model == "unity-Lewis-number":
            self.__preferential_diffusion = False
        else:
            raise Exception("Transport model should be \"multicomponent\", \"mixture-averaged\", or \"unity-Lewis-number\"")
        
        self.__transport_model=transport_model
        self.__SynchronizeSettings()
        return 
    
    def GetTransportModel(self):
        """Get the transport model for 1D flamelet computations.

        :return: flamelet transport model name.
        :rtype: str
        """
        return self.__transport_model
    
    def SetMixtureBounds(self, mix_lower:float=DefaultSettings_FGM.eq_ratio_min, mix_upper:float=DefaultSettings_FGM.eq_ratio_max):
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
            if mix_lower < 0.0:
                raise Exception("Mixture status should be positive.")
            if mix_upper > 1.0 and self.__run_mixture_fraction:
                raise Exception("Mixture fraction bounds should be between 0.0 and 1.0")
            
            self.__mix_status_lower = mix_lower
            self.__mix_status_upper = mix_upper

        if not self.__custom_Le_av_set:
            self.SetAverageLewisNumbers()

        return 
    
    def GetMixtureBounds(self):
        """
        Get the mixture status bounds.
        :return: List containing lower and upper mixture status values.
        :rtype: list[float]

        """
        return [self.__mix_status_lower, self.__mix_status_upper]
    
    def SetNpMix(self, input:int=DefaultSettings_FGM.Np_eq):
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

    def SetUnbTempBounds(self, T_unb_lower:float=DefaultSettings_FGM.T_min, T_unb_upper:float=DefaultSettings_FGM.T_max):
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

        if not self.__custom_Le_av_set:
            self.SetAverageLewisNumbers()

        return 
    
    def GetUnbTempBounds(self):
        """
        Get the reactant temperature bounds for flamelet generation.
        
        :return: lower and upper reactant temperature.
        :rtype: list[float]

        """
        return [self.__T_unb_lower, self.__T_unb_upper]
    
    def SetNpTemp(self, Np_Temp:int=DefaultSettings_FGM.Np_temp):
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
        return 
    
    def GetNpTemp(self):
        """
        Get the number of divisions for the reactant temperature range.

        :return: Number of divisions for the reactant temperature range.
        :rtype: int

        """
        return self.__Np_T_unb 

    def DefineMixtureStatus(self, run_as_mixture_fraction:bool=DefaultSettings_FGM.run_mixture_fraction):
        """
        Define the reactant mixture status as mixture fraction or equivalence ratio.

        :param run_as_mixture_fraction: If set to `True`, the mixture status is defined as mixture fraction, while `False` defines it as equivalence ratio.

        """
        self.__run_mixture_fraction = run_as_mixture_fraction
        if not self.__custom_Le_av_set:
            self.SetAverageLewisNumbers()

        return
    
    def GetMixtureStatus(self):
        """
        Get the mixture status definition of the current FlameletAIConfig class.

        :return: mixture status definition (`True` for mixture fraction, `False` for equivalence ratio)
        :rtype: bool

        """
        return self.__run_mixture_fraction
    
    def RunFreeFlames(self, input:bool=DefaultSettings_FGM.include_freeflames):
        """
        Include adiabatic free flame data in the manifold.

        :param input: enable generation of adabatic free flame data.
        :type input: bool

        """
        self.__generate_freeflames = input 
        return
    
    def RunBurnerFlames(self, input:bool=DefaultSettings_FGM.include_burnerflames):
        """
        Include burner-stabilized flame data in the manifold.

        :param input: enable generation of burner-stabilized flame data.
        :type input: bool

        """
        self.__generate_burnerflames = input 
        return
    
    def RunEquilibrium(self, input:bool=DefaultSettings_FGM.include_equilibrium):
        """
        Include chemical equilibrium (reactants and products) data in the manifold.

        :param input: enable generation of chemical equilibrium data.
        :type input: bool

        """
        self.__generate_equilibrium = input
        return
    
    def RunCounterFlames(self, input:bool=DefaultSettings_FGM.include_counterflames):
        """
        Include counter-flow diffusion flame data in the manifold.

        :param input: enable generation of counter-flow diffusion flamelets.
        :type input: bool

        """
        self.__generate_counterflames = input 
        return
    
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
        return
    
    def WriteMatlabFiles(self):
        """
        Save a copy of flamelet data as MATLAB TableMaster format.

        :return: save a MATLAB TableMaster copy of flamelet data.
        :rtype: bool
        """
        return self.__write_MATLAB_files
    
    def SetProgressVariableDefinition(self, pv_species:list[str]=DefaultSettings_FGM.pv_species, pv_weights:list[float]=DefaultSettings_FGM.pv_weights):
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
            self.__pv_definition = pv_species.copy()
            self.__pv_weights = pv_weights.copy()
            self.__custom_pv_set = True 
        return 
    
    def ResetProgressVariableDefinition(self):
        self.__pv_definition = []
        self.__pv_weights = []
        self.__custom_pv_set = False 
        self.SetDefaultProgressVariable()
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
    
    def SetDefaultProgressVariable(self):
        """Set progress variable to be weighted sum of fuel and oxidizer species (minus N2) and major product at stochiometry. 
           Weights are set as the inverse of specie molecular weight: negative for reactants, positive for product.
        """

        # Set mixture temperature and pressure at stochiometry.
        self.gas.TP=self.__T_unb_lower, DefaultSettings_FGM.pressure
        self.gas.set_equivalence_ratio(1.0, self.__fuel_string, self.__oxidizer_string)

        pv_species =[]
        pv_weights = []

        # Collect fuel reactant species and weights.
        for f in self.__fuel_species:
            if f != "N2":
                pv_species.append(f)
                pv_weights.append(-1.0/self.gas.molecular_weights[self.gas.species_index(f)])

        # Collect oxidizer reactant species and weights.
        for o in self.__oxidizer_species:
            if o != "N2":
                pv_species.append(o)
                pv_weights.append(-1.0/self.gas.molecular_weights[self.gas.species_index(o)])

        # Equilibrate at constant enthalpy and pressure.
        self.gas.equilibrate("HP")

        # Sort products.
        ix_species_sorted = np.argsort(self.gas.Y)[::-1]
        major_species = [self.gas.species_names[j] for j in ix_species_sorted]

        # Remove N2.
        major_species.remove("N2")

        # Select major product as progress variable species.
        major_product = major_species[0]
        pv_species.append(major_product)
        pv_weights.append(1.0 / self.gas.molecular_weights[self.gas.species_index(major_product)])

        return pv_species, pv_weights
    
    def SetPassiveSpecies(self, passive_species:list[str]=[]):
        """
        Set the passive transported species for which source terms should be saved in the manifold.

        :param __passive_species: list of species names.
        :type __passive_species: list[str]
        """
        self.__passive_species = passive_species
        return 
    
    def GetPassiveSpecies(self):
        """
        Get the list of passive species included in the manifold data.

        :return: list of passive species names.
        :rtype: list[str]
        """
        return self.__passive_species
    
    def SetLookUpVariables(self, lookup_vars:list[str]=[]):
        """
        Define passive look-up terms to be included in the manifold.

        :param lookup_vars: list of passive variable names.
        :type lookup_vars: list[str]
        """
        self.__lookup_variables = lookup_vars
        return 
    
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
        """Compute the progress variable based on the corresponding progress variable definition for an array of provided flamelet data or species mass fractions.

        :param variables: list of variable names in the flamelet data.
        :type variables: list[str]
        :param flamelet_data: flamelet data array.
        :type flamelet_data: np.ndarray
        :param Y_flamelet: species mass fraction array, defaults to None
        :type Y_flamelet: np.ndarray, optional
        :raise: Exception: if number of columns in Y_flamelet do not match the number of species in the reaction mechanism.
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
        :param net_production_rate_flamelet: species net production rates (kg m^{-3} s^{-1}), defaults to None
        :type net_production_rate_flamelet: np.ndarray
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
    
    def EnablePreferentialDiffusion(self, use_PD:bool=DefaultSettings_FGM.preferential_diffusion):
        """Include preferential diffusion scalars in flamelet data manifold.

        :param use_PD: include diffusion scalars, defaults to DefaultSettings_FGM.preferential_diffusion
        :type use_PD: bool, optional
        """

        self.__preferential_diffusion = use_PD 
        if not self.__custom_Le_av_set:
            self.SetAverageLewisNumbers()

        return
    
    def PreferentialDiffusion(self):
        """Inclusion of preferential diffusion in flamelet data manifold.

        :return: whether preferential diffusion scalars are included (True) or not (False).
        :rtype: bool
        """
        return self.__preferential_diffusion
    
    def SetAveragingMethod(self, avg_method=avg_Le_arythmic):
        self.__Le_avg_method = avg_method 
        return 
    
    def AverageLewisNumber(self, Le_sp:np.ndarray, iSp:int):
        if self.__Le_avg_method == avg_Le_const:
            Le_av = avg_Le_const(Le_sp, self.__Le_const_sp[iSp])
        else:
            Le_av = self.__Le_avg_method(Le_sp)
        return Le_av

    def SetAverageLewisNumbers(self, mixture_status:float=None, reactant_temperature:float=None):
        """Define the constant specie Lewis numbers for a given mixture status and reactant temperature.

        :param mixture_status: equivalence ratio or mixture fraction, defaults to the average of the manifold bounds.
        :type mixture_status: float, optional
        :param reactant_temperature: reactant temperature, defaults to the average of the manifold bounds.
        :type reactant_temperature: float, optional
        :raises Exception: if negative mixture status value or temperature is provided.
        """
        if mixture_status != None:
            if mixture_status < 0:
                raise Exception("Mixture status value should be positive.")
            if self.__run_mixture_fraction and mixture_status > 1:
                raise Exception("Mixture fraction should be between zero and one.")
        if reactant_temperature != None:
            if reactant_temperature < 0:
                raise Exception("Reactant temperature should be positive.")
        
        self.__Le_avg_method = avg_Le_const
        if reactant_temperature == None:
            T_reactants = 0.5*(self.__T_unb_lower + self.__T_unb_upper)
        else:
            T_reactants = reactant_temperature 
        
        if mixture_status == None:
            mixture_status_gas = 0.5*(self.__mix_status_lower + self.__mix_status_upper)
        else:
            mixture_status_gas = mixture_status

        self.gas.TP =T_reactants, DefaultSettings_FGM.pressure 
        if self.__run_mixture_fraction:
            self.gas.set_mixture_fraction(mixture_status_gas, self.__fuel_string, self.__oxidizer_string)
        else:
            self.gas.set_equivalence_ratio(mixture_status_gas, self.__fuel_string, self.__oxidizer_string)
        
        Le_reactants = ComputeLewisNumber(self.gas)
        self.gas.equilibrate("HP")
        Le_products = ComputeLewisNumber(self.gas)

        self.__Le_const_sp = 0.5*(Le_reactants + Le_products)
        self.__custom_Le_av_set = True 

        return 
    
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
            Le_av = self.AverageLewisNumber(Le_sp, iSp)

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
            Le_av = self.AverageLewisNumber(Le_sp, self.gas.species_index(self.__pv_definition[iPv]))
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
        :return: enth_unb: reactants total enthalpy value.
        :rtype: enth_unb: float
        :return: mixfrac_unb: reactants mixture fraction value.
        :rtype: mixfrac_unb: float
        """

        # if equivalence_ratio < 0:
        #     raise Exception("Equivalence ratio should be positive.")
        if temperature < 200:
            raise Exception("Temperature should be above 200 degrees Kelvin.")
        
        self.__fuel_string = ",".join(self.__fuel_species[i]+":"+str(self.__fuel_weights[i]) for i in range(len(self.__fuel_species)))
        self.__oxidizer_string = ",".join(self.__oxidizer_species[i]+":"+str(self.__oxidizer_weights[i]) for i in range(len(self.__oxidizer_species)))
        self.gas.set_equivalence_ratio(equivalence_ratio, self.__fuel_string, self.__oxidizer_string)
        self.gas.TP = temperature, DefaultSettings_FGM.pressure

        pv_unb = 0
        for iPV, pvSp in enumerate(self.__pv_definition):
            pv_unb += self.__pv_weights[iPV] * self.gas.Y[self.gas.species_index(pvSp)]
        
        enth_unb = self.gas.enthalpy_mass

        mixfrac_unb = self.gas.mixture_fraction(self.__fuel_string, self.__oxidizer_string)

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
        self.__fuel_string = ",".join(self.__fuel_species[i]+":"+str(self.__fuel_weights[i]) for i in range(len(self.__fuel_species)))
        self.__oxidizer_string = ",".join(self.__oxidizer_species[i]+":"+str(self.__oxidizer_weights[i]) for i in range(len(self.__oxidizer_species)))
        self.gas.set_equivalence_ratio(equivalence_ratio, self.__fuel_string, self.__oxidizer_string)
        self.gas.TP = temperature, DefaultSettings_FGM.pressure
        self.gas.equilibrate('TP')
        pv_b = 0
        for iPV, pvSp in enumerate(self.__pv_definition):
            pv_b += self.__pv_weights[iPV] * self.gas.Y[self.gas.species_index(pvSp)]
        
        enth_b = self.gas.enthalpy_mass

        mixfrac_b = self.gas.mixture_fraction(self.__fuel_string, self.__oxidizer_string)
        
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
        return 
    
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
        return 
    
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
        return 
    
    def GetTableMixtureFractionLimits(self):
        """Returns the lower and upper mixture fraction limits of the table between which table levels are generated.

        :return: lower mixture fraction value, upper mixture fraction value
        :rtype: float, float
        """
        return self.__Table_mixfrac_lower, self.__Table_mixfrac_upper
    
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
            self.__alpha_expo = []
            self.__lr_decay = []
            self.__batch_expo = []
            self.__activation_function = []
            self.__NN = []

        self.__MLP_output_groups.append([])
        for var in variable_names_in:
            self.__MLP_output_groups[-1].append(var)
        self.__alpha_expo.append(DefaultSettings_FGM.init_learning_rate_expo)
        self.__lr_decay.append(DefaultSettings_FGM.learning_rate_decay)
        self.__batch_expo.append(DefaultSettings_FGM.batch_size_exponent)
        self.__NN.append(DefaultSettings_FGM.hidden_layer_architecture)
        self.__activation_function.append(DefaultSettings_FGM.activation_function)
        return 
    
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
        return 
    
    def RemoveOutputGroup(self, i_group:int):
        if i_group > len(self.__MLP_output_groups):
            raise Exception("Group not present in MLP outputs.")
        print("Removing output group %i: %s" % (i_group, ",".join(s for s in self.__MLP_output_groups[i_group-1])))
        self.__MLP_output_groups.remove(self.__MLP_output_groups[i_group-1])
        return
    
    def ClearOutputGroups(self):
        self.__MLP_output_groups = None
        return 
    
    def DisplayOutputGroups(self):
        """Print the MLP output variables grouping arrangement.
        """
        for i_group, group in enumerate(self.__MLP_output_groups):
            print("Group %i: " % (i_group + 1))
            print("Output variables: "+",".join(var for var in group))
            print("Initial learning rate exponent: %+.6e" % self.__alpha_expo[i_group])
            print("Learning rate decay: %+.6e" % self.__lr_decay[i_group])
            print("Mini-batch exponent: %i" % self.__batch_expo[i_group])
            print("Activation function: %s" % self.__activation_function[i_group])
            print("Hidden layer architecture: " + ",".join(("%i" % n) for n in self.__NN[i_group]))
            print()
        return 
    
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
    
    def SetAlphaExpo(self, alpha_expo_in: float = DefaultSettings_FGM.init_learning_rate_expo, i_group:int=0):
        super().SetAlphaExpo(alpha_expo_in)
        self.__alpha_expo[i_group] = alpha_expo_in
        return 
    
    def GetAlphaExpo(self, i_group:int=0):
        return self.__alpha_expo[i_group]
    
    def SetLRDecay(self, lr_decay_in: float = DefaultSettings_FGM.learning_rate_decay, i_group:int=0):
        super().SetLRDecay(lr_decay_in)
        self.__lr_decay[i_group] = lr_decay_in
        return 
    
    def GetLRDecay(self, i_group:int=0):
        return self.__lr_decay[i_group]
    
    def SetBatchExpo(self, batch_expo_in: int = DefaultSettings_FGM.batch_size_exponent, i_group:int=0):
        super().SetBatchExpo(batch_expo_in)
        self.__batch_expo[i_group] = batch_expo_in
        return 
    
    def GetBatchExpo(self, i_group:int=0):
        return self.__batch_expo[i_group]
    
    def SetActivationFunction(self, activation_function_in: str = DefaultSettings_FGM.activation_function, i_group:int=0):
        super().SetActivationFunction(activation_function_in)
        self.__activation_function[i_group] = activation_function_in
        return 
    
    def GetActivationFunction(self, i_group:int=0):
        return self.__activation_function[i_group]
    
    def SetHiddenLayerArchitecture(self, hidden_layer_architecture: list[int] = DefaultSettings_FGM.hidden_layer_architecture, i_group:int=0):
        super().SetHiddenLayerArchitecture(hidden_layer_architecture)
        self.__NN[i_group] = []
        for N in hidden_layer_architecture:
            self.__NN[i_group].append(N)
        return 
    
    def GetHiddenLayerArchitecture(self, i_group:int=0):
        return self.__NN[i_group]
    
    def SaveConfig(self):
        """
        Save the current FlameletAI configuration.

        :param file_name: configuration file name.
        :type file_name: str
        """
        self.__SynchronizeSettings()

        file = open(self._config_name+'.cfg','wb')
        pickle.dump(self, file)
        file.close()
        return 
    

