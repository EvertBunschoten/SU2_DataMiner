import numpy as np 
import cantera as ct 
import pickle 
import os 
import CoolProp
from Common.Properties import DefaultProperties 

class EntropicAIConfig:
    """
    Define EntropicAIConfig class or load existing configuration. If `load_file` is set, the settings from an existing
    is loaded. If no file name is provided, a new `EntropicAIConfig` class is created.
    
    :param load_file: path to file of configuration to be loaded.
    :type load_file: str
    """
    # EntropicAI configuration class containing all relevant info on fluid manifold settings.

    __config_name:str = "EntropicAIConfig" # Configuration name.

    # Fluid definition settings
    __fluid_names:list[str] = ["MM"]
    __fluid_string:str="MM"
    __fluid_mole_fractions:list[float] = []
    __use_PT:bool = True 

    # Output directory for all large data files including fluid data.
    __output_dir:str = "./"

    __T_lower:float = DefaultProperties.T_min   # Lower temperature bound.
    __T_upper:float = DefaultProperties.T_max   # Upper temperature bound.
    __Np_T:int = DefaultProperties.Np_temp      # Number of temperature samples between bounds.

    __P_lower:float = DefaultProperties.P_min
    __P_upper:float = DefaultProperties.P_max
    __Np_P:int = DefaultProperties.Np_p

    __concatenated_file_header:str = DefaultProperties.output_file_header # File header for MLP training data files.

    # MLP Settings

    __train_fraction:float = DefaultProperties.train_fraction    # Fraction of fluid data used for training.
    __test_fraction:float = DefaultProperties.test_fraction    # Fraction of flamleet data used for test validation.

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
        """Class constructor
        """
        if load_file:
            print("Loading configuration for entropic model generation...")
            with open(load_file, "rb") as fid:
                loaded_config = pickle.load(fid)
            self.__dict__ = loaded_config.__dict__.copy()
            print("Loaded configuration file with name " + loaded_config.__config_name)
        else:
            print("Generating empty EntropicAI config")

        return 
    
    def PrintBanner(self):
        """Print banner visualizing EntropicAI configuration settings."""

        print("EntropicAIConfiguration: " + self.__config_name)
        print("")
        print("Fluid data generation settings:")
        print("Fluid data output directory: " + self.__output_dir)
        print("Fluid name(s): " + ",".join(self.__fluid_names))
        print("")
        print("Temperature range: %.2f K -> %.2f K (%i steps)" % (self.__T_lower, self.__T_upper, self.__Np_T))
        print("Pressure range: %.3e Pa -> %.3e Pa (%i steps)" % (self.__P_lower, self.__P_upper, self.__Np_P))
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
        return self.__config_name
    

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
    
    def SetOutputDir(self, output_dir:str):
        """
        Define the fluid data output directory. This directory is set as the default storage directory for all storage processes in the EntropicAI workflow.

        :param output_dir: storage directory.
        :raise: Exception: If the specified directory does not exist.

        """
        if not os.path.isdir(output_dir):
            raise Exception("Invalid output data directory")
        else:
            self.__output_dir = output_dir
        return 
    
    def GetOutputDir(self):
        """
        Get the current EntropicAI configuration fluid storage directory.

        :raises: Exception: if the storage directory in the current `EntropicAIConfig` class is not present on the current hardware.
        :return: Flamelet data storage directory.
        :rtype: str

        """
        if not os.path.isdir(self.__output_dir):
            raise Exception("Saved output directory not present on current machine.")
        else:
            return self.__output_dir
    
    def SetConcatenationFileHeader(self, header:str=DefaultProperties.output_file_header):
        """
        Define the file name header for the collection of fluid data.

        :param header: file name header.
        :type header: str
        """
        self.__concatenated_file_header = header 
        return 
    
    def GetConcatenationFileHeader(self):
        """
        Get the file name header for the fluid collection file.

        :return: fluid collection file header.
        :rtype: str
        """
        return self.__concatenated_file_header
    
    def SetTrainFraction(self, input:float=DefaultProperties.train_fraction):
        """
        Define the fraction of fluid data used for training multi-layer perceptrons.

        :param input: fluid data train fraction.
        :type input: float 
        :raise: Exception: if provided fraction is equal or higher than one.
        """
        if input >= 1:
            raise Exception("Training data fraction should be lower than one.")
        self.__train_fraction = input 
        return 
    
    def SetTestFraction(self, input:float=DefaultProperties.test_fraction):
        """
        Define the fraction of fluid data separate from the training data used for determining accuracy after training.

        :param input: fluid data test fraction.
        :type input: float
        :raise Exception: if provided fraction is equal or higher than one.
        """
        if input >= 1:
            raise Exception("Test data fraction should be lower than one.")
        self.__test_fraction = input 
        return 
    
    def GetTrainFraction(self):
        """
        Get fluid data fraction used for multi-layer perceptron training.

        :return: fluid data train fraction.
        :rtype: float 
        """
        return self.__train_fraction
    
    def GetTestFraction(self):
        """
        Get fluid data fraction used for determining accuracy after training.

        :return: fluid data test fraction.
        :rtype: float 
        """
        return self.__test_fraction
    
    
    
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

        self.__config_name = file_name
        file = open(file_name+'.cfg','wb')
        pickle.dump(self, file)
        file.close()

