import numpy as np 
import cantera as ct 
import pickle 
import os 
import CoolProp

class EntropicAIConfig:
    """
    Define EntropicAIConfig class or load existing configuration. If `load_file` is set, the settings from an existing
    is loaded. If no file name is provided, a new `EntropicAIConfig` class is created.
    
    :param load_file: path to file of configuration to be loaded.
    :type load_file: str
    """
    # EntropicAI configuration class containing all relevant info on flamelet manifold settings.

    __config_name:str = "EntropicAIConfig" # Configuration name.

    # Fluid definition settings
    __fluid_names:list[str] = ["MM"]
    __fluid_string:str="MM"
    __fluid_mole_fractions:list[float] = []
    __use_PT:bool = True 

    # Output directory for all large data files including fluid data.
    __output_dir:str = "./"

    __T_lower:float = 280   # Lower temperature bound.
    __T_upper:float = 500   # Upper temperature bound.
    __Np_T:int = 100      # Number of temperature samples between bounds.

    __P_lower:float = 1e4
    __P_upper:float = 2e6
    __Np_P:int = 100

    __concatenated_file_header:str = "entropic_data" # File header for MLP training data files.

    # MLP Settings

    __train_fraction:float = 0.8    # Fraction of flamelet data used for training.
    __test_fraction:float = 0.1     # Fraction of flamleet data used for test validation.

    # MLP output groups and architecture information.
    __MLP_output_groups:list[list[str]] = None  # Output variables for each MLP.
    __MLP_architectures:list[list[int]] = None  # Hidden layer architecture for each MLP.
    __MLP_trainparams:list[list[float]] = None  # Training parameters and activation function information for each MLP.

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
    
    # def PrintBanner(self):
    #     """Print banner visualizing EntropicAI configuration settings."""

    #     print("EntropicAIConfiguration: " + self.__config_name)
    #     print("")
    #     print("Flamelet generation settings:")
    #     print("Flamelet data output directory: " + self.__flamelet_output_dir)
    #     print("Reaction mechanism: " + self.__reaction_mechanism)
    #     print("Fuel definition: " + ",".join("%s: %.2e" % (self.__fuel_species[i], self.__fuel_weights[i]) for i in range(len(self.__fuel_species))))
    #     print("Oxidizer definition: " + ",".join("%s: %.2e" % (self.__oxidizer_species[i], self.__oxidizer_weights[i]) for i in range(len(self.__oxidizer_species))))
    #     print("")
    #     print("Reactant temperature range: %.2f K -> %.2f K (%i steps)" % (self.__T_unb_lower, self.__T_unb_upper, self.__Np_T_unb))
        
    #     if self.__run_mixture_fraction:
    #         print("Mixture status defined as mixture fraction")
    #     else: 
    #         print("Mixture status defined as equivalence ratio")
    #     print("Reactant mixture status range: %.2e -> %.2e  (%i steps)" % (self.__mix_status_lower, self.__mix_status_upper, self.__Np_mix_unb))
    #     print("")
    #     print("Flamelet types included in manifold:")
    #     if self.__generate_freeflames:
    #         print("-Adiabatic free-flamelet data")
    #     if self.__generate_burnerflames:
    #         print("-Burner-stabilized flamelet data")
    #     if self.__generate_equilibrium:
    #         print("-Chemical equilibrium data")
    #     if self.__generate_counterflames:
    #         print("-Counter-flow diffusion flamelet data")
    #     print("")

    #     print("Flamelet manifold data characteristics: ")
    #     print("Progress variable definition: " + ", ".join("%+.2e %s" % (self.__pv_weights[i], self.__pv_definition[i]) for i in range(len(self.__pv_weights))))
    #     if len(self.__passive_species) > 0:
    #         print("Passive species in manifold: " + ", ".join(s for s in self.__passive_species))
    #     print("")

    #     if self.__Table_level_count is not None:
    #         print("Table generation settings:")
    #         print("Table mixture fraction range: %.2e -> %.2e" % (self.__Table_mixfrac_lower, self.__Table_mixfrac_upper))
    #         print("Number of table levels: %i" % self.__Table_level_count)
    #         print("Table level base cell size: %.2e" % self.__Table_base_cell_size)
    #         print("Table level refined cell size: %.2e" % self.__Table_ref_cell_size)
    #         print("Table level refinement radius: %.2e" % self.__Table_ref_radius)
    #         print("Table level refinement threshold: %.2e" % self.__Table_curv_threshold)
    #     print("")

    #     if self.__MLP_output_groups is not None:
    #         print("MLP Output Groups:")
    #         self.DisplayOutputGroups()
    #     print("")
        
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

        :param fluid_name: CoolProp fluid name.
        :type fluid_name: str
        :raise: Exception: If the reaction mechanism could not be loaded from path.

        """
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
        if (mole_fraction_1 < 0) or (mole_fraction_2 < 0):
            raise Exception("Mole fractions should be positive")
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
        self.__use_PT = PT_grid 

    def GetPTGrid(self):
        return self.__use_PT 
    
    def SetTemperatureBounds(self, T_lower:float, T_upper:float):
        if (T_lower >= T_upper):
            raise Exception("Lower temperature should be lower than upper temperature.")
        else:
            self.__T_lower = T_lower
            self.__T_upper = T_upper
    
    def GetTemperatureBounds(self):
        return [self.__T_lower, self.__T_upper]
    
    
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
            self.__Np_T = Np_Temp
    
    def GetNpTemp(self):
        """
        Get the number of divisions for the reactant temperature range.

        :return: Number of divisions for the reactant temperature range.
        :rtype: int

        """
        return self.__Np_T
    

    def SetPressureBounds(self, P_lower:float, P_upper:float):
        if (P_lower >= P_upper):
            raise Exception("Lower temperature should be lower than upper temperature.")
        else:
            self.__P_lower = P_lower
            self.__P_upper = P_upper
    
    def GetPressureBounds(self):
        return [self.__P_lower, self.__P_upper]
    
    
    def SetNpPressure(self, Np_P:int):
        """
        Set number of divisions for the reactant temperature at each mixture fraction/equivalence ratio.

        :param Np_Temp: Number of divisions for the reactant temperature range.
        :type Np_Temp: int
        :rase: Exception: If the number of divisions is lower than one.

        """
        if (Np_P <= 0):
            raise Exception("Number of unburnt temperature samples should be higher than one.")
        else:
            self.__Np_P = Np_P 
    
    def GetNpPressure(self):
        """
        Get the number of divisions for the reactant temperature range.

        :return: Number of divisions for the reactant temperature range.
        :rtype: int

        """
        return self.__Np_P
    
    def SetOutputDir(self, output_dir:str):
        """
        Define the flamelet data output directory. This directory is set as the default storage directory for all storage processes in the EntropicAI workflow.

        :param output_dir: storage directory.
        :raise: Exception: If the specified directory does not exist.

        """
        if not os.path.isdir(output_dir):
            raise Exception("Invalid output data directory")
        else:
            self.__output_dir = output_dir
    
    def GetOutputDir(self):
        """
        Get the current EntropicAI configuration flamelet storage directory.

        :raises: Exception: if the storage directory in the current `FlameeltAIConfig` class is not present on the current hardware.
        :return: Flamelet data storage directory.
        :rtype: str

        """
        if not os.path.isdir(self.__output_dir):
            raise Exception("Saved output directory not present on current machine.")
        else:
            return self.__output_dir
    
    def SetConcatenationFileHeader(self, header:str):
        """
        Define the file name header for the collection of flamelet data.

        :param header: file name header.
        :type header: str
        """
        self.__concatenated_file_header = header 

    def GetConcatenationFileHeader(self):
        """
        Get the file name header for the flamelet collection file.

        :return: flamelet collection file header.
        :rtype: str
        """
        return self.__concatenated_file_header
    
    def SetTrainFraction(self, input:float):
        """
        Define the fraction of flamelet data used for training multi-layer perceptrons.

        :param input: flamelet data train fraction.
        :type input: float 
        :raise: Exception: if provided fraction is equal or higher than one.
        """
        if input >= 1:
            raise Exception("Training data fraction should be lower than one.")
        self.__train_fraction = input 

    def SetTestFraction(self, input:float):
        """
        Define the fraction of flamelet data separate from the training data used for determining accuracy after training.

        :param input: flamelet data test fraction.
        :type input: float
        :raise Exception: if provided fraction is equal or higher than one.
        """
        if input >= 1:
            raise Exception("Test data fraction should be lower than one.")
        self.__test_fraction = input 

    def GetTrainFraction(self):
        """
        Get flamelet data fraction used for multi-layer perceptron training.

        :return: flamelet data train fraction.
        :rtype: float 
        """
        return self.__train_fraction
    
    def GetTestFraction(self):
        """
        Get flamelet data fraction used for determining accuracy after training.

        :return: flamelet data test fraction.
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

