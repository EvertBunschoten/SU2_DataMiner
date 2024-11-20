###############################################################################################
#       #      _____ __  _____      ____        __        __  ____                   #        #
#       #     / ___// / / /__ \    / __ \____ _/ /_____ _/  |/  (_)___  ___  _____   #        #
#       #     \__ \/ / / /__/ /   / / / / __ `/ __/ __ `/ /|_/ / / __ \/ _ \/ ___/   #        #
#       #    ___/ / /_/ // __/   / /_/ / /_/ / /_/ /_/ / /  / / / / / /  __/ /       #        #
#       #   /____/\____//____/  /_____/\__,_/\__/\__,_/_/  /_/_/_/ /_/\___/_/        #        #
#       #                                                                            #        #
###############################################################################################

############################ FILE NAME: DataGenerator_NICFD.py ################################
#=============================================================================================#
# author: Evert Bunschoten                                                                    |
#    :PhD Candidate ,                                                                         |
#    :Flight Power and Propulsion                                                             |
#    :TU Delft,                                                                               |
#    :The Netherlands                                                                         |
#                                                                                             |
#                                                                                             |
# Description:                                                                                |
#  Class for generating fluid data for Flamelet-Generated Manifold data mining operations.    |
#                                                                                             |
# Version: 1.0.0                                                                              |
#                                                                                             |
#=============================================================================================#

#---------------------------------------------------------------------------------------------#
# Importing general packages
#---------------------------------------------------------------------------------------------#
import cantera as ct 
import numpy as np 
import csv 
from os import path, mkdir
from joblib import Parallel, delayed
np.random.seed(2)

#---------------------------------------------------------------------------------------------#
# Importing DataMiner classes and functions
#---------------------------------------------------------------------------------------------#
from Common.DataDrivenConfig import FlameletAIConfig
from Data_Generation.DataGenerator_Base import DataGenerator_Base
from Common.CommonMethods import ComputeLewisNumber
from Common.Properties import DefaultSettings_FGM

class FlameletGenerator_Cantera(DataGenerator_Base):
    """Generate flamelet data using Cantera.

    :param Config: FlameletAIConfig class describing the flamelet generation settings.
    :type: FlameletAIConfig
    """
    # Generate flamelet data from Cantera computation.
    _Config:FlameletAIConfig

    # Save directory for computed flamelet data
    __matlab__output_dir:str = "./"

    __fuel_definition:list[str] = DefaultSettings_FGM.fuel_definition    # Fuel species
    __fuel_weights:list[float] = DefaultSettings_FGM.fuel_weights        # Fuel molar weights
    __fuel_string:str = ''
    __oxidizer_definition:list[str] = DefaultSettings_FGM.oxidizer_definition  # Oxidizer species
    __oxidizer_weights:list[float] = DefaultSettings_FGM.oxidizer_weights      # Oxidizer molar weights
    __oxidizer_string:str = ''

    __n_flamelets:int = DefaultSettings_FGM.Np_temp       # Number of adiabatic and burner flame computations per mixture fraction
    __T_unburnt_upper:float = DefaultSettings_FGM.T_max   # Highest unburnt reactant temperature
    __T_unburnt_lower:float = DefaultSettings_FGM.T_min   # Lowest unburnt reactant temperature

    __reaction_mechanism:str = DefaultSettings_FGM.reaction_mechanism   # Cantera reaction mechanism
    __transport_model:str = DefaultSettings_FGM.transport_model

    __initial_grid_length:float = 1e-2  # Flamelet grid width
    __initial_grid_Np:int = 30          # Number of initial grid nodes.

    __define_equivalence_ratio:bool = not DefaultSettings_FGM.run_mixture_fraction # Define unburnt mixture via the equivalence ratio
    __unb_mixture_status:list[float] = [] 

    __translate_to_matlab:bool = False # Save a copy of the flamelet data file in Matlab table generator format

    __run_freeflames:bool = DefaultSettings_FGM.include_freeflames      # Run adiabatic flame computations
    __run_burnerflames:bool = DefaultSettings_FGM.include_burnerflames    # Run burner stabilized flame computations
    __run_equilibrium:bool = DefaultSettings_FGM.include_equilibrium    # Run chemical equilibrium computations
    __run_counterflames:bool = DefaultSettings_FGM.include_counterflames   # Run counter-flow diffusion flamelet simulations.
    __run_fuzzy:bool = False           # Add randomized data around flamelet solutions to manifold.

    __u_fuel:float = 1.0       # Fuel stream velocity in counter-flow diffusion flame.
    __u_oxidizer:float = None   # Oxidizer stream velocity in counter-flow diffusion flame.

    __fuzzy_delta:float = 0.1

    def __init__(self, Config:FlameletAIConfig=None):
        DataGenerator_Base.__init__(self, Config_in=Config)

        """Constructur, load flamelet generation settings from FlameletAIConfig.

        :param Config: FlameletAIConfig containing respective settings.
        :type Config: FlameletAIConfig
        """

        if Config is None:
            print("Initializing flamelet generator with default settings")
            self._Config = FlameletAIConfig()
        else:
            print("Initializing flamelet generator from FlameletAIConfig with name " + self._Config.GetConfigName())
            self.__SynchronizeSettings()
        
        return 
    
    def __SynchronizeSettings(self):
        """Update settings from configuration
        """
        self.__fuel_string = self._Config.GetFuelString()
        self.__oxidizer_string = self._Config.GetOxidizerString()
        
        self.__reaction_mechanism = self._Config.GetReactionMechanism()
        self.__transport_model = self._Config.GetTransportModel()

        self.gas = ct.Solution(self._Config.GetReactionMechanism())

        self.__n_flamelets = self._Config.GetNpTemp()
        [self.__T_unburnt_lower, self.__T_unburnt_upper] = self._Config.GetUnbTempBounds()

        self.__define_equivalence_ratio = (not self._Config.GetMixtureStatus())
        self.__unb_mixture_status = np.linspace(self._Config.GetMixtureBounds()[0], self._Config.GetMixtureBounds()[1], self._Config.GetNpMix())
        self.__run_freeflames = self._Config.GenerateFreeFlames()
        self.__run_burnerflames = self._Config.GenerateBurnerFlames()
        self.__run_equilibrium = self._Config.GenerateEquilibrium()
        self.__run_counterflames = self._Config.GenerateCounterFlames()
        
        self.__PrepareOutputDirectories()
        self.__translate_to_matlab = self._Config.WriteMatlabFiles()
        if self.__translate_to_matlab:
            self.__PrepareOutputDirectories_Matlab()
        self._Config.ComputeMixFracConstants()
        self.z_i = self._Config.GetMixtureFractionCoefficients()
        self.c = self._Config.GetMixtureFractionConstant()
        return 
    
    def SetFuelDefinition(self, fuel_species:list[str], fuel_weights:list[float]):
        """Manually define the fuel composition

        :param fuel_species: list of fuel species names.
        :type fuel_species: list[str]
        :param __fuel_weights: list of fuel molar fraction weights.
        :type __fuel_weights: list[float]
        :raises Exception: if no fuel species are provided.
        :raises Exception: if the number of species does not correspond to the number of weights.
        """
        self._Config.SetFuelDefinition(fuel_species, fuel_weights)
        self.__SynchronizeSettings()
        
        return 
    
    def SetOxidizerDefinition(self, oxidizer_species:list[str], oxidizer_weights:list[float]):
        """Manually define the oxidizer composition

        :param oxidizer_species: list of oxidizer species names.
        :type oxidizer_species: list[str]
        :param __oxidizer_weights: list of oxidizer molar fraction weights.
        :type __oxidizer_weights: list[float]
        :raises Exception: if no oxidizer species are provided.
        :raises Exception: if the number of species does not correspond to the number of weights.
        """
        self._Config.SetOxidizerDefinition(oxidizer_species, oxidizer_weights)
        self.__SynchronizeSettings()
        
        return 
    
    def SetNpTemp(self, n_flamelets_new:int):
        """Set the number of flamelets generated between the minimum and maximum reactant temperature manually.

        :param n_flamelets_new: number of flamelets generated between the minimum and maximum reactant temperature.
        :type n_flamelets_new: int
        :raises Exception: if the provided number is lower than one.
        """
        if n_flamelets_new < 1:
            raise Exception("Number of flamelets should be higher than one.")
        self.__n_flamelets = n_flamelets_new
        return 
    
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
            self.__T_unburnt_upper = T_unb_upper
            self.__T_unburnt_lower = T_unb_lower
        return 
    
    def RunMixtureFraction(self):
        """Define the mixture status as mixture fraction instead of equivalence ratio.
        """
        self.__define_equivalence_ratio = False 
        return 
    
    def RunEquivalenceRatio(self):
        """Define the mixture status as equivalence ratio instead of mixture fraction.
        """
        self.__define_equivalence_ratio = True
        return 
    
    def RunFreeFlames(self, input:bool=True):
        """Include adiabatic free-flame data in the manifold.

        :param input: Generate adiabatic free-flame data.
        :type input: bool
        """
        self.__run_freeflames = input
        return 
    
    def AddFuzz(self, input:bool=False):
        self.__run_fuzzy = input 
        return 
    
    def SetFuzzyMargin(self, fuzz_margin:float=0.1):
        if fuzz_margin < 0:
            raise Exception("Fuzzy margin should be positive")
        self.__fuzzy_delta = fuzz_margin
        return 
    
    def RunBurnerFlames(self, input:bool=True):
        """Include burner-stabilized flame data in the manifold.

        :param input: Generate burner-stabilized flamelet data.
        :type input: bool
        """
        self.__run_burnerflames = input
        return 
    
    def RunEquilibrium(self, input:bool=True):
        """Include chemical equilibrium data in the manifold.

        :param input: Generate chemical equilibrium data.
        :type input: bool
        """
        self.__run_equilibrium = input 
        return
    
    def RunCounterFlowFlames(self, input:bool=True):
        """Include counter-flow diffusion flame data in the manifold.

        :param input: Generate counter-flow diffusion flamelet data.
        :type input: bool
        """
        self.__run_counterflames = input
        return 
    
    def SetMixtureValues(self, mixture_values:list[float]):
        """Set the reactant mixture status values manually.

        :param mixture_values: list of equivalence ratio or mixture fraction values.
        :type mixture_values: list[float]
        :raises Exception: If an empty list is provided.
        """
        if len(mixture_values) == 0:
            raise Exception("At least one mixture status value should be provided.")
        self.__unb_mixture_status = []
        for phi in mixture_values:
            self.__unb_mixture_status.append(phi)
        return 
    
    def SetReactionMechanism(self, reaction_mechanism:str):
        """Define the reaction mechanism manually.

        :param __reaction_mechanism: name of the reaction mechanism.
        :type __reaction_mechanism: str
        """
        self._Config.SetReactionMechanism(reaction_mechanism)
        self.__SynchronizeSettings()
        return 
    
    def SetTransportModel(self, transport_model:str):
        self._Config.SetTransportModel(transport_model)
        self.__SynchronizeSettings()
        return 
    
    def SetTransportMechanism(self, transport_mechanism:str="multicomponent"):
        self.__transport_model = transport_mechanism 
        return 
    
    def TranslateToMatlab(self):
        """Save a copy of the flamelet data in Matlab TableMaster format.
        """
        self.__translate_to_matlab = True 
        return
    
    def SetOutputDir(self, output_dir_new:str):
        """Define the flamelet data output directory manually.

        :param output_dir_new: Flamelet data output directory.
        :type output_dir_new: str
        :raises Exception: If provided directory doesn't exist.
        """
        self._Config.SetOutputDir(output_dir=output_dir_new)
        self.__SynchronizeSettings()
        return
    
    def SetMatlabOutputDir(self, output_dir_new):
        self.__matlab__output_dir = output_dir_new
        self.__PrepareOutputDirectories_Matlab()

    def __PrepareOutputDirectories(self): 
        """Create sub-directories for the different types of flamelet data.
        """  
        if (not path.isdir(self.GetOutputDir()+'/freeflame_data')) and self.__run_freeflames:
            mkdir(self.GetOutputDir()+'/freeflame_data')
        if (not path.isdir(self.GetOutputDir()+'/burnerflame_data')) and self.__run_burnerflames:
            mkdir(self.GetOutputDir()+'/burnerflame_data')
        if (not path.isdir(self.GetOutputDir()+'/equilibrium_data')) and self.__run_equilibrium:
            mkdir(self.GetOutputDir()+'/equilibrium_data')
        if (not path.isdir(self.GetOutputDir()+'/counterflame_data')) and self.__run_counterflames:
            mkdir(self.GetOutputDir()+'/counterflame_data')
        return 
    
    def __PrepareOutputDirectories_Matlab(self):
        if (not path.isdir(self.__matlab__output_dir+'freeflame_data_MATLAB')) and self.__run_freeflames:
            mkdir(self.__matlab__output_dir+'freeflame_data_MATLAB')
        if (not path.isdir(self.__matlab__output_dir+'burnerflame_data_MATLAB')) and self.__run_burnerflames:
            mkdir(self.__matlab__output_dir+'burnerflame_data_MATLAB')
        if (not path.isdir(self.__matlab__output_dir+'equilibrium_data_MATLAB')) and self.__run_equilibrium:
            mkdir(self.__matlab__output_dir+'equilibrium_data_MATLAB')
        if (not path.isdir(self.__matlab__output_dir+'counterflame_data_MATLAB')) and self.__run_counterflames:
            mkdir(self.__matlab__output_dir+'counterflame_data_MATLAB')
        return 

    def AddRandomData(self, flame_solution, mix_status, T_ub, extra_header=""):

        if self.__define_equivalence_ratio:
            folder_header = "phi"
        else:
            folder_header = "mixfrac"

        if not path.isdir(self.GetOutputDir()+'/fuzzy_data/'):
            mkdir(self.GetOutputDir()+'/fuzzy_data/')
        if not path.isdir(self.GetOutputDir()+'/fuzzy_data/'+folder_header+'_'+str(round(mix_status, 6))):
            mkdir(self.GetOutputDir()+'/fuzzy_data/'+folder_header+'_'+str(round(mix_status, 6)))

        fileHeader = "fuzzy_data_"+folder_header+str(round(mix_status,6))+"_"+extra_header+"_Tu"+str(round(T_ub, 4))+".csv"

        Y_flamelet = flame_solution.Y 
        T_flamelet = flame_solution.T 
        h_flamelet = flame_solution.enthalpy_mass
        h_max, h_min = max(h_flamelet), min(h_flamelet)
        gas_eq = ct.Solution(self.__reaction_mechanism)
        gas_eq.Y = flame_solution.Y[:,0]
        gas_eq.TP = T_flamelet[0], ct.one_atm

        OH_ratio_base = gas_eq.elemental_mass_fraction("O")/gas_eq.elemental_mass_fraction("H")
        ON_ratio_base = gas_eq.elemental_mass_fraction("O")/gas_eq.elemental_mass_fraction("N")

        filepathname = self.GetOutputDir()+'/fuzzy_data/'+folder_header+'_'+str(round(mix_status, 6)) + "/" + fileHeader
        a = 8
        b = 5

        for iX in range(len(flame_solution.grid)):   

            y_local = Y_flamelet[:, iX]
            gas_eq.Y = y_local
            valid_mixture = False 
            while not valid_mixture:
                c = np.random.uniform(low=-1, high=1)
                
                h_perturbed = flame_solution.enthalpy_mass[iX] + (c/a) * (h_max - h_min)
                y_perturbed = np.power(np.abs(y_local), 1 + (c / b))
                y_perturbed = y_perturbed / np.sum(y_perturbed)
                gas_eq.HP = h_perturbed, ct.one_atm
                gas_eq.Y = y_perturbed

                OH_ratio = gas_eq.elemental_mass_fraction("O")/gas_eq.elemental_mass_fraction("H")
                ON_ratio = gas_eq.elemental_mass_fraction("O")/gas_eq.elemental_mass_fraction("N")
                valid_OH_ratio = (OH_ratio >= 0.98*OH_ratio_base and OH_ratio <= 1.02*OH_ratio_base)
                valid_ON_ratio = (ON_ratio >= 0.98*ON_ratio_base and ON_ratio <= 1.02*ON_ratio_base)

                if valid_OH_ratio and valid_ON_ratio:
                    valid_mixture = True
            
            gas_eq.Y = y_perturbed
            gas_eq.HP = h_perturbed, ct.one_atm
            if iX == 0:
                variables, data_calc = self.__SaveFlameletData(gas_eq, self.gas)
                fid = open(filepathname, 'w+')
                fid.write(variables + "\n")
                fid.close()
            else:
                variables, data_calc_2 = self.__SaveFlameletData(gas_eq, self.gas)
                data_calc = np.append(data_calc, data_calc_2, axis=0)
        fid = open(filepathname, 'a+')
        csvWriter = csv.writer(fid)
        csvWriter.writerows(data_calc)
        fid.close()

    def ComputeFreeFlames(self, mix_status:float, T_ub:float, i_freeflame:int=0):
        """Generate adiabatic free-flamelet data for a specific mixture fraction or equivalence ratio and reactant temperature.

        :param mix_status: Equivalence ratio or mixture fraction value.
        :type mix_status: float
        :param T_ub: Reactant temperature in Kelvin.
        :type T_ub: float
        :param i_freeflame: Solution index, defaults to 0
        :type i_freeflame: int, optional
        """
        if self.__define_equivalence_ratio:
            folder_header = "phi"
        else:
            folder_header = "mixfrac"
        # Setting unburnt temperature and pressure
        self.gas.TP = T_ub, ct.one_atm
        # Defining mixture ratio based on equivalence ratio or mixture fraction.
        if self.__define_equivalence_ratio:
            self.gas.set_equivalence_ratio(mix_status, self.__fuel_string, self.__oxidizer_string)
        else:
            self.gas.set_mixture_fraction(mix_status, self.__fuel_string, self.__oxidizer_string)

        # Define Cantera adiabatic flame object.
        initialgrid = np.linspace(0, self.__initial_grid_length, self.__initial_grid_Np)
        flame:ct.FreeFlame = ct.FreeFlame(self.gas, grid=initialgrid)
        flame.set_refine_criteria(ratio=2, slope=0.025, curve=0.025)

        # Multi-component diffusion for differential diffusion effects.
        flame.transport_model = self.__transport_model

        # Try to solve the flamelet solution. If solution diverges, move on to next flamelet.
        try:
            flame.solve(loglevel=0, refine_grid=True, auto=True)
            
            # Computing mass flow rate for later burner flame evaluation
            self.m_dot_free_flame = flame.velocity[0]*flame.density[0]
            
            # Check if mixture is burning
            if np.max(flame.T) <= DefaultSettings_FGM.T_threshold:
                print("Flamelet at %s %.3e, Tu %.3f is not burning" % (folder_header, mix_status, T_ub))
                return 
            
            variables, data_calc = self.__SaveFlameletData(flame, self.gas)

            if self.__run_fuzzy:
                self.AddRandomData(flame, mix_status, T_ub)

            # Generate sub-directory if it's not there.
            if not path.isdir(self.GetOutputDir()+'/freeflame_data/'):
                mkdir(self.GetOutputDir()+'/freeflame_data/')
            if not path.isdir(self.GetOutputDir()+'/freeflame_data/'+folder_header+'_'+str(round(mix_status, 6))):
                mkdir(self.GetOutputDir()+'/freeflame_data/'+folder_header+'_'+str(round(mix_status, 6)))

            if max(flame.grid) < 1.0:
                freeflame_filename = "freeflamelet_"+folder_header+str(round(mix_status,6))+"_Tu"+str(round(T_ub, 4))+".csv"
                filename_plus_folder = self.GetOutputDir()+"/freeflame_data/"+folder_header+'_'+str(round(mix_status, 6)) + "/"+freeflame_filename
                fid = open(filename_plus_folder, 'w+')
                fid.write(variables + "\n")
                csvWriter = csv.writer(fid)
                csvWriter.writerows(data_calc)
                fid.close()

                if self.__translate_to_matlab:
                    if not path.isdir(self.__matlab__output_dir+'/freeflame_data_MATLAB/'+folder_header+'_'+str(round(mix_status, 6))):
                            mkdir(self.__matlab__output_dir+'/freeflame_data_MATLAB/'+folder_header+'_'+str(round(mix_status, 6)))
                    self.__TranslateToMatlabFile(filename_plus_folder,freeflame_filename, self.__matlab__output_dir + "/freeflame_data_MATLAB/"+folder_header+'_'+str(round(mix_status, 6)) + "/")
                self.last_Y_flamelet = flame.Y
                self.last_h_flamelet = flame.enthalpy_mass 
                self.last_T_flamelet = flame.T 

                print("Successfull Freeflame simulation at "+folder_header+": "+str(mix_status)+ " T_u: " +str(T_ub) + " ("+str(i_freeflame+1)+"/"+str(self.__n_flamelets)+")")
            else:
                print("Unsuccessfull Freeflame simulation at "+folder_header+": "+str(mix_status)+ " T_u: " +str(T_ub) + " ("+str(i_freeflame+1)+"/"+str(self.__n_flamelets)+")")
            
        except:
            print("Unsuccessfull Freeflame simulation at "+folder_header+": "+str(mix_status)+ " T_u: " +str(T_ub) + " ("+str(i_freeflame+1)+"/"+str(self.__n_flamelets)+")")

    def compute_SingleBurnerFlame(self, mix_status:float, T_burner:float, m_dot:float):
        """Compute the solution of a single burner-stabilized flamelet.

        :param mix_status: mixture fraction or equivalence ratio.
        :type mix_status: float
        :param T_burner: burner plate temperature
        :type T_burner: float
        :param m_dot: mass flux [kg m/s]
        :type m_dot: float
        :return: converged burner flame object
        :rtype: cantera.BurnerFlame
        """
        self.gas.TP = T_burner, DefaultSettings_FGM.pressure
        if self.__define_equivalence_ratio:
            self.gas.set_equivalence_ratio(mix_status, self.__fuel_string, self.__oxidizer_string)
        else:
            self.gas.set_mixture_fraction(mix_status, self.__fuel_string, self.__oxidizer_string)
        
        # Definie initial grid.
        initialgrid = np.linspace(0, self.__initial_grid_length, self.__initial_grid_Np)

        # Initiate burner flame object.
        burner_flame = ct.BurnerFlame(self.gas, grid=initialgrid)
        burner_flame.burner.mdot = m_dot 
        burner_flame.set_refine_criteria(ratio=2, slope=0.025, curve=0.025)
        burner_flame.transport_model = self.__transport_model
        burner_flame.solve(loglevel=0, refine_grid=True, auto=False)

        return burner_flame
    
    def ComputeBurnerFlames(self, mix_status:float, m_dot:np.ndarray[float], T_burner:float=None):
        """Generate burner-stabilized flamelet data for a specific mixture fraction or equivalence ratio and mass flux.

        :param mix_status: Equivalence ratio or mixture fraction value.
        :type mix_status: float
        :param m_dot: Mass flux array (kg s^{-1} m^{-1})
        :type m_dot: np.ndarray[float]
        """
        if self.__define_equivalence_ratio:
            folder_header = "phi"
        else:
            folder_header = "mixfrac"

        if T_burner == None:
            T_burner = self.__T_unburnt_lower
        self.gas.TP = T_burner, ct.one_atm

        if self.__define_equivalence_ratio:
            self.gas.set_equivalence_ratio(mix_status, self.__fuel_string, self.__oxidizer_string)
        else:
            self.gas.set_mixture_fraction(mix_status, self.__fuel_string, self.__oxidizer_string)

        for i_burnerflame, m_dot_next in enumerate(m_dot):
            try:
                burner_flame = self.compute_SingleBurnerFlame(mix_status, self.__T_unburnt_lower, m_dot_next)
                if np.max(burner_flame.T) <= DefaultSettings_FGM.T_threshold:
                    print("Burnerflame at %s %.3e, mdot %.2e is not burning" % (folder_header, mix_status, m_dot_next))
                    return 
            
                # Extracting flamelet data
                variables, data_calc = self.__SaveFlameletData(burner_flame, self.gas)

                if self.__run_fuzzy:
                    self.AddRandomData(burner_flame, mix_status, T_burner, "mdot_"+str(round(m_dot_next,4)))
                    
                # Generate sub-directory if it's not there.
                if not path.isdir(self.GetOutputDir()+'/burnerflame_data/'):
                    mkdir(self.GetOutputDir()+'/burnerflame_data/')
                if not path.isdir(self.GetOutputDir()+'/burnerflame_data/'+folder_header+'_'+str(round(mix_status, 6))):
                    mkdir(self.GetOutputDir()+'/burnerflame_data/'+folder_header+'_'+str(round(mix_status, 6)))
                # burnerflame_filename = "burnerflamelet_"+folder_header+str(round(mix_status,6))+"_mdot"+str(round(m_dot_next, 4))+".csv"
                burnerflame_filename = "burnerflamelet_%s%.6f_mdot%.4f.csv" % (folder_header, mix_status, m_dot_next)
                filename_plus_folder = self.GetOutputDir()+"/burnerflame_data/"+folder_header+'_'+str(round(mix_status, 6)) + "/"+burnerflame_filename
                fid = open(filename_plus_folder, 'w+')
                fid.write(variables + "\n")
                csvWriter = csv.writer(fid)
                csvWriter.writerows(data_calc)
                fid.close()

                if self.__translate_to_matlab:
                    if not path.isdir(self.__matlab__output_dir+'/burnerflame_data_MATLAB/'+folder_header+'_'+str(round(mix_status, 6))):
                        mkdir(self.__matlab__output_dir+'/burnerflame_data_MATLAB/'+folder_header+'_'+str(round(mix_status, 6)))
                    self.__TranslateToMatlabFile(filename_plus_folder,burnerflame_filename, self.__matlab__output_dir + "/burnerflame_data_MATLAB/"+folder_header+'_'+str(round(mix_status, 6)) + "/")

                Y_max, Y_min = np.max(burner_flame.Y,axis=1), np.min(burner_flame.Y,axis=1)
                delta_Y_flamelet = Y_max - Y_min 
                if max(delta_Y_flamelet) > 1e-5:
                    self.last_Y_flamelet = burner_flame.Y
                    self.last_h_flamelet = burner_flame.enthalpy_mass 
                    self.last_T_flamelet = burner_flame.T 

                print("Successfull burnerflame simulation at "+folder_header+": "+ str(mix_status)+" mdot: " + str(m_dot_next)+ " ("+str(i_burnerflame+1)+"/"+str(self.__n_flamelets)+")")
                    
            # else:
            #     print("delta pv too small at "+folder_header+": "+str(mix_status)+" (" + str(i_burnerflame+1) + "/"+str(self.__n_flamelets)+")")    
            except:
                print("Unsuccessfull burnerflame simulation at "+folder_header+": "+ str(mix_status)+" mdot: " + str(m_dot_next)+ " ("+str(i_burnerflame+1)+"/"+str(self.__n_flamelets)+")")
                pass
    
    def ComputeCounterFlowFlames(self, v_fuel:float, v_ox:float, T_ub:float):
        """Generate counter-flow diffusion flamelet data for a given temperature, and reactant velocities. 
        Strain rate is gradually increased until extinction in order to distribute data over the progress variable spectrum.

        :param v_fuel: Fuel reactant velocity in meters per second.
        :type v_fuel: float
        :param v_ox: Oxidizer reactant velocity in meters per second.
        :type v_ox: float
        :param T_ub: Reactant temperature in Kelvin.
        :type T_ub: float
        :raises Exception: If either of the velocity values is lower than zero.
        :raises Exception: If the reactant temperature is lower than 200 K.
        """
        if (v_fuel <= 0) or (v_ox <= 0):
            raise Exception("Reactant velocities should be higher than zero.")
        if T_ub < 200:
            raise Exception("Reactant temperature should be higher than 200K.")
        flame = ct.CounterflowDiffusionFlame(self.gas, width=18e-3)

        self.gas.set_mixture_fraction(1.0, self.__fuel_string, self.__oxidizer_string)
        self.gas.TP = T_ub, ct.one_atm
        rho_fuel = self.gas.density

        self.gas.set_mixture_fraction(0.0, self.__fuel_string, self.__oxidizer_string)
        self.gas.TP = T_ub, ct.one_atm
        rho_oxidizer = self.gas.density

        flame.P = ct.one_atm
        flame.fuel_inlet.Y = self.__fuel_string
        flame.fuel_inlet.T = T_ub
        flame.fuel_inlet.mdot = rho_fuel*v_fuel
        flame.oxidizer_inlet.Y = self.__oxidizer_string
        flame.oxidizer_inlet.T = T_ub
        flame.oxidizer_inlet.mdot = rho_oxidizer*v_ox
        flame.set_refine_criteria(ratio=3, slope=0.04, curve=0.06, prune=0.02)

        flame.solve(loglevel=0, auto=True)
        variables, data_calc = self.__SaveFlameletData(flame, self.gas)

        counterflame_filename = "counterflamelet_strain_0_Tu"+str(round(T_ub, 4))+".csv"
        if not path.isdir(self.GetOutputDir()+"/counterflame_data"):
            mkdir(self.GetOutputDir()+"/counterflame_data")
        fid = open(self.GetOutputDir()+"/counterflame_data/"+counterflame_filename, 'w+')
        fid.write(variables + "\n")
        csvWriter = csv.writer(fid)
        csvWriter.writerows(data_calc)
        fid.close()
        # Compute counterflow diffusion flames at increasing strain rates at 1 bar
        # The strain rate is assumed to increase by 25% in each step until the flame is
        # extinguished
        strain_factor = 1.25
        # Exponents for the initial solution variation with changes in strain rate
        # Taken from Fiala and Sattelmayer (2014)
        exp_d_a = -0.05
        exp_u_a = 1. / 2.
        exp_V_a = 1.
        exp_lam_a = 2.
        exp_mdot_a = 1. / 2.

        n_iter = 1
        strain_overload = False
        while not strain_overload:
            # Create an initial guess based on the previous solution
            # Update grid
            flame.flame.grid *= strain_factor ** exp_d_a
            normalized_grid = flame.grid / (flame.grid[-1] - flame.grid[0])
            # Update mass fluxes
            flame.fuel_inlet.mdot *= strain_factor ** exp_mdot_a
            flame.oxidizer_inlet.mdot *= strain_factor ** exp_mdot_a
            # Update velocities
            flame.set_profile('velocity', normalized_grid,
                        flame.velocity * strain_factor ** exp_u_a)
            flame.set_profile('spread_rate', normalized_grid,
                        flame.spread_rate * strain_factor ** exp_V_a)
            # Update pressure curvature
            flame.set_profile('lambda', normalized_grid, flame.L * strain_factor ** exp_lam_a)

            try:
                # Try solving the flame
                flame.solve(loglevel=0)
                self.last_counterflame_massfraction = flame.Y
                variables, data_calc = self.__SaveFlameletData(flame, self.gas)

                counterflame_filename = "counterflamelet_strain_"+str(n_iter)+"_Tu"+str(round(T_ub, 4))+".csv"
                fid = open(self.GetOutputDir()+"/counterflame_data/"+counterflame_filename, 'w+')
                fid.write(variables + "\n")
                csvWriter = csv.writer(fid)
                csvWriter.writerows(data_calc)
                fid.close()
                print("Successful Counter-Flow Diffusion Flame at Strain Iteration " + str(n_iter))
            except:
                print("Unsuccessful Counter-Flow Diffusion Flame at Strain Iteration " + str(n_iter))
                strain_overload = True
            n_iter += 1

    def ComputeEquilibrium(self, mix_status:float, T_range:np.ndarray[float], burnt:bool=False):
        """Generate chemical equilibrium data for a given mixture status and temperature range.

        :param mix_status: Mixture fraction or equivalence ratio.
        :type mix_status: float
        :param T_range: Reactant or product temperature range.
        :type T_range: np.array[float]
        :param burnt: Compute reaction product properties, defaults to False
        :type burnt: bool, optional
        """
        if self.__define_equivalence_ratio:
            folder_header = "phi"
        else:
            folder_header = "mixfrac"

        gas_eq = ct.Solution(self.__reaction_mechanism)

        if burnt:
            fileHeader = "equilibrium_b_"
        else:
            fileHeader = "equilibrium_ub_"
        if not path.isdir(self.GetOutputDir()+'/equilibrium_data/'):
                        mkdir(self.GetOutputDir()+'/equilibrium_data/')
        if not path.isdir(self.GetOutputDir() + "/equilibrium_data/" + folder_header+"_"+str(round(mix_status,6))):
            mkdir(self.GetOutputDir() + "/equilibrium_data/" + folder_header+"_"+str(round(mix_status,6)))
        
        if self.__define_equivalence_ratio:
            gas_eq.set_equivalence_ratio(mix_status, self.__fuel_string, self.__oxidizer_string)
        else:
            gas_eq.set_mixture_fraction(mix_status, self.__fuel_string, self.__oxidizer_string)

        gas_eq.TP = max(T_range), ct.one_atm 
        H_max = gas_eq.enthalpy_mass
        # In case of reaction products, set the maximum enthalpy to that of the reactants at the maximum temperature.
        if burnt:
            gas_eq.TP = min(T_range), ct.one_atm 
            gas_eq.equilibrate('TP')
            gas_eq.HP = H_max, ct.one_atm
            T_range = np.linspace(min(T_range), gas_eq.T, len(T_range))

        for i, T in enumerate(T_range):
            
            gas_eq.TP = T, ct.one_atm
  
            if i == 0:
                if not path.isdir(self.GetOutputDir()+'/equilibrium_data/'+folder_header+'_'+str(round(mix_status, 6))):
                    mkdir(self.GetOutputDir()+'/equilibrium_data/'+folder_header+'_'+str(round(mix_status, 6)))
                variables, data_calc = self.__SaveFlameletData(gas_eq, self.gas)
                fid = open(self.GetOutputDir()+"/equilibrium_data/"+folder_header+"_"+str(round(mix_status,6))+"/"+ fileHeader +folder_header+"_"+str(round(mix_status,6))+".csv", 'w+')
                fid.write(variables + "\n")
                fid.close()
            else:
                variables, data_calc_2 = self.__SaveFlameletData(gas_eq, self.gas)
                data_calc = np.append(data_calc, data_calc_2, axis=0)

        eq_filename = fileHeader +folder_header+"_"+str(round(mix_status,6))+".csv"
        filename_plus_folder = self.GetOutputDir()+"/equilibrium_data/"+folder_header+"_"+str(round(mix_status,6))+"/"+ eq_filename
        fid = open(filename_plus_folder, 'a+')
        csvWriter = csv.writer(fid)
        csvWriter.writerows(data_calc)
        fid.close()

        if self.__translate_to_matlab:
            if not path.isdir(self.__matlab__output_dir+'/equilibrium_data_MATLAB/'+folder_header+'_'+str(round(mix_status, 6))):
                    mkdir(self.__matlab__output_dir+'/equilibrium_data_MATLAB/'+folder_header+'_'+str(round(mix_status, 6)))
            self.__TranslateToMatlabFile(filename_plus_folder, eq_filename, self.__matlab__output_dir + "/equilibrium_data_MATLAB/"+folder_header+'_'+str(round(mix_status, 6)) + "/")

    def ComputeHardCorner(self, mix_status:float, T_u:float):
        print("Starting interpolation process...")
        gas_eq = ct.Solution(self.__reaction_mechanism)
        

        if self.__define_equivalence_ratio:
            folder_header = "phi"
        else:
            folder_header = "mixfrac"

        fileHeader = "corner_data_"
        if not path.isdir(self.GetOutputDir()+'/equilibrium_data/'):
                        mkdir(self.GetOutputDir()+'/equilibrium_data/')
        if not path.isdir(self.GetOutputDir() + "/equilibrium_data/" + folder_header+"_"+str(round(mix_status,6))):
            mkdir(self.GetOutputDir() + "/equilibrium_data/" + folder_header+"_"+str(round(mix_status,6)))
        
        if self.__define_equivalence_ratio:
            gas_eq.set_equivalence_ratio(mix_status, self.__fuel_string, self.__oxidizer_string)
        else:
            gas_eq.set_mixture_fraction(mix_status, self.__fuel_string, self.__oxidizer_string)
        gas_eq.TP = T_u, ct.one_atm 
        gas_eq.equilibrate("HP")
        gas_eq.TP = T_u, ct.one_atm 
        Y_target = gas_eq.Y 
        N_enth_grid = self.__n_flamelets+1
        Np_last_flamelet = len(self.last_h_flamelet)

        for i in range(Np_last_flamelet):
            enth_range = np.linspace(0, 1, N_enth_grid)
            Y_interpolated = np.zeros([gas_eq.n_species, N_enth_grid])
            T_interpolated = np.interp(enth_range, xp=np.array([0,1]),fp=np.array([self.last_T_flamelet[i], T_u]))
            for iSp in range(gas_eq.n_species):
                Y_interpolated[iSp, :] = np.interp(enth_range, xp=np.array([0, 1]),\
                                                fp=np.array([self.last_Y_flamelet[iSp, i], Y_target[iSp]]))
            
            max_Y, min_Y = np.max(Y_interpolated, axis=1), np.min(Y_interpolated, axis=1)
            min_T, max_T = max(T_interpolated), min(T_interpolated)
            for j in range(N_enth_grid):
                # T_fuzzy = T_interpolated[j] + (max_T - min_T)*self.__fuzzy_delta*(np.random.rand()-0.5)
                # Y_fuzzy = Y_interpolated[:, j] + (max_Y - min_Y)*self.__fuzzy_delta*(np.random.rand()-0.5)
                # Y_fuzzy = Y_fuzzy / np.sum(Y_fuzzy)
                gas_eq.TP = T_interpolated[j], ct.one_atm 
                gas_eq.Y = Y_interpolated[:, j]   
                if j == 0:
                    if not path.isdir(self.GetOutputDir()+'/equilibrium_data/'+folder_header+'_'+str(round(mix_status, 6))):
                        mkdir(self.GetOutputDir()+'/equilibrium_data/'+folder_header+'_'+str(round(mix_status, 6)))
                    variables, data_calc = self.__SaveFlameletData(gas_eq, self.gas)
                    fid = open(self.GetOutputDir()+"/equilibrium_data/"+folder_header+"_"+str(round(mix_status,6))+"/"+ fileHeader +folder_header+"_"+str(round(mix_status,6))+"_"+str(i)+".csv", 'w+')
                    fid.write(variables + "\n")
                    fid.close()
                else:
                    _, data_calc_2 = self.__SaveFlameletData(gas_eq, self.gas)
                    data_calc = np.append(data_calc, data_calc_2, axis=0)

            eq_filename = fileHeader +folder_header+"_"+str(round(mix_status,6))+"_"+str(i)+".csv"
            filename_plus_folder = self.GetOutputDir()+"/equilibrium_data/"+folder_header+"_"+str(round(mix_status,6))+"/"+ eq_filename
            fid = open(filename_plus_folder, 'a+')
            csvWriter = csv.writer(fid)
            csvWriter.writerows(data_calc)
            fid.close()

    def ComputeFlameletsOnMixStatus(self, mix_status:float):
        """Generate flamelet data for a given mixture fraction or equivalence ratio.

        :param mix_status: Mixture fraction or equivalence ratio value.
        :type mix_status: float
        :raises Exception: If mixture status value is below zero.
        """

        if mix_status < 0:
            raise Exception("Mixture status value should be positive.")
        
        T_unburnt_range = np.linspace(self.__T_unburnt_upper, self.__T_unburnt_lower, self.__n_flamelets)
        # Generate adiabatic freeflame data
        if self.__run_freeflames:
            # Generate and safe adiabatic flamelet data.
            for i_freeflame, T_ub in enumerate(T_unburnt_range):
                self.ComputeFreeFlames(mix_status=mix_status, T_ub=T_ub, i_freeflame=i_freeflame)

        # Generate burner-stabilized flamelet data
        if self.__run_burnerflames:
            # Generate a single freeflamelet solution for reference
            if not self.__run_freeflames:
                self.ComputeFreeFlames(mix_status=mix_status, T_ub=self.__T_unburnt_lower, i_freeflame=0)

            # Define mass flow rate range.
            m_dot_range = np.linspace(self.m_dot_free_flame, 0.001*self.m_dot_free_flame, self.__n_flamelets+1)
            m_dot_range = m_dot_range[:-1]

            # Generate and safe adiabatic flamelet data.
            self.ComputeBurnerFlames(mix_status=mix_status, m_dot=m_dot_range)

        # Generate chemical equilibrium data
        if self.__run_equilibrium:

            # Generate unburnt reactants data.
            self.ComputeEquilibrium(mix_status=mix_status,\
                                    T_range=np.linspace(self.__T_unburnt_lower, self.__T_unburnt_upper, 2*self.__n_flamelets),\
                                    burnt=False)
            
            # Generate reaction products data.
            self.ComputeEquilibrium(mix_status=mix_status,\
                                    T_range=np.linspace(self.__T_unburnt_lower, self.__T_unburnt_upper, 2*self.__n_flamelets),\
                                    burnt=True)

            # if self.__run_freeflames or self.__run_burnerflames:
            #     self.ComputeHardCorner(mix_status=mix_status, T_u=self.__T_unburnt_lower)

    def ComputeFlamelets(self):
        """Generate and store all flamelet data for the current settings.
        """

        T_unburnt_range = np.linspace(self.__T_unburnt_upper, self.__T_unburnt_lower, self.__n_flamelets)

        # Generate counter-flow diffusion flamelet data
        if self.__run_counterflames:
            
            if not path.isdir(self.GetOutputDir()+'counterflame_data'):
                mkdir(self.GetOutputDir()+'counterflame_data')
            for T_ub in T_unburnt_range:
                self.gas.TP = T_ub, 101325 
                self.gas.set_mixture_fraction(1.0, self.__fuel_string, self.__oxidizer_string)
                rho_fuel = self.gas.density_mass
                rhou_fuel = rho_fuel * self.__u_fuel 
                self.gas.set_mixture_fraction(0.0, self.__fuel_string, self.__oxidizer_string)
                rho_ox = self.gas.density_mass 
                self.__u_oxidizer = rhou_fuel / rho_ox 
                self.ComputeCounterFlowFlames(v_fuel=self.__u_fuel, v_ox=self.__u_oxidizer, T_ub=T_ub)

        # Generate all other flamelet types.
        for mix_status in self.__unb_mixture_status:
            self.ComputeFlameletsOnMixStatus(mix_status)

    def __SaveFlameletData(self,flame, gas:ct.Solution):
        """Save flamelet or chemical equilibrium data in csv file.

        :param flame: Converged Cantera flamelet class.
        :type flame: cantera.FreeFlame, cantera.BurnerFlame, or cantera.CounterFlowDiffusionFlame
        :param gas: Cantera Solution object containing molecular properties of the respective mixture.
        :type gas: cantera.Solution
        :return: Flamelet variables string and data array
        :rtype: str, np.ndarray
        """
        
        # Check if chemical equilibrium or flamelet data are supplied.
        flame_is_gas = (np.shape(flame.Y) == np.shape(gas.Y))
        molar_weights = np.reshape(gas.molecular_weights, [gas.n_species, 1])

        # Extract species mass and molar fractions, reaction rates, and species specific heat values.
        if flame_is_gas:
            Y = np.reshape(flame.Y, [gas.n_species, 1])
            X = np.reshape(flame.X, [gas.n_species, 1])
            net_reaction_rate = np.zeros(np.shape(Y))#flame.net_production_rates[:,np.newaxis]
            neg_reaction_rate =np.zeros(np.shape(Y))#flame.destruction_rates[:,np.newaxis]
            pos_reaction_rate = np.zeros(np.shape(Y))#net_reaction_rate- neg_reaction_rate
            cp_i = np.reshape(flame.partial_molar_cp/gas.molecular_weights, [gas.n_species, 1])
            enth_i = np.reshape(flame.partial_molar_enthalpies/gas.molecular_weights, [gas.n_species, 1])
            grid = np.zeros([1,1])
            velocity = np.zeros([1,1])
        else:
            Y = flame.Y
            X = flame.X
            net_reaction_rate = flame.net_production_rates
            neg_reaction_rate =flame.destruction_rates
            pos_reaction_rate = flame.net_production_rates - neg_reaction_rate
            cp_i = (flame.partial_molar_cp.T/gas.molecular_weights)
            enth_i = (flame.partial_molar_enthalpies.T/gas.molecular_weights)
            grid= flame.grid
            velocity = flame.velocity[:,np.newaxis]
        Y = Y.T
        try:
            mixture_fraction = flame.mixture_fraction("Bilger")
        except:
            mixture_fraction = np.sum(Y.T * np.reshape(self.z_i, [self.gas.n_species, 1]), axis=0) + self.c 
        
        mean_molar_weights = np.dot(molar_weights.T, X)
        enthalpy = flame.enthalpy_mass 

        density = flame.density
        cp = flame.cp_mass
        k = flame.thermal_conductivity

        T = flame.T
        
        viscosity = flame.viscosity
        
        Y_dot_net = net_reaction_rate * molar_weights
        Y_dot_pos = pos_reaction_rate * molar_weights
        Y_dot_neg = neg_reaction_rate * molar_weights / (Y.T+1e-11) 

        Le_i = ComputeLewisNumber(flame)
        if self.__transport_model == "unity-Lewis-number":
            Le_i = Le_i / Le_i

        cp_i = np.reshape(cp_i, np.shape(Y))
        enth_i = np.reshape(enth_i, np.shape(Y))
        
        Le_i = Le_i.T

        if flame_is_gas:
            Le_i = np.reshape(Le_i, [1, gas.n_species])

        if flame_is_gas:
            heat_rel = 0.0
        else:
            heat_rel = flame.heat_release_rate
        
        # Define variables and output data array.
        variables = 'Distance,'
        data_matrix = np.reshape(grid, [len(grid), 1])
        variables += 'Velocity,'
        data_matrix = np.append(data_matrix, velocity,axis=1)
        variables += ','.join("Y-"+s for s in gas.species_names)
        data_matrix = np.append(data_matrix, Y,axis=1)
        variables += ',' + ','.join("Y_dot_net-"+s for s in gas.species_names) 
        data_matrix = np.append(data_matrix, Y_dot_net.T, axis=1)
        variables += ',' + ','.join("Y_dot_pos-"+s for s in gas.species_names) 
        data_matrix = np.append(data_matrix, Y_dot_pos.T, axis=1)
        variables += ',' + ','.join("Y_dot_neg-"+s for s in gas.species_names) 
        data_matrix = np.append(data_matrix, Y_dot_neg.T, axis=1)
        variables += ',' + ','.join("Cp-"+s for s in gas.species_names)
        data_matrix = np.append(data_matrix, cp_i, axis=1)
        variables += ',' + ','.join("h-"+s for s in gas.species_names)
        data_matrix = np.append(data_matrix, enth_i, axis=1)
        variables += ',' + ','.join("Le-"+s for s in gas.species_names)
        data_matrix = np.append(data_matrix, Le_i, axis=1)


        if flame_is_gas:
            variables += ','+DefaultSettings_FGM.name_enth+','
            data_matrix = np.append(data_matrix, np.array([[enthalpy]]), axis=1)
            variables += DefaultSettings_FGM.name_mixfrac+','
            data_matrix = np.append(data_matrix, np.array([mixture_fraction]), axis=1)
            variables += 'Temperature,'
            data_matrix = np.append(data_matrix, np.array([[T]]), axis=1)
            variables += 'Density,'
            data_matrix = np.append(data_matrix, np.array([[density]]), axis=1)
            variables += 'MolarWeightMix,'
            data_matrix = np.append(data_matrix, mean_molar_weights.T, axis=1)
            variables += 'Cp,'
            data_matrix = np.append(data_matrix, np.array([[cp]]), axis=1)
            variables += 'Conductivity,'
            data_matrix = np.append(data_matrix, np.array([[k]]), axis=1)
            variables += 'ViscosityDyn,'
            data_matrix = np.append(data_matrix, np.array([[viscosity]]), axis=1)
            variables += 'Heat_Release'
            data_matrix = np.append(data_matrix, np.array([[heat_rel]]), axis=1)
        else:
            variables += ','+DefaultSettings_FGM.name_enth+','
            data_matrix = np.append(data_matrix, np.reshape(enthalpy, [len(enthalpy),1]), axis=1)
            variables += DefaultSettings_FGM.name_mixfrac+','
            data_matrix = np.append(data_matrix, np.reshape(mixture_fraction, [len(mixture_fraction),1]), axis=1)
            variables += 'Temperature,'
            data_matrix = np.append(data_matrix, np.reshape(T, [len(T), 1]), axis=1)
            variables += 'Density,'
            data_matrix = np.append(data_matrix, np.reshape(density, [len(density), 1]), axis=1)
            variables += 'MolarWeightMix,'
            data_matrix = np.append(data_matrix, mean_molar_weights.T, axis=1)
            variables += 'Cp,'
            data_matrix = np.append(data_matrix, np.reshape(cp, [len(cp), 1]), axis=1)
            variables += 'Conductivity,'
            data_matrix = np.append(data_matrix, np.reshape(k, [len(k), 1]), axis=1)
            variables += 'ViscosityDyn,'
            data_matrix = np.append(data_matrix, np.reshape(viscosity, [len(viscosity), 1]), axis=1)
            variables += 'Heat_Release'
            data_matrix = np.append(data_matrix, np.reshape(heat_rel, [len(heat_rel), 1]), axis=1)

        return variables, data_matrix

    def __TranslateToMatlabFile(self, filename:str, filename_out:str, output_dir:str):
        """Translate default FlameletAI output file to TableMaster compatible file.

        :param filename: default FlameletAI output file name.
        :type filename: str
        :param filename_out: output file name.
        :type filename_out: str
        :param output_dir: folder where to store the translated file.
        :type output_dir: str
        """
        fid = open(filename, "r")
        variables = fid.readline().strip().split(',')
        fid.close()

        data_flamelet = np.loadtxt(filename,delimiter=',',skiprows=1)

        species_in_flamelet = []
        species_molecular_weights = []
        for v in variables:
            if v[:2] == 'Y-':
                species_in_flamelet.append(v[2:])
                species_molecular_weights.append(self.gas.molecular_weights[self.gas.species_index(v[2:])])

        variables_1 = ['Distance',\
            'Temperature',\
            'Density',\
            'Conductivity',\
            'Dynamic_Viscosity',\
            'Cp',\
            'Total_Enthalpy',\
            'Heat_Release',\
            'Mixture_Fraction']

        variables_translated = ['Distance',\
                                'T',\
                                'rho',\
                                'Conductivity',\
                                'ViscosityDyn',\
                                'cp',\
                                'Enthalpy total',\
                                'Heat release rate',\
                                'Mixture Fraction']

        units = ['m',\
                'K', \
                'kg m^-3',\
                'W/m/K',\
                'kg/m/s',\
                'J/kg/K',\
                'J/kg',\
                'W/m^3',\
                '-']

        fid = open(output_dir + "/" + filename_out, 'w+')
        fid.write("Cantera (Bosch edit) flamelet\n\n")
        fid.write("Molecular weights:\n")
        fid.write(",".join(species_in_flamelet) + "\n")
        fid.write(",".join([str(m) for m in species_molecular_weights]) + "\n\n")
        fid.write(",".join([variables_translated[i] + " ("+units[i]+")" for i in range(len(variables_translated))]) + ",")
        fid.write(",".join(["Y-"+s for s in species_in_flamelet]) + ",")
        fid.write(",".join(["ReacRatePos-"+s for s in species_in_flamelet]) + ",")
        fid.write(",".join(["ReacRateNeg-"+s for s in species_in_flamelet]) + ",")
        fid.write(",".join(["cp-"+s for s in species_in_flamelet]) + ",")
        fid.write(",".join(["Enthalpy-"+s for s in species_in_flamelet]) + ",")
        fid.write(",".join(["Le-"+s for s in species_in_flamelet]))

        fid.write('\n\n')
        fid.close()

        idx_vars = [variables.index(v) for v in variables_1]
        idx_massfrac = [variables.index("Y-"+s) for s in species_in_flamelet]
        idx_pos_reacrate = [variables.index("Y_dot_pos-"+s) for s in species_in_flamelet]
        idx_neg_reacrate = [variables.index("Y_dot_neg-"+s) for s in species_in_flamelet]
        idx_cp_sp = [variables.index("Cp-"+s) for s in species_in_flamelet]
        idx_h_sp = [variables.index("h-"+s) for s in species_in_flamelet]
        idx_le_sp = [variables.index("Le-"+s) for s in species_in_flamelet]

        thermophysical_props = data_flamelet[:, [i for i in idx_vars]]
        massfracs = data_flamelet[:, [i for i in idx_massfrac]]
        pos_reacrate = data_flamelet[:, [i for i in idx_pos_reacrate]] / np.array([species_molecular_weights])
        neg_reacrate = data_flamelet[:, [i for i in idx_neg_reacrate]] / np.array([species_molecular_weights])
        cp_sp = data_flamelet[:, [i for i in idx_cp_sp]]
        h_sp = data_flamelet[:, [i for i in idx_h_sp]]
        le_sp = data_flamelet[:, [i for i in idx_le_sp]]

        total_data = np.hstack([thermophysical_props,\
                            massfracs,\
                            pos_reacrate,\
                            neg_reacrate,\
                            cp_sp,\
                            h_sp,le_sp])

        with open(output_dir + "/" + filename_out, "a+") as fid:
            csvWriter = csv.writer(fid)
            csvWriter.writerows(total_data)

def ComputeFlameletData(Config:FlameletAIConfig, run_parallel:bool=False, N_processors:int=2):
    """Generate flamelet data according to FlameletAIConfig settings either in serial or parallel.

    :param Config: FlameletAIConfig class containing manifold and flamelet generation settings.
    :type Config: FlameletAIConfig
    :param run_parallel: Generate flamelet data in parallel, defaults to False
    :type run_parallel: bool, optional
    :param N_processors: Number of parallel jobs when generating flamelet data in parallel, defaults to 0
    :type N_processors: int, optional
    :raises Exception: If number of processors is set to zero when running in parallel.
    """

    if run_parallel and (N_processors == 0):
        raise Exception("Number of processors should be higher than zero when running in parallel.")

    mix_bounds = Config.GetMixtureBounds()
    Np_unb_mix = Config.GetNpMix()
    Config.gas.TP=300,101325
    Config.gas.set_equivalence_ratio(1.0, Config.GetFuelString(), Config.GetOxidizerString())
    if Config.GetMixtureStatus():
        mix_status_stoch = Config.gas.mixture_fraction(Config.GetFuelString(), Config.GetOxidizerString())
    else:
        mix_status_stoch = Config.gas.equivalence_ratio(Config.GetFuelString(), Config.GetOxidizerString())
    if mix_bounds[0] < mix_status_stoch and mix_bounds[1] > mix_status_stoch:
        mixture_range_lean = np.linspace(mix_bounds[0], mix_status_stoch, int(Np_unb_mix/2))
        mixture_range_rich = np.linspace(mix_status_stoch, mix_bounds[1], int(Np_unb_mix/2)+1)
        mixture_range = np.append(mixture_range_lean, mixture_range_rich[1:])
    else:
        # Equivalence ratios to calculate flamelets for are system inputs
        mixture_range = np.linspace(mix_bounds[0], mix_bounds[1], Np_unb_mix)

    # Set up Cantera flamelet generator object

    def ComputeFlameletData(mix_input):

        F = FlameletGenerator_Cantera(Config)
        F.ComputeFlameletsOnMixStatus(mix_input)

    if run_parallel:
        Parallel(n_jobs=N_processors)(delayed(ComputeFlameletData)(mix_status) for mix_status in mixture_range)
    else:
        F = FlameletGenerator_Cantera(Config)
        F.SetMixtureValues(mixture_range)
        F.ComputeFlamelets()

def ComputeBoundaryData(Config:FlameletAIConfig, run_parallel:bool=False, N_processors:int=2):

    def ComputeEquilibriumData(mix_input):
        F = FlameletGenerator_Cantera(Config)
        F.RunMixtureFraction()
        F.RunEquilibrium(True)
        F.RunFreeFlames(False)
        F.RunBurnerFlames(False)
        F.RunCounterFlowFlames(False)
        F.ComputeFlameletsOnMixStatus(mix_input)


    mix_bounds = Config.GetMixtureBounds()
    Np_unb_mix = Config.GetNpMix()
    Config.gas.TP=300,101325
    Config.gas.set_equivalence_ratio(1.0, Config.GetFuelString(), Config.GetOxidizerString())
    #if Config.GetMixtureStatus():
    mix_status_stoch = Config.gas.mixture_fraction(Config.GetFuelString(), Config.GetOxidizerString())
    # else:
    #     mix_status_stoch = Config.gas.equivalence_ratio(Config.GetFuelString(), Config.GetOxidizerString())
    #if mix_bounds[0] < mix_status_stoch and mix_bounds[1] > mix_status_stoch:
    mixture_range_lean = np.linspace(0, mix_status_stoch, int(Np_unb_mix/2))
    mixture_range_rich = np.linspace(mix_status_stoch, 1, int(Np_unb_mix/2)+1)
    mixture_range = np.append(mixture_range_lean, mixture_range_rich[1:])
    # else:
    #     # Equivalence ratios to calculate flamelets for are system inputs
    #     mixture_range = np.linspace(mix_bounds[0], mix_bounds[1], Np_unb_mix)
    
    if run_parallel:
        Parallel(n_jobs=N_processors)(delayed(ComputeEquilibriumData)(mix_status) for mix_status in mixture_range)
    else:
        for z in mixture_range:
            ComputeEquilibriumData(z)