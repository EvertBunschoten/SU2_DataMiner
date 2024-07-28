import CoolProp.CoolProp as CP
import cantera as ct 
import numpy as np 
from tqdm import tqdm
import csv 
from os import path, mkdir
import matplotlib.pyplot as plt 
from joblib import Parallel, delayed

from Common.EntropicAIConfig import EntropicAIConfig, FlameletAIConfig
from Common.Properties import DefaultProperties
from DataGenerator_Base import DataGenerator_Base
from Common.CommonMethods import ComputeLewisNumber

class DataGenerator_CoolProp(DataGenerator_Base):
    """Class for generating fluid data using CoolProp
    """

    __Config:EntropicAIConfig = None 
    __fluid = None 

    # Pressure and temperature limits
    __use_PT:bool = DefaultProperties.use_PT_grid
    __T_min:float = DefaultProperties.T_min
    __T_max:float = DefaultProperties.T_max
    __Np_Y:int = DefaultProperties.Np_temp

    __P_min:float = DefaultProperties.P_min
    __P_max:float = DefaultProperties.P_max
    __Np_X:int = DefaultProperties.Np_p

    # Density and static energy limits
    __rho_min:float = DefaultProperties.Rho_min
    __rho_max:float = DefaultProperties.Rho_max
    __e_min:float = DefaultProperties.Energy_min 
    __e_max:float = DefaultProperties.Energy_max 
    __X_grid:np.ndarray[float] = None
    __Y_grid:np.ndarray[float] = None 

    # Entropy derivatives.
    __s_fluid:np.ndarray[float] = None 
    __dsdrho_e_fluid:np.ndarray[float] = None 
    __dsde_rho_fluid:np.ndarray[float] = None 
    __d2sdrho2_fluid:np.ndarray[float] = None 
    __d2sde2_fluid:np.ndarray[float] = None 
    __d2sdedrho_fluid:np.ndarray[float] = None 

    # Computed thermodynamic fluid properties.
    __T_fluid:np.ndarray[float] = None 
    __P_fluid:np.ndarray[float] = None 
    __e_fluid:np.ndarray[float] = None 
    __rho_fluid:np.ndarray[float] = None 
    __c2_fluid:np.ndarray[float] = None 
    __success_locations:np.ndarray[bool] = None 

    def __init__(self, Config_in:EntropicAIConfig):
        DataGenerator_Base.__init__(self, Config_in=Config_in)
        # Load configuration and set default properties.
        self.__use_PT = self.__Config.GetPTGrid()
        if len(self.__Config.GetFluidNames()) > 1:
            fluid_names = self.__Config.GetFluidNames()
            CAS_1 = CP.get_fluid_param_string(fluid_names[0], "CAS")
            CAS_2 = CP.get_fluid_param_string(fluid_names[1], "CAS")
            CP.apply_simple_mixing_rule(CAS_1, CAS_2,'linear')
        self.__fluid = CP.AbstractState("HEOS", self.__Config.GetFluidName())
        if len(self.__Config.GetFluidNames()) > 1:
            mole_fractions = self.__Config.GetMoleFractions()
            self.__fluid.set_mole_fractions(mole_fractions)
        self.__use_PT = self.__Config.GetPTGrid()
        P_bounds = self.__Config.GetPressureBounds()
        T_bounds = self.__Config.GetTemperatureBounds()
        rho_bounds = self.__Config.GetDensityBounds()
        e_bounds = self.__Config.GetEnergyBounds()

        self.__P_min, self.__P_max = P_bounds[0], P_bounds[1]
        self.__rho_min, self.__rho_max = rho_bounds[0], rho_bounds[1]
        self.__Np_X = self.__Config.GetNpPressure()

        self.__T_min, self.__T_max = T_bounds[0], T_bounds[1]
        self.__e_min, self.__e_max = e_bounds[0], e_bounds[1]
        self.__Np_Y = self.__Config.GetNpTemp()

        return 
    
    def PreprocessData(self):
        """Generate density and static energy grid at which to evaluate fluid properties.
        """

        if self.__use_PT:
            X_min = self.__P_min
            X_max = self.__P_max
            Y_min = self.__T_min
            Y_max = self.__T_max
        else:
            X_min = self.__rho_min
            X_max = self.__rho_max
            Y_min = self.__e_min
            Y_max = self.__e_max 
        
        X_range = (X_min - X_max) * np.cos(np.linspace(0, 0.5*np.pi, self.__Np_X)) + X_max
        Y_range = np.linspace(Y_min, Y_max, self.__Np_Y)

        self.__X_grid, self.__Y_grid = np.meshgrid(X_range, Y_range)
        return 
    
    def VisualizeDataGrid(self):
        """Visualize query points at which fluid data are evaluated.
        """
        fig = plt.figure(figsize=[10,10])
        ax = plt.axes()
        ax.plot(self.__X_grid.flatten(), self.__Y_grid.flatten(), 'k.')
        if self. __use_PT:
            ax.set_xlabel(r"Pressure $(p)[Pa]",fontsize=20)
            ax.set_ylabel(r"Temperature $(T)[K]",fontsize=20)
        else:
            ax.set_xlabel(r"Density $(\rho)[kg m^{-3}]$",fontsize=20)
            ax.set_ylabel(r"Static energy $(e)[J kg^{-1}]$",fontsize=20)
        ax.tick_params(which='both',labelsize=18)
        ax.grid()
        
        plt.show()
        return 
    
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
        """Get fluid temperature limits.

        :return: list with minimum and maximum temperature.
        :rtype: list[float]
        """
        return [self.__T_lower, self.__T_upper]
    
    def SetNpDensity(self, Np_density:int=DefaultProperties.Np_p):
        self.SetNpPressure(Np_P=Np_density)
        return 

    def GetNpDensity(self):
        return self.GetNpPressure()
    
    def SetDensityBounds(self, Density_lower:float=DefaultProperties.Rho_min, Density_upper:float=DefaultProperties.Rho_max):
        self.__rho_min = Density_lower
        self.__rho_max = Density_upper
        return 
    
    def SetNpEnergy(self, Np_energy:int=DefaultProperties.Np_temp):
        self.SetNpTemp(Np_Temp=Np_energy)
        return 
    
    def GetNpEnergy(self):
        return self.GetNpTemp()
    
    def SetEnergyBounds(self, Energy_lower:float=DefaultProperties.Energy_min, Energy_upper:float=DefaultProperties.Energy_max):
        self.__e_min = Energy_lower
        self.__e_max = Energy_upper
        return 
    
    
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
            self.__Np_Y = Np_Temp
        return 
    
    def GetNpTemp(self):
        """
        Get the number of divisions for the fluid temperature range.

        :return: Number of divisions for the fluid temperature range.
        :rtype: int

        """
        return self.__Np_Y
    

    
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
            self.__Np_X = Np_P 
        return 
    
    def GetNpPressure(self):
        return self.__Np_X
    
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
        """Get minimum and maximum pressure.

        :return: list with minimum and maximum fluid pressure.
        :rtype: list[float]
        """
        return [self.__P_lower, self.__P_upper]
    

    def ComputeData(self):
        super().ComputeData()
        """Evaluate fluid properties on the density-energy grid or pressure-temperature grid.
        """

        # Initiate empty fluid property arrays.
        self.__s_fluid = np.zeros([self.__Np_X, self.__Np_Y])
        self.__dsde_rho_fluid = np.zeros([self.__Np_X, self.__Np_Y])
        self.__dsdrho_e_fluid = np.zeros([self.__Np_X, self.__Np_Y])
        self.__d2sde2_fluid = np.zeros([self.__Np_X, self.__Np_Y])
        self.__d2sdrho2_fluid = np.zeros([self.__Np_X, self.__Np_Y])
        self.__d2sdedrho_fluid = np.zeros([self.__Np_X, self.__Np_Y])
        self.__P_fluid = np.zeros([self.__Np_X, self.__Np_Y])
        self.__T_fluid = np.zeros([self.__Np_X, self.__Np_Y])
        self.__c2_fluid = np.zeros([self.__Np_X, self.__Np_Y])
        self.__rho_fluid = np.zeros([self.__Np_X, self.__Np_Y])
        self.__e_fluid = np.zeros([self.__Np_X, self.__Np_Y])

        self.__success_locations = np.ones([self.__Np_X, self.__Np_Y],dtype=bool)
        
        # Loop over density-based or pressure-based grid.
        for i in tqdm(range(self.__Np_X)):
            for j in range(self.__Np_Y):
                try:
                    if self.__use_PT:
                        self.__fluid.update(CP.PT_INPUTS, self.__X_grid[i,j], self.__Y_grid[i,j])
                    else:
                        self.__fluid.update(CP.DmassUmass_INPUTS, self.__X_grid[j,i], self.__Y_grid[j,i])

                    # Check if fluid phase is not vapor or liquid
                    if (self.__fluid.phase() != 0) and (self.__fluid.phase() != 3) and (self.__fluid.phase() != 6):
                        self.__s_fluid[i,j] = self.__fluid.smass()
                        self.__dsde_rho_fluid[i,j] = self.__fluid.first_partial_deriv(CP.iSmass, CP.iUmass, CP.iDmass)
                        self.__dsdrho_e_fluid[i,j] = self.__fluid.first_partial_deriv(CP.iSmass, CP.iDmass, CP.iUmass)
                        self.__d2sde2_fluid[i,j] = self.__fluid.second_partial_deriv(CP.iSmass, CP.iUmass, CP.iDmass, CP.iUmass, CP.iDmass)
                        self.__d2sdedrho_fluid[i,j] = self.__fluid.second_partial_deriv(CP.iSmass, CP.iUmass, CP.iDmass, CP.iDmass, CP.iUmass)
                        self.__d2sdrho2_fluid[i,j] = self.__fluid.second_partial_deriv(CP.iSmass, CP.iDmass, CP.iUmass, CP.iDmass, CP.iUmass)
                        self.__P_fluid[i,j] = self.__fluid.p()
                        self.__T_fluid[i,j] = self.__fluid.T()
                        self.__c2_fluid[i,j] = self.__fluid.speed_sound()**2
                        self.__rho_fluid[i,j] = self.__fluid.rhomass()
                        self.__e_fluid[i,j] = self.__fluid.umass()
                    else:
                        self.__success_locations[i,j] = False
                        self.__s_fluid[i, j] = None
                        self.__dsde_rho_fluid[i,j] = None 
                        self.__dsdrho_e_fluid[i,j] = None 
                        self.__d2sde2_fluid[i,j] = None 
                        self.__d2sdrho2_fluid[i,j] = None 
                        self.__d2sdedrho_fluid[i,j] = None 
                        self.__c2_fluid[i,j] = None 
                        self.__P_fluid[i,j] = None 
                        self.__T_fluid[i,j] = None 
                        self.__rho_fluid[i,j] = None 
                        self.__e_fluid[i,j] = None 
                except:
                    self.__success_locations[i,j] = False 
                    self.__s_fluid[i, j] = None
                    self.__dsde_rho_fluid[i,j] = None 
                    self.__dsdrho_e_fluid[i,j] = None 
                    self.__d2sde2_fluid[i,j] = None 
                    self.__d2sdrho2_fluid[i,j] = None 
                    self.__d2sdedrho_fluid[i,j] = None 
                    self.__c2_fluid[i,j] = None 
                    self.__P_fluid[i,j] = None 
                    self.__T_fluid[i,j] = None 
                    self.__rho_fluid[i,j] = None 
                    self.__e_fluid[i,j] = None 
        return 
    
    def VisualizeFluidData(self):
        """Visualize computed fluid data.
        """

        fig = plt.figure(figsize=[27, 9])
        ax0 = fig.add_subplot(1, 3, 1, projection='3d')
        ax0.plot_surface(self.__rho_fluid, self.__e_fluid, self.__P_fluid)
        ax0.set_xlabel("Density [kg/m3]",fontsize=20)
        ax0.set_ylabel("Static Energy [J/kg]",fontsize=20)
        ax0.set_zlabel("Pressure [Pa]",fontsize=20)
        ax0.tick_params(which='both',labelsize=18)
        ax0.grid()
        ax0.set_title("Fluid pressure data",fontsize=22)

        ax1 = fig.add_subplot(1, 3, 2, projection='3d')
        ax1.plot_surface(self.__rho_fluid, self.__e_fluid, self.__T_fluid)
        ax1.set_xlabel("Density [kg/m3]",fontsize=20)
        ax1.set_ylabel("Static Energy [J/kg]",fontsize=20)
        ax1.set_zlabel("Temperature [K]",fontsize=20)
        ax1.tick_params(which='both',labelsize=18)
        ax1.grid()
        ax1.set_title("Fluid temperature data",fontsize=22)

        ax2 = fig.add_subplot(1, 3, 3, projection='3d')
        ax2.plot_surface(self.__rho_fluid, self.__e_fluid, np.sqrt(self.__c2_fluid))
        ax2.set_xlabel("Density [kg/m3]",fontsize=20)
        ax2.set_ylabel("Static Energy [J/kg]",fontsize=20)
        ax2.set_zlabel("Speed of sound [m/s]",fontsize=20)
        ax2.tick_params(which='both',labelsize=18)
        ax2.grid()
        ax2.set_title("Fluid speed of sound data",fontsize=22)

        plt.show()
        return 
    
    def SaveData(self):
        """Save fluid data in separate files for train, test and validation.
        """

        output_dir = self.__output_dir

        # Define output files for all, train, test, and validation data.
        full_file = output_dir + "/" + self.__output_file_header + "_full.csv"
        train_file = output_dir + "/" + self.__output_file_header + "_train.csv"
        test_file = output_dir + "/" + self.__output_file_header + "_test.csv"
        val_file = output_dir + "/" + self.__output_file_header + "_val.csv"

        # Append controlling and training variable data.
        controlling_vars = ["Density", "Energy"]
        entropic_vars = ["s","dsdrho_e","dsde_rho","d2sdrho2","d2sde2","d2sdedrho"]
        TD_vars = ["T","p","c2"]

        all_vars = controlling_vars + entropic_vars + TD_vars

        CV_data = np.vstack((self.__rho_fluid.flatten(), \
                             self.__e_fluid.flatten())).T 
        entropic_data = np.vstack((self.__s_fluid.flatten(),\
                                   self.__dsdrho_e_fluid.flatten(),\
                                   self.__dsde_rho_fluid.flatten(),\
                                   self.__d2sdrho2_fluid.flatten(),\
                                   self.__d2sde2_fluid.flatten(),\
                                   self.__d2sdedrho_fluid.flatten())).T 
        TD_data = np.vstack((self.__T_fluid.flatten(),\
                             self.__P_fluid.flatten(),\
                             self.__c2_fluid.flatten())).T
        
        full_data = np.hstack((CV_data, entropic_data, TD_data))
        full_data = full_data[self.__success_locations.flatten(), :]

        # Shuffle data array.
        np.random.shuffle(full_data)

        # Define number of training and test data points.
        Np_full = np.shape(full_data)[0]
        Np_train = int(self.__train_fraction*Np_full)
        Np_test = int(self.__test_fraction*Np_full)

        train_data = full_data[:Np_train, :]
        test_data = full_data[Np_train:Np_train+Np_test, :]
        val_data = full_data[Np_train+Np_test:, :]

        # Write output data files.
        with open(full_file,"w+") as fid:
            fid.write(",".join(v for v in all_vars) + "\n")
            csvWriter = csv.writer(fid)
            csvWriter.writerows(full_data)
        
        with open(train_file,"w+") as fid:
            fid.write(",".join(v for v in all_vars) + "\n")
            csvWriter = csv.writer(fid)
            csvWriter.writerows(train_data)

        with open(test_file,"w+") as fid:
            fid.write(",".join(v for v in all_vars) + "\n")
            csvWriter = csv.writer(fid)
            csvWriter.writerows(test_data)

        with open(val_file,"w+") as fid:
            fid.write(",".join(v for v in all_vars) + "\n")
            csvWriter = csv.writer(fid)
            csvWriter.writerows(val_data)
            
        return 
    

class FlameletGenerator_Cantera(DataGenerator_Base):
    """Generate flamelet data using Cantera.

    :param Config: FlameletAIConfig class describing the flamelet generation settings.
    :type: FlameletAIConfig
    """
    # Generate flamelet data from Cantera computation.
    __Config:FlameletAIConfig

    # Save directory for computed flamelet data
    __matlab_output_dir:str = "./"

    __fuel_definition:list[str] = ['H2']    # Fuel species
    __fuel_weights:list[float] = [1.0]        # Fuel molar weights
    __fuel_string:str = ''
    __oxidizer_definition:list[str] = ['O2', 'N2']  # Oxidizer species
    __oxidizer_weights:list[float] = [1.0, 3.76]      # Oxidizer molar weights
    __oxidizer_string:str = ''

    __n_flamelets:int = 100       # Number of adiabatic and burner flame computations per mixture fraction
    __T_unburnt_upper:float = 800   # Highest unburnt reactant temperature
    __T_unburnt_lower:float = 350   # Lowest unburnt reactant temperature

    __reaction_mechanism:str = 'gri30.yaml'   # Cantera reaction mechanism
    __transport_model:str = "multicomponent"

    __define_equivalence_ratio:bool = True # Define unburnt mixture via the equivalence ratio
    __unb_mixture_status:list[float] = [] 

    __translate_to_matlab:bool = False # Save a copy of the flamelet data file in Matlab table generator format

    __run_freeflames:bool = False      # Run adiabatic flame computations
    __run_burnerflames:bool = False    # Run burner stabilized flame computations
    __run_equilibrium:bool = False     # Run chemical equilibrium computations
    __run_counterflames:bool = False   # Run counter-flow diffusion flamelet simulations.
    __run_fuzzy:bool = False           # Add randomized data around flamelet solutions to manifold.

    __u_fuel:float = 1.0       # Fuel stream velocity in counter-flow diffusion flame.
    __u_oxidizer:float = None   # Oxidizer stream velocity in counter-flow diffusion flame.

    __fuzzy_delta:float = 0.1

    def __init__(self, Config:FlameletAIConfig):
        DataGenerator_Base.__init__(self, Config_in=Config)

        """Constructur, load flamelet generation settings from FlameletAIConfig.

        :param Config: FlameletAIConfig containing respective settings.
        :type Config: FlameletAIConfig
        """

        print("Initializing flamelet generator from FlameletAIConfig with name " + self.__Config.GetConfigName())
        self.__fuel_definition = self.__Config.GetFuelDefinition()
        self.__fuel_weights = self.__Config.GetFuelWeights()
        
        self.__oxidizer_definition = self.__Config.GetOxidizerDefinition()
        self.__oxidizer_weights = self.__Config.GetOxidizerWeights()

        self.__fuel_string = ",".join([self.__fuel_definition[i] + ":" + str(self.__fuel_weights[i]) for i in range(len(self.__fuel_definition))])
        self.__oxidizer_string = ",".join([self.__oxidizer_definition[i] + ":" + str(self.__oxidizer_weights[i]) for i in range(len(self.__oxidizer_definition))])

        self.__reaction_mechanism = self.__Config.GetReactionMechanism()
        self.gas = ct.Solution(self.__Config.GetReactionMechanism())

        self.__n_flamelets = self.__Config.GetNpTemp()
        [self.__T_unburnt_lower, self.__T_unburnt_upper] = self.__Config.GetUnbTempBounds()

        self.__define_equivalence_ratio = (not self.__Config.GetMixtureStatus())
        self.__unb_mixture_status = np.linspace(self.__Config.GetMixtureBounds()[0], self.__Config.GetMixtureBounds()[1], self.__Config.GetNpMix())
        self.__run_freeflames = self.__Config.GenerateFreeFlames()
        self.__run_burnerflames = self.__Config.GenerateBurnerFlames()
        self.__run_equilibrium = self.__Config.GenerateEquilibrium()
        self.__run_counterflames = self.__Config.GenerateCounterFlames()
        
        self.__PrepareOutputDirectories()
        self.__translate_to_matlab = self.__Config.WriteMatlabFiles()
        if self.__translate_to_matlab:
            self.__PrepareOutputDirectories_Matlab()

        self.z_i = self.__Config.GetMixtureFractionCoefficients()
        self.c = self.__Config.GetMixtureFractionConstant()
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
        # Set the premixed fuel for the flame computations
        if len(fuel_species) == 0:
            raise Exception("Fuel definition should contain at least one specie.")
        if len(fuel_species) != (len(fuel_weights)):
            raise Exception("The number of fuel species and weights should be equal.")
        self.__fuel_definition = []
        self.__fuel_weights = []
        for f in fuel_species:
            self.__fuel_definition.append(f)
        for w in fuel_weights:
            self.__fuel_weights.append(w)
        self.__fuel_string = ",".join([self.__fuel_definition[i] + ":" + str(self.__fuel_weights[i]) for i in range(len(self.__fuel_definition))])
        self.__Config.SetFuelDefinition(fuel_species, fuel_weights)
        self.__Config.ComputeMixFracConstants()
        self.z_i = self.__Config.GetMixtureFractionCoefficients()
        self.c = self.__Config.GetMixtureFractionConstant()
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

        if len(oxidizer_species) == 0:
            raise Exception("Oxidizer definition should contain at least one specie.")
        if len(oxidizer_species) != (len(oxidizer_weights)):
            raise Exception("The number of oxidizer species and weights should be equal.")
        self.__oxidizer_definition = []
        self.__oxidizer_weights = []
        for o in oxidizer_species:
            self.__oxidizer_definition.append(o)
        for w in oxidizer_weights:
            self.__oxidizer_weights.append(w)
        self.__oxidizer_string = ",".join([self.__oxidizer_definition[i] + ":" + str(self.__oxidizer_weights[i]) for i in range(len(self.__oxidizer_definition))])
        self.__Config.SetOxidizerDefinition(oxidizer_species, oxidizer_weights)
        self.__Config.ComputeMixFracConstants()
        self.z_i = self.__Config.GetMixtureFractionCoefficients()
        self.c = self.__Config.GetMixtureFractionConstant()
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
        self.__reaction_mechanism = reaction_mechanism
        self.gas = ct.Solution(self.__reaction_mechanism)
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
        if not path.isdir(output_dir_new):
            raise Exception("Provided output path doesn't exist.")
        self.__output_dir = output_dir_new
        self.__PrepareOutputDirectories()
        return
    
    def SetMatlabOutputDir(self, output_dir_new):
        self.__matlab_output_dir = output_dir_new
        self.__PrepareOutputDirectories_Matlab()

    def __PrepareOutputDirectories(self): 
        """Create sub-directories for the different types of flamelet data.
        """  
        if (not path.isdir(self.__output_dir+'/freeflame_data')) and self.__run_freeflames:
            mkdir(self.__output_dir+'/freeflame_data')
        if (not path.isdir(self.__output_dir+'/burnerflame_data')) and self.__run_burnerflames:
            mkdir(self.__output_dir+'/burnerflame_data')
        if (not path.isdir(self.__output_dir+'/equilibrium_data')) and self.__run_equilibrium:
            mkdir(self.__output_dir+'/equilibrium_data')
        if (not path.isdir(self.__output_dir+'/counterflame_data')) and self.__run_counterflames:
            mkdir(self.__output_dir+'/counterflame_data')
        return 
    
    def __PrepareOutputDirectories_Matlab(self):
        if (not path.isdir(self.__matlab_output_dir+'freeflame_data_MATLAB')) and self.__run_freeflames:
            mkdir(self.__matlab_output_dir+'freeflame_data_MATLAB')
        if (not path.isdir(self.__matlab_output_dir+'burnerflame_data_MATLAB')) and self.__run_burnerflames:
            mkdir(self.__matlab_output_dir+'burnerflame_data_MATLAB')
        if (not path.isdir(self.__matlab_output_dir+'equilibrium_data_MATLAB')) and self.__run_equilibrium:
            mkdir(self.__matlab_output_dir+'equilibrium_data_MATLAB')
        if (not path.isdir(self.__matlab_output_dir+'counterflame_data_MATLAB')) and self.__run_counterflames:
            mkdir(self.__matlab_output_dir+'counterflame_data_MATLAB')
        return 

    def AddRandomData(self, flame_solution, mix_status, T_ub, extra_header=""):

        if self.__define_equivalence_ratio:
            folder_header = "phi"
        else:
            folder_header = "mixfrac"

        if not path.isdir(self.__output_dir+'/fuzzy_data/'):
            mkdir(self.__output_dir+'/fuzzy_data/')
        if not path.isdir(self.__output_dir+'/fuzzy_data/'+folder_header+'_'+str(round(mix_status, 6))):
            mkdir(self.__output_dir+'/fuzzy_data/'+folder_header+'_'+str(round(mix_status, 6)))

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

        filepathname = self.__output_dir+'/fuzzy_data/'+folder_header+'_'+str(round(mix_status, 6)) + "/" + fileHeader
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
        flame:ct.FreeFlame = ct.FreeFlame(self.gas, width=1e-2)
        flame.set_refine_criteria(ratio=3, slope=0.04, curve=0.06, prune=0.02)

        # Multi-component diffusion for differential diffusion effects.
        flame.transport_model = self.__transport_model

        # Try to solve the flamelet solution. If solution diverges, move on to next flamelet.
        try:
            flame.solve(loglevel=0, auto=True)
            
            # Computing mass flow rate for later burner flame evaluation
            self.m_dot_free_flame = flame.velocity[0]*flame.density[0]
            
            variables, data_calc = self.__SaveFlameletData(flame, self.gas)

            if self.__run_fuzzy:
                self.AddRandomData(flame, mix_status, T_ub)

            # Generate sub-directory if it's not there.
            if not path.isdir(self.__output_dir+'/freeflame_data/'):
                mkdir(self.__output_dir+'/freeflame_data/')
            if not path.isdir(self.__output_dir+'/freeflame_data/'+folder_header+'_'+str(round(mix_status, 6))):
                mkdir(self.__output_dir+'/freeflame_data/'+folder_header+'_'+str(round(mix_status, 6)))

            if max(flame.grid) < 1.0:
                freeflame_filename = "freeflamelet_"+folder_header+str(round(mix_status,6))+"_Tu"+str(round(T_ub, 4))+".csv"
                filename_plus_folder = self.__output_dir+"/freeflame_data/"+folder_header+'_'+str(round(mix_status, 6)) + "/"+freeflame_filename
                fid = open(filename_plus_folder, 'w+')
                fid.write(variables + "\n")
                csvWriter = csv.writer(fid)
                csvWriter.writerows(data_calc)
                fid.close()

                if self.__translate_to_matlab:
                    if not path.isdir(self.__matlab_output_dir+'/freeflame_data_MATLAB/'+folder_header+'_'+str(round(mix_status, 6))):
                            mkdir(self.__matlab_output_dir+'/freeflame_data_MATLAB/'+folder_header+'_'+str(round(mix_status, 6)))
                    self.__TranslateToMatlabFile(filename_plus_folder,freeflame_filename, self.__matlab_output_dir + "/freeflame_data_MATLAB/"+folder_header+'_'+str(round(mix_status, 6)) + "/")
                self.last_Y_flamelet = flame.Y
                self.last_h_flamelet = flame.enthalpy_mass 
                self.last_T_flamelet = flame.T 

                print("Successfull Freeflame simulation at "+folder_header+": "+str(mix_status)+ " T_u: " +str(T_ub) + " ("+str(i_freeflame+1)+"/"+str(self.__n_flamelets)+")")
            else:
                print("Unsuccessfull Freeflame simulation at "+folder_header+": "+str(mix_status)+ " T_u: " +str(T_ub) + " ("+str(i_freeflame+1)+"/"+str(self.__n_flamelets)+")")
            
        except:
            print("Unsuccessfull Freeflame simulation at "+folder_header+": "+str(mix_status)+ " T_u: " +str(T_ub) + " ("+str(i_freeflame+1)+"/"+str(self.__n_flamelets)+")")
        
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

        burner_flame = ct.BurnerFlame(self.gas, width=18e-3)
        burner_flame.set_refine_criteria(ratio=3.0, slope=0.05, curve=0.1)
        burner_flame.transport_model = self.__transport_model

        for i_burnerflame, m_dot_next in enumerate(m_dot):
            try:
                burner_flame.burner.mdot = m_dot_next
                # Attempt to solve the burner flame simulation
                burner_flame.solve(loglevel=0, auto=True)
            
                # Computing ANN flamelet data
                variables, data_calc = self.__SaveFlameletData(burner_flame, self.gas)

                if self.__run_fuzzy:
                    self.AddRandomData(burner_flame, mix_status, T_burner, "mdot_"+str(round(m_dot_next,4)))
                    
                # Generate sub-directory if it's not there.
                if not path.isdir(self.__output_dir+'/burnerflame_data/'):
                    mkdir(self.__output_dir+'/burnerflame_data/')
                if not path.isdir(self.__output_dir+'/burnerflame_data/'+folder_header+'_'+str(round(mix_status, 6))):
                    mkdir(self.__output_dir+'/burnerflame_data/'+folder_header+'_'+str(round(mix_status, 6)))
                burnerflame_filename = "burnerflamelet_"+folder_header+str(round(mix_status,6))+"_mdot"+str(round(m_dot_next, 4))+".csv"
                filename_plus_folder = self.__output_dir+"/burnerflame_data/"+folder_header+'_'+str(round(mix_status, 6)) + "/"+burnerflame_filename
                fid = open(filename_plus_folder, 'w+')
                fid.write(variables + "\n")
                csvWriter = csv.writer(fid)
                csvWriter.writerows(data_calc)
                fid.close()

                if self.__translate_to_matlab:
                    if not path.isdir(self.__matlab_output_dir+'/burnerflame_data_MATLAB/'+folder_header+'_'+str(round(mix_status, 6))):
                        mkdir(self.__matlab_output_dir+'/burnerflame_data_MATLAB/'+folder_header+'_'+str(round(mix_status, 6)))
                    self.__TranslateToMatlabFile(filename_plus_folder,burnerflame_filename, self.__matlab_output_dir + "/burnerflame_data_MATLAB/"+folder_header+'_'+str(round(mix_status, 6)) + "/")

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
        if not path.isdir(self.__output_dir+"/counterflame_data"):
            mkdir(self.__output_dir+"/counterflame_data")
        fid = open(self.__output_dir+"/counterflame_data/"+counterflame_filename, 'w+')
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
                fid = open(self.__output_dir+"/counterflame_data/"+counterflame_filename, 'w+')
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
        if not path.isdir(self.__output_dir+'/equilibrium_data/'):
                        mkdir(self.__output_dir+'/equilibrium_data/')
        if not path.isdir(self.__output_dir + "/equilibrium_data/" + folder_header+"_"+str(round(mix_status,6))):
            mkdir(self.__output_dir + "/equilibrium_data/" + folder_header+"_"+str(round(mix_status,6)))
        
        if self.__define_equivalence_ratio:
            gas_eq.set_equivalence_ratio(mix_status, self.__fuel_string, self.__oxidizer_string)
        else:
            gas_eq.set_mixture_fraction(mix_status, self.__fuel_string, self.__oxidizer_string)

        gas_eq.TP = max(T_range), ct.one_atm 
        H_max = gas_eq.enthalpy_mass
        # In case of reaction products, set the maximum enthalpy to that of the reactants at the maximum temperature.
        if burnt:
            gas_eq.equilibrate('TP')
            gas_eq.HP = H_max, ct.one_atm
            T_range = np.linspace(min(T_range), gas_eq.T, len(T_range))

        for i, T in enumerate(T_range):
            
            gas_eq.TP = T, ct.one_atm
  
            if i == 0:
                if not path.isdir(self.__output_dir+'/equilibrium_data/'+folder_header+'_'+str(round(mix_status, 6))):
                    mkdir(self.__output_dir+'/equilibrium_data/'+folder_header+'_'+str(round(mix_status, 6)))
                variables, data_calc = self.__SaveFlameletData(gas_eq, self.gas)
                fid = open(self.__output_dir+"/equilibrium_data/"+folder_header+"_"+str(round(mix_status,6))+"/"+ fileHeader +folder_header+"_"+str(round(mix_status,6))+".csv", 'w+')
                fid.write(variables + "\n")
                fid.close()
            else:
                variables, data_calc_2 = self.__SaveFlameletData(gas_eq, self.gas)
                data_calc = np.append(data_calc, data_calc_2, axis=0)

        eq_filename = fileHeader +folder_header+"_"+str(round(mix_status,6))+".csv"
        filename_plus_folder = self.__output_dir+"/equilibrium_data/"+folder_header+"_"+str(round(mix_status,6))+"/"+ eq_filename
        fid = open(filename_plus_folder, 'a+')
        csvWriter = csv.writer(fid)
        csvWriter.writerows(data_calc)
        fid.close()

        if self.__translate_to_matlab:
            if not path.isdir(self.__matlab_output_dir+'/equilibrium_data_MATLAB/'+folder_header+'_'+str(round(mix_status, 6))):
                    mkdir(self.__matlab_output_dir+'/equilibrium_data_MATLAB/'+folder_header+'_'+str(round(mix_status, 6)))
            self.__TranslateToMatlabFile(filename_plus_folder, eq_filename, self.__matlab_output_dir + "/equilibrium_data_MATLAB/"+folder_header+'_'+str(round(mix_status, 6)) + "/")

    def ComputeHardCorner(self, mix_status:float, T_u:float):
        print("Starting interpolation process...")
        gas_eq = ct.Solution(self.__reaction_mechanism)
        

        if self.__define_equivalence_ratio:
            folder_header = "phi"
        else:
            folder_header = "mixfrac"

        fileHeader = "corner_data_"
        if not path.isdir(self.__output_dir+'/equilibrium_data/'):
                        mkdir(self.__output_dir+'/equilibrium_data/')
        if not path.isdir(self.__output_dir + "/equilibrium_data/" + folder_header+"_"+str(round(mix_status,6))):
            mkdir(self.__output_dir + "/equilibrium_data/" + folder_header+"_"+str(round(mix_status,6)))
        
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
                    if not path.isdir(self.__output_dir+'/equilibrium_data/'+folder_header+'_'+str(round(mix_status, 6))):
                        mkdir(self.__output_dir+'/equilibrium_data/'+folder_header+'_'+str(round(mix_status, 6)))
                    variables, data_calc = self.__SaveFlameletData(gas_eq, self.gas)
                    fid = open(self.__output_dir+"/equilibrium_data/"+folder_header+"_"+str(round(mix_status,6))+"/"+ fileHeader +folder_header+"_"+str(round(mix_status,6))+"_"+str(i)+".csv", 'w+')
                    fid.write(variables + "\n")
                    fid.close()
                else:
                    _, data_calc_2 = self.__SaveFlameletData(gas_eq, self.gas)
                    data_calc = np.append(data_calc, data_calc_2, axis=0)

            eq_filename = fileHeader +folder_header+"_"+str(round(mix_status,6))+"_"+str(i)+".csv"
            filename_plus_folder = self.__output_dir+"/equilibrium_data/"+folder_header+"_"+str(round(mix_status,6))+"/"+ eq_filename
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
            
            if not path.isdir(self.__output_dir+'counterflame_data'):
                mkdir(self.__output_dir+'counterflame_data')
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
            net_reaction_rate = flame.net_production_rates[:,np.newaxis]
            neg_reaction_rate =flame.destruction_rates[:,np.newaxis]
            pos_reaction_rate = net_reaction_rate- neg_reaction_rate
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
            variables += ',EnthalpyTot,'
            data_matrix = np.append(data_matrix, np.array([[enthalpy]]), axis=1)
            variables += 'MixtureFraction,'
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
            variables += ',EnthalpyTot,'
            data_matrix = np.append(data_matrix, np.reshape(enthalpy, [len(enthalpy),1]), axis=1)
            variables += 'MixtureFraction,'
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