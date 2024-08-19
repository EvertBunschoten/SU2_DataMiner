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
#  Class for generating fluid data for NI-CFD data mining operations.                         |                                                               
#                                                                                             |  
# Version: 1.0.0                                                                              |
#                                                                                             |
#=============================================================================================#

#---------------------------------------------------------------------------------------------#
# Importing general packages
#---------------------------------------------------------------------------------------------#
import CoolProp.CoolProp as CP
import numpy as np 
from tqdm import tqdm
import csv 
import matplotlib.pyplot as plt 
np.random.seed(2)

#---------------------------------------------------------------------------------------------#
# Importing DataMiner classes and functions
#---------------------------------------------------------------------------------------------#
from Common.DataDrivenConfig import EntropicAIConfig
from Common.Properties import DefaultSettings_NICFD
from Data_Generation.DataGenerator_Base import DataGenerator_Base

class DataGenerator_CoolProp(DataGenerator_Base):
    """Class for generating fluid data using CoolProp
    """

    __fluid = None 

    # Pressure and temperature limits
    __use_PT:bool = DefaultSettings_NICFD.use_PT_grid
    __T_min:float = DefaultSettings_NICFD.T_min
    __T_max:float = DefaultSettings_NICFD.T_max
    __Np_Y:int = DefaultSettings_NICFD.Np_temp

    __P_min:float = DefaultSettings_NICFD.P_min
    __P_max:float = DefaultSettings_NICFD.P_max
    __Np_X:int = DefaultSettings_NICFD.Np_p

    # Density and static energy limits
    __rho_min:float = DefaultSettings_NICFD.Rho_min
    __rho_max:float = DefaultSettings_NICFD.Rho_max
    __e_min:float = DefaultSettings_NICFD.Energy_min 
    __e_max:float = DefaultSettings_NICFD.Energy_max 
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

    def __init__(self, Config_in:EntropicAIConfig=None):
        DataGenerator_Base.__init__(self, Config_in=Config_in)

        if Config_in is None:
            print("Initializing NICFD data generator with default settings.")
            self._Config = EntropicAIConfig()
        else:
            # Load configuration and set default properties.
            self.__use_PT = self._Config.GetPTGrid()
            if len(self._Config.GetFluidNames()) > 1:
                fluid_names = self._Config.GetFluidNames()
                CAS_1 = CP.get_fluid_param_string(fluid_names[0], "CAS")
                CAS_2 = CP.get_fluid_param_string(fluid_names[1], "CAS")
                CP.apply_simple_mixing_rule(CAS_1, CAS_2,'linear')
            self.__fluid = CP.AbstractState("HEOS", self._Config.GetFluid())
            if len(self._Config.GetFluidNames()) > 1:
                mole_fractions = self._Config.GetMoleFractions()
                self.__fluid.set_mole_fractions(mole_fractions)
            self.__use_PT = self._Config.GetPTGrid()
            P_bounds = self._Config.GetPressureBounds()
            T_bounds = self._Config.GetTemperatureBounds()
            rho_bounds = self._Config.GetDensityBounds()
            e_bounds = self._Config.GetEnergyBounds()

            self.__P_min, self.__P_max = P_bounds[0], P_bounds[1]
            self.__rho_min, self.__rho_max = rho_bounds[0], rho_bounds[1]
            self.__Np_X = self._Config.GetNpPressure()

            self.__T_min, self.__T_max = T_bounds[0], T_bounds[1]
            self.__e_min, self.__e_max = e_bounds[0], e_bounds[1]
            self.__Np_Y = self._Config.GetNpTemp()

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
    
    def GetTemperatureBounds(self):
        """Get fluid temperature limits.

        :return: list with minimum and maximum temperature.
        :rtype: list[float]
        """
        return [self.__T_lower, self.__T_upper]
    
    def SetNpDensity(self, Np_density:int=DefaultSettings_NICFD.Np_p):
        self.SetNpPressure(Np_P=Np_density)
        return 

    def GetNpDensity(self):
        return self.GetNpPressure()
    
    def SetDensityBounds(self, Density_lower:float=DefaultSettings_NICFD.Rho_min, Density_upper:float=DefaultSettings_NICFD.Rho_max):
        self.__rho_min = Density_lower
        self.__rho_max = Density_upper
        return 
    
    def SetNpEnergy(self, Np_energy:int=DefaultSettings_NICFD.Np_temp):
        self.SetNpTemp(Np_Temp=Np_energy)
        return 
    
    def GetNpEnergy(self):
        return self.GetNpTemp()
    
    def SetEnergyBounds(self, Energy_lower:float=DefaultSettings_NICFD.Energy_min, Energy_upper:float=DefaultSettings_NICFD.Energy_max):
        self.__e_min = Energy_lower
        self.__e_max = Energy_upper
        return 
    
    
    def SetNpTemp(self, Np_Temp:int=DefaultSettings_NICFD.Np_temp):
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
    

    
    def SetNpPressure(self, Np_P:int=DefaultSettings_NICFD.Np_p):
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

        # Define output files for all, train, test, and validation data.
        full_file = self.GetOutputDir() + "/" + self.GetConcatenationFileHeader() + "_full.csv"
        train_file = self.GetOutputDir() + "/" + self.GetConcatenationFileHeader() + "_train.csv"
        test_file = self.GetOutputDir() + "/" + self.GetConcatenationFileHeader() + "_test.csv"
        val_file = self.GetOutputDir() + "/" + self.GetConcatenationFileHeader() + "_val.csv"

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
        Np_train = int(self.GetTrainFraction()*Np_full)
        Np_test = int(self.GetTestFraction()*Np_full)

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
    
