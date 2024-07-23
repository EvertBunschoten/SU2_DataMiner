import CoolProp.CoolProp as CP
import numpy as np 
from tqdm import tqdm
import csv 
import os 
import matplotlib.pyplot as plt 
from Common.EntropicAIConfig import EntropicAIConfig 
from Common.Properties import DefaultProperties


class DataGenerator_CoolProp:
    """Class for generating fluid data using CoolProp
    """

    __Config:EntropicAIConfig = None 
    __fluid = None 

    # Pressure and temperature limits
    __use_PT:bool = DefaultProperties.use_PT_grid
    __T_min:float = DefaultProperties.T_min
    __T_max:float = DefaultProperties.T_max
    __Np_T:int = DefaultProperties.Np_temp

    __P_min:float = DefaultProperties.P_min
    __P_max:float = DefaultProperties.P_max
    __Np_P:int = DefaultProperties.Np_p

    __T_grid:np.ndarray[float] = None 
    __P_grid:np.ndarray[float] = None 
    

    # Output train and test fractions.
    __train_fraction:float = DefaultProperties.train_fraction
    __test_fraction:float = DefaultProperties.test_fraction 

    # Output data properties.
    __output_file_header:str = DefaultProperties.output_file_header 
    __output_dir:str 

    # Density and static energy limits
    __rho_min:float = None 
    __rho_max:float = None 
    __e_min:float = None 
    __e_max:float = None 
    __e_grid:np.ndarray[float] = None
    __rho_grid:np.ndarray[float] = None 

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

        # Load configuration and set default properties.
        self.__Config = Config_in 
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
        self.__P_min, self.__P_max = P_bounds[0], P_bounds[1]
        self.__Np_P = self.__Config.GetNpPressure()

        self.__T_min, self.__T_max = T_bounds[0], T_bounds[1]
        self.__Np_T = self.__Config.GetNpTemp()

        self.__output_file_header = self.__Config.GetConcatenationFileHeader()
        self.__output_dir = self.__Config.GetOutputDir() 

        self.__train_fraction = self.__Config.GetTrainFraction()
        self.__test_fraction = self.__Config.GetTestFraction()

        return 
    
    def PreprocessData(self):
        """Generate density and static energy grid at which to evaluate fluid properties.
        """
        P_range = np.linspace(self.__P_min, self.__P_max, self.__Np_P)
        T_range = np.linspace(self.__T_min, self.__T_max, self.__Np_T)
        self.__P_grid, self.__T_grid = np.meshgrid(P_range, T_range)

        if not self.__use_PT:
            P_dataset = self.__P_grid.flatten()
            T_dataset = self.__T_grid.flatten()
            self.__rho_min = 1e32
            self.__rho_max = -1e32
            self.__e_min = 1e32
            self.__e_max = -1e32 
            for p,T in zip(P_dataset, T_dataset):
                try:
                    self.__fluid.update(CP.PT_INPUTS, p, T)
                    rho = self.__fluid.rhomass()
                    e = self.__fluid.umass()
                    self.__rho_max = max(rho, self.__rho_max)
                    self.__rho_min = min(rho, self.__rho_min)
                    self.__e_max = max(e, self.__e_max)
                    self.__e_min = min(e, self.__e_min)
                except:
                    pass 
            rho_range = (self.__rho_min - self.__rho_max)* (np.cos(np.linspace(0, 0.5*np.pi, self.__Np_P))) + self.__rho_max
            e_range = np.linspace(self.__e_min, self.__e_max, self.__Np_T)
            self.__rho_grid, self.__e_grid = np.meshgrid(rho_range, e_range)
        return 
    
    def VisualizeDataGrid(self):
        """Visualize query points at which fluid data are evaluated.
        """
        if self.__use_PT:
            fig = plt.figure(figsize=[10,10])
            ax = plt.axes()
            ax.plot(self.__P_grid.flatten(), self.__T_grid.flatten(), 'k.')
            ax.set_xlabel(r"Pressure $(p)[Pa]",fontsize=20)
            ax.set_ylabel(r"Temperature $(T)[K]",fontsize=20)
            ax.tick_params(which='both',labelsize=18)
            ax.grid()
        else:
            fig = plt.figure(figsize=[10,10])
            ax = plt.axes()
            ax.plot(self.__rho_grid.flatten(), self.__e_grid.flatten(), 'k.')
            ax.set_xlabel(r"Density $(\rho)[kg m^{-3}]",fontsize=20)
            ax.set_ylabel(r"Internal energy $(e)[J kg^{-1}]",fontsize=20)
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
        """Get minimum and maximum pressure.

        :return: list with minimum and maximum fluid pressure.
        :rtype: list[float]
        """
        return [self.__P_lower, self.__P_upper]
    
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
        self.__output_file_header = header 
        return 
    
    def GetConcatenationFileHeader(self):
        """Get fluid data output file header.

        :return: output file header.
        :rtype: str
        """
        return self.__output_file_header 
    

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
    

    def ComputeFluidData(self):
        """Evaluate fluid properties on the density-energy grid or pressure-temperature grid.
        """

        # Initiate empty fluid property arrays.
        self.__s_fluid = np.zeros([self.__Np_P, self.__Np_T])
        self.__dsde_rho_fluid = np.zeros([self.__Np_P, self.__Np_T])
        self.__dsdrho_e_fluid = np.zeros([self.__Np_P, self.__Np_T])
        self.__d2sde2_fluid = np.zeros([self.__Np_P, self.__Np_T])
        self.__d2sdrho2_fluid = np.zeros([self.__Np_P, self.__Np_T])
        self.__d2sdedrho_fluid = np.zeros([self.__Np_P, self.__Np_T])
        self.__P_fluid = np.zeros([self.__Np_P, self.__Np_T])
        self.__T_fluid = np.zeros([self.__Np_P, self.__Np_T])
        self.__c2_fluid = np.zeros([self.__Np_P, self.__Np_T])
        self.__rho_fluid = np.zeros([self.__Np_P, self.__Np_T])
        self.__e_fluid = np.zeros([self.__Np_P, self.__Np_T])

        self.__success_locations = np.ones([self.__Np_P, self.__Np_T],dtype=bool)
        
        # Loop over density-based or pressure-based grid.
        for i in tqdm(range(self.__Np_P)):
            for j in range(self.__Np_T):
                try:
                    if self.__use_PT:
                        self.__fluid.update(CP.PT_INPUTS, self.__P_grid[i,j], self.__T_grid[i,j])
                    else:
                        self.__fluid.update(CP.DmassUmass_INPUTS, self.__rho_grid[j,i], self.__e_grid[j,i])

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
    
    def SaveFluidData(self):
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
    