import numpy as np 
import tkinter as tk
from tkinter.filedialog import askopenfilenames
import os 

from Common.DataDrivenConfig import FlameletAIConfig,EntropicAIConfig
from Common.Properties import DefaultSettings_FGM
from Data_Processing.DataPlotter_Base import DataPlotter_Base

class DataPlotter_FGM(DataPlotter_Base):

    _Config:FlameletAIConfig = None 

    __data_dir:str = None 
    __plot_freeflames:bool = DefaultSettings_FGM.include_freeflames
    __plot_burnerflames:bool = DefaultSettings_FGM.include_burnerflames
    __plot_equilibrium:bool = DefaultSettings_FGM.include_equilibrium
    __manual_select:bool = True 

    __color_freeflames:str = 'r'
    __color_burnerflames:str = 'm'
    __color_equilibrium:str = 'b'

    __freeflame_displayname = r"Adiabatic flame data"
    __burnerflame_displayname = r"Burner-stabilized data"
    __equilibrium_displayname = r"Chemical equilibrium data"

    __mix_status:list[float] = []

    _plot_label_default_x:str=r"Progress Variable $(\mathcal{Y})[-]$"
    _plot_label_default_y:str=r"Total Enthalpy $(h)[J kg^{-1}]$"
    _plot_label_default_z:str=r"Mixture Fraction $(Z)[-]$"

    _label_map = { DefaultSettings_FGM.name_pv : r"Progress Variable $(\mathcal{Y})[-]$",\
                   DefaultSettings_FGM.name_enth : r"Total Enthalpy $(h)[J kg^{-1}]$",\
                   DefaultSettings_FGM.name_mixfrac : r"Mixture Fraction $(Z)[-]$",\
                  "Temperature" : r"Temperature $(T)[K]$",\
                  "ViscosityDyn" : r"Dynamic Viscosity $(\mu)[kg m^{-1}s^{-2}]$",\
                  "Cp" : r"Specific heat $(c_p)[J kg^{-1} K^{-1}]$",\
                  "MolarWeightMix" : r"Mean Molar Weight $(W_M)[kg kmol^{-1}]$",\
                  "ProdRateTot_PV" : r"PV Source Term $(\rho\dot{\omega}_{\mathcal{Y}})[kg m^{-3}s^{-1}]$",\
                  "Beta_ProgVar" : r"PV Preferential Diffusion Term $(\beta_\mathcal{Y})[-]$",\
                  "Beta_Enth_Thermal" : r"Specific Heat Preferential Diffusion Term $(\beta_{h,1})[J kg^{-1} K^{-1}]$",\
                  "Beta_Enth" : r"Enthalpy Prefertial Diffusion Term $(\beta_{h,2})[J kg^{-1}]$",\
                  "Beta_MixFrac" : r"Mixture Fraction Preferential Diffusion Term $(\beta_Z)[-]$"}
    
    def __init__(self, Config_in:FlameletAIConfig=None):
        DataPlotter_Base.__init__(self,Config_in)
        if Config_in is None:
            self._Config = FlameletAIConfig()

        self.__data_dir = self._Config.GetOutputDir()

        return 

    def ManualSelection(self, input:bool=False):
        """Select flamelets to plot manually.

        :param input: select flamelets manually(True) or all flamelets within 
        :type input: bool
        """
        self.__manual_select = input 
        return 
    
    def SetFlameletDataDir(self, input:str):
        """Set the data directory from which to read flamelet data.

        :param input: folder from which to read flamelet data.
        :type input: str
        :raises Exception: if specified directory doesn't exist.
        """
        if not os.path.isdir(input):
            raise Exception("Provided data directory does not exist.")
        self.__data_dir = input 
        return 
    
    def PlotFreeflames(self, input:bool=DefaultSettings_FGM.include_freeflames):
        """Plot data under freeflame_data directory in the flamelet data directory.

        :param input: plot adiabatic free-flame data.
        :type input: bool
        """
        self.__plot_freeflames = input 
        return 
    
    def PlotBurnerflames(self, input:bool=DefaultSettings_FGM.include_burnerflames):
        """Plot data under burnerflame_data directory in the flamelet data directory.

        :param input: plot burner-stabilized data.
        :type input: bool
        """
        self.__plot_burnerflames = input 
        return
    
    def PlotEquilibrium(self, input:bool=DefaultSettings_FGM.include_equilibrium):
        """Plot data under equilibrium_data directory in the flamelet data directory.

        :param input: plot chemical equilibrium data.
        :type input: bool
        """
        self.__plot_equilibrium = input
        return 
    
    def SetMixtureStatus(self, mixture_status:list[float]):
        """Set the mixture status value for which to plot flamelet data.

        :param mixture_status: mixture status values (equivalence ratio or mixture fraction) for which to plot flamelet data.
        :type mixture_status: list[float]
        :raises Exception: if the mixture status value is negative.
        """
        for z in mixture_status:
            if z < 0:
                raise Exception("Mixture status should be positive.")
        self.__mix_status = []
        for z in mixture_status:
            self.__mix_status.append(z)
        return 
    
    def SetProgressVariableDefinition(self, pv_species:list[str]=DefaultSettings_FGM.pv_species, pv_weights:list[float]=DefaultSettings_FGM.pv_weights):
        self._Config.SetProgressVariableDefinition(pv_species, pv_weights)
        return 
    
    def Plot2D(self, y_variable: str, x_variable: str=DefaultSettings_FGM.name_pv, show:bool=True):
        return super().Plot2D(x_variable, y_variable, show)
    
    def Plot3D(self, z_variable:str, y_variable: str=DefaultSettings_FGM.name_enth, x_variable: str=DefaultSettings_FGM.name_pv, show:bool=True):
        return super().Plot3D(x_variable, y_variable, z_variable, show)
        
    def _PlotBody(self, plot_variables: list[str]):
        # if len(self.__mix_status) == 0:
        #     raise Exception("No mixture status values provided.")
        self.__GetFileNames()
        plot_3D = super()._PlotBody(plot_variables)

        plot_data_freeflame = []
        if self.__plot_freeflames:
            plot_label=self.__freeflame_displayname
            for f in self.freeflame_files:
                plot_data = self.__GeneratePlotData(f, plot_variables)
                if plot_3D:
                    self._ax.plot3D(plot_data[:,0],plot_data[:,1],plot_data[:,2],color=self.__color_freeflames, label=plot_label, linewidth=2)
                else:
                    self._ax.plot(plot_data[:,0],plot_data[:,1],color=self.__color_freeflames, label=plot_label, linewidth=2)
                plot_label=""
                plot_data_freeflame.append(plot_data)
        
        plot_data_burnerflame = []
        if self.__plot_burnerflames:
            plot_label=self.__burnerflame_displayname
            for f in self.burnerflame_files:
                plot_data = self.__GeneratePlotData(f, plot_variables)
                if plot_3D:
                    self._ax.plot3D(plot_data[:,0],plot_data[:,1],plot_data[:,2],color=self.__color_burnerflames, label=plot_label, linewidth=2)
                else:
                    self._ax.plot(plot_data[:,0],plot_data[:,1],color=self.__color_burnerflames, label=plot_label, linewidth=2)
                plot_label=""
                plot_data_burnerflame.append(plot_data)
        
        plot_data_eq = []
        if self.__plot_equilibrium:
            plot_label=self.__equilibrium_displayname
            for f in self.equilibrium_files:
                plot_data=self.__GeneratePlotData(f, plot_variables)
                if plot_3D:
                    self._ax.plot3D(plot_data[:,0],plot_data[:,1],plot_data[:,2],color=self.__color_equilibrium, label=plot_label, linewidth=2)
                else:
                    self._ax.plot(plot_data[:,0],plot_data[:,1],color=self.__color_equilibrium, label=plot_label, linewidth=2)
                plot_label=""
                plot_data_eq.append(plot_data)

        return [plot_data_freeflame, plot_data_burnerflame, plot_data_eq]
    
    
    def __GetFileNames(self):
        """Collect the list of flamelet data files of which to plot the data.
        """
        if self._Config.GetMixtureStatus():
            header = "mixfrac_"
        else:
            header = "phi_"

        tk.Tk().withdraw()
        if self.__plot_freeflames:
            self.freeflame_files = []
            freeflame_dir = self.__data_dir + "/freeflame_data/"
            if self.__manual_select and len(self.__mix_status) == 0:
                filenames = askopenfilenames(initialdir=freeflame_dir, title="Choose freeflame files to plot")
                for file in filenames:
                    self.freeflame_files.append(file)
            else:
                for i in self.__mix_status:
                    if self.__manual_select:
                        filenames = askopenfilenames(initialdir=freeflame_dir+ header + str(round(i, 6)), title="Choose freeflame files to plot")
                        for file in filenames:
                            self.freeflame_files.append(file)
                    else:
                        filenames = next(os.walk(freeflame_dir + header + str(round(i, 6))), (None, None, []))[2]
                        filenames.sort()
                        for file in filenames:
                            self.freeflame_files.append(freeflame_dir + header + str(round(i, 6)) + "/" +file)

        if self.__plot_burnerflames:
            self.burnerflame_files = []
            burnerflame_dir = self.__data_dir + "/burnerflame_data/"
            if self.__manual_select and len(self.__mix_status) == 0:
                filenames = askopenfilenames(initialdir=burnerflame_dir, title="Choose burnerflame files to plot")
                for file in filenames:
                    self.burnerflame_files.append(file)
            else:
                for i in self.__mix_status:
                    if self.__manual_select:
                        filenames = askopenfilenames(initialdir=burnerflame_dir+ header + str(round(i, 6)), title="Choose burnerflame files to plot")
                        for file in filenames:
                            self.burnerflame_files.append(file)
                    else:
                        filenames = next(os.walk(burnerflame_dir + header + str(round(i, 6))), (None, None, []))[2]
                        filenames.sort()
                        for file in filenames:
                            self.burnerflame_files.append(burnerflame_dir + header + str(round(i, 6)) + "/" +file)

        if self.__plot_equilibrium:
            self.equilibrium_files = []
            equilibrium_dir = self.__data_dir + "/equilibrium_data/"
            if self.__manual_select and len(self.__mix_status) == 0:
                filenames = askopenfilenames(initialdir=equilibrium_dir, title="Choose equilibrium files to plot")
                for file in filenames:
                    self.equilibrium_files.append(file)
            else:
                for i in self.__mix_status:
                    if self.__manual_select:
                        filenames = askopenfilenames(initialdir=equilibrium_dir+ header + str(round(i, 6)), title="Choose equilibrium files to plot")
                        for file in filenames:
                            self.equilibrium_files.append(file)
                    else:
                        filenames = next(os.walk(equilibrium_dir + header + str(round(i, 6))), (None, None, []))[2]
                        for file in filenames:
                            self.equilibrium_files.append(equilibrium_dir + header + str(round(i, 6)) + "/" +file)
        return
    
    def __GeneratePlotData(self, filepathname:str, plot_variables:list[str]):
        """Read specific variables from flamelet data file.

        :param filepathname: file name and path to flamelet data file.
        :type filepathname: str
        :param plot_variables: list of plot variables to read from file.
        :type plot_variables: list[str]
        :return: array with flamelet data read from file.
        :rtype: np.ndarray
        """
        with open(filepathname, "r") as fid:
                variables = fid.readline().strip().split(',')
        flamelet_data = np.loadtxt(filepathname, delimiter=',',skiprows=1)

        plot_data = self.__ExtractPlotData(variables, flamelet_data, plot_variables)

        return plot_data 
    
    def __ExtractPlotData(self, flamelet_variables:list[str], flamelet_data_array:np.ndarray[float], variables_to_plot:list[str]):
        """Apply operations on flamelet data depending on the plot variables.

        :param flamelet_variables: variables to plot.
        :type flamelet_variables: list[str]
        :param flamelet_data_array: array of loaded flamelet data.
        :type flamelet_data_array: np.ndarray
        :param variables_to_plot: list of variables to extract from flamelet data.
        :type variables_to_plot: list[str]
        :return: array of data to be plotted.
        :rtype: np.ndarray[float]
        """
        plot_data_out = np.zeros([np.shape(flamelet_data_array)[0], len(variables_to_plot)])
        for iVar, var in enumerate(variables_to_plot):
            if var == DefaultSettings_FGM.name_pv:
                plot_data = self._Config.ComputeProgressVariable(variables=flamelet_variables, flamelet_data=flamelet_data_array)
            elif var == "NOx":
                plot_data = np.zeros(np.shape(flamelet_data_array)[0])
                for s in self._Config.gas.species_names:
                    if ("N" in s) and ("O" in s) and not (("H" in s) or ("C" in s)):
                        plot_data += flamelet_data_array[:, flamelet_variables.index("Y-"+s)]            
            else:
                if var == "ProdRateTot_PV":
                    plot_data = self._Config.ComputeProgressVariable_Source(variables=flamelet_variables, flamelet_data=flamelet_data_array)
                elif "Beta_" in var:
                    beta_pv, beta_enth_1, beta_enth_2, beta_mixfrac = self._Config.ComputeBetaTerms(flamelet_variables, flamelet_data_array)
                    if var == "Beta_ProgVar":
                        plot_data = beta_pv 
                    elif var == "Beta_Enth_Thermal":
                        plot_data = beta_enth_1
                    elif var == "Beta_Enth":
                        plot_data = beta_enth_2
                    else:
                        plot_data = beta_mixfrac
                else:
                    if "ProdRateTot_" in var:
                        Sp_name = var[len("ProdRateTot_"):]
                        plot_data = self.__ComputeReactionRate(flamelet_variables, flamelet_data_array, Sp_name)
                    else:
                        idx_var = flamelet_variables.index(var)
                        plot_data = flamelet_data_array[:, idx_var]
            
            plot_data_out[:, iVar] = plot_data 
        return plot_data_out
    
    def __ComputeReactionRate(self, variables:list[str], flamelet_data:np.ndarray[float], Sp_name:str):
        """Compute the reaction rate of a specified specie.

        :param variables: flamelet data variables.
        :type variables: list[str]
        :param flamelet_data: flamelet data array.
        :type flamelet_data: np.ndarray
        :param Sp_name: name of the specie for which to compute the total reaction rate.
        :type Sp_name: str
        :raises Exception: if specie is not present in current reaction mechanism.
        :return: species reaction rate throughout the flamelet solution.
        :rtype: np.ndarray[float]
        """
        if Sp_name == "NOx":
            RR = np.zeros(np.shape(flamelet_data)[0])
            for s in self._Config.gas.species_names:
                if ("N" in s) and ("O" in s) and not ("C" in s) and not ("H" in s):
                    RR += flamelet_data[:, variables.index("Y_dot_net-"+s)]
        else:          
            if Sp_name not in self._Config.gas.species_names:
                raise Exception("Specie "+Sp_name+" not present in reaction mechanism.")
            RR = flamelet_data[:, variables.index("Y_dot_net-"+Sp_name)]
        return RR
    
class DataPlotter_NICFD(DataPlotter_Base):

    _Config:EntropicAIConfig=None

    def __init__(self, Config_in:EntropicAIConfig=None):
        DataPlotter_Base.__init__(self, Config_in)

        if Config_in is None:
            self._Config = EntropicAIConfig()
        return 
    
    
    def _PlotBody(self, plot_variables: list[str]):
        plot_3D = super()._PlotBody(plot_variables)
        
        full_filename = self._Config.GetOutputDir()+"/"+self._Config.GetConcatenationFileHeader()+"_full.csv"
        with open(full_filename, 'r') as fid:
            vars_in_data = fid.readline().strip().split(',')
        D_fluid = np.loadtxt(full_filename,delimiter=',',skiprows=1)

        for var in plot_variables:
            if var not in vars_in_data:
                raise Exception(var + " not present in fluid data.")
        
        plot_data_x = D_fluid[:, vars_in_data.index(plot_variables[0])]
        plot_data_y = D_fluid[:, vars_in_data.index(plot_variables[1])]
        if plot_3D:
            plot_data_z = D_fluid[:, vars_in_data.index(plot_variables[2])]
        
        if plot_3D:
            self._ax.plot3D(plot_data_x,plot_data_y,plot_data_z,'k.')
        else:
            self._ax.plot(plot_data_x, plot_data_y, 'k.')
        return 
    