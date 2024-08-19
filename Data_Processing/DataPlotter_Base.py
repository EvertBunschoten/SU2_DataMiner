###############################################################################################
#       #      _____ __  _____      ____        __        __  ____                   #        #
#       #     / ___// / / /__ \    / __ \____ _/ /_____ _/  |/  (_)___  ___  _____   #        #
#       #     \__ \/ / / /__/ /   / / / / __ `/ __/ __ `/ /|_/ / / __ \/ _ \/ ___/   #        #
#       #    ___/ / /_/ // __/   / /_/ / /_/ / /_/ /_/ / /  / / / / / /  __/ /       #        #
#       #   /____/\____//____/  /_____/\__,_/\__/\__,_/_/  /_/_/_/ /_/\___/_/        #        #
#       #                                                                            #        #
###############################################################################################

############################# FILE NAME: DataPlotter_Base.py ##################################
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


import numpy as np 
import matplotlib.pyplot as plt 
import os 

from Common.Config_base import Config 

class DataPlotter_Base:
    _Config:Config = None 
    __plot_vars:list[str] = None 

    __plot_title:str = ""
    __plot_label_custom_x:str = ""
    _custom_plot_label_x_set:bool = False 
    __plot_label_custom_y:str = ""
    _custom_plot_label_y_set:bool = False 
    __plot_label_custom_y:str = ""
    _custom_plot_label_z_set:bool = False 
    
    __save_images:bool = False
    __fig_format:str = "png"
    __fig_window:plt.figure = None 
    _ax:plt.axes = None

    def __init__(self, Config_in:Config=None):
        if Config_in is None:
            self._Config = Config()
        else:
            self._Config = Config_in

        return
    
    def SetPlotTitle(self, title_in:str):
        self.__plot_title = title_in
        return
    
    def SetPlotLabelX(self, input:str):
        """Set a custom x-axis label.

        :param input: custom x-axis label for the current plot.
        :type input: str
        """
        self.__plot_label_custom_x = input 
        self._custom_plot_label_x_set = True 
        return 
    
    def SetPlotLabelY(self, input:str):
        """Set a custom y-axis label.

        :param input: custom y-axis label for the current plot.
        :type input: str
        """
        self.__plot_label_custom_y = input 
        self._custom_plot_label_y_set = True 
        return
    
    def SetPlotLabelZ(self, input:str):
        """Set a custom z-axis label.

        :param input: custom z-axis label for the current plot.
        :type input: str
        """
        self.__plot_label_custom_z = input 
        self._custom_plot_label_z_set = True 
        return
    
    def SaveImages(self, save:bool):
        """Save generated images.

        :param save: save generated images (True) or only show (False)
        :type save: bool
        """
        self.__save_images=save
        return

    def SetFigFormat(self, fig_format:str="png"):
        self.__fig_format = fig_format
        return
    
    def _PrepareOutputDir(self):
        if not os.path.isdir(self._Config.GetOutputDir()+"/Plots"):
            os.mkdir(self._Config.GetOutputDir()+"/Plots")
        return
    
    def SetOutputDir(self, output_dir:str):
        self._Config.SetOutputDir(output_dir)
        return 
    

    
    def _Initiate2DPlot(self):
        self.__fig_window= plt.figure(figsize=[10,10])
        self._ax = plt.axes()
        return 
    
    def _Initiate3DPlot(self):
        self.__fig_window= plt.figure(figsize=[10,10])
        self._ax = plt.axes(projection='3d')
        return 
    
    def _FinalizePlot(self, fig_title:str):
        if self._custom_plot_label_y_set:
            self._ax.set_ylabel(self.__plot_label_custom_y, fontsize=20)
        if self._custom_plot_label_x_set:
            self._ax.set_xlabel(self.__plot_label_custom_x, fontsize=20)
        if self._custom_plot_label_z_set:
            self._ax.set_zlabel(self.__plot_label_custom_z, fontsize=20)
        self._ax.tick_params(which='both',labelsize=18)
        self._ax.set_title(self.__plot_title, fontsize=20)
        self._ax.legend(fontsize=20,loc='upper center', bbox_to_anchor=(0.5, -0.12),
          ncol=2, fancybox=True, shadow=True)
        self._ax.grid()
        if self.__save_images:
            self.__fig_window.savefig(self._Config.GetOutputDir()+"/Plots/"+fig_title+"."+self.__fig_format, format=self.__fig_format, bbox_inches='tight')
        plt.show() 
        return 
    
    def _PlotBody(self, plot_variables:list[str]):
        plot_3D = (len(plot_variables) == 3)
        return plot_3D
    
    
    def Plot3D(self, x_variable:str, y_variable:str, z_variable:str):
        if self.__save_images:
            self._PrepareOutputDir()
        self._Initiate3DPlot()
        self._PlotBody([x_variable, y_variable, z_variable])
        self._FinalizePlot("_".join((x_variable,y_variable,z_variable)) + "_3D")
        return 

    def Plot2D(self, x_variable:str, y_variable:str):
        if self.__save_images:
            self._PrepareOutputDir()
        self._Initiate2DPlot()
        self._PlotBody([x_variable, y_variable])
        self._FinalizePlot("_".join((x_variable,y_variable)) + "_2D")
        return
