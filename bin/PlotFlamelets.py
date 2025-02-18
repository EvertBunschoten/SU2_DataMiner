#!/usr/bin/env python3
###############################################################################################
#       #      _____ __  _____      ____        __        __  ____                   #        #  
#       #     / ___// / / /__ \    / __ \____ _/ /_____ _/  |/  (_)___  ___  _____   #        #  
#       #     \__ \/ / / /__/ /   / / / / __ `/ __/ __ `/ /|_/ / / __ \/ _ \/ ___/   #        #      
#       #    ___/ / /_/ // __/   / /_/ / /_/ / /_/ /_/ / /  / / / / / /  __/ /       #        #  
#       #   /____/\____//____/  /_____/\__,_/\__/\__,_/_/  /_/_/_/ /_/\___/_/        #        #
#       #                                                                            #        #
###############################################################################################

############################### FILE NAME: PlotFlamelets.py ###################################
#=============================================================================================#
# author: Evert Bunschoten                                                                    |
#    :PhD Candidate ,                                                                         |
#    :Flight Power and Propulsion                                                             |
#    :TU Delft,                                                                               |
#    :The Netherlands                                                                         |
#                                                                                             |
#                                                                                             |
# Description:                                                                                |
#  Plot flamelet data trends.                                                                 |
#                                                                                             |  
# Version: 1.0.0                                                                              |
#                                                                                             |
#=============================================================================================#

import argparse 
import sys

#---------------------------------------------------------------------------------------------#
# Importing DataMiner classes and functions
#---------------------------------------------------------------------------------------------#
from Common.Properties import DefaultSettings_FGM
from Common.DataDrivenConfig import Config_FGM 
from Data_Processing.DataPlotters import DataPlotter_FGM

#---------------------------------------------------------------------------------------------#
# Parse arguments
#---------------------------------------------------------------------------------------------#
parser = argparse.ArgumentParser()
parser.add_argument('--c', dest='config_name', type=str, help='FlameletAI configuration file name.', default=DefaultSettings_FGM.config_name+".cfg")
parser.add_argument('--m', dest='manual_select', action='store_true', help='Manually select flamelet data files to plot')
parser.add_argument('--x', dest='x_var', type=str, help="X-variable (default: ProgressVariable)", default=DefaultSettings_FGM.name_pv)
parser.add_argument('--y', dest='y_var', type=str, help="Y-variable (default: EnthalpyTot)", default=DefaultSettings_FGM.name_enth)
parser.add_argument('--z', dest='z_var', type=str, help="Z-variable", default=None)
parser.add_argument('--Mix', nargs='+',dest='mixture_status', help="Mixture status values for which to plot flamelet data.", default=[])
parser.add_argument('--save',action='store_true',dest='save_images', help="Save generated images in flamelet data folder.")

args = parser.parse_args(args=None if sys.argv[1:] else ['--help']) 

#---------------------------------------------------------------------------------------------#
# Load FlameletAI configuration
#---------------------------------------------------------------------------------------------#
Config = Config_FGM(args.config_name)
Config.PrintBanner()


#---------------------------------------------------------------------------------------------#
# Initiate flamelet data plotter
#---------------------------------------------------------------------------------------------#
Plotter = DataPlotter_FGM(Config)
Plotter.ManualSelection(args.manual_select)
Plotter.SaveImages(args.save_images)
mix_status = []
if len(args.mixture_status) > 0:
    mix_status = [float(s) for s in args.mixture_status]
Plotter.SetMixtureStatus(mix_status)

if args.z_var == None:
    Plotter.Plot2D(x_variable=args.x_var, y_variable=args.y_var)
else:
    Plotter.Plot3D(x_variable=args.x_var, y_variable=args.y_var, z_variable=args.z_var)
