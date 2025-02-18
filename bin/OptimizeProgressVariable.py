#!/usr/bin/env python3
###############################################################################################
#       #      _____ __  _____      ____        __        __  ____                   #        #  
#       #     / ___// / / /__ \    / __ \____ _/ /_____ _/  |/  (_)___  ___  _____   #        #  
#       #     \__ \/ / / /__/ /   / / / / __ `/ __/ __ `/ /|_/ / / __ \/ _ \/ ___/   #        #      
#       #    ___/ / /_/ // __/   / /_/ / /_/ / /_/ /_/ / /  / / / / / /  __/ /       #        #  
#       #   /____/\____//____/  /_____/\__,_/\__/\__,_/_/  /_/_/_/ /_/\___/_/        #        #
#       #                                                                            #        #
###############################################################################################

########################## FILE NAME: OptimizeProgressVariable.py #############################
#=============================================================================================#
# author: Evert Bunschoten                                                                    |
#    :PhD Candidate ,                                                                         |
#    :Flight Power and Propulsion                                                             |
#    :TU Delft,                                                                               |
#    :The Netherlands                                                                         |
#                                                                                             |
#                                                                                             |
# Description:                                                                                |
#  Optimize the progress variable definition and weights.                                     |
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
from Data_Processing.OptimizeProgressVariable import PVOptimizer

#---------------------------------------------------------------------------------------------#
# Parse arguments
#---------------------------------------------------------------------------------------------#
parser = argparse.ArgumentParser()
parser.add_argument('--c', dest='config_name', type=str, help='FlameletAI configuration file name.', default=DefaultSettings_FGM.config_name+".cfg")
parser.add_argument('--np', dest='Np', type=int, help='Number of processors to use for flamelet data generation.', default=1)
args = parser.parse_args(args=None if sys.argv[1:] else ['--help']) 

if args.Np <= 0:
    raise Exception("Number of processors should be positive.")

#---------------------------------------------------------------------------------------------#
# Load FlameletAI configuration
#---------------------------------------------------------------------------------------------#
Config = Config_FGM(args.config_name)
Config.PrintBanner()

#---------------------------------------------------------------------------------------------#
# Initialize progress variable optimizer
#---------------------------------------------------------------------------------------------#
PVO = PVOptimizer(Config)
PVO.SetNWorkers(args.Np)
PVO.OptimizePV()

#---------------------------------------------------------------------------------------------#
# Update progress variable definition in the FlameletAI configuration
#---------------------------------------------------------------------------------------------#
Config.SetProgressVariableDefinition(pv_species=PVO.GetOptimizedSpecies(), \
                                     pv_weights=PVO.GetOptimizedWeights())

Config.PrintBanner()
Config.SaveConfig()
