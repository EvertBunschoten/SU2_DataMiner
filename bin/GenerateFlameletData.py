#!/usr/bin/env python3
###############################################################################################
#       #      _____ __  _____      ____        __        __  ____                   #        #  
#       #     / ___// / / /__ \    / __ \____ _/ /_____ _/  |/  (_)___  ___  _____   #        #  
#       #     \__ \/ / / /__/ /   / / / / __ `/ __/ __ `/ /|_/ / / __ \/ _ \/ ___/   #        #      
#       #    ___/ / /_/ // __/   / /_/ / /_/ / /_/ /_/ / /  / / / / / /  __/ /       #        #  
#       #   /____/\____//____/  /_____/\__,_/\__/\__,_/_/  /_/_/_/ /_/\___/_/        #        #
#       #                                                                            #        #
###############################################################################################

############################ FILE NAME: GenerateFlameletData.py ###############################
#=============================================================================================#
# author: Evert Bunschoten                                                                    |
#    :PhD Candidate ,                                                                         |
#    :Flight Power and Propulsion                                                             |
#    :TU Delft,                                                                               |
#    :The Netherlands                                                                         |
#                                                                                             |
#                                                                                             |
# Description:                                                                                |
#  Generate flamelet data as per settings in the FlameletAI configuration                     |
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
from Common.DataDrivenConfig import FlameletAIConfig 
from Data_Generation.DataGenerator_FGM import ComputeFlameletData, ComputeBoundaryData

#---------------------------------------------------------------------------------------------#
# Parse arguments
#---------------------------------------------------------------------------------------------#
parser = argparse.ArgumentParser()
parser.add_argument('--c', dest='config_name', type=str, help='FlameletAI configuration file name.', default=DefaultSettings_FGM.config_name+".cfg")
parser.add_argument('--np', dest='Np', type=int, help='Number of processors to use for flamelet data generation.', default=1)
parser.add_argument('--b', dest='boundary_data', action='store_true', help='Generate chemical equilibrium boundary data over the full mixture range (0.0 <= Z <= 1.0).')
args = parser.parse_args(args=None if sys.argv[1:] else ['--help']) 

if args.Np <= 0:
    raise Exception("Number of processors should be positive.")

#---------------------------------------------------------------------------------------------#
# Load FlameletAI configuration
#---------------------------------------------------------------------------------------------#
Config = FlameletAIConfig(args.config_name)
Config.PrintBanner()

#---------------------------------------------------------------------------------------------#
# Generate flamelet manifold data
#---------------------------------------------------------------------------------------------#
if args.Np > 1:
    run_parallel = True 
else:
    run_parallel = False

ComputeFlameletData(Config, run_parallel=run_parallel, N_processors=args.Np)

#---------------------------------------------------------------------------------------------#
# Load equilibrium boundary data
#---------------------------------------------------------------------------------------------#
if args.boundary_data:
    ComputeBoundaryData(Config, run_parallel=run_parallel, N_processors=args.Np)
