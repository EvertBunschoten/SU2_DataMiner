#!/usr/bin/env python3
###############################################################################################
#       #      _____ __  _____      ____        __        __  ____                   #        #  
#       #     / ___// / / /__ \    / __ \____ _/ /_____ _/  |/  (_)___  ___  _____   #        #  
#       #     \__ \/ / / /__/ /   / / / / __ `/ __/ __ `/ /|_/ / / __ \/ _ \/ ___/   #        #      
#       #    ___/ / /_/ // __/   / /_/ / /_/ / /_/ /_/ / /  / / / / / /  __/ /       #        #  
#       #   /____/\____//____/  /_____/\__,_/\__/\__,_/_/  /_/_/_/ /_/\___/_/        #        #
#       #                                                                            #        #
###############################################################################################

############################# FILE NAME: GenerateFluidData.py #################################
#=============================================================================================#
# author: Evert Bunschoten                                                                    |
#    :PhD Candidate ,                                                                         |
#    :Flight Power and Propulsion                                                             |
#    :TU Delft,                                                                               |
#    :The Netherlands                                                                         |
#                                                                                             |
#                                                                                             |
# Description:                                                                                |
#  Generate flamelet data as per settings in the SU2 DataMiner configuration.                 |
#                                                                                             |  
# Version: 1.0.0                                                                              |
#                                                                                             |
#=============================================================================================#

import argparse 
import sys 

#---------------------------------------------------------------------------------------------#
# Importing DataMiner classes and functions
#---------------------------------------------------------------------------------------------#
from Common.Properties import *
from Common.DataDrivenConfig import *
from Data_Generation.DataGenerator_FGM import *
from Data_Generation.DataGenerator_NICFD import *
#---------------------------------------------------------------------------------------------#
# Parse arguments
#---------------------------------------------------------------------------------------------#
parser = argparse.ArgumentParser()
parser.add_argument('--c', dest='config_name', type=str, help='Configuration file name.', default=DefaultProperties.config_name+".cfg")
parser.add_argument('--np', dest='Np', type=int, help='Number of processors to use for flamelet data generation.', default=1)
parser.add_argument('--b', dest='boundary_data', action='store_true', help='Generate chemical equilibrium boundary data over the full mixture range (0.0 <= Z <= 1.0).')
parser.add_argument('--t', dest='type', type=int, help='Data type to generate: (1:FGM, 2:NICFD)', default=1)
args = parser.parse_args(args=None if sys.argv[1:] else ['--help']) 

if args.Np <= 0:
    raise Exception("Number of processors should be positive.")

if args.type == 1:
    #---------------------------------------------------------------------------------------------#
    # Load FlameletAI configuration
    #---------------------------------------------------------------------------------------------#
    try:
        config = FlameletAIConfig(args.config_name)
        config.PrintBanner()
    except:
        raise Exception("Improper configuration file for FlameletAI configuration.")
    #---------------------------------------------------------------------------------------------#
    # Generate flamelet manifold data
    #---------------------------------------------------------------------------------------------#
    if args.Np > 1:
        run_parallel = True 
    else:
        run_parallel = False

    ComputeFlameletData(config, run_parallel=run_parallel, N_processors=args.Np)

    #---------------------------------------------------------------------------------------------#
    # Load equilibrium boundary data
    #---------------------------------------------------------------------------------------------#
    if args.boundary_data:
        ComputeBoundaryData(config, run_parallel=run_parallel, N_processors=args.Np)

elif args.type == 2:

    #---------------------------------------------------------------------------------------------#
    # Load EntropicAI
    #---------------------------------------------------------------------------------------------#
    try:
        config = EntropicAIConfig(args.config_name)
        config.PrintBanner()
    except:
        raise Exception("Improper configuration file for EntropicAI configuration.")
    
    #---------------------------------------------------------------------------------------------#
    # Initiate NICFD data generator
    #---------------------------------------------------------------------------------------------#
    D = DataGenerator_CoolProp(Config_in=config)
    D.PreprocessData()

    #---------------------------------------------------------------------------------------------#
    # Compute and save fluid data
    #---------------------------------------------------------------------------------------------#
    D.ComputeData()
    D.SaveData()

else:
    raise Exception("Data type should be FGM (1) or NCIFD (2).")