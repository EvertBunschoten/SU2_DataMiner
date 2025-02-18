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
#---------------------------------------------------------------------------------------------#
# Parse arguments
#---------------------------------------------------------------------------------------------#
parser = argparse.ArgumentParser()
parser.add_argument('--c', dest='config_name', type=str, help='Configuration file name.', default=DefaultProperties.config_name+".cfg")
parser.add_argument('--t', dest='type', type=int, help='Data type to generate: (1:FGM, 2:NICFD)', default=1)
args = parser.parse_args(args=None if sys.argv[1:] else ['--help']) 

if args.type == 1:
    config = Config_FGM(args.config_name)
elif args.type == 2:
    config = Config_NICFD(args.config_name)
else:
    raise Exception("Config type should be 1 or 2")
config.PrintBanner()
