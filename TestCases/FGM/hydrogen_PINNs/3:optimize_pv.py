#!/usr/bin/env python3
###############################################################################################
#       #      _____ __  _____      ____        __        __  ____                   #        #  
#       #     / ___// / / /__ \    / __ \____ _/ /_____ _/  |/  (_)___  ___  _____   #        #  
#       #     \__ \/ / / /__/ /   / / / / __ `/ __/ __ `/ /|_/ / / __ \/ _ \/ ___/   #        #      
#       #    ___/ / /_/ // __/   / /_/ / /_/ / /_/ /_/ / /  / / / / / /  __/ /       #        #  
#       #   /____/\____//____/  /_____/\__,_/\__/\__,_/_/  /_/_/_/ /_/\___/_/        #        #
#       #                                                                            #        #
###############################################################################################

############################## FILE NAME: 3:optimize_pv.py ####################################
#=============================================================================================#
# author: Evert Bunschoten                                                                    |
#    :PhD Candidate ,                                                                         |
#    :Flight Power and Propulsion                                                             |
#    :TU Delft,                                                                               |
#    :The Netherlands                                                                         |
#                                                                                             |
#                                                                                             |
# Description:                                                                                |
#  Optimize the progress variable definition for hydrogen flamelets.                          |
#                                                                                             |  
# Version: 1.0.0                                                                              |
#                                                                                             |
#=============================================================================================#

from Common.DataDrivenConfig import FlameletAIConfig 
from Data_Processing.OptimizeProgressVariable import PVOptimizer

# Load FlameletAI configuration
Config = FlameletAIConfig("Hydrogen_PINNs.cfg")

# Initiate progress variable definition optimizer
PVO = PVOptimizer(Config)

# Commence pv optimization
PVO.OptimizePV()

# Update progress variable definition in configuration
Config.SetProgressVariableDefinition(PVO.GetOptimizedSpecies(), PVO.GetOptimizedWeights())
Config.PrintBanner()
Config.SaveConfig()
