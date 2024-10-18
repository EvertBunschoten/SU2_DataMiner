#!/usr/bin/env python3
###############################################################################################
#       #      _____ __  _____      ____        __        __  ____                   #        #  
#       #     / ___// / / /__ \    / __ \____ _/ /_____ _/  |/  (_)___  ___  _____   #        #  
#       #     \__ \/ / / /__/ /   / / / / __ `/ __/ __ `/ /|_/ / / __ \/ _ \/ ___/   #        #      
#       #    ___/ / /_/ // __/   / /_/ / /_/ / /_/ /_/ / /  / / / / / /  __/ /       #        #  
#       #   /____/\____//____/  /_____/\__,_/\__/\__,_/_/  /_/_/_/ /_/\___/_/        #        #
#       #                                                                            #        #
###############################################################################################

######################### FILE NAME: 2:generate_flamelet_data.py ##############################
#=============================================================================================#
# author: Evert Bunschoten                                                                    |
#    :PhD Candidate ,                                                                         |
#    :Flight Power and Propulsion                                                             |
#    :TU Delft,                                                                               |
#    :The Netherlands                                                                         |
#                                                                                             |
#                                                                                             |
# Description:                                                                                |
#  Generate flamelet data for training physics-informed neural networks.                      |
#                                                                                             |  
# Version: 1.0.0                                                                              |
#                                                                                             |
#=============================================================================================#
from Common.DataDrivenConfig import FlameletAIConfig 
from Data_Generation.DataGenerator_FGM import ComputeFlameletData, ComputeBoundaryData

# Load FlameletAI configuration
Config = FlameletAIConfig("Hydrogen_PINNs.cfg")

# Distribute flamelet data generation process over 20 cores.
ComputeFlameletData(Config, run_parallel=True, N_processors=20)

# Compute boundary data for PINN training
ComputeBoundaryData(Config)