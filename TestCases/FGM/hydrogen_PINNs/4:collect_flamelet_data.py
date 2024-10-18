#!/usr/bin/env python3
###############################################################################################
#       #      _____ __  _____      ____        __        __  ____                   #        #  
#       #     / ___// / / /__ \    / __ \____ _/ /_____ _/  |/  (_)___  ___  _____   #        #  
#       #     \__ \/ / / /__/ /   / / / / __ `/ __/ __ `/ /|_/ / / __ \/ _ \/ ___/   #        #      
#       #    ___/ / /_/ // __/   / /_/ / /_/ / /_/ /_/ / /  / / / / / /  __/ /       #        #  
#       #   /____/\____//____/  /_____/\__,_/\__/\__,_/_/  /_/_/_/ /_/\___/_/        #        #
#       #                                                                            #        #
###############################################################################################

######################### FILE NAME: 4:collect_flamelet_data.py ###############################
#=============================================================================================#
# author: Evert Bunschoten                                                                    |
#    :PhD Candidate ,                                                                         |
#    :Flight Power and Propulsion                                                             |
#    :TU Delft,                                                                               |
#    :The Netherlands                                                                         |
#                                                                                             |
#                                                                                             |
# Description:                                                                                |
#  Concatenate flamelet data and group outputs for training physics-informed neural networks. |
#                                                                                             |  
# Version: 1.0.0                                                                              |
#                                                                                             |
#=============================================================================================#

from Common.DataDrivenConfig import FlameletAIConfig 
from Data_Processing.collectFlameletData import FlameletConcatenator, GroupOutputs

# Load FlameletAI configuration
Config = FlameletAIConfig("Hydrogen_PINNs.cfg")

# Initiate flamelet data concatenator
Concat = FlameletConcatenator(Config)

# Reduce number of points per flamelet to speed up training
Concat.SetNFlameletNodes(2**Config.GetBatchExpo())

# Include H2O reaction rates and heat release in flamelet data set 
Concat.SetAuxilarySpecies(["H2O"])
Concat.SetLookUpVars(["Heat_Release"])

# Read and concatenate flamelet data
Concat.ConcatenateFlameletData()

# Read and concatenate equilibrium boundary data
Concat.CollectBoundaryData()

# Compute affinity between flamelet data trends and group into MLP outputs.
Grouper = GroupOutputs(Config)

# Only consider net reaction rate for H2O
Grouper.ExcludeVariables(["Y_dot_neg-H2O","Y_dot_pos-H2O"])
Grouper.EvaluateGroups()
Grouper.UpdateConfig()

