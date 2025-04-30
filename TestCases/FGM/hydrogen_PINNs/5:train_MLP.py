#!/usr/bin/env python3
###############################################################################################
#       #      _____ __  _____      ____        __        __  ____                   #        #  
#       #     / ___// / / /__ \    / __ \____ _/ /_____ _/  |/  (_)___  ___  _____   #        #  
#       #     \__ \/ / / /__/ /   / / / / __ `/ __/ __ `/ /|_/ / / __ \/ _ \/ ___/   #        #      
#       #    ___/ / /_/ // __/   / /_/ / /_/ / /_/ /_/ / /  / / / / / /  __/ /       #        #  
#       #   /____/\____//____/  /_____/\__,_/\__/\__,_/_/  /_/_/_/ /_/\___/_/        #        #
#       #                                                                            #        #
###############################################################################################

############################### FILE NAME: 5:train_MLP.py #####################################
#=============================================================================================#
# author: Evert Bunschoten                                                                    |
#    :PhD Candidate ,                                                                         |
#    :Flight Power and Propulsion                                                             |
#    :TU Delft,                                                                               |
#    :The Netherlands                                                                         |
#                                                                                             |
#                                                                                             |
# Description:                                                                                |
#  Train physics-informed neural networks for FGM applications of hydrogen-air problems.      |
#                                                                                             |  
# Version: 2.0.0                                                                              |
#                                                                                             |
#=============================================================================================#
from su2dataminer.config import Config_FGM 
from su2dataminer.manifold import TrainMLP_FGM

# Load FlameletAI configuration
Config = Config_FGM("Hydrogen_PINNs.cfg")

# For every output group, train an MLP
for iGroup in range(Config.GetNMLPOutputGroups()):
    Eval = TrainMLP_FGM(Config, iGroup)
    Eval.SetVerbose(1)
    Eval.CommenceTraining()
    Config.UpdateMLPHyperParams(Eval)
    Config.SaveConfig()

    