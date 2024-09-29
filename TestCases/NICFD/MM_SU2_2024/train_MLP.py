###############################################################################################
#       #      _____ __  _____      ____        __        __  ____                   #        #
#       #     / ___// / / /__ \    / __ \____ _/ /_____ _/  |/  (_)___  ___  _____   #        #
#       #     \__ \/ / / /__/ /   / / / / __ `/ __/ __ `/ /|_/ / / __ \/ _ \/ ___/   #        #
#       #    ___/ / /_/ // __/   / /_/ / /_/ / /_/ /_/ / /  / / / / / /  __/ /       #        #
#       #   /____/\____//____/  /_____/\__,_/\__/\__,_/_/  /_/_/_/ /_/\___/_/        #        #
#       #                                                                            #        #
###############################################################################################

################################ FILE NAME: train_MLP.py ######################################
#=============================================================================================#
# author: Evert Bunschoten                                                                    |
#    :PhD Candidate ,                                                                         |
#    :Flight Power and Propulsion                                                             |
#    :TU Delft,                                                                               |
#    :The Netherlands                                                                         |
#                                                                                             |
#                                                                                             |
# Description:                                                                                |
#  Perform the physics-informed learning on MM fluid data by Evert Bunschoten during his      |
# contribution to the fifth annual SU2 conference in 2024.                                    |
#                                                                                             |
# Version: 1.0.0                                                                              |
#                                                                                             |
#=============================================================================================#

# Import SU2 DataMiner modules
from Manifold_Generation.MLP.Trainers_NICFD.Trainers import EvaluateArchitecture_NICFD
from Common.DataDrivenConfig import EntropicAIConfig 

# Load test case configuration.
Config = EntropicAIConfig("MM_config_SU2_2024.cfg")

# Initate trainer module.
Eval = EvaluateArchitecture_NICFD(Config)

# Train for 1000 epochs
Eval.SetNEpochs(2000)
Eval.SetActivationFunction("exponential")

# Display training progress in terminal.
Eval.SetVerbose(1)

# Train on CPU
Eval.SetTrainHardware("CPU",0)

# Start the training process.
Eval.CommenceTraining()
