###############################################################################################
#       #      _____ __  _____      ____        __        __  ____                   #        #  
#       #     / ___// / / /__ \    / __ \____ _/ /_____ _/  |/  (_)___  ___  _____   #        #  
#       #     \__ \/ / / /__/ /   / / / / __ `/ __/ __ `/ /|_/ / / __ \/ _ \/ ___/   #        #      
#       #    ___/ / /_/ // __/   / /_/ / /_/ / /_/ /_/ / /  / / / / / /  __/ /       #        #  
#       #   /____/\____//____/  /_____/\__,_/\__/\__,_/_/  /_/_/_/ /_/\___/_/        #        #
#       #                                                                            #        #
###############################################################################################

################################# FILE NAME: train_MLP.py #####################################
#=============================================================================================#
# author: Evert Bunschoten                                                                    |
#    :PhD Candidate ,                                                                         |
#    :Flight Power and Propulsion                                                             |
#    :TU Delft,                                                                               |
#    :The Netherlands                                                                         |
#                                                                                             |
#                                                                                             |
# Description:                                                                                |
#  Train a multi-layer perceptron for evaluation of thermodynamic properties through physics- |
#  informed learning to be used for SU2 simulations.                                          |                                                             
# Version: 1.0.0                                                                              |
#                                                                                             |
#=============================================================================================#

# Import EntropicAI configuration and MLP training module.
from Manifold_Generation.MLP.Trainers_NICFD.Trainers import EvaluateArchitecture_NICFD
from Common.DataDrivenConfig import EntropicAIConfig 

# Load test case configuration.
Config = EntropicAIConfig("MM_test.cfg")

# Define MLP trainer class. This class first trains the MLP to predict the entropy based on 
# density and internal energy. Subsequently, it initiates a physics-informed training approach
# where the MLP is trained to predict the temperature, pressure, and speed of sound based on
# the entropic equation of state.
Eval = EvaluateArchitecture_NICFD(Config)

# Set activation function. For the physics-informed approach, the "exponential" function is 
# recommended as it saves computational resourses during SU2 simulations.
Eval.SetActivationFunction("exponential")

# Set hidden layer architecture through a list of integers. In this case, only a single hidden
# layer with 30 perceptrons is used.
Eval.SetHiddenLayers([30])

# The step size during training is adjusted through an exponential decay algorithm. 
# The user can specify the initial learning rate exponent and learning rate decay
# parameter. The default parameters work well, although the optimum set of parameters
# depends on the data set and architecture! 

# Set initial learning rate exponent (base 10).
Eval.SetAlphaExpo(-1.8261e+00)

# Set learning rate decay parameter.
Eval.SetLRDecay(+9.8787e-01)

# Training is performed in batches. The user can set the batch size exponent (base 2).
# The optimum choice depends on the data set.
Eval.SetBatchExpo(6)

# Set output verbose level. A value of 1 displays training information per epoch in the terminal. 
# A value of 0 ommits all outputs.
Eval.SetVerbose(1)

# Train for 1000 epochs both direct and physics-informed.
Eval.SetNEpochs(1000)

# Start the training process. During the training process, convergence history and other relevant
# information are saved in the main output directory as specified in the configuration. 
Eval.CommenceTraining()