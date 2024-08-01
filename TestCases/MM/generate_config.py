###############################################################################################
#       #      _____ __  _____      ____        __        __  ____                   #        #  
#       #     / ___// / / /__ \    / __ \____ _/ /_____ _/  |/  (_)___  ___  _____   #        #  
#       #     \__ \/ / / /__/ /   / / / / __ `/ __/ __ `/ /|_/ / / __ \/ _ \/ ___/   #        #      
#       #    ___/ / /_/ // __/   / /_/ / /_/ / /_/ /_/ / /  / / / / / /  __/ /       #        #  
#       #   /____/\____//____/  /_____/\__,_/\__/\__,_/_/  /_/_/_/ /_/\___/_/        #        #
#       #                                                                            #        #
###############################################################################################

############################## FILE NAME: generate_config.py ##################################
#=============================================================================================#
# author: Evert Bunschoten                                                                    |
#    :PhD Candidate ,                                                                         |
#    :Flight Power and Propulsion                                                             |
#    :TU Delft,                                                                               |
#    :The Netherlands                                                                         |
#                                                                                             |
#                                                                                             |
# Description:                                                                                |
#  Configuration generator for generating a data-driven manifold for MM.                      |
#                                                                                             |  
# Version: 1.0.0                                                                              |
#                                                                                             |
#=============================================================================================#

# Test case for MM fluid: generate manifold generation configuration 

# Import EntropicAI configuration module
from Common.DataDrivenConfig import EntropicAIConfig 
import os 

# Initiating empty configuration
Config = EntropicAIConfig()

# Define fluid name. Fluid is selected from RefProp library. Mixtures are not yet supported!
Config.SetFluid("MM")

# Set bounds and discretization for density grid. In this case the density grid contains
# 700 values between 0.5 kg/m3 and 300 kg/m3 (cosine distribution)
Config.SetDensityBounds(Rho_lower=0.5, Rho_upper=300)
Config.SetNpDensity(Np_rho=700)

# Set bounds and discretization for energy grid. In this case the energy grid contains
# 600 values between 300 kJ/kg and 550 kJ/kg (linear distribution)
Config.SetEnergyBounds(E_lower=3e5, E_upper=5.5e5)
Config.SetNpEnergy(Np_Energy=600)

# Define data grid in the density-energy space rather than pressure-temperature.
Config.UsePTGrid(False)

# Set file header for fluid data. Four files are generated: full, train, 
# validation, and test data. 
Config.SetConcatenationFileHeader("fluid_data")

# The fraction of the fluid data used for hyper-parameter training.
Config.SetTrainFraction(0.8)

# The fraction of the fluid data used for accuracy evaluation upon completion of 
# the training process. The fraction of data apart from training and testing is
# used for monitoring the convergence of the training process (validation).
Config.SetTestFraction(0.1)

# Set main output directory where fluid data are stored and manifolds are saved.
Config.SetOutputDir(os.getcwd())

# Print configuration information to terminal.
Config.PrintBanner()

# Save current configuration.
Config.SaveConfig("MM_test")
 