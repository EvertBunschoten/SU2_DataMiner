###############################################################################################
#       #      _____ __  _____      ____        __        __  ____                   #        #  
#       #     / ___// / / /__ \    / __ \____ _/ /_____ _/  |/  (_)___  ___  _____   #        #  
#       #     \__ \/ / / /__/ /   / / / / __ `/ __/ __ `/ /|_/ / / __ \/ _ \/ ___/   #        #      
#       #    ___/ / /_/ // __/   / /_/ / /_/ / /_/ /_/ / /  / / / / / /  __/ /       #        #  
#       #   /____/\____//____/  /_____/\__,_/\__/\__,_/_/  /_/_/_/ /_/\___/_/        #        #
#       #                                                                            #        #
###############################################################################################

############################ FILE NAME: generate_fluid_data.py ################################
#=============================================================================================#
# author: Evert Bunschoten                                                                    |
#    :PhD Candidate ,                                                                         |
#    :Flight Power and Propulsion                                                             |
#    :TU Delft,                                                                               |
#    :The Netherlands                                                                         |
#                                                                                             |
#                                                                                             |
# Description:                                                                                |
#  Routine for fluid data generation for the MM test case.                                    |  
# Version: 1.0.0                                                                              |
#                                                                                             |
#=============================================================================================#

# Import EntropicAI configuration and data generation module.
from Common.DataDrivenConfig import EntropicAIConfig 
from Data_Generation.DataGenerator_NICFD import DataGenerator_CoolProp 

# Load configuration.
Config = EntropicAIConfig("MM_test.cfg")

# Initiate data-generator object.
D = DataGenerator_CoolProp(Config)

# Define density-energy grid used for the evaluation of fluid properties.
D.PreprocessData()

# Optional: visualize data grid prior to data generation.
D.VisualizeDataGrid()

# For every node in the data grid, compute thermodynamic state.
D.ComputeData()

# Optional: visualize thermodynamic data: temperature, pressure, and speed of sound.
D.VisualizeFluidData()

# Save all fluid data in respective files.
D.SaveData()

# By default, the options from the configuration are used in the fluid data generation
# process. However, it's possible to overwrite these, allowing for the generation of 
# different fluid data files.
# For example, one could generate a data set of fluid data with a different resolution
# compared to the resolution in the Config and save it under a different name.

# D.SetNpDensity(400)
# D.SetNpEnergy(200)
# D.ComputeData()
# D.SetConcatenationFileHeader("fluid_data_reduced")
# D.SaveData()