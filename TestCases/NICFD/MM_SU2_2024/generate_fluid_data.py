###############################################################################################
#       #      _____ __  _____      ____        __        __  ____                   #        #
#       #     / ___// / / /__ \    / __ \____ _/ /_____ _/  |/  (_)___  ___  _____   #        #
#       #     \__ \/ / / /__/ /   / / / / __ `/ __/ __ `/ /|_/ / / __ \/ _ \/ ___/   #        #
#       #    ___/ / /_/ // __/   / /_/ / /_/ / /_/ /_/ / /  / / / / / /  __/ /       #        #
#       #   /____/\____//____/  /_____/\__,_/\__/\__,_/_/  /_/_/_/ /_/\___/_/        #        #
#       #                                                                            #        #
###############################################################################################

########################### FILE NAME: generate_fluid_data.py #################################
#=============================================================================================#
# author: Evert Bunschoten                                                                    |
#    :PhD Candidate ,                                                                         |
#    :Flight Power and Propulsion                                                             |
#    :TU Delft,                                                                               |
#    :The Netherlands                                                                         |
#                                                                                             |
#                                                                                             |
# Description:                                                                                |
#  Generate the MM siloxane fluid data set used for physics-informed neural network training  |
#  by Evert Bunschoten during his contribution to the fifth annual SU2 conference in 2024.    |
#                                                                                             |
# Version: 1.0.0                                                                              |
#                                                                                             |
#=============================================================================================#

# Import SU2 DataMiner modules
from Common.DataDrivenConfig import EntropicAIConfig 
from Data_Generation.DataGenerator_NICFD import DataGenerator_CoolProp 

# Load configuration.
Config = EntropicAIConfig("MM_config_SU2_2024.cfg")

D = DataGenerator_CoolProp(Config)

# Define data grid.
D.PreprocessData()

# Visualize data grid prior to data generation.
D.VisualizeDataGrid()

# For every node in the data grid, compute thermodynamic state.
D.ComputeData()

# Visualize entropic data.
D.VisualizeFluidData()

# Save all fluid data in respective files.
D.SaveData()