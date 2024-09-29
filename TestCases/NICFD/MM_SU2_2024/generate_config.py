###############################################################################################
#       #      _____ __  _____      ____        __        __  ____                   #        #
#       #     / ___// / / /__ \    / __ \____ _/ /_____ _/  |/  (_)___  ___  _____   #        #
#       #     \__ \/ / / /__/ /   / / / / __ `/ __/ __ `/ /|_/ / / __ \/ _ \/ ___/   #        #
#       #    ___/ / /_/ // __/   / /_/ / /_/ / /_/ /_/ / /  / / / / / /  __/ /       #        #
#       #   /____/\____//____/  /_____/\__,_/\__/\__,_/_/  /_/_/_/ /_/\___/_/        #        #
#       #                                                                            #        #
###############################################################################################

############################# FILE NAME: generate_config.py ###################################
#=============================================================================================#
# author: Evert Bunschoten                                                                    |
#    :PhD Candidate ,                                                                         |
#    :Flight Power and Propulsion                                                             |
#    :TU Delft,                                                                               |
#    :The Netherlands                                                                         |
#                                                                                             |
#                                                                                             |
# Description:                                                                                |
#  Generate the SU2 DataMiner configuration class used by Evert Bunschoten for his            |
#  contribution to the fifth annual SU2 conference in 2024.                                   |
#                                                                                             |
# Version: 1.0.0                                                                              |
#                                                                                             |
#=============================================================================================#
import os 

# Import EntropicAI configuration module
from Common.DataDrivenConfig import EntropicAIConfig 


Config = EntropicAIConfig()

# Using a density-energy based fluid grid for MM siloxane using the RefProp Coolprop wrapper.
Config.SetFluid("MM")
Config.SetEquationOfState("REFPROP")
Config.UsePTGrid(False)

# Set bounds and discretization for pressure grid
Config.SetDensityBounds(Rho_lower=0.5, Rho_upper=300.0)
Config.SetNpDensity(Np_rho=700)

Config.SetEnergyBounds(E_lower=2.5e5, E_upper=5.5e5)
Config.SetNpEnergy(Np_Energy=600)

# Apply physics-informed training on temperature, pressure, and squred speed of sound.
Config.SetStateVars(["T", "p", "c2"])

# Hyper-parameters for network training. These were manually selected.
Config.SetHiddenLayerArchitecture([40])
Config.SetAlphaExpo(-1.8261e+00)
Config.SetLRDecay(+9.8787e-01)
Config.SetBatchExpo(6)

Config.SetOutputDir(os.getcwd())

Config.PrintBanner()

Config.SetConfigName("MM_config_SU2_2024")

Config.SaveConfig()
 