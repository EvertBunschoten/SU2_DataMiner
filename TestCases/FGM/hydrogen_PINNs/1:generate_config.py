#!/usr/bin/env python3
###############################################################################################
#       #      _____ __  _____      ____        __        __  ____                   #        #  
#       #     / ___// / / /__ \    / __ \____ _/ /_____ _/  |/  (_)___  ___  _____   #        #  
#       #     \__ \/ / / /__/ /   / / / / __ `/ __/ __ `/ /|_/ / / __ \/ _ \/ ___/   #        #      
#       #    ___/ / /_/ // __/   / /_/ / /_/ / /_/ /_/ / /  / / / / / /  __/ /       #        #  
#       #   /____/\____//____/  /_____/\__,_/\__/\__,_/_/  /_/_/_/ /_/\___/_/        #        #
#       #                                                                            #        #
###############################################################################################

############################ FILE NAME: 1:generate_config.py ##################################
#=============================================================================================#
# author: Evert Bunschoten                                                                    |
#    :PhD Candidate ,                                                                         |
#    :Flight Power and Propulsion                                                             |
#    :TU Delft,                                                                               |
#    :The Netherlands                                                                         |
#                                                                                             |
#                                                                                             |
# Description:                                                                                |
#  Initiate the FlameletAI configuration for training physics-informed neural networks for    |
#  FGM applications.                                                                          |
#                                                                                             |  
# Version: 1.0.0                                                                              |
#                                                                                             |
#=============================================================================================#

from su2dataminer.config import Config_FGM
import os 

# Manifold bounds:

T_unb_lower = 300.0 # Reactant lower temperature
T_unb_upper = 800.0 # Reactant upper temperature
phi_lower = 0.25    # Lower equivalence ratio
phi_upper = 20.0    # Upper equivalence ratio

Np_T = 10   # Number of reactant temperature divisions
Np_mix = 10 # Number of mixture divisions

# Initiate Config_FGM
Config = Config_FGM()

Config.SetConfigName("Hydrogen_PINNs")

# Hydrogen-air flamelets using simple h2o2 mechanism.
Config.SetFuelDefinition(fuel_species=["H2"],fuel_weights=[1.0])
Config.SetReactionMechanism('h2o2.yaml')

# Set manifold bounds
Config.SetMixtureBounds(phi_lower, phi_upper)
Config.SetNpMix(Np_mix)
Config.SetUnbTempBounds(T_unb_lower,T_unb_upper)
Config.SetNpTemp(Np_T)

# Enable preferential diffusion through selecting the "multicomponent" transport model.
Config.SetTransportModel('multicomponent')

Config.SetConcatenationFileHeader("MLP_data")

# Preparing flamelet output directory.
flamelet_data_dir = os.getcwd() + "/flamelet_data/"
if not os.path.isdir(flamelet_data_dir):
    os.mkdir(flamelet_data_dir)
Config.SetOutputDir(flamelet_data_dir) 
 
# Display configuration info in terminal.
Config.PrintBanner()
Config.SaveConfig()
