#!/usr/bin/env python3
###############################################################################################
#       #      _____ __  _____      ____        __        __  ____                   #        #  
#       #     / ___// / / /__ \    / __ \____ _/ /_____ _/  |/  (_)___  ___  _____   #        #  
#       #     \__ \/ / / /__/ /   / / / / __ `/ __/ __ `/ /|_/ / / __ \/ _ \/ ___/   #        #      
#       #    ___/ / /_/ // __/   / /_/ / /_/ / /_/ /_/ / /  / / / / / /  __/ /       #        #  
#       #   /____/\____//____/  /_____/\__,_/\__/\__,_/_/  /_/_/_/ /_/\___/_/        #        #
#       #                                                                            #        #
###############################################################################################

############################## FILE NAME: GenerateConfig.py ###################################
#=============================================================================================#
# author: Evert Bunschoten                                                                    |
#    :PhD Candidate ,                                                                         |
#    :Flight Power and Propulsion                                                             |
#    :TU Delft,                                                                               |
#    :The Netherlands                                                                         |
#                                                                                             |
#                                                                                             |
# Description:                                                                                |
#  Initiate a configuration object through the interactive terminal for the SU2 DataMiner     |
#  workflow.                                                                                  |
#                                                                                             |  
# Version: 2.0.0                                                                              |
#                                                                                             |
#=============================================================================================#

import pyfiglet
from Common.DataDrivenConfig import Config_FGM, Config_NICFD

from manual_configuration import *
customfig = pyfiglet.Figlet(font="slant")
print(customfig.renderText("SU2 DataMiner"))

printhbar()
print("Welcome to the SU2 DataMiner interactive configuration menu.")
print("Generate a configuration for a data-driven fluid manifold through terminal inputs.")
printhbar()

correct_config_type = False
while not correct_config_type:
    configuration_type = input("Type of SU2 DataMiner configuration (1:Flamelet,2:NICFD): ")

    if configuration_type != "1" and configuration_type != "2":
        print("Please insert \"1\" or \"2\"")
    else:
        correct_config_type = True 

Config_in:Config_FGM = None 
if configuration_type=="1":
    ManualFlameletConfiguration()
elif configuration_type == "2":
    ManualNICFDConfiguration()
