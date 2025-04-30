#!/usr/bin/env python3
###############################################################################################
#       #      _____ __  _____      ____        __        __  ____                   #        #  
#       #     / ___// / / /__ \    / __ \____ _/ /_____ _/  |/  (_)___  ___  _____   #        #  
#       #     \__ \/ / / /__/ /   / / / / __ `/ __/ __ `/ /|_/ / / __ \/ _ \/ ___/   #        #      
#       #    ___/ / /_/ // __/   / /_/ / /_/ / /_/ /_/ / /  / / / / / /  __/ /       #        #  
#       #   /____/\____//____/  /_____/\__,_/\__/\__,_/_/  /_/_/_/ /_/\___/_/        #        #
#       #                                                                            #        #
###############################################################################################

######################### FILE NAME: mixturefraction_computation.py ###########################
#=============================================================================================#
# author: Evert Bunschoten                                                                    |
#    :PhD Candidate ,                                                                         |
#    :Flight Power and Propulsion                                                             |
#    :TU Delft,                                                                               |
#    :The Netherlands                                                                         |
#                                                                                             |
#                                                                                             |
# Description:                                                                                |
#  Regression test checking the computation of the mixture fraction coefficients.             |
#  FlameletAI computes the mixture fraction coefficients based on Bilgers definition of the   |
#  mixture fraction and are used in the computation of the preferential diffusion scalars.    |
#  These coefficients are correct if sum z_i*Y + z_c = Z, where Z is the mixture fraction     |
#  , and z_i and z_c are the mixture fraction coefficients and constant respectively.         |
#  This regression test compares the mixture fraction according to Cantera and to FlameletAI  |
#                                                                                             |  
# Version: 2.0.0                                                                              |
#                                                                                             |
#=============================================================================================#

import cantera as ct 
import numpy as np
from su2dataminer.config import Config_FGM 
np.random.seed(0)

# Reaction mechanism, temperature, pressure, and equivalence ratio.
T = 300
p = ct.one_atm
phi = 0.5 

def VerifyMixtureFraction(fuel_species:list[str], fuel_weights:list[float], reaction_mechanism:str, fid):
    # Initiate empty FlameletAI.
    Config = Config_FGM()
    Config.SetFuelDefinition(fuel_species, fuel_weights)
    Config.SetReactionMechanism(reaction_mechanism)

    # Compute and extract mixture fraction coefficients and constant.
    Config.ComputeMixFracConstants()
    z_sp, z_c = Config.GetMixtureFractionCoefficients(), Config.GetMixtureFractionConstant()

    # Initiate Cantera solution.
    gas = ct.Solution(reaction_mechanism)
    gas.TP=T, p 

    # Extract mixture fraction for unburnt reactants according to Cantera.
    fuel_string, oxidizer_string = Config.GetFuelString(), Config.GetOxidizerString()

    gas.set_equivalence_ratio(phi, fuel_string, oxidizer_string)
    mixture_fraction_cantera_unb = gas.mixture_fraction(fuel_string, oxidizer_string)
    Y_unb = gas.Y 

    # Comptue mixture fraction for unburnt reactants according to FlameletAI.
    mixture_fraction_config_unb = sum(z_sp * Y_unb) + z_c 

    # Extract mixture fraction for burnt products according to Cantera.
    gas.equilibrate("HP")
    mixture_fraction_cantera_b = gas.mixture_fraction(fuel_string, oxidizer_string)
    Y_b = gas.Y

    # Comptue mixture fraction for burnt products according to FlameletAI.
    mixture_fraction_config_b = sum(z_sp * Y_b) + z_c 

    # Compare mixture fraction values with a random combination of species.
    Y_random = np.random.rand(gas.n_species)
    Y_random /= np.sum(Y_random)

    gas.Y = Y_random 
    gas.TP = T, p 
    mixture_fraction_cantera_random = gas.mixture_fraction(fuel_string, oxidizer_string)
    mixture_fraction_config_random = sum(z_sp * Y_random) + z_c

    # Print outputs to terminal
    print("Fuel: %s Oxidizer: %s Reaction mechanism: %s" % (fuel_string, oxidizer_string, reaction_mechanism))
    print("Mixture fraction according to Cantera (reactants): %.6e" % mixture_fraction_cantera_unb)
    print("Mixture fraction according to SU2 DataMiner (reactants): %.6e" % mixture_fraction_config_unb)

    print("Mixture fraction according to Cantera (products): %.6e" % mixture_fraction_cantera_b)
    print("Mixture fraction according to SU2 DataMiner (products): %.6e" % mixture_fraction_config_b)

    print("Mixture fraction according to Cantera (random mixture): %.6e" % mixture_fraction_cantera_random)
    print("Mixture fraction according to SU2 DataMiner (random mixture): %.6e" % mixture_fraction_config_random)

    fid.write("%.6f\n" % abs(mixture_fraction_cantera_unb - mixture_fraction_config_unb))
    fid.write("%.6f\n" % abs(mixture_fraction_cantera_b - mixture_fraction_config_b))
    fid.write("%.6f\n" % abs(mixture_fraction_cantera_random - mixture_fraction_config_random))
    
    return [mixture_fraction_cantera_unb, mixture_fraction_config_unb], \
           [mixture_fraction_cantera_b, mixture_fraction_config_b], \
           [mixture_fraction_cantera_random, mixture_fraction_config_random]

with open("mixture_fraction_verification.csv", "w+") as fid:
    z_H_unb, z_H_b, z_H_r = VerifyMixtureFraction(["H2"],[1.0],"h2o2.yaml", fid)
    z_CH4_unb, z_CH4_b, z_CH4_r = VerifyMixtureFraction(["CH4"],[1.0],"gri30.yaml",fid)
    z_comp_unb, z_comp_b, z_comp_r = VerifyMixtureFraction(["H2", "CH4"],[0.5, 0.5],"gri30.yaml",fid)