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
from Common.DataDrivenConfig import Config_FGM 
np.random.seed(0)

# Reaction mechanism, temperature, pressure, and equivalence ratio.
reaction_mechanism = "h2o2.yaml"
T = 300
p = ct.one_atm
phi = 0.5 

# Initiate empty FlameletAI.
Config = Config_FGM()
Config.SetFuelDefinition(["H2"], [1.0])
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
y_random = np.random.rand(gas.n_species)
Y_random = y_random/np.sum(y_random)

gas.Y = Y_random 
gas.TP = T, p 
mixture_fraction_cantera_random = gas.mixture_fraction(fuel_string, oxidizer_string)
mixture_fraction_config_random = sum(z_sp * Y_random) + z_c

# Print outputs to terminal
print("Mixture fraction according to Cantera (reactants): %.6e" % mixture_fraction_cantera_unb)
print("Mixture fraction according to FlameletAI (reactants): %.6e" % mixture_fraction_config_unb)

print("Mixture fraction according to Cantera (products): %.6e" % mixture_fraction_cantera_b)
print("Mixture fraction according to FlameletAI (products): %.6e" % mixture_fraction_config_b)

print("Mixture fraction according to Cantera (random mixture): %.6e" % mixture_fraction_cantera_random)
print("Mixture fraction according to FlameletAI (random mixture): %.6e" % mixture_fraction_config_random)

with open("mixture_fraction_verification.csv", "w+") as fid:
    diff_unb = mixture_fraction_cantera_unb - mixture_fraction_config_unb
    diff_b = mixture_fraction_cantera_b - mixture_fraction_config_b
    diff_r = mixture_fraction_cantera_random - mixture_fraction_config_random
    fid.write("%.6f\n" % abs(diff_unb))
    fid.write("%.6f\n" % abs(diff_b))
    fid.write("%.6f\n" % abs(diff_r))
    