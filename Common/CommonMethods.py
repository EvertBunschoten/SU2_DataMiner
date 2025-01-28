###############################################################################################
#       #      _____ __  _____      ____        __        __  ____                   #        #
#       #     / ___// / / /__ \    / __ \____ _/ /_____ _/  |/  (_)___  ___  _____   #        #
#       #     \__ \/ / / /__/ /   / / / / __ `/ __/ __ `/ /|_/ / / __ \/ _ \/ ___/   #        #
#       #    ___/ / /_/ // __/   / /_/ / /_/ / /_/ /_/ / /  / / / / / /  __/ /       #        #
#       #   /____/\____//____/  /_____/\__,_/\__/\__,_/_/  /_/_/_/ /_/\___/_/        #        #
#       #                                                                            #        #
###############################################################################################

############################### FILE NAME: CommonMethods.py ###################################
#=============================================================================================#
# author: Evert Bunschoten                                                                    |
#    :PhD Candidate ,                                                                         |
#    :Flight Power and Propulsion                                                             |
#    :TU Delft,                                                                               |
#    :The Netherlands                                                                         |
#                                                                                             |
#                                                                                             |
# Description:                                                                                |
#  Common methods used during fluid data computation and processing steps.                    |
#                                                                                             |
# Version: 1.0.0                                                                              |
#                                                                                             |
#=============================================================================================#

import numpy as np
import cantera as ct 

def GetUnburntScalars(gas:ct.Solution, fuel_definition:str, oxidizer_definition:str, pv_species:list[str], pv_weights:list[float], equivalence_ratio:float, T_unb:float):
    if len(pv_weights) != len(pv_species):
        raise Exception("Progress variable weights and species should contain same number of elements.")
    if equivalence_ratio < 0:
        raise Exception("Equivalence ratio should be positive.")
    if T_unb < 0:
        raise Exception("Temperature should be positive.")
    
    gas.set_equivalence_ratio(equivalence_ratio, fuel_definition, oxidizer_definition)
    gas.TP= T_unb, 101325


    return 

def ComputeLewisNumber(flame:ct.Solution):
    Le_species = flame.thermal_conductivity/flame.cp_mass/flame.density_mass/(flame.mix_diff_coeffs+1e-15)
    return Le_species

def avg_Le_start_end(Le_sp:np.ndarray):
    Le_av = 0.5*(Le_sp[0] + Le_sp[-1])
    return Le_av 

def avg_Le_arythmic(Le_sp:np.ndarray):
    Le_av = np.average(Le_sp)
    return Le_av 

def avg_Le_min_max(Le_sp:np.ndarray):
    Le_av = 0.5*(np.min(Le_sp)+np.max(Le_sp))
    return Le_av

def avg_Le_unity(Le_sp:np.ndarray):
    Le_av = np.ones(np.shape(Le_sp))
    return Le_av

def avg_Le_const(Le_sp:np.ndarray, Le_const:float):
    Le_av = Le_const * np.ones(np.shape(Le_sp))
    return Le_av 

def avg_Le_local(Le_sp:np.ndarray):
    return Le_sp


def GetReferenceData(dataset_file, x_vars, train_variables,dtype=np.float32):
    # Open data file and get variable names from the first line
    fid = open(dataset_file, 'r')
    line = fid.readline()
    fid.close()
    line = line.strip()
    line_split = line.split(',')
    if(line_split[0][0] == '"'):
        varnames = [s[1:-1] for s in line_split]
    else:
        varnames = line_split
    
    # Get indices of controlling and train variables
    iVar_x = [varnames.index(v) for v in x_vars]
    iVar_y = [varnames.index(v) for v in train_variables]

    # Retrieve respective data from data set
    D = np.loadtxt(dataset_file, delimiter=',', skiprows=1, dtype=dtype)
    X_data = D[:, iVar_x]
    Y_data = D[:, iVar_y]

    return X_data, Y_data