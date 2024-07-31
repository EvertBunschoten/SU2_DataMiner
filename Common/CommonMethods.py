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


def GetReferenceData(dataset_file, x_vars, train_variables):
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
    D = np.loadtxt(dataset_file, delimiter=',', skiprows=1, dtype=np.float32)
    X_data = D[:, iVar_x]
    Y_data = D[:, iVar_y]

    return X_data, Y_data
