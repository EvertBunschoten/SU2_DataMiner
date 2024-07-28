import numpy as np
import cantera as ct 

from DataDrivenConfig import FlameletAIConfig 


def ComputeProgressVariable(Config:FlameletAIConfig, variables:list[str], flamelet_data:np.ndarray, Y_flamelet:np.ndarray=None):
    """
    Compute the progress variable based on the corresponding progress variable definition for an array of provided flamelet data.

    :param variables: list of variable names in the flamelet data.
    :type variables: list[str]
    :param flamelet_data: flamelet data array.
    :type flamelet_data: np.ndarray
    :raise: Exception: if number of variables does not match number of columns in flamelet data.
    :return: Progress variable array.
    :rtype: np.array
    """

    pv_species = Config.GetProgressVariableSpecies()
    pv_weights = Config.GetProgressVariableWeights()
    if Y_flamelet is not None:
        if np.shape(Y_flamelet)[0] != Config.gas.n_species:
            raise Exception("Number of species does not match mass fraction array content.")
        pv = np.zeros(np.shape(Y_flamelet)[1])
        for pv_w, pv_sp in zip(pv_weights, pv_species):
            pv += pv_w * Y_flamelet[Config.gas.species_index(pv_sp), :]
        return pv 
    else:
        if len(variables) != np.shape(flamelet_data)[1]:
            raise Exception("Number of variables does not match data array.")
        
        pv = np.zeros(np.shape(flamelet_data)[0])
        for iPv, pvSp in enumerate(pv_species):
            pv += pv_weights[iPv] * flamelet_data[:, variables.index("Y-"+pvSp)]
        return pv 

def ComputeProgressVariable_Source(Config:FlameletAIConfig, variables:list[str], flamelet_data:np.ndarray,net_production_rate_flamelet:np.ndarray=None):
        """
        Compute the progress variable source term based on the corresponding progress variable definition for an array of provided flamelet data.

        :param variables: list of variable names in the flamelet data.
        :type variables: list[str]
        :param flamelet_data: flamelet data array.
        :type flamelet_data: np.ndarray
        :raise: Exception: if number of variables does not match number of columns in flamelet data.
        :return: Progress variable source terms.
        :rtype: np.array
        """

        pv_species = Config.GetProgressVariableSpecies()
        pv_weights = Config.GetProgressVariableWeights()
        if net_production_rate_flamelet is not None:
            if np.shape(net_production_rate_flamelet)[0] != Config.gas.n_species:
                raise Exception("Number of species does not match mass fraction array content.")
            ppv = np.zeros(np.shape(net_production_rate_flamelet)[1])
            for pv_w, pv_sp in zip(pv_weights, pv_species):
                ppv += pv_w * net_production_rate_flamelet[Config.gas.species_index(pv_sp), :]\
                    * Config.gas.molecular_weights[Config.gas.species_index(pv_sp)]
            return ppv
        else:
            if len(variables) != np.shape(flamelet_data)[1]:
                raise Exception("Number of variables does not match data array.")
            ppv = np.zeros(np.shape(flamelet_data)[0])
            for iPv, pvSp in enumerate(pv_species):
                prodrate_pos = flamelet_data[:, variables.index('Y_dot_pos-'+pvSp)]
                prodrate_neg = flamelet_data[:, variables.index('Y_dot_neg-'+pvSp)]
                mass_fraction = flamelet_data[:, variables.index('Y-'+pvSp)]
                ppv += pv_weights[iPv] * (prodrate_pos + prodrate_neg * mass_fraction)
            return ppv 
        
def ComputeMixFracConstants(Config:FlameletAIConfig):
        """
        
        Compute the species mass fraction coefficients according to the Bilger mixture fraction definition.
        
        """

        # Number of species in fuel and oxidizer definition.
        fuel_species = Config.GetFuelDefinition()
        fuel_weights = Config.GetFuelWeights()
        oxidizer_species = Config.GetOxidizerDefinition()
        oxidizer_weights = Config.GetOxidizerWeights()

        n_fuel = len(fuel_species)
        n_ox = len(oxidizer_species)

        # Joining fuel and oxidizer definitions into a single string
        fuel_string = ','.join([fuel_species[i] + ':'+str(fuel_weights[i]) for i in range(n_fuel)])
        oxidizer_string = ','.join([oxidizer_species[i] + ':'+str(oxidizer_weights[i]) for i in range(n_ox)])

        #--- Computing mixture fraction coefficients ---#
        # setting up mixture in stochiometric condition
        Config.gas.TP = 300, ct.one_atm
        Config.gas.set_equivalence_ratio(1.0, fuel_string, oxidizer_string)
        Config.gas.equilibrate('TP')


        # number of atoms occurrances in fuel
        atoms_in_fuel = np.zeros(Config.gas.n_elements)
        for i_e in range(Config.gas.n_elements):
            for i_f in range(n_fuel):
                if Config.gas.n_atoms(fuel_species[i_f], Config.gas.element_names[i_e]) > 0:
                    atoms_in_fuel[i_e] += fuel_weights[i_f]

        # Computing the element mass fractions in the equilibrated mixture
        Z_elements = np.zeros(Config.gas.n_elements)
        for i_e in range(Config.gas.n_elements):
            for i_s in range(Config.gas.n_species):
                Z_elements[i_e] += Config.gas.n_atoms(Config.gas.species_name(i_s), Config.gas.element_name(i_e)) * Config.gas.atomic_weights[i_e] * Config.gas.Y[i_s]/Config.gas.molecular_weights[i_s]

        # Getting element index of oxygen
        idx_O = Config.gas.element_index('O')

        # Computing the elemental mass fractions in the fuel
        Z_fuel_elements = 0
        for i_e in range(Config.gas.n_elements):
            if i_e != idx_O:
                    Z_fuel_elements += atoms_in_fuel[i_e] * Z_elements[i_e]/Config.gas.atomic_weights[i_e]

        # Computing the oxygen stochimetric coefficient
        nu_O = Z_fuel_elements * Config.gas.atomic_weights[idx_O]/Z_elements[idx_O]

        # Filling in fuel specie mass fraction array
        __fuel_weights_s = np.zeros(Config.gas.n_species)
        for i_fuel in range(n_fuel):
            idx_sp = Config.gas.species_index(fuel_species[i_fuel])
            __fuel_weights_s[idx_sp] = fuel_weights[i_fuel]
        Y_fuel_s = __fuel_weights_s * Config.gas.molecular_weights/np.sum(__fuel_weights_s * Config.gas.molecular_weights)

        # Filling in oxidizer specie mass fraction array
        __oxidizer_weights_s = np.zeros(Config.gas.n_species)
        for i_oxidizer in range(n_ox):
            idx_sp = Config.gas.species_index(oxidizer_species[i_oxidizer])
            __oxidizer_weights_s[idx_sp] = oxidizer_weights[i_oxidizer]
        Y_oxidizer_s = __oxidizer_weights_s * Config.gas.molecular_weights/np.sum(__oxidizer_weights_s * Config.gas.molecular_weights)

        # Computing elemental mass fractions of pure fuel stream
        Z_elements_1 = np.zeros(Config.gas.n_elements)
        for i_e in range(Config.gas.n_elements):
            for i_s in range(Config.gas.n_species):
                Z_elements_1[i_e] += Config.gas.n_atoms(Config.gas.species_name(i_s), Config.gas.element_name(i_e)) * Config.gas.atomic_weights[i_e] * Y_fuel_s[i_s] / Config.gas.molecular_weights[i_s]

        # Computing elemental mass fractions of pure oxidizer stream
        Z_elements_2 = np.zeros(Config.gas.n_elements)
        for i_e in range(Config.gas.n_elements):
            for i_s in range(Config.gas.n_species):
                Z_elements_2[i_e] += Config.gas.n_atoms(Config.gas.species_name(i_s), Config.gas.element_name(i_e)) * Config.gas.atomic_weights[i_e] * Y_oxidizer_s[i_s] / Config.gas.molecular_weights[i_s]

        # Computing stochimetric coefficient of pure fuel stream
        beta_1 = 0
        for i_e in range(Config.gas.n_elements):
            beta_1 += atoms_in_fuel[i_e]*Z_elements_1[i_e]/Config.gas.atomic_weights[i_e]
        beta_1 -= nu_O * Z_elements_1[idx_O]/Config.gas.atomic_weights[idx_O]

        # Computing stochimetric coefficient of pure oxidizer stream
        beta_2 = 0
        for i_e in range(Config.gas.n_elements):
            beta_2 += atoms_in_fuel[i_e] * Z_elements_2[i_e]/Config.gas.atomic_weights[i_e]
        beta_2 -= nu_O * Z_elements_2[idx_O]/Config.gas.atomic_weights[idx_O]

        # Computing mixture fraction coefficient
        mixfrac_coefficients = np.zeros(Config.gas.n_species)
        for i_s in range(Config.gas.n_species):
            z_fuel = 0
            for i_e in range(Config.gas.n_elements):
                z_fuel += atoms_in_fuel[i_e] * Config.gas.n_atoms(Config.gas.species_name(i_s), Config.gas.element_name(i_e))/Config.gas.molecular_weights[i_s]
            z_ox = -nu_O * Config.gas.n_atoms(Config.gas.species_name(i_s), 'O')/Config.gas.molecular_weights[i_s]

            mixfrac_coefficients[i_s] = (1/(beta_1 - beta_2)) * (z_fuel + z_ox)

        # Constant term in mixture fraction equation
        mixfrac_constant = -beta_2 / (beta_1 - beta_2)

        return mixfrac_coefficients, mixfrac_constant

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
