###############################################################################################
#       #      _____ __  _____      ____        __        __  ____                   #        #  
#       #     / ___// / / /__ \    / __ \____ _/ /_____ _/  |/  (_)___  ___  _____   #        #  
#       #     \__ \/ / / /__/ /   / / / / __ `/ __/ __ `/ /|_/ / / __ \/ _ \/ ___/   #        #      
#       #    ___/ / /_/ // __/   / /_/ / /_/ / /_/ /_/ / /  / / / / / /  __/ /       #        #  
#       #   /____/\____//____/  /_____/\__,_/\__/\__,_/_/  /_/_/_/ /_/\___/_/        #        #
#       #                                                                            #        #
###############################################################################################

################################# FILE NAME: Properties.py ####################################
#=============================================================================================#
# author: Evert Bunschoten                                                                    |
#    :PhD Candidate ,                                                                         |
#    :Flight Power and Propulsion                                                             |
#    :TU Delft,                                                                               |
#    :The Netherlands                                                                         |
#                                                                                             |
#                                                                                             |
# Description:                                                                                |
#  Default settings/names/properties for the various steps within the DataMiner workflow.     |                                                                          
#                                                                                             |  
# Version: 1.0.0                                                                              |
#                                                                                             |
#=============================================================================================#


class DefaultProperties:
    
    train_fraction:float = 0.8
    test_fraction:float = 0.1

    init_learning_rate_expo:float = -1.8269e+00
    learning_rate_decay:float =  +9.8959e-01
    batch_size_exponent:int = 6
    NN_hidden:int = 30

    N_epochs:int = 1000
    hidden_layer_architecture:list[int] = [20,20,20]
    activation_function:str = "gelu"
    output_file_header:str = "fluid_data"

class DefaultSettings_NICFD(DefaultProperties):
    T_min:float = 300
    T_max:float = 600
    Np_temp:float = 600
    
    P_min:float = 2e4
    P_max:float = 2e6
    Np_p:float = 700 

    Rho_min:float = 0.5
    Rho_max:float = 300 
    
    Energy_min:float = 3e5
    Energy_max:float = 5.5e5 
    
    fluid_name:str = "Air"

    use_PT_grid:bool = False 

    controlling_variables:list[str] = ["Density", "Energy"]

    hidden_layer_architecture:list[int] = [30]

    activation_function:str = "exponential"

class DefaultSettings_FGM(DefaultProperties):
    config_name:str = "config_FGM"

    pressure:float = 101325

    T_min:float = 300.0
    T_max:float = 800.0
    Np_temp:int = 30

    eq_ratio_min:float = 0.2
    eq_ratio_max:float = 20.0
    Np_eq:int = 30

    reaction_mechanism:str = 'gri30.yaml'
    transport_model:str = 'multicomponent'

    fuel_definition:list[str] = ["CH4"]
    fuel_weights:list[float] = [1.0]
    oxidizer_definition:list[str] = ["O2","N2"]
    oxidizer_weights:list[float] = [1.0, 3.76]

    carrier_specie:str = 'N2'

    pv_species:list[str] = ["CH4", "H2", "O2", "H2O", "CO2"]
    pv_weights:list[float] = [-6.2332e-02, -4.9603e-01, -3.1252e-02, 5.5509e-02, 2.2723e-02]

    name_pv:str = "ProgressVariable"
    name_enth:str = "EnthalpyTot"
    name_mixfrac:str = "MixtureFraction"
    
    controlling_variables:list[str] = [name_pv, name_enth, name_mixfrac]

    init_learning_rate_expo:float = -3.0
    learning_rate_decay:float =  +9.8959e-01
    batch_size_exponent:int = 6
    hidden_layer_architecture:list[int] = [16, 20, 28, 34, 30, 24, 20]
    activation_function:str = "gelu"

    preferential_diffusion:bool = False
    run_mixture_fraction:bool = False 

    include_freeflames:bool = True 
    include_burnerflames:bool = True 
    include_equilibrium:bool = True 
    include_counterflames:bool = False 
    
    affinity_threshold:float = 0.7
    output_file_header:str = "flamelet_data"
    boundary_file_header:str = "boundary_data"