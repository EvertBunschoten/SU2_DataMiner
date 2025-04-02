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
# Version: 2.0.0                                                                              |
#                                                                                             |
#=============================================================================================#

import tensorflow as tf 
from enum import Enum, auto

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
    config_name:str = "DataMiner"

class EntropicVars(Enum):
    Density=0
    Energy=auto()
    T=auto()
    p=auto()
    c2=auto()
    s=auto()
    dsdrho_e=auto()
    dsde_rho=auto()
    d2sdrho2=auto()
    d2sdedrho=auto()
    d2sde2=auto()
    dTdrho_e=auto()
    dTde_rho=auto()
    dpdrho_e=auto()
    dpde_rho=auto()
    dhdrho_e=auto()
    dhde_rho=auto()
    dhdp_rho=auto()
    dhdrho_p=auto()
    dsdp_rho=auto()
    dsdrho_p=auto()
    cp=auto()
    N_STATE_VARS=auto()

class FGMVars(Enum):
    ProgressVariable=0
    EnthalpyTot=auto()
    MixtureFraction=auto()
    Distance=auto()
    Velocity=auto()
    Temperature=auto()
    Density=auto()
    MolarWeightMix=auto()
    Cp=auto()
    Conductivity=auto()
    ViscosityDyn=auto()
    Heat_Release=auto()
    DiffusionCoefficient=auto()
    ProdRateTot_PV=auto()
    Beta_ProgVar=auto()
    Beta_Enth_Thermal=auto()
    Beta_Enth=auto()
    Beta_MixFrac=auto() 

FGMSymbols:dict = {FGMVars.ProgressVariable.name : r"Progress variable $(\mathcal{Y})[-]$",\
                   FGMVars.EnthalpyTot.name : r"Total enthalpy $(h)[J \mathrm{kg}^{-1}]$",\
                   FGMVars.MixtureFraction.name : r"Mixture fraction $(Z)[-]$",\
                   FGMVars.Distance.name : r"Flamelet solution grid $(x)[m]$",\
                   FGMVars.Velocity.name : r"Flamelet velocity $(u)[m s^{-1}]$",\
                   FGMVars.Temperature.name : r"Temperature $(T)[K]$",\
                   FGMVars.Density.name : r"Density $(\rho)[\mathrm{kg} m^{-3}]$",\
                   FGMVars.MolarWeightMix.name : r"Mean, molar weight $(W_M)[\mathrm{kg} \mathrm{mol}^{-1}]$",\
                   FGMVars.Cp.name : r"Specific heat $(c_p)[J \mathrm{kg}^{-1}K^{-1}]$",\
                   FGMVars.Conductivity.name : r"Thermal conductivity $(k)[W m^{-1} K^{-1}]$",\
                   FGMVars.ViscosityDyn.name : r"Dynamic viscosity $(\mu)[m^2 s^{-2}]$",\
                   FGMVars.Heat_Release.name : r"Heat release rate $\left(\dot{Q}\right)[W \mathrm{kg}^{-1} m^3]$",\
                   FGMVars.DiffusionCoefficient.name : r"Diffusion coefficient $\left(D\right)[\mathrm{kg} m]$",\
                   FGMVars.ProdRateTot_PV.name : r"PV source term $\left(\rho\dot{\omega}_\mathcal{Y}\right)[\mathrm{kg} m^{-3} s^{-1}]$",\
                   FGMVars.Beta_ProgVar.name : r"PV pref. diffusion scalar $\left(\beta_\mathcal{Y}\right)[-]$",\
                   FGMVars.Beta_Enth_Thermal.name : r"Cp pref. diffusion scalar $\left(\beta_{h,1}\right)[J \mathrm{kg}^{-1} K^{-1}]$",\
                   FGMVars.Beta_Enth.name : r"Specific enthalpy pref. diffusion scalar $\left(\beta_{h,2}\right)[J \mathrm{kg}^{-1}]$",\
                   FGMVars.Beta_MixFrac.name : r"Mixture fraction pref. diffusion scalar $\left(\beta_Z\right)[-]$"}

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
    EOS_type:str = "HEOS"
    use_PT_grid:bool = False 

    controlling_variables:list[str] = [EntropicVars.Density.name, \
                                       EntropicVars.Energy.name]
    name_density:str = EntropicVars.Density.name
    name_energy:str = EntropicVars.Energy.name

    hidden_layer_architecture:list[int] = [12,12]

    init_learning_rate_expo:float = -3.0
    learning_rate_decay:float = 0.98985
    activation_function:str = "exponential"
    config_type:str = "EntropicAI"
    supported_state_vars:list[str] = ["s","T","p","c2","dTdrho_e","dTde_rho","dpdrho_e","dpde_rho"]

class DefaultSettings_FGM(DefaultProperties):
    config_name:str = "config_FGM"

    pressure:float = 101325

    T_min:float = 300.0
    T_max:float = 800.0
    Np_temp:int = 30

    T_threshold:float = 600.0
    
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

    init_learning_rate_expo:float = -2.8
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
    config_type:str = "FlameletAI"

ActivationFunctionOptions = {"linear" : tf.keras.activations.linear,\
                             "elu" : tf.keras.activations.elu,\
                             "relu" : tf.keras.activations.relu,\
                             "tanh" : tf.keras.activations.tanh,\
                             "exponential" : tf.keras.activations.exponential,\
                             "gelu" : tf.keras.activations.gelu,\
                             "sigmoid" : tf.keras.activations.sigmoid,\
                             "swish" : tf.keras.activations.swish}