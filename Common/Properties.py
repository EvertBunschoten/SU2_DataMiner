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

    output_file_header:str = "fluid_data"

    train_fraction:float = 0.8
    test_fraction:float = 0.1

    init_learning_rate_expo:float = -1.8261e+00
    learning_rate_decay:float =  +9.8787e-01
    batch_size_exponent:int = 6
    NN_hidden:int = 30

    N_epochs:int = 1000
