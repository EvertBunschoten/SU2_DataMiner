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

    init_learning_rate_expo:float = -1.7838e+00
    learning_rate_decay:float =  +9.8972e-01
    batch_size_exponent:int = 6
    NN_hidden:int = 40

    N_epochs:int = 1000
    