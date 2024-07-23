class DefaultProperties:
    T_min:float = 300
    T_max:float = 600
    Np_temp:float = 100
    
    P_min:float = 2e4
    P_max:float = 2e6
    Np_p:float = 300 

    fluid_name:str = "Air"

    use_PT_grid:bool = False 

    output_file_header:str = "fluid_data"

    train_fraction:float = 0.8
    test_fraction:float = 0.1

    init_learning_rate_expo:float = -2.6
    learning_rate_decay:float =  0.9985
    batch_size_exponent:int = 5 
    NN_hidden:int = 30

    N_epochs:int = 1000
    