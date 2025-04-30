from TestCase import TestCase 
import sys 

def main():
    
    test_list:list[TestCase] = [] 

    fluid_air = TestCase("Fluid_Air")
    fluid_air.config_dir = "FluidGeneration/Air/"
    fluid_air.config_file = "config_air.cfg"
    fluid_air.exec_command = "./generate_fluid_data.py"
    fluid_air.reference_files = ["MLP_data_test_ref.csv"]
    fluid_air.test_files = ["fluid_data_test.csv"]
    test_list.append(fluid_air)

    # fluid_MM = TestCase("Fluid_MM")
    # fluid_MM.config_dir = "FluidGeneration/MM/"
    # fluid_MM.config_file = "config_MM.cfg"
    # fluid_MM.exec_command = "./generate_fluid_data.py"
    # fluid_MM.reference_files = ["MLP_data_test_ref.csv"]
    # fluid_MM.test_files = ["fluid_data_test.csv"]
    # test_list.append(fluid_MM)

    # training_MM_direct = TestCase("Training_MM_Direct")
    # training_MM_direct.config_dir = "FluidTraining/MM/"
    # training_MM_direct.config_file = "config_MM.cfg"
    # training_MM_direct.exec_command = "./train_MLP.py"
    # training_MM_direct.reference_files = ["TrainingHistory_ref.csv"]
    # training_MM_direct.test_files = ["Model_0/TrainingHistory.csv"]
    # test_list.append(training_MM_direct)

    # training_MM_PINN = TestCase("Training_MM_PhysicsInformed")
    # training_MM_PINN.config_dir = "FluidTraining/MM_PINN/"
    # training_MM_PINN.config_file = "config_MM.cfg"
    # training_MM_PINN.exec_command = "./train_MLP.py"
    # training_MM_PINN.reference_files = ["SU2_MLP_ref.mlp"]
    # training_MM_PINN.test_files = ["Model_0/SU2_MLP.mlp"]

    # test_list.append(training_MM_PINN)
    pass_list = [test.run_test() for test in test_list]

    # Tests summary
    print('==================================================================')
    print('Summary of the serial tests')
    print('python version:', sys.version)
    for i, test in enumerate(test_list):
        if (pass_list[i]):
            print('  passed - %s'%test.tag)
        else:
            print('* FAILED - %s'%test.tag)

    if all(pass_list):
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == '__main__':
    main()
