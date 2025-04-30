from TestCase import TestCase 
import sys 

def main():
    
    test_list_NICFD:list[TestCase] = [] 
    test_list_FGM:list[TestCase] = []

    fluid_air = TestCase("Fluid_Air")
    fluid_air.config_dir = "FluidGeneration/Air/"
    fluid_air.config_file = "config_air.cfg"
    fluid_air.exec_command = "./generate_fluid_data.py"
    fluid_air.reference_files = ["MLP_data_test_ref.csv"]
    fluid_air.test_files = ["fluid_data_test.csv"]
    test_list_NICFD.append(fluid_air)

    fluid_MM = TestCase("Fluid_MM")
    fluid_MM.config_dir = "FluidGeneration/MM/"
    fluid_MM.config_file = "config_MM.cfg"
    fluid_MM.exec_command = "./generate_fluid_data.py"
    fluid_MM.reference_files = ["MLP_data_test_ref.csv"]
    fluid_MM.test_files = ["fluid_data_test.csv"]
    test_list_NICFD.append(fluid_MM)

    unittest_mixturefraction = TestCase("Mixture fraction")
    unittest_mixturefraction.config_dir = "Physics/"
    unittest_mixturefraction.config_file = ""
    unittest_mixturefraction.exec_command = "./mixturefraction_computation.py"
    unittest_mixturefraction.reference_files = ["mixture_fraction_verification.ref"]
    unittest_mixturefraction.test_files = ["mixture_fraction_verification.csv"]
    test_list_FGM.append(unittest_mixturefraction)
    
    # test_list.append(training_MM_PINN)
    pass_list_NICFD = [test.run_test() for test in test_list_NICFD]
    pass_list_FGM = [test.run_test() for test in test_list_FGM]

    # Tests summary
    print('==================================================================')
    print('Summary of the serial tests for NICFD')
    print('python version:', sys.version)
    for i, test in enumerate(test_list_NICFD):
        if (pass_list_NICFD[i]):
            print('  passed - %s'%test.tag)
        else:
            print('* FAILED - %s'%test.tag)

    print('==================================================================')
    print('Summary of the serial tests for FGM')
    print('python version:', sys.version)
    for i, test in enumerate(test_list_FGM):
        if (pass_list_FGM[i]):
            print('  passed - %s'%test.tag)
        else:
            print('* FAILED - %s'%test.tag)

    if all(pass_list_NICFD) and all(pass_list_FGM):
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == '__main__':
    main()
