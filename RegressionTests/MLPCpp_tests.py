from TestCase import TestCase 
import sys 

def main():
    MLPCpp_tests:list[TestCase] = []
    tf_mlpcpp_equivalence = TestCase("MLP_eval_tensorflow_MLPCpp")
    tf_mlpcpp_equivalence.config_dir = "MLPCppwrapper"
    tf_mlpcpp_equivalence.config_file = ""
    tf_mlpcpp_equivalence.timeout = 600
    tf_mlpcpp_equivalence.exec_command = "./test_wrapper.py"
    tf_mlpcpp_equivalence.reference_files = []
    tf_mlpcpp_equivalence.test_files = []
    MLPCpp_tests.append(tf_mlpcpp_equivalence)

    pass_list_mlcpp = [test.run_test() for test in MLPCpp_tests]

    # Tests summary
    print('==================================================================')
    print('Summary of the serial tests for MLPCpp integration:')
    print('python version:', sys.version)
    for i, test in enumerate(MLPCpp_tests):
        if (pass_list_mlcpp[i]):
            print('  passed - %s'%test.tag)
        else:
            print('* FAILED - %s'%test.tag)
    if all(pass_list_mlcpp):
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == '__main__':
    main()
