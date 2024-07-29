import os,sys 
import subprocess 
import datetime
import time 
import difflib 

def is_float(test_string):
    try:
        float(test_string)
        return True
    except ValueError:
        return False
    
class TestCase:
    reference_files:list[str]
    test_files:list[str] 
    config_dir:str
    config_file:str 
    exec_command:str 
    timeout:float = 10.0 
    comp_threshold:float = 0.0 
    tolerance:float = 0.0 
    
    def __init__(self, tag_in:str):
          self.tag = tag_in 
          self.config_dir = "."
          self.config_file = "config.cfg"

          self.tol = 0.0


    def run_test(self):
        print('==================== Start Test: %s ===================='%self.tag)
        passed = True 
        timed_out = False 

        logfilename = "%s.log" % os.path.splitext(self.config_file)[0]

        shell_command = "%s %s > %s" % (self.exec_command, self.config_file, logfilename)
        
        workdir = os.getcwd()
        os.chdir(self.config_dir)
        print(shell_command)
        print(os.getcwd())
        start = datetime.datetime.now()
        process = subprocess.Popen(shell_command, shell=True)

        while process.poll() is None:
            time.sleep(0.1)
            now = datetime.datetime.now()
            running_time = (now - start).seconds 
            if running_time > self.timeout:
                try:
                    process.kill()
                except AttributeError:
                    pass
                timed_out = True
                passed = False 
        
        if process.poll() != 0:
            passed = False 
            print("ERROR")
            print("Output from the failed case:")
            subprocess.call(["cat", logfilename])

        if not timed_out and passed:
            diff = []
            for iFile, fromfile in enumerate(self.reference_files):
                tofile = self.test_files[iFile]
                
                with open(fromfile,'r') as fid:
                    fromlines = fid.readlines()
                with open(tofile, 'r') as fid:
                    tolines = fid.readlines() 
                
                max_delta = 0
                compare_counter = 0 
                ignore_counter = 0 

                if len(fromlines) != len(tolines):
                    diff = ["ERROR: Number of lines in " + fromfile + " and " + tofile + " differ."]
                    passed = False 

                for i_line in range(0, len(fromlines)):
                    if passed == False: break 

                    from_line = fromlines[i_line].split()
                    to_line = tolines[i_line].split()
                    if len(from_line) != len(to_line):
                        diff = ["ERROR: Number of words in file " + fromfile + "line " + str(i_line + 1) + " differ."]
                        passed = False 
                        break 

                    for i_word in range(len(from_line)):
                        from_word = from_line[i_word].strip().strip(',')
                        to_word = to_line[i_word].strip().strip(',')

                        from_isfloat = is_float(from_word)
                        to_isfloat = is_float(to_word)
                        if from_isfloat != to_isfloat:
                            diff = ["ERROR: File entries in " + fromfile + "'" + from_word + "' and '" + to_word + "' in line " + str(i_line+1) + ", word " + str(i_word+1) + " differ."]
                            passed = False 
                            delta = 0.0 
                            max_delta = "not applicable"
                            break 
                        if from_isfloat:
                            try:
                                # Only do a relative comparison when the threshold is met.
                                # This is to prevent large relative differences for very small numbers.
                                if (abs(float(from_word)) > self.comp_threshold):
                                    delta = abs( (float(from_word) - float(to_word)) / float(from_word) ) * 100
                                    compare_counter += 1
                                else:
                                    delta = 0.0
                                    ignore_counter += 1

                                max_delta = max(max_delta, delta)

                            except ZeroDivisionError:
                                ignore_counter += 1
                                continue
                        # Compare non-floats
                        else:
                            delta = 0.0
                            compare_counter += 1
                            if from_word != to_word:
                                diff = ["ERROR: File entries '" + from_word + "' and '" + to_word + "' in line " + str(i_line+1) + ", word " + str(i_word+1) + " differ."]
                                passed = False
                                max_delta = "Not applicable"
                                break

                        if delta > self.tolerance:
                            diff = ["ERROR: File entries '" + from_word + "' and '" + to_word + "' in line " + str(i_line+1) + ", word " + str(i_word+1) + " differ."]
                            passed = False
                            break
                
                if diff == []:
                    passed = True 
                else:
                    print(diff)
        print('==================== End Test: %s ====================\n'%self.tag)
        sys.stdout.flush()
        os.chdir(workdir)
        return passed 
