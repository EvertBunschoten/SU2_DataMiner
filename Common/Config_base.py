import os 

from Properties import DefaultProperties 
class Config:

    __output_dir:str 
    __config_name:str = "config"

    __train_fraction:float = DefaultProperties.train_fraction
    __test_fraction:float = DefaultProperties.test_fraction
    
    def __init__(self):
        return 
    
    def SetOutputDir(self, output_dir:str):
        """
        Define the fluid data output directory. This directory is set as the default storage directory for all storage processes in the EntropicAI workflow.

        :param output_dir: storage directory.
        :raise: Exception: If the specified directory does not exist.

        """
        if not os.path.isdir(output_dir):
            raise Exception("Invalid output data directory")
        else:
            self.__output_dir = output_dir
        return 
    
    def GetOutputDir(self):
        """
        Get the current EntropicAI configuration fluid storage directory.

        :raises: Exception: if the storage directory in the current `EntropicAIConfig` class is not present on the current hardware.
        :return: Flamelet data storage directory.
        :rtype: str

        """
        if not os.path.isdir(self.__output_dir):
            raise Exception("Saved output directory not present on current machine.")
        else:
            return self.__output_dir
    
    def SetConcatenationFileHeader(self, header:str=DefaultProperties.output_file_header):
        """
        Define the file name header for the collection of fluid data.

        :param header: file name header.
        :type header: str
        """
        self.__concatenated_file_header = header 
        return 
    
    def GetConcatenationFileHeader(self):
        """
        Get the file name header for the fluid collection file.

        :return: fluid collection file header.
        :rtype: str
        """
        return self.__concatenated_file_header
    
    def SetTrainFraction(self, input:float=DefaultProperties.train_fraction):
        """
        Define the fraction of fluid data used for training multi-layer perceptrons.

        :param input: fluid data train fraction.
        :type input: float 
        :raise: Exception: if provided fraction is equal or higher than one.
        """
        if input >= 1:
            raise Exception("Training data fraction should be lower than one.")
        self.__train_fraction = input 
        return 
    
    def SetTestFraction(self, input:float=DefaultProperties.test_fraction):
        """
        Define the fraction of fluid data separate from the training data used for determining accuracy after training.

        :param input: fluid data test fraction.
        :type input: float
        :raise Exception: if provided fraction is equal or higher than one.
        """
        if input >= 1:
            raise Exception("Test data fraction should be lower than one.")
        self.__test_fraction = input 
        return 
    
    def GetTrainFraction(self):
        """
        Get fluid data fraction used for multi-layer perceptron training.

        :return: fluid data train fraction.
        :rtype: float 
        """
        return self.__train_fraction
    
    def GetTestFraction(self):
        """
        Get fluid data fraction used for determining accuracy after training.

        :return: fluid data test fraction.
        :rtype: float 
        """
        return self.__test_fraction
    