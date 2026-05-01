###############################################################################################
#       #      _____ __  _____      ____        __        __  ____                   #        #
#       #     / ___// / / /__ \    / __ \____ _/ /_____ _/  |/  (_)___  ___  _____   #        #
#       #     \__ \/ / / /__/ /   / / / / __ `/ __/ __ `/ /|_/ / / __ \/ _ \/ ___/   #        #
#       #    ___/ / /_/ // __/   / /_/ / /_/ / /_/ /_/ / /  / / / / / /  __/ /       #        #
#       #   /____/\____//____/  /_____/\__,_/\__/\__,_/_/  /_/_/_/ /_/\___/_/        #        #
#       #                                                                            #        #
###############################################################################################

################################ FILE NAME: Config_base.py ####################################
#=============================================================================================#
# author: Evert Bunschoten                                                                    |
#    :PhD Candidate ,                                                                         |
#    :Flight Power and Propulsion                                                             |
#    :TU Delft,                                                                               |
#    :The Netherlands                                                                         |
#                                                                                             |
#                                                                                             |
# Description:                                                                                |
#  Base class for DataMiner configuration                                                     |
#                                                                                             |
# Version: 3.1.0                                                                              |
#                                                                                             |
#=============================================================================================#

import os 
import pyfiglet
import pickle
import numpy as np 

from Common.Properties import DefaultProperties, ActivationFunctionOptions
from Common.CommonMethods import write_SU2_MLP

class Config:
    """Base class for the SU2 DataMiner configuration.
    """
    
    __banner_header:str = "SU2 DataMiner"            # Main banner message to be printed in fancy text.

    # Output settings
    _output_dir:str                                  # Data output directory.
    _config_name:str = DefaultProperties.config_name # SU2 DataMiner configuration name.
    __concatenated_file_header:str=DefaultProperties.output_file_header # Processed fluid data file name header.
    _controlling_variables:list[str]                                    # Controlling variable names.

    # MLP-based manifold settings
    __train_fraction:float = DefaultProperties.train_fraction           # Fluid data train fraction.
    __test_fraction:float = DefaultProperties.test_fraction             # Fluid data test fraction.
    _alpha_expo:float = DefaultProperties.init_learning_rate_expo       # Initial learning rate exponent (base 10)
    _lr_decay:float = DefaultProperties.learning_rate_decay             # Learning rate decay parameter.
    _batch_expo:int = DefaultProperties.batch_size_exponent             # Mini-batch size exponent (base 2).
    _n_epochs:int = DefaultProperties.N_epochs                          # Maximum number of training epochs.
    _hidden_layer_architecture:list[int] = DefaultProperties.hidden_layer_architecture  # Hidden layer perceptron count.
    _activation_function:str = DefaultProperties.activation_function    # Hidden layer activation function name.

    _scaler_function_name:str = "minmax"        # Scaler function by which MLP train data is scaled.
    _scaler_function_vals_in:list[list[float]]  # Linear scaling function values for controlling variable data.
    _scaler_function_vals_out:list[list[float]] # Linear scaling function values for dependent variable data.
    _train_vars:list[str]                       # Dependent variables.
    _control_vars:list[str]                     # Controlling variables.
    _MLP_weights:list[np.ndarray[float]]        # MLP weights values.
    _MLP_biases:list[np.ndarray[float]]         # MLP biases values.

    _config_type:str= None  # SU2 DataMiner configuration type.
    
    def __init__(self):
        self._output_dir = os.getcwd()
        return 
    
    def PrintBanner(self):
        """Print the main banner for the SU2 DataMiner configuration in the terminal.
        """

        customfig = pyfiglet.Figlet(font="slant")
        print(customfig.renderText(self.__banner_header))

        return 
    
    def SetOutputDir(self, output_dir:str):
        """Define the output directory where all raw and processed fluid data and manifold data are saved.

        :param output_dir: output directory.
        :type output_dir: str
        :raises Exception: if provided directory does not exist on the current hardware.
        """
        if not os.path.isdir(output_dir):
            raise Exception("Invalid output data directory")
        
        self._output_dir = output_dir

        return 
    
    def GetOutputDir(self):
        """Get the output directory where raw and processed fluid data and manifold data are stored.

        :raises Exception: if the output directory of the current SU2 DataMiner configuration is not present on the current hardware.
        :return: output directory.
        :rtype: str
        """

        if not os.path.isdir(self._output_dir):
            raise Exception("Saved output directory not present on current machine.")
        else:
            return self._output_dir
    
    def SetConcatenationFileHeader(self, header:str=DefaultProperties.output_file_header):
        """Define the file name header of the processed fluid manifold data.

        :param header: manifold data file header, defaults to "fluid_data"
        :type header: str, optional
        """

        self.__concatenated_file_header = header 

        return 
    
    def GetConcatenationFileHeader(self):
        """Get the file name header of the processed fluid manifold data.

        :return: fluid manifold data file header.
        :rtype: str
        """

        return self.__concatenated_file_header
    
    def SetConfigName(self, config_name:str):
        """Set the name for the current SU2 DataMiner configuration. When saving the configuration, it will be saved under this name.

        :param config_name: SU2 DataMiner configuration name.
        :type config_name: str
        """

        self._config_name = config_name 

        return 
    
    def GetConfigName(self):
        """Get the name of the current SU2 DataMiner configuration.

        :return: SU2 DataMiner configuration name.
        :rtype: str
        """
        return self._config_name 
    
    def SetControllingVariables(self, names_cv:list[str]):
        """Define the set of controlling variable names used as inputs for the networks of the data-driven fluid model.

        :param names_cv: list with controlling variable names.
        :type names_cv: list[str]
        """

        self._controlling_variables = []
        for c in names_cv:
            self._controlling_variables.append(c)

        return 
    
    def GetControllingVariables(self):
        """Retrieve the set of controlling variable names used as inputs for the networks of the data-driven fluid model.

        :return: list of controlling variable names.
        :rtype: list[str]
        """

        return self._controlling_variables
    
    def SetTrainFraction(self, input:float=DefaultProperties.train_fraction):
        """Define the fraction of fluid data used for MLP training.

        :param input: fluid data train fraction, defaults to 0.8
        :type input: float, optional
        :raises Exception: if provided value lies outside 0-1.
        """

        if input >= 1 or input <=0:
            raise Exception("Training data fraction should be between zero and one.")
        self.__train_fraction = input 

        return 
    
    def SetTestFraction(self, input:float=DefaultProperties.test_fraction):
        """Define the fraction of fluid data used for MLP prediction accuracy evaluation.

        :param input: fluid data test set fraction, defaults to 0.1
        :type input: float, optional
        :raises Exception: if provided value lies outside 0-1.
        """

        if input >= 1 or input <=0:
            raise Exception("Test data fraction should be between zero and one.")
        self.__test_fraction = input 

        return 
    
    def GetTrainFraction(self):
        """Get the fraction of fluid data used for MLP training.

        :return: fluid data train fraction.
        :rtype: float
        """
        return self.__train_fraction
    
    def GetTestFraction(self):
        """Get the fraction of fluid data used for MLP accuracy evaluation.

        :return: fluid data test fraction.
        :rtype: float
        """

        return self.__test_fraction
    
    def GetAlphaExpo(self):
        """Get the initial learning rate exponent (base 10).

        :return: log10 of initial learning rate.
        :rtype: float
        """
        
        return self._alpha_expo
    
    def SetAlphaExpo(self, alpha_expo_in:float=DefaultProperties.init_learning_rate_expo):
        """Define the initial learning rate exponent (base 10).

        :param alpha_expo_in: log10 of initial learning rate, defaults to -1.8269
        :type alpha_expo_in: float, optional
        :raises Exception: if provided value is positive.
        """

        if alpha_expo_in >= 0:
            raise Exception("Initial learning rate exponent should be negative.")
        self._alpha_expo = alpha_expo_in
        return 
    
    def GetLRDecay(self):
        """Get the exponential learning rate decay parameter for MLP training.

        :return: Exponential learning rate decay parameter.
        :rtype: float
        """
        return self._lr_decay
    
    def SetLRDecay(self, lr_decay_in:float=DefaultProperties.learning_rate_decay):
        """Set the exponential learning rate decay parameter for MLP training.

        :param lr_decay_in: Exponential learning rate decay parameter, defaults to DefaultProperties.learning_rate_decay
        :type lr_decay_in: float, optional
        :raises Exception: if the learning rate decay parameter is not within zero and one.
        """
        
        if lr_decay_in <= 0 or lr_decay_in > 1.0:
            raise Exception("Learning rate decay parameter should be between zero and one.")
        self._lr_decay = lr_decay_in
        return 
    
    def SetNEpochs(self, n_epochs_in:int=DefaultProperties.N_epochs):
        """Specify the maximum number of epochs for training of the networks.

        :param n_epochs_in: maximum number of epochs, defaults to 1000
        :type n_epochs_in: int, optional
        :raises Exception: if the number is lower than 1.
        """
        if n_epochs_in < 1:
            raise Exception("Number of epochs should be higher than 1")
        
        self._n_epochs = int(n_epochs_in)
        return 
    
    def GetNEpochs(self):
        """Retrieve the maximum number of epochs the networks are trained for.

        :return: maximum number of training epochs.
        :rtype: int
        """
        return self._n_epochs
    
    def SetBatchExpo(self, batch_expo_in:int=DefaultProperties.batch_size_exponent):
        """Set the mini-batch size exponent (base 2) for MLP training.

        :param batch_expo_in: Mini-batch size exponent (base 2) used for MLP training, defaults to DefaultProperties.batch_size_exponent
        :type batch_expo_in: int, optional
        :raises Exception: if provided value is lower than or equal to zero.
        """

        if batch_expo_in <= 0:
            raise Exception("Mini-batch size exponent should be positive.")
        self._batch_expo = int(batch_expo_in)
        return 
    
    def GetBatchExpo(self):
        """Get the MLP training mini-batch size exponent.

        :return: mini-batch size exponent (base 2)
        :rtype: int
        """
        return self._batch_expo 
    
    def __HiddenLayerChecks(self,hidden_layer_architecture:list[int]):
        if not hidden_layer_architecture:
            raise Exception("At least one hidden layer should be specified.")
        for h in hidden_layer_architecture:
            if h < 1:
                raise Exception("Number of nodes in the hidden layers should be positive.")
            if type(h) is not int:
                raise Exception("Nodes in the hidden layers should be integers.")
        return 
    
    def SetHiddenLayerArchitecture(self, hidden_layer_architecture:list[int]=DefaultProperties.hidden_layer_architecture):
        """
        Define the hidden layer architecture of the multi-layer perceptron used for the MLP-based manifold.

        :param hidden_layer_architecture: listed neuron count per hidden layer, defaults to [20,20,20]
        :type hidden_layer_architecture: list[int], optional
        :raises Exception: if an empty list is provided or if input contains non-integer data or the number of nodes is less than 1.
        """

        self.__HiddenLayerChecks(hidden_layer_architecture)

        self._hidden_layer_architecture = []
        for n in hidden_layer_architecture:
            self._hidden_layer_architecture.append(n)
        return 
    
    def GetHiddenLayerArchitecture(self):
        """Get the hidden layer architecture of the multi-layer perceptron used for the MLP-based manifold.

        :return: list with number of neurons per hidden layer.
        :rtype: list[str]
        """
        return self._hidden_layer_architecture
    
    def __WeightsCheck(self, weights:list[np.ndarray[float]]):
        if not weights:
            raise Exception("Weights list should contain at least one array.")
        for i in range(len(weights)-1):
            w_i = weights[i]
            w_ip = weights[i+1]
            if np.shape(w_i)[1] != np.shape(w_ip)[0]:
                raise Exception("Weight arrays are improperly formatted. Check rows and columns.")
        return 
    
        
    def SetWeights(self, weights:list[np.ndarray[float]]):
        """Store the weight values of the neural network.

        :param weights: weight arrays for the network hidden layers.
        :type weights: list[np.ndarray[float]]
        :raises: Exception: if an empty list is provided or the weights arrays are improperly formatted.
        """
        self.__WeightsCheck(weights)

        self._MLP_weights = []
        for w in weights:
            self._MLP_weights.append(w)
        return 
    
    def __BiasesCheck(self, biases:list[np.ndarray[float]]):
        if not biases:
            raise Exception("Biases list should contain at least one entry.")
        for b in biases:
            if b.size == 0:
                raise Exception("Biases for hidden layers should contain at least one value.")
        return 
    
    def SetBiases(self, biases:list[np.ndarray[float]]):
        """Store the bias values of the neural network.

        :param weights: bias arrays for the network hidden layers.
        :type weights: list[np.ndarray[float]]
        :raises: Exception: if an empty list is provided or contains empty arrays.
        """
        self.__BiasesCheck(biases)

        self._MLP_biases = []
        for w in biases:
            self._MLP_biases.append(w)
        return 
    
    def SetActivationFunction(self, activation_function_in:str=DefaultProperties.activation_function):
        """Define the hidden layer activation function for the MLP-based manifold. See Common.Properties.ActivationFunctionOptions for the supported options.

        :param activation_function_in: hidden layer activation function name, defaults to "gelu"
        :type activation_function_in: str, optional
        :raises Exception: if the provided name does not appear in the list of available activation function options.
        """
        if activation_function_in not in ActivationFunctionOptions.keys():
            raise Exception("Activation function " + activation_function_in + " not in available options.")
        self._activation_function = activation_function_in 
        return 
    
    def GetActivationFunction(self):
        """Get the hidden layer activation function name.

        :return: hidden layer activation function name.
        :rtype: str
        """
        return self._activation_function
    
    def UpdateMLPHyperParams(self, trainer):
        """Retrieve the weights and biases from the MLP trainer class and store them in the configuration class.

        :param trainer: reference to trainer class used to train the network.
        :type trainer: TrainMLP
        """

        # Store train parameters
        self._alpha_expo = trainer.alpha_expo
        self._lr_decay = trainer.lr_decay
        self._batch_expo = trainer.batch_expo
        self._n_epochs = trainer.n_epochs
        self._hidden_layer_architecture = trainer.architecture.copy()
        self._activation_function = trainer.activation_function

        # Retrieve MLP definition data
        self._train_vars = trainer.GetTrainVars().copy()
        self._control_vars = trainer.GetControlVars().copy()
        self._scaler_function_name, self._scaler_function_vals_in,self._scaler_function_vals_out = trainer.GetScalerFunctionParams()
        self._MLP_weights = trainer.GetWeights().copy()
        self._MLP_biases = trainer.GetBiases().copy()
        self._hidden_layer_architecture =[h for h in trainer.architecture]
        
        return 
    
    def GetWeightsBiases(self):
        """Return values for weights and biases for the hidden layers in the MLP.

        :return: weight arrays, biases arrays
        :rtype: tuple(np.ndarray[float])
        """
        return self._MLP_weights, self._MLP_biases
    
    def WriteSU2MLP(self, file_name_out:str):
        """Write ASCII MLP file containing the network weights and biases from the data stored in the configuration.

        :param file_name_out: MLP file name
        :type file_name_out: str
        """
        return write_SU2_MLP(file_name_out,\
                             weights=self._MLP_weights,\
                             biases=self._MLP_biases,\
                             activation_function_name=self._activation_function,\
                             train_vars=self._train_vars,\
                             controlling_vars=self._control_vars,\
                             scaler_function=self._scaler_function_name,\
                             scaler_function_vals_in=self._scaler_function_vals_in,\
                             scaler_function_vals_out=self._scaler_function_vals_out)

    def SaveConfig(self):
        """
        Save the current SU2 DataMiner configuration.

        """

        file = open(self._config_name+'.cfg','wb')
        pickle.dump(self, file)
        file.close()
        return 
    