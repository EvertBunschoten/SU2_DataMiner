###############################################################################################
#       #      _____ __  _____      ____        __        __  ____                   #        #
#       #     / ___// / / /__ \    / __ \____ _/ /_____ _/  |/  (_)___  ___  _____   #        #
#       #     \__ \/ / / /__/ /   / / / / __ `/ __/ __ `/ /|_/ / / __ \/ _ \/ ___/   #        #
#       #    ___/ / /_/ // __/   / /_/ / /_/ / /_/ /_/ / /  / / / / / /  __/ /       #        #
#       #   /____/\____//____/  /_____/\__,_/\__/\__,_/_/  /_/_/_/ /_/\___/_/        #        #
#       #                                                                            #        #
###############################################################################################

############################## FILE NAME: Trainer_Base.py #####################################
#=============================================================================================#
# author: Evert Bunschoten                                                                    |
#    :PhD Candidate ,                                                                         |
#    :Flight Power and Propulsion                                                             |
#    :TU Delft,                                                                               |
#    :The Netherlands                                                                         |
#                                                                                             |
#                                                                                             |
# Description:                                                                                |
#   Base class for the various MLP trainer types and MLP evaluator class.                     |
#                                                                                             |
# Version: 2.0.0                                                                              |
#                                                                                             |
#=============================================================================================#

# Set seed values.
seed_value = 2
import os
os.environ['PYTHONASHSEED'] = str(seed_value)
seed_value += 1
import random
random.seed(seed_value)
seed_value += 1
import numpy as np
np.random.seed(seed_value)
seed_value += 1 
import tensorflow as tf
tf.random.set_seed(seed_value)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

import time 
from tensorflow import keras
import matplotlib.pyplot as plt 
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from keras.initializers import HeUniform,RandomUniform
import csv 

from Common.Config_base import Config
from Common.Properties import DefaultProperties 
from Common.CommonMethods import GetReferenceData, write_SU2_MLP

# Activation function options
activation_function_names_options:list[str] = ["linear","elu","relu","tanh","exponential","gelu","sigmoid", "swish"]
activation_function_options = [tf.keras.activations.linear,
                            tf.keras.activations.elu,\
                            tf.keras.activations.relu,\
                            tf.keras.activations.tanh,\
                            tf.keras.activations.exponential,\
                            tf.keras.activations.gelu,\
                            tf.keras.activations.sigmoid,\
                            tf.keras.activations.swish]

scaler_functions = {"robust":RobustScaler,\
                    "standard":StandardScaler,\
                    "minmax":MinMaxScaler}
class MLPTrainer:
    # Base class for flamelet MLP trainer
    _dt = tf.float32
    _dt_np = np.float32 

    _n_epochs:int = DefaultProperties.N_epochs      # Number of epochs to train for.
    _alpha_expo:float = DefaultProperties.init_learning_rate_expo  # Alpha training exponent parameter.
    _lr_decay:float = DefaultProperties.learning_rate_decay      # Learning rate decay parameter.
    _batch_expo:int = DefaultProperties.batch_size_exponent     # Mini-batch size exponent.
    _decay_steps:int = 1e4

    _i_activation_function:int = 0   # Activation function index.
    _activation_function_name:str = DefaultProperties.activation_function
    _activation_function = None
    _restart_training:bool = False # Restart training process

    # Hardware info:
    _kind_device:str = "CPU" # Device type used to train (CPU or GPU)
    _device_index:int = 0    # Device index (core index or GPU card index)
    
    # MLP input (controlling) variables.
    _controlling_vars:list[str] = ["Density", 
                        "Energy"]
    
    # Variable names to train for.
    _train_vars:list[str] = []

    # Train, test, and validation data.
    _filedata_train:str 
    _Np_train:int = None
    _Np_test:int = None
    _Np_val:int = None
    _X_train_norm:np.ndarray = None 
    _Y_train_norm:np.ndarray = None
    _X_test_norm:np.ndarray = None 
    _Y_test_norm:np.ndarray = None 
    _X_val_norm:np.ndarray = None 
    _Y_val_norm:np.ndarray = None 

    # Dataset normalization bounds.
    _X_mean:np.ndarray = None 
    _X_scale:np.ndarray = None 
    _X_offset:np.ndarray = None 

    _Y_mean:np.ndarray = None 
    _Y_scale:np.ndarray = None 
    _Y_offset:np.ndarray = None 

    # Hidden layers neuron count.
    _hidden_layers:list[int] = None

    # Weights and biases matrices.
    _weights:list[np.ndarray] = []
    _biases:list[np.ndarray] = []

    _test_score:float = None       # Loss function on test set upon training termination.
    _cost_parameter:float = None   # Cost parameter of the trimmed network.

    _train_time:float = 0  # Training time in minutes.
    _test_time:float = 0   # Test set evaluation time in seconds.

    _save_dir:str = "/"  # Directory to save trained network information to.
    _figformat:str="png"
    _mlp_output_file_name:str = "SU2_MLP"

    # Intermediate history update window settings.
    history_plot_window = None 
    history_plot_axes = None
    history_epochs = []
    history_loss = []
    history_val_loss = []

    _stagnation_tolerance:float = 1e-11 
    _stagnation_patience:int = 200
    _verbose:int = 1

    callback_every:int = 20 

    scaler_function_name:str = "robust"
    scaler_function_x = scaler_functions[scaler_function_name]()
    scaler_function_y = scaler_functions[scaler_function_name]()

    weights_initializer:str = "he_uniform"

    _loaded_custom_weights:bool = False
    _custom_weights:list[np.ndarray[float]] = None 
    _custom_biases:list[np.ndarray[float]] = None 
    
    def __init__(self):
        """Initiate MLP trainer object.
        """
        return    
        
    def SetVerbose(self, verbose_level:int=1):
        """Set the trainer output verbose level. 0 = no outputs, 1 = output per epoch, 2 = output for every batch.

        :param verbose_level: output verbose level, defaults to 1
        :type verbose_level: int, optional
        :raises Exception: if the verbose level is not 0, 1, or 2
        """
        if verbose_level < 0 or verbose_level > 2:
            raise Exception("Verbose level should be 0, 1, or 2.")
        self._verbose = int(verbose_level)
        return 
    
    def SetFigFormat(self, fig_format:str="png"):
        """Set the format by which to save any generated images during training.

        :param fig_format: image file format, defaults to "png"
        :type fig_format: str, optional
        """

        self._figformat = fig_format
        return 
    
    def SetTrainFileHeader(self, train_filepathname:str):
        """Set a custom MLP train file path header.

        :param train_filepathname: MLP train data file header plus path.
        :type train_filepathname: str
        """
        self._filedata_train = train_filepathname
        return 
    
    def SetMLPFileHeader(self, mlp_fileheader:str="MLP_SU2"):
        """Set the SU2 MLP output file header.

        :param mlp_fileheader: header of the .mlp output file, defaults to "MLP_SU2"
        :type mlp_fileheader: str, optional
        """
        self._mlp_output_file_name = mlp_fileheader
        return 
    
    def SetScaler(self, scaler_function:str="robust",scale_x:float=None,scale_y:float=None,offset_x:float=None,offset_y:float=None):
        if scaler_function not in scaler_functions.keys():
            raise Exception("Train data scaling function should be one of the following: "+",".join(f for f in scaler_functions.keys()))
        self.scaler_function_name = scaler_function
        self.scaler_function_x = scaler_functions[scaler_function]()
        self.scaler_function_y = scaler_functions[scaler_function]()
        return
    
    def SetInitializer(self, initializer_function:str="he_uniform"):
        self.weights_initializer = initializer_function
        return 
    
    def SetSaveDir(self, save_dir_in:str):
        """Define directory in which trained MLP information is saved.

        :param save_dir_in: main directory in which to save the trained models and outputs.
        :type save_dir_in: str
        """
        self._save_dir = save_dir_in
        return 
    
    def SetModelIndex(self, idx_input:int):
        """Define model index under which MLP info is saved.

        :param idx_input: MLP model index
        :type idx_input: int
        """
        self._model_index = idx_input
        return 
    
    def SetNEpochs(self, n_input:int=DefaultProperties.N_epochs):
        """Set number of training epochs

        :param n_input: epoch count.
        :type n_input: int
        :raises Exception: if the specified number of epochs is lower than zero.
        """
        if n_input <= 0:
            raise Exception("Epoch count should be higher than zero.")
        self._n_epochs = n_input
        return 
    
    def SetActivationFunction(self, name_function:str=DefaultProperties.activation_function):
        """Define the hidden layer activation function name.

        :param name_function: hidden layer activation function name, defaults to "gelu"
        :type name_function: str, optional
        :raises Exception: if the option is not in the list of supported activation functions.
        """
        if name_function not in activation_function_names_options:
            raise Exception("Activation function not supported")
        self._activation_function_name = name_function 
        self._i_activation_function = activation_function_names_options.index(name_function)
        self._activation_function = activation_function_options[self._i_activation_function]
        return 
    
    def SetDeviceKind(self, kind_device:str):
        """Define computational hardware on which to train the network.

        :param kind_device: device kind (should be "CPU" or "GPU")
        :type kind_device: str
        :raises Exception: if specified device is neither "CPU" or "GPU".
        """
        if kind_device != "CPU" and kind_device != "GPU":
            raise Exception("Device should be \"CPU\" or \"GPU\"")
        self._kind_device = kind_device
        return 
    
    def SetDeviceIndex(self, device_index:int):
        """Define device index on which to train (CPU core or GPU card).

        :param device_index: CPU node or GPU card index to use for training.
        :type device_index: int
        """
        self._device_index = device_index
        return 
    
    def SetControllingVariables(self, x_vars:list[str]):
        """Specify MLP input or controlling variable names.

        :param x_vars: list of controlling variable names on which to train the MLP's.
        :type x_vars: list[str]
        """
        self._controlling_vars = []
        for var in x_vars:
            self._controlling_vars.append(var)
        return 
    
    def SetTrainVariables(self, train_vars:list[str]):
        """Specify MLP input or controlling variable names.

        :param x_vars: list of controlling variable names on which to train the MLP's.
        :type x_vars: list[str]
        """
        self._train_vars = []
        for var in train_vars:
            self._train_vars.append(var)
        return 
    
    def SetLRDecay(self, lr_decay:float=DefaultProperties.learning_rate_decay):
        """Specify learning rate decay parameter for exponential decay scheduler.

        :param lr_decay: learning rate decay factor.
        :type lr_decay: float
        :raises Exception: if specified learning rate decay factor is not between zero and one.
        """
        if lr_decay < 0 or lr_decay > 1.0:
            raise Exception("Learning rate decay factor should be between zero and one, not "+str(lr_decay))
        self._lr_decay = lr_decay
        return 
    
    def SetAlphaExpo(self, alpha_expo:float=DefaultProperties.init_learning_rate_expo):
        """Specify exponent of initial learning rate for exponential decay scheduler.

        :param alpha_expo: initial learning rate exponent.
        :type alpha_expo: float
        :raises Exception: if specified exponent is higher than zero.
        """
        if alpha_expo > 0:
            raise Exception("Initial learning rate exponent should be below zero.")
        self._alpha_expo = alpha_expo
        return 
    
    def SetBatchExpo(self, batch_expo:int=DefaultProperties.batch_size_exponent):
        """Specify exponent of mini-batch size.

        :param batch_expo: mini-batch exponent (base 2) to be used during training.
        :type batch_expo: int
        :raises Exception: if the specified exponent is lower than zero.
        """
        if batch_expo < 0:
            raise Exception("Mini-batch exponent should be higher than zero.")
        self._batch_expo = batch_expo
        return 
    
    def SetHiddenLayers(self, layers_input:list[int]=DefaultProperties.hidden_layer_architecture):
        """Define hidden layer architecture.

        :param layers_input: list of neuron count per hidden layer.
        :type layers_input: list[int]
        :raises Exception: if any of the supplied neuron counts is lower or equal to zero.
        """
        self._hidden_layers = []
        for NN in layers_input:
            if NN <=0:
                raise Exception("Neuron count in hidden layers should be higher than zero.")
            self._hidden_layers.append(NN)
        return 
    
    def EvaluateMLP(self, input_data_norm:np.ndarray):
        """Evaluate MLP for a given set of normalized input data.

        :param input_data_norm: array of normalized controlling variable data.
        :type input_data_norm: np.ndarray
        :raises Exception: if the number of columns in the input data does not equate the number of controlling variables.
        :return: MLP output data for the given inputs.
        :rtype: np.ndarray
        """
        if np.shape(input_data_norm)[1] != len(self._controlling_vars):
            raise Exception("Number of input variables ("+str(np.shape(input_data_norm)[1]) + ") \
                            does not equal the MLP input dimension ("+str(len(self._controlling_vars))+")")
        return np.zeros(1)
    
    def SaveWeights(self):
        """Save the weights of the current network as numpy arrays.
        """
        for iW, w in enumerate(self._weights):
            np.save(self._save_dir + "/Model_"+str(self._model_index) + "/W_"+str(iW)+".npy", w.numpy(), allow_pickle=True)
            np.save(self._save_dir + "/Model_"+str(self._model_index) + "/b_"+str(iW)+".npy", self._biases[iW].numpy(), allow_pickle=True)
        return
    
    def SetDecaySteps(self):
        self._decay_steps = int(float(self._Np_train) / (2**self._batch_expo))
        return 
    
    def RestartTraining(self):
        """Restart the training process.
        """
        self._restart_training = True 
        return 
    
    def Train_MLP(self):
        """Commence network training.
        """
        return 
    
    def GetCostParameter(self):
        """Retrieve MLP evaluation cost parameter.
        :return: MLP evaluation cost parameter.
        :rtype: float
        """
        return self._cost_parameter
    
    def GetTestScore(self):
        """Retrieve loss value of test set upon training finalization.
        :return: loss value of test set.
        :rtype: float
        """
        return self._test_score
    
    def GetWeights(self):
        """Get the trainable weights from the network.

        :return: list of weight arrays.
        :rtype: list[np.ndarray]
        """
        return [w.numpy() for w in self._weights]
    
    def GetBiases(self):
        """Get the trainable biases from the network.

        :return: list of bias arrays.
        :rtype: list[np.ndarray]
        """
        return [b.numpy() for b in self._biases] 
    
    def PlotR2Data(self):
        """Plot the MLP prediction in the form of R2-plots w.r.t. the reference data, and along each of the 
        normalized controlling variables.
        """

        # Evaluate the MLP on the input test set data.
        X_test = self.scaler_function_x.inverse_transform(self._X_test_norm)
        pred_data_dim = self.EvaluateMLP(X_test)
        pred_data_norm = self.scaler_function_y.transform(self.TransformData(pred_data_dim))
        ref_data_norm = self._Y_test_norm 

        pred_data_norm[np.isnan(pred_data_norm)] = 10.0
        # Generate and save R2-plots for each of the output parameters.
        fig, axs = plt.subplots(nrows=len(self._train_vars), ncols=1,figsize=[5,5*len(self._train_vars)])
        for iVar in range(len(self._train_vars)):
            R2_score = r2_score(ref_data_norm[:, iVar], pred_data_norm[:, iVar])
            if len(self._train_vars) == 1:
                ax = axs
            else:
                ax = axs[iVar]
            ref_min, ref_max = min(ref_data_norm[:,iVar]), max(ref_data_norm[:, iVar])
            ax.plot([ref_min, ref_max],[ref_min,ref_max],'r')
            ax.plot(ref_data_norm[:, iVar], pred_data_norm[:, iVar], 'k.')
            ax.grid()
            ax.set_title(self._train_vars[iVar] + ": %.3e" % R2_score)
            ax.set_xlabel("Labeled data",fontsize=20)
            ax.set_ylabel("Predicted data",fontsize=20)
        fig.savefig(self._save_dir + "/Model_"+str(self._model_index) + "/R2."+self._figformat, format=self._figformat, bbox_inches='tight')
        plt.close(fig)

        # Generate and save the MLP predictions along each of the controlling variable ranges.
        fig, axs = plt.subplots(nrows=len(self._train_vars), ncols=len(self._controlling_vars),figsize=[5*len(self._controlling_vars),5*len(self._train_vars)])
        for iVar in range(len(self._train_vars)):
            for iInput in range(len(self._controlling_vars)):
                if len(self._train_vars) == 1:
                    ax = axs[iInput]
                else:
                    ax = axs[iVar, iInput]
                ax.plot(self._X_test_norm[:, iInput],ref_data_norm[:, iVar],'k.')
                ax.plot(self._X_test_norm[:, iInput], pred_data_norm[:, iVar], 'r.')
                ax.grid()
                ax.set_title(self._train_vars[iVar])
                ax.set_xlabel(self._controlling_vars[iInput])
        fig.savefig(self._save_dir + "/Model_"+str(self._model_index) + "/Predict_along_CVs."+self._figformat, format=self._figformat, bbox_inches='tight')
        plt.close(fig)
        return 
    
    def TransformData(self, Y_untransformed):
        return Y_untransformed
    
    def TransformData_Inv(self, Y_transformed):
        return Y_transformed 
    
    def GetTrainData(self):
        """
        Read train, test, and validation data sets according to flameletAI configuration and normalize data sets
        with a feature range of 0-1.
        """

        self._X_train_norm, self._X_test_norm, self._X_val_norm,\
        self._Y_train_norm, self._Y_test_norm, self._Y_val_norm = self.GetTrainTestValData()

        if self.scaler_function_name == "minmax":
            X_min,X_max = self.scaler_function_x.data_min_, self.scaler_function_x.data_max_
            Y_min,Y_max = self.scaler_function_y.data_min_, self.scaler_function_y.data_max_
            self._X_scale = X_max - X_min 
            self._X_offset = X_min 
            self._Y_scale = Y_max - Y_min 
            self._Y_offset = Y_min 
        elif self.scaler_function_name == "robust":
            self._X_scale = self.scaler_function_x.scale_
            self._Y_scale = self.scaler_function_y.scale_
            self._X_offset = self.scaler_function_x.center_
            self._Y_offset = self.scaler_function_y.center_
        elif self.scaler_function_name == "standard":
            self._X_scale = self.scaler_function_x.scale_
            self._Y_scale = self.scaler_function_y.scale_
            self._X_offset = self.scaler_function_x.mean_
            self._Y_offset = self.scaler_function_y.mean_

        self._Np_train = np.shape(self._X_train_norm)[0]
        self._Np_test = np.shape(self._X_test_norm)[0]
        self._Np_val = np.shape(self._X_val_norm)[0]
        
        
        return 
    
    def GetTrainTestValData(self, x_vars:list[str]=None, y_vars:list[str]=None, scaler_x=None, scaler_y=None):

        if x_vars == None:
            x_vars = [c for c in self._controlling_vars]
            scaler_x = self.scaler_function_x
        if y_vars == None:
            y_vars = [c for c in self._train_vars]
            scaler_y = self.scaler_function_y

        MLPData_filepath = self._filedata_train
        
        is_nullMLP = ("null" in y_vars)
        
        if self._verbose > 0:
            print("Reading train, test, and validation data...")
        
        if is_nullMLP:
            X_full, _ = GetReferenceData(MLPData_filepath + "_full.csv", x_vars, [],dtype=self._dt_np)
            Y_full = np.zeros(np.shape(X_full)[0])
        else:
            X_full, Y_full = GetReferenceData(MLPData_filepath + "_full.csv", x_vars, y_vars,dtype=self._dt_np)
        Y_full = self.TransformData(Y_full)

        scaler_x.fit(X_full)
        scaler_y.fit(Y_full)

        # Free up memory
        del X_full
        del Y_full

        if is_nullMLP:
            return 
        else:
            X_train, Y_train = GetReferenceData(MLPData_filepath + "_train.csv", x_vars, y_vars,dtype=self._dt_np)
            X_test, Y_test = GetReferenceData(MLPData_filepath + "_test.csv", x_vars, y_vars,dtype=self._dt_np)
            X_val, Y_val = GetReferenceData(MLPData_filepath + "_val.csv",x_vars, y_vars,dtype=self._dt_np)
            if self._verbose > 0:
                print("Done!")

            Y_train = self.TransformData(Y_train)
            Y_test = self.TransformData(Y_test)
            Y_val = self.TransformData(Y_val)

            # Normalize train, test, and validation controlling variables
            X_train_norm = scaler_x.transform(X_train)
            X_test_norm = scaler_x.transform(X_test)
            X_val_norm = scaler_x.transform(X_val)

            # Normalize train, test, and validation data
            Y_train_norm = scaler_y.transform(Y_train)
            Y_test_norm = scaler_y.transform(Y_test)
            Y_val_norm = scaler_y.transform(Y_val)

            return X_train_norm, X_test_norm, X_val_norm, Y_train_norm, Y_test_norm, Y_val_norm
    
    def GetScalerFunctionParams(self):
        if self.scaler_function_name == "minmax":
            scaler_function_vals_in = [[mi,ma] for mi, ma in zip(self.scaler_function_x.data_min_, self.scaler_function_x.data_max_)]
            scaler_function_vals_out = [[mi,ma] for mi, ma in zip(self.scaler_function_y.data_min_, self.scaler_function_y.data_max_)]
        else:
            scaler_function_vals_in = [[mi,ma] for mi, ma in zip(self._X_offset, self._X_scale)]
            scaler_function_vals_out = [[mi,ma] for mi, ma in zip(self._Y_offset, self._Y_scale)]
        return self.scaler_function_name, scaler_function_vals_in, scaler_function_vals_out
    
    def write_SU2_MLP(self, file_out:str):
        """Write the network to ASCII format readable by the MLPCpp module in SU2.

        :param file_out: MLP output path and file name.
        :type file_out: str
        """
        weights = [w.numpy() for w in self._weights]
        biases = [b.numpy() for b in self._biases]
        if self.scaler_function_name == "minmax":
            scaler_function_vals_in = [[mi,ma] for mi, ma in zip(self.scaler_function_x.data_min_, self.scaler_function_x.data_max_)]
            scaler_function_vals_out = [[mi,ma] for mi, ma in zip(self.scaler_function_y.data_min_, self.scaler_function_y.data_max_)]
        else:
            scaler_function_vals_in = [[mi,ma] for mi, ma in zip(self._X_offset, self._X_scale)]
            scaler_function_vals_out = [[mi,ma] for mi, ma in zip(self._Y_offset, self._Y_scale)]
        return write_SU2_MLP(file_out, weights=weights,\
                                       biases=biases, \
                                       activation_function_name=self._activation_function_name,\
                                       train_vars=self._train_vars,\
                                       controlling_vars=self._controlling_vars,\
                                       scaler_function=self.scaler_function_name,\
                                       scaler_function_vals_in=scaler_function_vals_in,\
                                       scaler_function_vals_out=scaler_function_vals_out,\
                                       additional_header_info_function=self.add_additional_header_info)
        # n_layers = len(self._weights)+1

        # # Select trimmed weight matrices for output.
        # weights_for_output = self._weights
        # biases_for_output = self._biases

        # # Opening output file
        # fid = open(file_out+'.mlp', 'w+')
        # fid.write("<header>\n\n")
        
        # self.add_additional_header_info(fid)
        # # Writing number of neurons per layer
        # fid.write('[number of layers]\n%i\n\n' % n_layers)
        # fid.write('[neurons per layer]\n')
        # activation_functions = []

        # for iLayer in range(n_layers-1):
        #     if iLayer == 0:
        #         activation_functions.append('linear')
        #     else:
        #         activation_functions.append(self._activation_function_name)
        #     n_neurons = np.shape(weights_for_output[iLayer])[0]
        #     fid.write('%i\n' % n_neurons)
        # fid.write('%i\n' % len(self._train_vars))

        # activation_functions.append('linear')

        # # Writing the activation function for each layer
        # fid.write('\n[activation function]\n')
        # for iLayer in range(n_layers):
        #     fid.write(activation_functions[iLayer] + '\n')

        # # Writing the input and output names
        # fid.write('\n[input names]\n')
        # for input in self._controlling_vars:
        #         fid.write(input + '\n')
        
        # fid.write('\n[input regularization method]\n%s\n' % self.scaler_function_name)

        # fid.write('\n[input normalization]\n')
        # for i in range(len(self._controlling_vars)):
        #     if self.scaler_function_name == "minmax":
        #         fid.write('%+.16e\t%+.16e\n' % (self.scaler_function_x.data_min_[i], self.scaler_function_x.data_max_[i]))
        #     else:
        #         fid.write('%+.16e\t%+.16e\n' % (self._X_offset[i], self._X_scale[i]))

        # fid.write('\n[output names]\n')
        # for output in self._train_vars:
        #     fid.write(output+'\n')
        
        # fid.write('\n[output regularization method]\n%s\n' % self.scaler_function_name)

        # fid.write('\n[output normalization]\n')
        # for i in range(len(self._train_vars)):
        #     if self.scaler_function_name == "minmax":
        #         fid.write('%+.16e\t%+.16e\n' % (self.scaler_function_y.data_min_[i], self.scaler_function_y.data_max_[i]))
        #     else:
        #         fid.write('%+.16e\t%+.16e\n' % (self._Y_offset[i], self._Y_scale[i]))

        # fid.write("\n</header>\n")
        # # Writing the weights of each layer
        # fid.write('\n[weights per layer]\n')
        # for W in weights_for_output:
        #     fid.write("<layer>\n")
        #     for i in range(np.shape(W)[0]):
        #         fid.write("\t".join("%+.16e" % float(w) for w in W[i, :]) + "\n")
        #     fid.write("</layer>\n")
        
        # # Writing the biases of each layer
        # fid.write('\n[biases per layer]\n')
        
        # # Input layer biases are set to zero
        # fid.write("\t".join("%+.16e" % 0 for _ in self._controlling_vars) + "\n")

        # #for B in self.biases:
        # for B in biases_for_output:
        #     try:
        #         fid.write("\t".join("%+.16e" % float(b) for b in B.numpy()) + "\n")
        #     except:
        #         fid.write("\t".join("%+.16e" % float(B.numpy())) + "\n")

        # fid.close()
        # return 
    
    def add_additional_header_info(self, fid):
        return 
    
    def Save_Relevant_Data(self):
        """Save network performance characteristics in text file and write SU2 MLP input file.
        """
        fid = open(self._save_dir + "/Model_"+str(self._model_index)+"/MLP_performance.txt", "w+")
        fid.write("Training time[minutes]: %+.3e\n" % self._train_time)
        fid.write("Validation score: %+.16e\n" % self._test_score)
        fid.write("Total neuron count:  %i\n" % np.sum(np.array(self._hidden_layers)))
        fid.write("Evaluation time[seconds]: %+.3e\n" % (self._test_time))
        fid.write("Evaluation cost parameter: %+.6e\n" % (self._cost_parameter))
        fid.write("Alpha exponent: %+.16e\n" % self._alpha_expo)
        fid.write("Learning rate decay: %+.16e\n" % self._lr_decay)
        fid.write("Batch size exponent: %i\n" % self._batch_expo)
        fid.write("Decay steps: %i\n" % self._decay_steps)
        fid.write("Activation function index: %i\n" % self._i_activation_function)
        fid.write("Number of hidden layers: %i\n" % len(self._hidden_layers))
        fid.write("Architecture: " + " ".join(str(n) for n in self._hidden_layers) + "\n")
        fid.close()

        self.write_SU2_MLP(self._save_dir + "/Model_"+str(self._model_index)+"/"+self._mlp_output_file_name)
        return 
    
    def Plot_Architecture(self):
        """Visualize the MLP architecture by plotting the neurons in each of the hidden layers.
        """
        fig = plt.figure()
        plt.plot(np.zeros(len(self._controlling_vars)), np.arange(len(self._controlling_vars)) - 0.5*len(self._controlling_vars), 'bo')
        for i in range(len(self._hidden_layers)):
            plt.plot((i+1)*np.ones(int(self._hidden_layers[i])), np.arange(int(self._hidden_layers[i])) - 0.5*self._hidden_layers[i], 'ko')
        plt.plot((i+2)*np.ones(len(self._train_vars)), np.arange(len(self._train_vars)) - 0.5*len(self._train_vars), 'go')
        plt.axis('equal')
        fig.savefig(self._save_dir +"/Model_"+str(self._model_index) + "/architecture."+self._figformat,format=self._figformat, bbox_inches='tight')
        plt.close(fig)
        return 
    
    def CustomCallback(self):
        return 
    
    def Plot_and_Save_History(self):
        return
    
    def SetWeightsBiases(self, custom_weights:list[np.ndarray[float]], custom_biases:list[np.ndarray[float]]):
        self._custom_weights = custom_weights.copy()
        self._custom_biases = custom_biases.copy()
        self._loaded_custom_weights = True
        return 
    
class TensorFlowFit(MLPTrainer):
    _model:keras.models.Sequential
    history_epochs = []
    history_loss = []
    history_val_loss = []

    def __init__(self):
        MLPTrainer.__init__(self)
        return 
    
    # Construct MLP based on architecture information
    def DefineMLP(self):

        # Construct MLP on specified device
        with tf.device("/"+self._kind_device+":"+str(self._device_index)):

            # Initialize sequential model
            self._model = keras.models.Sequential()
            self.history = None 

            # Add input layer
            self._model.add(keras.layers.Input([len(self._controlling_vars, )]))

            # Add hidden layersSetTrainFileHeader
            iLayer = 0
            while iLayer < len(self._hidden_layers):
                self._model.add(keras.layers.Dense(self._hidden_layers[iLayer], activation=self._activation_function_name, kernel_initializer=self.weights_initializer))
                iLayer += 1
            
            # Add output layer
            self._model.add(keras.layers.Dense(len(self._train_vars), activation='linear'))

            if self._loaded_custom_weights:
                weights_and_biases = []
                for w,b in zip(self._custom_weights,self._custom_biases):
                    weights_and_biases.append(w)
                    weights_and_biases.append(b)
                self._model.set_weights(weights_and_biases)

            # Define learning rate schedule and optimizer
            self.SetDecaySteps()
            #self._decay_steps = 1e4
            _lr_schedule = keras.optimizers.schedules.ExponentialDecay(10**self._alpha_expo, decay_steps=self._decay_steps,
                                                                    decay_rate=self._lr_decay, staircase=False)
            opt = keras.optimizers.Adam(learning_rate=_lr_schedule, beta_1=0.9, beta_2=0.999, epsilon=1e-8, amsgrad=False) 

            # Compile model on device
            self._model.compile(optimizer=opt, loss="mean_squared_error", metrics=["mape"])
        return 
    
    def EvaluateMLP(self,input_data_dim):
        input_data_norm = self.scaler_function_x.transform(input_data_dim)
        pred_data_norm = self._model.predict(input_data_norm, verbose=0)
        pred_data_dim = self.scaler_function_y.inverse_transform(pred_data_norm)
        pred_data_dim_out = self.TransformData_Inv(pred_data_dim)
        return pred_data_dim_out
    
    # Initialize MLP training 
    def Train_MLP(self):
        """Commence network training.
        """

        self.history_epochs = []
        self.history_loss=[]
        self.history_val_loss=[]
        
        # Read train,test, and validation data.
        self.GetTrainData()

        # Pre-processing of model before training.
        self.DefineMLP()

        self._weights = []
        self._biases = []
        for layer in self._model.layers:
            self._weights.append(layer.weights[0])
            self._biases.append(layer.weights[1])


        if not os.path.isdir(self._save_dir + "/Model_"+str(self._model_index)):
            os.mkdir(self._save_dir + "/Model_"+str(self._model_index))
        
        self.Plot_Architecture()

        with tf.device("/"+self._kind_device+":"+str(self._device_index)):
            t_start = time.time()
            StagnationStop = tf.keras.callbacks.EarlyStopping(monitor="loss", \
                                                      min_delta=self._stagnation_tolerance, \
                                                      patience=self._stagnation_patience,\
                                                      start_from_epoch=1,\
                                                      mode="min",\
                                                      verbose=self._verbose)
            self.history = self._model.fit(self._X_train_norm, self._Y_train_norm, \
                                          epochs=self._n_epochs, \
                                          batch_size=2**self._batch_expo,\
                                          verbose=self._verbose, \
                                          validation_data=(self._X_val_norm, self._Y_val_norm), \
                                          shuffle=True,\
                                          callbacks=[StagnationStop, self.PlotCallback(self)])
            t_end = time.time()
            # Store training time in minutes
            self._train_time = (t_end - t_start) / 60

            t_start = time.time()
            self._test_score = self._model.evaluate(self._X_test_norm, self._Y_test_norm, verbose=0)[0]
            t_end = time.time()
            self._test_time = (t_end - t_start)
            self._weights = []
            self._biases = []
            for layer in self._model.layers:
                self._weights.append(layer.weights[0])
                self._biases.append(layer.weights[1])
            self.SaveWeights()

        self._cost_parameter = 0
        for w in self._weights:
            self._cost_parameter += np.shape(w)[0] * np.shape(w)[1]
        return 

    def Plot_and_Save_History(self):
        """Plot the training convergence trends.
        """

        with open(self._save_dir + "/Model_"+str(self._model_index)+"/TrainingHistory.csv", "w+") as fid:
            fid.write("epoch,loss,validation_loss\n")
            csvWriter = csv.writer(fid, delimiter=',')
            csvWriter.writerows(np.array([self.history_epochs, self.history_loss, self.history_val_loss]).T)

        fig = plt.figure(figsize=[10,10])
        ax = plt.axes()
        ax.plot(self.history_loss, 'b', linewidth=2,label=r'Training score')
        ax.plot(self.history_val_loss, 'r', linewidth=2,label=r"Validation score")
        ax.grid()
        ax.set_yscale('log')
        ax.legend(fontsize=20)
        ax.set_xlabel(r"Iteration[-]", fontsize=20)
        ax.set_ylabel(r"Training loss function [-]", fontsize=20)
        ax.set_title(r"Direct Training History", fontsize=22)
        ax.tick_params(axis='both', which='major', labelsize=18)
        fig.savefig(self._save_dir + "/Model_"+str(self._model_index)+ "/History_Plot_Direct."+self._figformat, format=self._figformat, bbox_inches='tight')
        plt.close(fig)
        return 
        
    class PlotCallback(tf.keras.callbacks.Callback):
            FitClass = None
            def __init__(self, TensorFlowFit:MLPTrainer):
                self.FitClass = TensorFlowFit

            def on_epoch_end(self, epoch, logs=None):
                self.FitClass.history_epochs.append(epoch)
                self.FitClass.history_loss.append(logs["loss"])
                self.FitClass.history_val_loss.append(logs["val_loss"])
                
                if (epoch+1) % self.FitClass.callback_every == 0:
                    self.FitClass.CustomCallback()
                    self.FitClass.Plot_and_Save_History()

                return super().on_epoch_end(epoch, logs)

class CustomTrainer(MLPTrainer):
    """MLP trainer class with full customization of loss functions.
    """

    _dt = tf.float32            # tensor data type used to cast training data.
    _trainable_hyperparams:list[tf.Variable]=[] # MLP hyper-parameters to adjust during training.
    _optimizer = None   # optimization algorithm used to adjust the MLP hyper-parameters during training.
    _lr_schedule = None # learning rate decay schedule.
    _train_name:str = ""    # MLP network name.

    # Training stagnation parameters.
    __keep_training:bool = True 
    __stagnation_iter:int = 0

    _include_regularization:bool = False
    _regularization_param:float = 1e-5

    def __init__(self):
        MLPTrainer.__init__(self)
        return
    
    def SetWeights(self, weights_input:list[np.ndarray]):
        """Manually set the network weights values.

        :param weights_input: list of network weights values.
        :type weights_input: list[np.ndarray]
        """

        self._weights = []
        for W in weights_input:
            self._weights.append(tf.Variable(tf.cast(W, self._dt), self._dt))
        return 
    
    def SetBiases(self, biases_input:list[np.ndarray]):
        """Manually set the network biases values.

        :param biases_input: list of network bias values.
        :type biases_input: list[np.ndarray]
        """

        self._biases = []
        for b in biases_input:
            self._biases.append(tf.Variable(tf.cast(b,self._dt), self._dt))
        return 
    
    def InitializeWeights_and_Biases(self):
        """Initialize network weights and biases using He-invariance initialization.
        """

        self._weights = []
        self._biases = []

        NN = [len(self._controlling_vars)]
        for N in self._hidden_layers:
            NN.append(N)
        NN.append(len(self._train_vars))

        if self.weights_initializer == "he_uniform":
            initializer = HeUniform()
        elif self.weights_initializer == "random_uniform":
            initializer = RandomUniform()
        if self._loaded_custom_weights:
            for i in range(len(self._custom_weights)):
                if self._loaded_custom_weights:
                    self._weights.append(tf.Variable(tf.cast(self._custom_weights[i], self._dt),self._dt))
                    self._biases.append(tf.Variable(tf.cast(self._custom_biases[i], self._dt),self._dt))
        else:     
            for i in range(len(NN)-1):
                self._weights.append(tf.Variable(tf.cast((initializer(shape=(NN[i],NN[i+1]))),self._dt),self._dt))
                self._biases.append(tf.Variable(tf.cast((initializer(shape=(NN[i+1],))),self._dt),self._dt))             
            
        return 
    
    
    @tf.function
    def CollectVariables(self):
        """Define trainable hyper-parameters.
        """
        self._trainable_hyperparams = []
        for W in self._weights:
            self._trainable_hyperparams.append(W)
        for b in self._biases:
            self._trainable_hyperparams.append(b)
        return 
    
    def SetOptimizer(self):
        """Set weights and biases training algorithm.
        """

        # Set number of gradient descend steps.
        self.SetDecaySteps()

        # Define learning rate decay schedule.
        self._lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(10**self._alpha_expo, decay_steps=self._decay_steps,
                                                                            decay_rate=self._lr_decay, staircase=False)

        self._optimizer = tf.keras.optimizers.Adam(self._lr_schedule, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False)
        return 
    
    @tf.function
    def ComputeLayerInput(self, x:tf.constant, W:tf.Variable, b:tf.Variable):
        X = tf.matmul(x, W) + b 
        return X 
    
    @tf.function
    def _MLP_Evaluation(self, x_norm:tf.Tensor):
        w = self._weights
        b = self._biases
        Y = x_norm 
        for iLayer in range(len(w)-1):
            Y = self._activation_function(self.ComputeLayerInput(Y, w[iLayer], b[iLayer]))
        Y = self.ComputeLayerInput(Y, w[-1], b[-1])
        return Y 
    
    def EvaluateMLP(self, input_data_dim:np.ndarray):
        """Evaluate MLP for a given set of normalized input data.

        :param input_data_norm: array of normalized controlling variable data.
        :type input_data_norm: np.ndarray
        :raises Exception: if the number of columns in the input data does not equate the number of controlling variables.
        :return: MLP output data for the given inputs.
        :rtype: np.ndarray
        """
        if np.shape(input_data_dim)[1] != len(self._controlling_vars):
            raise Exception("Number of input variables ("+str(np.shape(input_data_dim)[1]) + ") \
                            does not equal the MLP input dimension ("+str(len(self._controlling_vars))+")")
        input_data_norm = self.scaler_function_x.transform(input_data_dim)
        input_data_tf = tf.constant(input_data_norm, self._dt)
        output_data_norm = self._MLP_Evaluation(input_data_tf).numpy()
        output_data_dim = self.scaler_function_y.inverse_transform(output_data_norm)
        output_data_dim_transformed = self.TransformData_Inv(output_data_dim)
        return output_data_dim_transformed
    
    @tf.function
    def mean_square_error(self, y_true, y_pred):
        return tf.reduce_mean(tf.pow(y_pred - y_true, 2), axis=0)
    
    @tf.function
    def Compute_Direct_Error(self, x_norm:tf.constant, y_label_norm:tf.constant):
        y_pred_norm = self._MLP_Evaluation(x_norm)
        return tf.reduce_mean(tf.pow(y_pred_norm - y_label_norm, 2),axis=0)
    
    @tf.function 
    def TrainingLoss_error(self, x_norm:tf.constant, y_label_norm:tf.constant):
        pred_error_outputs = self.Compute_Direct_Error(x_norm, y_label_norm)
        mean_pred_error = tf.reduce_mean(pred_error_outputs)
        if self._include_regularization:
            reg_error = self.RegularizationLoss()
            mean_pred_error += reg_error
        return mean_pred_error 
    
    @tf.function
    def ComputeGradients_Direct_Error(self, x_norm:tf.constant, y_label_norm:tf.constant):
        with tf.GradientTape() as tape:
            tape.watch(self._trainable_hyperparams)
            y_norm_loss = self.TrainingLoss_error(x_norm, y_label_norm)
            total_loss = y_norm_loss
            grads_loss = tape.gradient(total_loss, self._trainable_hyperparams)
            
        return total_loss, grads_loss
    
    @tf.function
    def ComputeJacobian_Direct_Error(self, x_norm, y_label_norm):
        with tf.GradientTape() as tape:
            tape.watch(x_norm)
            y_norm_loss = self.Compute_Direct_Error(x_norm, y_label_norm)
            jac = tape.jacobian(y_norm_loss, x_norm)
        return y_norm_loss, jac
    
    @tf.function
    def Train_Step(self, x_norm_batch, y_label_norm_batch):
        y_norm_loss, grads_loss = self.ComputeGradients_Direct_Error(x_norm_batch, y_label_norm_batch)
        self._optimizer.apply_gradients(zip(grads_loss, self._trainable_hyperparams))
        
        return y_norm_loss 
    
    def Train_MLP(self):
        """Commence network training.
        """
        self.Preprocessing()

        self.LoopEpochs()

        self.PostProcessing()
        return 
    
    def Preprocessing(self):
        if not os.path.isdir(self._save_dir + "/Model_"+str(self._model_index)):
            os.mkdir(self._save_dir + "/Model_"+str(self._model_index))
        
        self.Plot_Architecture()

        self._cost_parameter = 0
        for w in self._weights:
            self._cost_parameter += np.shape(w)[0] * np.shape(w)[1]

        # Read train,test, and validation data.
        self.GetTrainData()
        # Read data from training variables according to network outputs.
        self.CollectVariables()

        # Pre-process model before training.
        self.SetOptimizer()

        self.PrepareValidationHistory()
        return 
    
    def PrepareValidationHistory(self):
        self.val_loss_history=[]
        for _ in self._train_vars:
            self.val_loss_history.append([])
        return 
    
    def SetTrainBatches(self):
        train_batches = tf.data.Dataset.from_tensor_slices((self._X_train_norm, self._Y_train_norm)).batch(2**self._batch_expo)
        return train_batches
    
    def LoopEpochs(self):
        t_start = time.time()
        worst_error = 1e32
        self._i_epoch = 0
        train_batches = self.SetTrainBatches()
        while (self._i_epoch < self._n_epochs) and self.__keep_training:
            self.LoopBatches(train_batches=train_batches)

            val_loss = self.ValidationLoss()
            
            if (self._i_epoch + 1) % self.callback_every == 0:
                self.TestLoss()
                self.CustomCallback()
            
            worst_error = self.__CheckEarlyStopping(val_loss, worst_error)

            self.PrintEpochInfo(self._i_epoch, val_loss)
            self._i_epoch += 1
        t_end = time.time()
        self._train_time = (t_end - t_start)/60
        return 
    
    def PrintEpochInfo(self, i_epoch, val_loss):
        if self._verbose > 0:
            print("Epoch: ", str(i_epoch), " Validation loss: ", ", ".join("%s : %.8e" % (s, v) for s, v in zip(self._train_vars, val_loss)))
        return 
    
    def LoopBatches(self, train_batches):
        for x_norm_batch, y_norm_batch in train_batches:
            self.Train_Step(x_norm_batch, y_norm_batch)
        return
    
    def ValidationLoss(self):
        val_loss = self.Compute_Direct_Error(tf.constant(self._X_val_norm, self._dt), tf.constant(self._Y_val_norm, self._dt))
        for iVar in range(len(self._train_vars)):
            self.val_loss_history[iVar].append(val_loss[iVar])
        return val_loss
    
    @tf.function
    def RegularizationLoss(self):
        reg_loss = 0.0
        for w in self._trainable_hyperparams:
            reg_loss += self._regularization_param*tf.reduce_sum(tf.pow(w, 2))
        
        return reg_loss 
    
    def TestLoss(self):

        t_start = time.time()
        self._test_score = tf.reduce_mean(self.Compute_Direct_Error(tf.constant(self._X_test_norm, self._dt), tf.constant(self._Y_test_norm, self._dt))).numpy()
        t_end = time.time()
        self._test_time = (t_end - t_start)/60
        return 
    
    def Plot_and_Save_History(self, vars=None):
        """Plot the training convergence trends.
        """

        if vars == None:
            vars_to_plot=self._train_vars 
        else:
            vars_to_plot = vars 

        fig = plt.figure(figsize=[10,10])
        ax = plt.axes()
        H = np.array(self.val_loss_history)
        for i in range(len(vars_to_plot)):
            ax.plot(H[i,:], label=r"Validation score "+vars_to_plot[i])
        ax.grid()
        ax.set_yscale('log')
        ax.legend(fontsize=20)
        ax.set_xlabel(r"Iteration[-]", fontsize=20)
        ax.set_ylabel(r"Training loss function [-]", fontsize=20)
        ax.set_title(r""+self._train_name+r" Training History", fontsize=22)
        ax.tick_params(axis='both', which='major', labelsize=18)
        fig.savefig(self._save_dir + "/Model_"+str(self._model_index)+ "/History_Plot_"+self._train_name+"."+self._figformat, format=self._figformat, bbox_inches='tight')
        plt.close(fig)

        with open(self._save_dir + "/Model_"+str(self._model_index)+"/TrainingHistory.csv", "w+") as fid:
            fid.write("epoch,"+ ",".join(("validation_loss_"+var for var in vars_to_plot))+"\n")
            epochs = np.arange(np.shape(H)[1])
            H_concat = np.vstack((epochs, H)).T 
            csvWriter = csv.writer(fid, delimiter=',')
            csvWriter.writerows(H_concat)

        return 
    
    def __CheckEarlyStopping(self, val_loss, worst_error):
        current_error = tf.reduce_max(val_loss)
        if current_error < worst_error - self._stagnation_tolerance:
            self.__keep_training = True 
            self.__stagnation_iter = 0
            
            worst_error = current_error
        else:
            self.__stagnation_iter += 1
            if self.__stagnation_iter > self._stagnation_patience and self._verbose > 0:
                self.__keep_training = False 
                print("Early stopping due to stagnation")
        return worst_error 
    
    def PostProcessing(self):
        self.TestLoss()
        self.CustomCallback()
        self.SaveWeights()
        self.Save_Relevant_Data()
        return 
    
class PhysicsInformedTrainer(CustomTrainer):

    _Y_state_full:np.ndarray 
    _Y_state_train_norm:np.ndarray 
    _Y_state_test_norm:np.ndarray
    _Y_state_val_norm:np.ndarray 

    _scaler_state = RobustScaler()
    _Y_state_scale:np.ndarray = None 
    _Y_state_offset:np.ndarray = None 

    _X_boundary_norm:np.ndarray[float] = None 
    _Y_boundary_norm:np.ndarray[float] = None 
    
    _state_vars:list[str]

    vals_lambda:list[float] = None
    projection_vals:list[tf.constant] = None 

    projection_arrays:list[np.ndarray] = None 
    target_arrays:list[np.ndarray] = None 
    idx_PIvar:list[int] = None 
    lamba_labels:list[str] = None 

    _boundary_data_file:str = None 


    _enable_boundary_loss:bool=True
    _include_boundary_loss:bool=True
    _boundary_loss_patience:int = 10
    _N_bc:int=None 

    _train_step_type:str="Jacobi"
    __train_step_type_options:list[str] = ["Gauss-Seidel","Jacobi"]
    j_gradient_update:int = 0
    lambda_history:list = []
    update_lambda_every_iter:int = 10
    def __init__(self):
        CustomTrainer.__init__(self)
        return 
    
    def GetStateTrainData(self):
        """
        Read train, test, and validation data sets according to flameletAI configuration and normalize data sets
        with a feature range of 0-1.
        """

        _, _, _, self._Y_state_train_norm, \
            self._Y_state_test_norm, self._Y_state_val_norm = self.GetTrainTestValData(self._controlling_vars, \
                                                                                       self._state_vars, \
                                                                                       self.scaler_function_x, \
                                                                                       self._scaler_state)
        if self.scaler_function_name == "minmax":
            Y_state_max, Y_state_min = self._scaler_state.data_max_, self._scaler_state.data_min_
            self._Y_state_scale = Y_state_max - Y_state_min
            self._Y_state_offset = Y_state_min
        elif self.scaler_function_name == "robust":
            self._Y_state_scale = self._scaler_state.scale_
            self._Y_state_offset = self._scaler_state.center_
        elif self.scaler_function_name == "standard":
            self._Y_state_scale = self._scaler_state.scale_
            self._Y_state_offset = self._scaler_state.mean_

        return 
    
    def SetBoundaryDataFile(self, boundary_file_name:str):
        self._boundary_data_file = boundary_file_name
        return 
    
    def GetBoundaryData(self, y_vars=None):
        """Load flamelet data equilibrium boundary data.
        """
        if y_vars == None:
            y_vars = self._train_vars
        # Load controlling and train variables from boundary data.
        X_boundary, Y_boundary = GetReferenceData(self._boundary_data_file, x_vars=self._controlling_vars, train_variables=y_vars,dtype=self._dt_np)
        
        
        # Normalize controlling and labeled data with respect to domain data.
        self._X_boundary_norm = self.scaler_function_x.transform(X_boundary)
        self._Y_boundary_norm = self.scaler_function_y.transform(Y_boundary)

        return 
    
    def SetTrainStepType(self, train_step_type:str="Jacobi"):
        if train_step_type not in self.__train_step_type_options:
            raise Exception("Weights update step type should be one of the following : "+ ",".join(s for s in self.__train_step_type_options))
        self._train_step_type = train_step_type
        return 
    
    def SetDecaySteps(self):
        super().SetDecaySteps()
        if self._train_step_type=="Gauss-Seidel":
            self._decay_steps *= len(self._state_vars)
        return 
    
    def PreprocessPINNVars(self):
        
        return 
    
    def SetScaler(self, scaler_function: str = "robust"):
        super().SetScaler(scaler_function)
        self._scaler_state = scaler_functions[scaler_function]()
        return 
    
    def GetTrainData(self):
        super().GetTrainData()
        self.GetBoundaryData()
        self.CollectPIVars()
        self.GetStateTrainData()
        self._Y_state_scale_tf = tf.cast(self._Y_state_scale, self._dt)
        self._Y_state_offset_tf = tf.cast(self._Y_state_offset, self._dt)
        return 
    
    def Plot_and_Save_History(self, vars=None):
        super().Plot_and_Save_History(vars=self._state_vars)
        return 
    
    def PrepareValidationHistory(self):
        self.val_loss_history=[]
        for _ in self._state_vars:
            self.val_loss_history.append([])
        return 
    
    def SetTrainBatches(self):
        train_batches_domain = tf.data.Dataset.from_tensor_slices((self._X_train_norm, self._Y_state_train_norm)).batch(2**self._batch_expo)
        domain_batches_list = [b for b in train_batches_domain]

        batch_size_train = 2**self._batch_expo

        # Collect projection array data.
        p_concatenated = tf.stack([tf.constant(p, dtype=self._dt) for p in self.projection_arrays],axis=2)
        
        # Collect target projection gradient data.
        Y_target_concatenated = tf.stack([tf.constant(t, dtype=self._dt) for t in self.target_arrays], axis=1)

        # Collect boundary controlling variable data.
        X_boundary_tf = tf.constant(self._X_boundary_norm, dtype=self._dt)

        # Forumulate batches.
        batches_concat = tf.data.Dataset.from_tensor_slices((X_boundary_tf, p_concatenated, Y_target_concatenated)).batch(batch_size_train)
        batches_concat_list = [b for b in batches_concat]

        # Re-size boundary data batches to that of the domain batches such that both data can be evaluated simultaneously during training.
        Nb_boundary = len(batches_concat_list)
        batches_concat_list_resized = [batches_concat_list[i % Nb_boundary] for i in range(len(domain_batches_list))]

        return (domain_batches_list, batches_concat_list_resized)
    
    @tf.function
    def ComputeFirstOrderDerivatives(self, x_norm_input:tf.constant,idx_out:int=0):
        x_var = x_norm_input
        with tf.GradientTape(watch_accessed_variables=False) as tape_con:
            tape_con.watch(x_var)
            Y_norm = self._MLP_Evaluation(x_var)
            dY_norm = tape_con.gradient(tf.gather(Y_norm, indices=idx_out, axis=1), x_var)
        return Y_norm, dY_norm
    
    @tf.function
    def ComputeSecondOrderDerivatives(self, x_norm_input:tf.constant,iVar:int=0, jVar:int=0):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(x_norm_input)
            Y_norm, dY_norm = self.ComputeFirstOrderDerivatives(x_norm_input, iVar)
            d2Y_norm = tape.gradient(tf.gather(dY_norm, indices=jVar, axis=1), x_norm_input)
        return Y_norm, dY_norm, d2Y_norm
    
    @tf.function
    def update_lambda(self, grads_direct, grads_ub, val_lambda_old):
        max_grad_direct = 0.0
        for g in grads_direct:
            max_grad_direct = tf.maximum(max_grad_direct, tf.reduce_max(tf.abs(g)))

        mean_grad_ub = 0.0
        for g_ub in grads_ub:
            mean_grad_ub += val_lambda_old * tf.reduce_mean(tf.abs(g_ub))
        mean_grad_ub /= len(self._weights)

        lambda_prime = max_grad_direct / (mean_grad_ub + 1e-7)
        val_lambda_new = 0.9 * val_lambda_old + 0.1 * lambda_prime
        if tf.math.is_nan(val_lambda_new):
            val_lambda_new = 1.0
        return val_lambda_new

    def CollectPIVars(self):
        self.projection_arrays = []
        self.target_arrays = []
        self.vals_lambda = []
        self.idx_PIvar = []
        self.lamba_labels = []
        val_lambda_default = tf.constant(1.0,dtype=self._dt)
        return val_lambda_default
    
    def LoopEpochs(self):
        self.j_gradient_update = 0
        self.lambda_history.clear()
        return super().LoopEpochs()
    
    def LoopBatches(self, train_batches):
        """Loop over domain and boundary batches for each epoch.

        :param train_batches: tuple of domain and boundary data batches.
        :type train_batches: tuple
        """
        domain_batches = train_batches[0]
        boundary_batches = train_batches[1]
        vals_lambda = self.vals_lambda.copy()
        for batch_domain, batch_boundary in zip(domain_batches, boundary_batches):

            # Extract domain batch data.
            X_domain_batch = batch_domain[0]
            Y_domain_batch = batch_domain[1]
            # Extract boundary batch data.
            X_boundary_batch = batch_boundary[0]
            P_boundary_batch = batch_boundary[1]
            Yt_boundary_batch = batch_boundary[2]

            if self._i_epoch > self._boundary_loss_patience and self._enable_boundary_loss:
                self._include_boundary_loss = True
            else:
                self._include_boundary_loss = False
            # Run train step and adjust weights.
            if self._train_step_type == "Gauss-Seidel":
                self.Train_Step_Gauss_Seidel(X_domain_batch, Y_domain_batch, X_boundary_batch,P_boundary_batch, Yt_boundary_batch, vals_lambda, self._include_boundary_loss)
            else:
                self.Train_Step(X_domain_batch, Y_domain_batch, X_boundary_batch,P_boundary_batch, Yt_boundary_batch, vals_lambda, self._include_boundary_loss)

            # Update boundary condition penalty values.
            if ((self.j_gradient_update + 1)%self.update_lambda_every_iter ==0) and self._include_boundary_loss:
                vals_lambda_updated = self.UpdateLambdas(X_domain_batch, Y_domain_batch, X_boundary_batch, P_boundary_batch, Yt_boundary_batch, vals_lambda)
                self.vals_lambda = [v for v in vals_lambda_updated]

            self.j_gradient_update += 1
        if self._enable_boundary_loss:  
            self.lambda_history.append([lamb.numpy() for lamb in self.vals_lambda])

        return 
    
    @tf.function
    def Train_Step(self, X_domain_batch:tf.constant, Y_domain_batch:tf.constant, \
                   X_boundary_batch:tf.constant, P_boundary_batch:tf.constant, Yt_boundary_batch:tf.constant, vals_lambda:list[tf.constant], include_boundary:bool):

        # Compute training loss for the current batch and extract HP sensitivities.
        batch_loss, sens_batch = self.Train_sensitivity_function(X_domain_batch, Y_domain_batch, X_boundary_batch, P_boundary_batch, Yt_boundary_batch, vals_lambda,include_boundary)
        
        # Update network weigths and biases.
        self.UpdateWeights(sens_batch)

        return batch_loss
    
    @tf.function
    def Train_Step_Gauss_Seidel(self, X_domain_batch:tf.constant, Y_domain_batch:tf.constant, \
                   X_boundary_batch:tf.constant, P_boundary_batch:tf.constant, Yt_boundary_batch:tf.constant, vals_lambda:list[tf.constant], include_boundary:bool):
        for iVar in range(len(self._state_vars)):
            with tf.GradientTape() as tape:
                tape.watch(self._trainable_hyperparams)
                Y_state_pred = self.EvaluateState(X_domain_batch)[:, iVar]
                Y_state_pred_norm = (Y_state_pred - self._Y_state_offset[iVar]) / self._Y_state_scale[iVar]
                Y_state_ref= Y_domain_batch[:, iVar]
                state_error_norm = tf.reduce_mean(tf.pow((Y_state_pred_norm - Y_state_ref), 2))
                grads_state_error = tape.gradient(state_error_norm, self._trainable_hyperparams)
            self.UpdateWeights(grads_state_error)
            if include_boundary:
                with tf.GradientTape() as tape:
                    boundary_loss = 0.0
                    for iBc in range(self._N_bc):
                        bc_loss = self.ComputeNeumannPenalty(X_boundary_batch, Yt_boundary_batch, P_boundary_batch,iBc)
                        boundary_loss += vals_lambda[iBc] * bc_loss
                    grads_boundary_error = tape.gradient(boundary_loss, self._trainable_hyperparams)
                    self.UpdateWeights(grads_boundary_error)
        return 
    
    @tf.function
    def UpdateWeights(self, grads):
        self._optimizer.apply_gradients(zip(grads, self._trainable_hyperparams))
        return
    
    @tf.function 
    def UpdateLambdas(self, X_domain_batch:tf.constant, Y_domain_batch:tf.constant,\
                            X_boundary_batch:tf.constant,  \
                            P_boundary_batch:tf.constant, Yt_boundary_batch:tf.constant, vals_lambda_old:list[tf.constant]):
        """Update boundary condition penalty values.

        :return: list of updated penalty values.
        :rtype: list[tf.constant]
        """

        _, grads_domain, _, grads_bc_list = self.ComputeGradients(X_domain_batch, Y_domain_batch, X_boundary_batch, P_boundary_batch, Yt_boundary_batch)
        
        vals_lambda_new = []
        for iBc, lambda_old in enumerate(vals_lambda_old):
            lambda_new = self.update_lambda(grads_domain, grads_bc_list[iBc], lambda_old)
            vals_lambda_new.append(lambda_new)
        return vals_lambda_new
    
    @tf.function 
    def EvaluateState(self, X_norm:tf.constant):
        return 
    
    @tf.function 
    def ComputeStateError(self, X_label_norm:tf.constant,Y_state_label_norm:tf.constant):
        Y_state_pred = self.EvaluateState(X_label_norm)
        Y_state_pred_norm = (Y_state_pred - self._Y_state_offset_tf)/self._Y_state_scale_tf
        pred_error = tf.reduce_mean(tf.pow(Y_state_pred_norm - Y_state_label_norm, 2), axis=0)
        return pred_error
    
    @tf.function 
    def ComputeGradients_State_error(self, Y_state_label_norm:tf.constant, X_label_norm:tf.constant):
        with tf.GradientTape() as tape:
            tape.watch(self._trainable_hyperparams)
            state_loss = self.ComputeStateError(X_label_norm, Y_state_label_norm)

            grads_state = tape.gradient(tf.reduce_mean(state_loss), self._trainable_hyperparams)

        return state_loss, grads_state 
    
    @tf.function
    def ComputeGradients(self, X_domain_batch, Y_domain_batch, X_boundary_batch,  P_boundary_batch, Yt_boundary_batch):
        y_domain_loss, grads_domain = self.ComputeGradients_State_error(X_label_norm=X_domain_batch, Y_state_label_norm=Y_domain_batch)

        grads_bc_list = []
        loss_bc_list = []
        for iBC in range(self._N_bc):
            boundary_loss, grads_boundary_loss = self.ComputeGradients_Boundary_Error(X_boundary_batch, Yt_boundary_batch,P_boundary_batch,iBC)
            
            grads_bc_list.append(grads_boundary_loss)
            loss_bc_list.append(boundary_loss)
        return y_domain_loss, grads_domain, loss_bc_list, grads_bc_list
    
    @tf.function
    def ComputeGradients_Boundary_Error(self, X_boundary_batch, Yt_boundary_batch,P_boundary_batch,iVar):
        with tf.GradientTape() as tape:
            tape.watch(self._trainable_hyperparams)
            neumann_penalty_var = self.ComputeNeumannPenalty(X_boundary_batch, Yt_boundary_batch, P_boundary_batch,iVar)
            grads_neumann = tape.gradient(neumann_penalty_var, self._trainable_hyperparams)
        return neumann_penalty_var, grads_neumann
    
    @tf.function
    def ComputeNeumannPenalty(self, x_norm_boundary:tf.constant, dy_norm_boundary_target:tf.constant,precon_gradient:tf.constant, iVar:int=0): 
        """Neumann penalty function for projected MLP Jacobians along boundary data.

        :param x_norm_boundary: boundary data controlling variable values.
        :type x_norm_boundary: tf.constant
        :param y_norm_boundary_target: labeled boundary data.
        :type y_norm_boundary_target: tf.constant
        :param dy_norm_boundary_target: target projected gradient.
        :type dy_norm_boundary_target: tf.constant
        :param precon_gradient: MLP Jacobian pre-conditioner.
        :type precon_gradient: tf.constant
        :param iVar: boundary condition index, defaults to 0
        :type iVar: int, optional
        :return: direct evaluation and Neumann penalty values.
        :rtype: tf.constant
        """

        # Evaluate MLP Jacobian on boundary data.
        _, dy_pred_norm = self.ComputeFirstOrderDerivatives(x_norm_boundary, self.idx_PIvar[iVar])

        # Project Jacobian along boundary data according to penalty function.
        project_dy_pred_norm = tf.reduce_sum(tf.multiply(precon_gradient[:,:,iVar], dy_pred_norm), axis=1)

        # Compute direct and Neumann penalty values.
        penalty = tf.reduce_mean(tf.pow(project_dy_pred_norm - dy_norm_boundary_target[:, iVar], 2))
        return penalty
    
    @tf.function 
    def Train_loss_function(self, X_domain_batch, Y_domain_batch, X_boundary_batch, P_boundary_batch, Yt_boundary_batch, vals_lambda, include_boundary):
        
        domain_loss = self.TrainingLoss_error(X_domain_batch, Y_domain_batch)
        total_loss = domain_loss
        bc_loss = 0.0
        if include_boundary:
            bc_loss = self.ComputeBCLoss(X_boundary_batch, Yt_boundary_batch, P_boundary_batch, vals_lambda)

        total_loss += bc_loss
        return [total_loss, domain_loss, bc_loss]
    
    
    @tf.function 
    def ComputeBCLoss(self, X_boundary_batch, Yt_boundary_batch, P_boundary_batch, vals_lambda):
        boundary_loss = 0.0
        for iBc in range(self._N_bc):
            boundary_loss += vals_lambda[iBc] * self.ComputeNeumannPenalty(X_boundary_batch, Yt_boundary_batch, P_boundary_batch,iBc)
        return boundary_loss 
    
    @tf.function 
    def Train_sensitivity_function(self, X_domain_batch, Y_domain_batch, X_boundary_batch, P_boundary_batch, Yt_boundary_batch, vals_lambda, include_boundary):
        with tf.GradientTape() as tape:
            tape.watch(self._trainable_hyperparams)
            train_losses = self.Train_loss_function(X_domain_batch, Y_domain_batch, X_boundary_batch, P_boundary_batch, Yt_boundary_batch, vals_lambda,include_boundary)
            total_loss = train_losses[0]
            domain_loss = train_losses[1]
            boundary_loss = train_losses[2]
            grads_loss = tape.gradient(total_loss, self._trainable_hyperparams)
        return [total_loss, domain_loss, boundary_loss], grads_loss 
    
    @tf.function 
    def TrainingLoss_error(self, x_norm:tf.constant, y_state_label_norm:tf.constant):
        pred_error_outputs = self.ComputeStateError(x_norm, y_state_label_norm)
        mean_pred_error = tf.reduce_mean(pred_error_outputs)
        return mean_pred_error
    
    def ValidationLoss(self):
        val_error_state = self.ComputeStateError(tf.cast(self._X_val_norm, dtype=self._dt), tf.cast(self._Y_state_val_norm,dtype=self._dt))
        val_error_state = val_error_state.numpy()
        for iVar in range(len(self._state_vars)):
            self.val_loss_history[iVar].append(val_error_state[iVar])
        return val_error_state
    
    def TestLoss(self):
        test_error_state = self.ComputeStateError(tf.cast(self._X_test_norm, dtype=self._dt), tf.cast(self._Y_state_test_norm,dtype=self._dt))
        self.state_test_loss = test_error_state.numpy()
        self._test_score = np.average(self.state_test_loss)
        return test_error_state
    
    def PlotLambdaHistory(self):
        fig = plt.figure(figsize=[10,10])
        ax = plt.axes()
        for iBc in range(self._N_bc):
            ax.plot([self.lambda_history[i][iBc] for i in range(len(self.lambda_history))], label=self.lamba_labels[iBc])
        ax.grid()
        ax.set_xlabel(r"Epoch [-]",fontsize=20)
        ax.set_ylabel(r"Lambda value[-]",fontsize=20)
        ax.tick_params(which='both',labelsize=18)
        ax.legend(fontsize=20)
        ax.set_title(r"Neumann penalty modifier history", fontsize=20)
        fig.savefig(self._save_dir+"/Model_"+str(self._model_index)+"/Lambda_history."+self._figformat,format=self._figformat,bbox_inches='tight')
        plt.close(fig)
        return 
    
    def CustomCallback(self):
        if self._enable_boundary_loss:
            self.PlotLambdaHistory()
        return 
    
    def EnableBCLoss(self, enable_bc_loss:bool=True):
        self._enable_boundary_loss = enable_bc_loss
        return 
    
class TrainMLP:
    """Class for training MLP architectures
    """

    _Config:Config = None 
    _trainer_direct:TensorFlowFit = None 

    architecture:list[int] = DefaultProperties.hidden_layer_architecture  # Hidden layer architecture.
    alpha_expo:float = DefaultProperties.init_learning_rate_expo # Initial learning rate exponent (base 10)
    lr_decay:float = DefaultProperties.learning_rate_decay # Learning rate decay parameter.
    batch_expo:int = DefaultProperties.batch_size_exponent      # Mini-batch exponent (base 2)
    activation_function:str = DefaultProperties.activation_function    # Activation function name applied to hidden layers.

    n_epochs:int = DefaultProperties.N_epochs # Number of epochs to train for.
    save_dir:str        # Directory to save trained networks in.

    device:str = "CPU"      # Hardware to train on.
    process_index:int = 0   # Hardware index.

    current_iter:int=0
    verbose:int=0
    _test_score:float        # MLP evaluation score on test set.
    _cost_parameter:float    # MLP evaluation cost parameter.

    _train_file_header:str = None 
    main_save_dir:str = "./"

    _fig_format:str = "png"
    _scaler:str = "robust"
    __set_custom_weights:bool = False
    __weights_custom:list[np.ndarray[float]] = None 
    __biases_custom:list[np.ndarray[float]] = None 
    
    def __init__(self, Config_in:Config):
        """Define TrainMLP instance and prepare MLP trainer with
        default settings.

        :param Config: Config_FGM object describing the flamelet data manifold.
        :type Config: Config_FGM
        :param group_idx: MLP output group index, defaults to 0
        :type group_idx: int, optional
        :raises Exception: if MLP output group index is undefined by flameletAI configuration.
        """

        self._Config=Config_in
        self.alpha_expo = self._Config.GetAlphaExpo()
        self.lr_decay = self._Config.GetLRDecay()
        self.batch_expo = self._Config.GetBatchExpo()
        self.activation_function = self._Config.GetActivationFunction()
        self.architecture = []
        for n in self._Config.GetHiddenLayerArchitecture():
            self.architecture.append(n)
            
        self.main_save_dir = self._Config.GetOutputDir()
        
        # Define MLPTrainer object with default settings (currently only supports TensorFlowFit)
        self._train_file_header = self._Config.GetOutputDir()+"/"+self._Config.GetConcatenationFileHeader()
        self.SynchronizeTrainer()
        pass

    def SynchronizeTrainer(self):
        """Synchronize all MLP trainer settings with locally stored settings.
        """
        self.worker_dir = self.main_save_dir + ("/Worker_%i/" % self.process_index)
        self._trainer_direct.SetModelIndex(self.current_iter)
        self._trainer_direct.SetSaveDir(self.worker_dir)

        self._trainer_direct.SetDeviceKind(self.device)
        self._trainer_direct.SetDeviceIndex(self.process_index)

        self._trainer_direct.SetNEpochs(self.n_epochs)
        self._trainer_direct.SetActivationFunction(self.activation_function)
        self._trainer_direct.SetAlphaExpo(self.alpha_expo)
        self._trainer_direct.SetLRDecay(self.lr_decay)
        self._trainer_direct.SetBatchExpo(self.batch_expo)
        self._trainer_direct.SetHiddenLayers(self.architecture)
        if self.__set_custom_weights:
            self._trainer_direct.SetWeightsBiases(self.__weights_custom,self.__biases_custom)
        self._trainer_direct.SetTrainFileHeader(self._train_file_header)
        self._trainer_direct.SetVerbose(self.verbose)
        self._trainer_direct.SetFigFormat(self._fig_format)
        self._trainer_direct.SetScaler(self._scaler)
        return 
    
    def SetSaveDir(self, save_dir:str):
        """Define directory in which to save trained MLP data.

        :param save_dir: file path directory in which to save trained MLP data.
        :type save_dir: str
        :raises Exception: if specified directory doesn't exist.
        """
        if not os.path.isdir(save_dir):
            raise Exception("Specified directory is not present on current machine.")
        self.main_save_dir = save_dir
        return 
    
    def SetNEpochs(self, n_epochs:int=DefaultProperties.N_epochs):
        """Set the number of epochs to train for.

        :param n_epochs: Number of training epoch, defaults to 250.
        :type n_epochs: int
        :raises Exception: provided number of epochs is negative.
        """
        if n_epochs <= 0:
            raise Exception("Number of epochs should be at least one.")
        self.n_epochs = n_epochs
        self.SynchronizeTrainer()
        return 
    
    def SetHiddenLayers(self, NN_hidden_layers:list[int]=DefaultProperties.hidden_layer_architecture):
        """Define hidden layer architecture.

        :param NN_hidden_layers: list with neuron count for each hidden layer.
        :type NN_hidden_layers: list[int]
        :raises Exception: if any of the entries is lower or equal to zero.
        """
        self.architecture = []
        for NN in NN_hidden_layers:
            if NN <= 0:
                raise Exception("Number of neurons in hidden layers should be higher than zero.")
            self.architecture.append(NN)
        self.__set_custom_weights = False
        self.SynchronizeTrainer()
        return 
    
    def SetBatchExpo(self, batch_expo_in:int=DefaultProperties.batch_size_exponent):
        """Set the mini-batch size exponent.

        :param batch_expo_in: exponent of mini-batch size (base 2)
        :type batch_expo_in: int
        :raises Exception: if entry is lower than or equal to zero.
        """
        if batch_expo_in <=0:
            raise Exception("Mini-batch exponent should be higher than zero.")
        self.batch_expo = batch_expo_in
        self.SynchronizeTrainer()
        return 
    
    def SetActivationFunction(self, activation_function_name:str=DefaultProperties.activation_function):
        """Define hidden layer activation function.

        :param activation_function_name: activation function name.
        :type activation_function_name: str
        :raises Exception: if activation function is not in the list of supported functions.
        """
        if activation_function_name not in activation_function_names_options:
            raise Exception("Activation function name should be one of the following: "\
                             + ",".join(p for p in activation_function_names_options))
        self.activation_function = activation_function_name
        self.SynchronizeTrainer()
        return 
    
    def SetAlphaExpo(self, alpha_expo_in:float=DefaultProperties.init_learning_rate_expo):
        """Define initial learning rate exponent.

        :param alpha_expo_in: initial learning rate exponent.
        :type alpha_expo_in: float
        :raises Exception: if entry is higher than zero.
        """
        if alpha_expo_in > 0:
            raise Exception("Initial learning rate exponent should be negative.")
        self.alpha_expo = alpha_expo_in
        self.SynchronizeTrainer()
        return 
    
    def SetLRDecay(self, lr_decay_in:float=DefaultProperties.learning_rate_decay):
        """Define learning rate decay parameter.

        :param lr_decay_in: learning rate decay parameter.
        :type lr_decay_in: float
        :raises Exception: if entry is not between zero and one.
        """
        if lr_decay_in < 0 or lr_decay_in > 1:
            raise Exception("Learning rate decay parameter should be between zero and one.")
        
        self.lr_decay = lr_decay_in
        self.SynchronizeTrainer()
        return 
        
    def SetTrainHardware(self, device:str, process:int=0):
        """Define hardware to train on.

        :param device: device, should be "CPU" or "GPU"
        :type device: str
        :param process: device index, defaults to 0
        :type process: int, optional
        :raises Exception: if device is anything other than "GPU" or "CPU" or device index is negative.
        """
        if device != "CPU" and device != "GPU":
            raise Exception("Device should be GPU or CPU.")
        if process < 0:
            raise Exception("Device index should be positive.")
        
        self.device=device
        self.process_index = process
        self.SynchronizeTrainer()
        return
    
    def SetWeightsBiases(self, weights_input_custom:list[np.ndarray[float]],biases_input_custom:list[np.ndarray[float]]):
        if len(weights_input_custom) != len(biases_input_custom):
            raise Exception("Weights and biases should be same size.")
        if len(weights_input_custom) != (len(self.architecture)+1):
            raise Exception("Weights should be compatible with hidden layer architecture.")
        self.__weights_custom = weights_input_custom.copy()
        self.__biases_custom = biases_input_custom.copy()
        self.__set_custom_weights = True 
        self.SynchronizeTrainer()
        return 
    
    def PrepareOutputDir(self):
        """Prepare output directory in which to save trained MLP data.
        """
        worker_idx = self.process_index
        if not os.path.isdir(self.main_save_dir):
            os.mkdir(self.main_save_dir)
        if not os.path.isdir(self.main_save_dir + "/Worker_"+str(worker_idx)):
            os.mkdir(self.main_save_dir + "/Worker_"+str(worker_idx))
            self.current_iter = 0
        else:
            try:
                fid = open(self.main_save_dir + "/Worker_"+str(worker_idx)+"/current_iter.txt", "r")
                line = fid.readline()
                fid.close()
                self.current_iter = int(line.strip()) + 1
            except:
                self.current_iter = 0
        self.SynchronizeTrainer()
        return 
    
    def CommenceTraining(self):
        """Initiate the training process.
        """
        self.PrepareOutputDir()
        self._trainer_direct.Train_MLP()
        self.TrainPostprocessing()

        fid = open(self.worker_dir + "/current_iter.txt", "w+")
        fid.write(str(self.current_iter) + "\n")
        fid.close()

        self._test_score = self._trainer_direct.GetTestScore()
        if np.isnan(self._test_score):
            self._test_score = 1e2
        self._cost_parameter = self._trainer_direct.GetCostParameter()
        return 
    
    def TrainPostprocessing(self):
        """Post-process MLP training by saving all relevant data/figures.
        """
        self._trainer_direct.Save_Relevant_Data()
        self._trainer_direct.Plot_and_Save_History()
        return 
    
    def GetCostParameter(self):
        """Get MLP evaluation cost parameter.

        :return: MLP cost parameter
        :rtype: float
        """
        return self._cost_parameter
    
    def GetTestScore(self):
        """Get MLP evaluation test score upon completion of training.

        :return: MLP evaluation test score.
        :rtype: float
        """
        return self._test_score
    
    def GetWeights(self):
        return self._trainer_direct.GetWeights()
    def GetBiases(self):
        return self._trainer_direct.GetBiases()
    
    def GetScalerFunctionParams(self):
        return self._trainer_direct.GetScalerFunctionParams()
    def GetControlVars(self):
        return self._trainer_direct._controlling_vars
    def GetTrainVars(self):
        return self._trainer_direct._train_vars
    
    def SetTrainFileHeader(self, fileheader:str):
        """Set a custom training data file header.

        :param fileheader: file path and name
        :type fileheader: str
        """
        self._train_file_header = fileheader
        self.SynchronizeTrainer()

        return 
    
    def SetVerbose(self, val_verbose:int=1):
        """Set verbose level during training. 0 means no information, 1 means minimal information every epoch, 2 means detailed information.

        :param val_verbose: verbose level (0, 1, or 2), defaults to 1
        :type val_verbose: int, optional
        """

        self.verbose = val_verbose
        self.SynchronizeTrainer()

        return                
    
    def SetFigFormat(self, fig_format:str="png"):
        self._fig_format = fig_format
        return 
    
    def SetScaler(self, scaler_name:str="robust"):
        if scaler_name not in scaler_functions.keys():
            raise Exception("Input-output scaler function should be one of the following: "+",".join(s for s in scaler_functions.keys()))
        self._scaler = scaler_name
        self._trainer_direct.SetScaler(scaler_name)
        return 
