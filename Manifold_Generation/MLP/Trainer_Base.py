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
import csv 
from Common.Config_base import Config
from Common.Properties import DefaultProperties 
from Common.CommonMethods import GetReferenceData

activation_function_names_options:list[str] = ["linear","elu","relu","tanh","exponential","gelu"]
activation_function_options = [tf.keras.activations.linear,
                            tf.keras.activations.elu,\
                            tf.keras.activations.relu,\
                            tf.keras.activations.tanh,\
                            tf.keras.activations.exponential,\
                            tf.keras.activations.gelu]
class MLPTrainer:
    # Base class for flamelet MLP trainer

    _n_epochs:int = DefaultProperties.N_epochs      # Number of epochs to train for.
    _alpha_expo:float = DefaultProperties.init_learning_rate_expo  # Alpha training exponent parameter.
    _lr_decay:float = DefaultProperties.learning_rate_decay      # Learning rate decay parameter.
    _batch_expo:int = DefaultProperties.batch_size_exponent     # Mini-batch size exponent.
    _decay_steps:int = 3e5

    _i_activation_function:int = 0   # Activation function index.
    _activation_function_name:str = "exponential"
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
    _X_train:np.ndarray = None 
    _Y_train:np.ndarray = None 
    _Np_train:int = 30000
    _X_train_norm:np.ndarray = None 
    _Y_train_norm:np.ndarray = None
    _X_test:np.ndarray = None 
    _Y_test:np.ndarray = None 
    _X_test_norm:np.ndarray = None 
    _Y_test_norm:np.ndarray = None 
    _X_val:np.ndarray = None 
    _Y_val:np.ndarray = None 
    _X_val_norm:np.ndarray = None 
    _Y_val_norm:np.ndarray = None 

    # Dataset normalization bounds.
    _X_min:np.ndarray = None
    _X_max:np.ndarray = None
    _Y_min:np.ndarray = None 
    _Y_max:np.ndarray = None

    # Hidden layers neuron count.
    _hidden_layers:list[int] = None

    # Weights and biases matrices.
    _weights:list[np.ndarray] = []
    _biases:list[np.ndarray] = []

    # Weights and biases with redundant neurons removed.
    _trimmed_weights:list[np.ndarray] = []
    _trimmed_biases:list[np.ndarray] = []

    _test_score:float = None       # Loss function on test set upon training termination.
    _cost_parameter:float = None   # Cost parameter of the trimmed network.

    _train_time:float = 0  # Training time in minutes.
    _test_time:float = 0   # Test set evaluation time in seconds.

    _save_dir:str = "/"  # Directory to save trained network information to.
    _figformat:str="png"
    optuna_trial = None 

    # Intermediate history update window settings.
    history_plot_window = None 
    history_plot_axes = None
    history_epochs = []
    history_loss = []
    history_val_loss = []

    _stagnation_tolerance:float = 1e-11 
    _stagnation_patience:int = 1000 
    _verbose:int = 1

    callback_every:int = 20 

    def __init__(self):
        """Initiate MLP trainer object.
        """
        return    
    
    def SetVerbose(self, verbose_level:int=1):
        if verbose_level < 0 or verbose_level > 2:
            raise Exception("Verbose level should be 0, 1, or 2.")
        self._verbose = int(verbose_level)
        return 
    
        
    def SetVerbose(self, verbose_level:int=1):
        if verbose_level < 0 or verbose_level > 2:
            raise Exception("Verbose level should be 0, 1, or 2.")
        self._verbose = int(verbose_level)
        return 
    
    def SetTrainFileHeader(self, train_filepathname:str):
        self._filedata_train = train_filepathname
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
    
    def SetActivationFunction(self, name_function:str="exponential"):
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
    
    def SetHiddenLayers(self, layers_input:list[int]=[DefaultProperties.NN_hidden]):
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
    
    # Load previously trained MLP (did not try this yet!)
    def LoadWeights(self):
        """Load the weights from a previous run in order to restart training from a previous training result.
        """
        self._weights = []
        self._biases = []
        for i in range(len(self.__model.layers)):
            loaded_W = np.load(self._save_dir + "/Model_"+str(self._model_index) + "/W_"+str(i)+".npy", allow_pickle=True)
            loaded_b = np.load(self._save_dir + "/Model_"+str(self._model_index) + "/b_"+str(i)+".npy", allow_pickle=True)
            self._weights.append(loaded_W)
            self._biases.append(loaded_b)
        return 
    
    def SaveWeights(self):
        """Save the weights of the current network as numpy arrays.
        """
        self._weights = []
        self._biases = []
        for layer in self.__model.layers:
            self._weights.append(layer.weights[0])
            self._biases.append(layer.weights[1])

        for iW, w in enumerate(self._weights):
            np.save(self._save_dir + "/Model_"+str(self._model_index) + "/W_"+str(iW)+".npy", w, allow_pickle=True)
            np.save(self._save_dir + "/Model_"+str(self._model_index) + "/b_"+str(iW)+".npy", self._biases[iW], allow_pickle=True)
        return
    
    def SetDecaySteps(self):
        self._decay_steps = int(self._n_epochs * self._Np_train / (2**self._batch_expo))
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
    
    def SaveWeights(self):
        """Save weight arrays as numpy arrays.
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
        return self._weights
    
    def GetBiases(self):
        """Get the trainable biases from the network.

        :return: list of bias arrays.
        :rtype: list[np.ndarray]
        """
        return self._biases 
    
    def PlotR2Data(self):
        """Plot the MLP prediction in the form of R2-plots w.r.t. the reference data, and along each of the 
        normalized controlling variables.
        """

        # Evaluate the MLP on the input test set data.
        pred_data_norm = self.EvaluateMLP(self._X_test_norm)
        ref_data_norm = self._Y_test_norm 

        # Generate and save R2-plots for each of the output parameters.
        fig, axs = plt.subplots(nrows=len(self._train_vars), ncols=1,figsize=[5,5*len(self._train_vars)])
        for iVar in range(len(self._train_vars)):
            R2_score = r2_score(ref_data_norm[:, iVar], pred_data_norm[:, iVar])
            if len(self._train_vars) == 1:
                axs.plot([0, 1],[0,1],'r')
                axs.plot(ref_data_norm[:, iVar], pred_data_norm[:, iVar], 'k.')
                axs.grid()
                axs.set_title(self._train_vars[iVar] + ": %.3e" % R2_score)
            else:
                axs[iVar].plot([0, 1],[0,1],'r')
                axs[iVar].plot(ref_data_norm[:, iVar], pred_data_norm[:, iVar], 'k.')
                axs[iVar].grid()
                axs[iVar].set_title(self._train_vars[iVar] + ": %.3e" % R2_score)
        fig.savefig(self._save_dir + "/Model_"+str(self._model_index) + "/R2.pdf", format='pdf', bbox_inches='tight')
        plt.close(fig)

        # Generate and save the MLP predictions along each of the controlling variable ranges.
        fig, axs = plt.subplots(nrows=len(self._train_vars), ncols=len(self._controlling_vars),figsize=[5*len(self._controlling_vars),5*len(self._train_vars)])
        for iVar in range(len(self._train_vars)):
            for iInput in range(len(self._controlling_vars)):
                if len(self._train_vars) == 1:
                    axs[iInput].plot(self._X_test_norm[:, iInput],ref_data_norm[:, iVar],'k.')
                    axs[iInput].plot(self._X_test_norm[:, iInput], pred_data_norm[:, iVar], 'r.')
                    axs[iInput].grid()
                    axs[iInput].set_title(self._train_vars[iVar])
                    axs[iInput].set_xlabel(self._controlling_vars[iInput])
                else:
                    axs[iVar, iInput].plot(self._X_test_norm[:, iInput],ref_data_norm[:, iVar],'k.')
                    axs[iVar, iInput].plot(self._X_test_norm[:, iInput], pred_data_norm[:, iVar], 'r.')
                    axs[iVar, iInput].grid()
                    axs[iVar, iInput].set_title(self._train_vars[iVar])
                    axs[iVar, iInput].set_xlabel(self._controlling_vars[iInput])
        fig.savefig(self._save_dir + "/Model_"+str(self._model_index) + "/Predict_along_CVs.pdf", format='pdf', bbox_inches='tight')
        plt.close(fig)
        return 
    
    def GetTrainData(self):
        """
        Read train, test, and validation data sets according to flameletAI configuration and normalize data sets
        with a feature range of 0-1.
        """

        MLPData_filepath = self._filedata_train
        
        print("Reading train, test, and validation data...")
        X_full, Y_full = GetReferenceData(MLPData_filepath + "_full.csv", self._controlling_vars, self._train_vars)
        
        self._X_train, self._Y_train = GetReferenceData(MLPData_filepath + "_train.csv", self._controlling_vars, self._train_vars)
        self._X_test, self._Y_test = GetReferenceData(MLPData_filepath + "_test.csv", self._controlling_vars, self._train_vars)
        self._X_val, self._Y_val = GetReferenceData(MLPData_filepath + "_val.csv", self._controlling_vars, self._train_vars)
        print("Done!")


        # Calculate normalization bounds of full data set
        self._X_min, self._X_max = np.min(X_full, 0), np.max(X_full, 0)
        self._Y_min, self._Y_max = np.min(Y_full, 0), np.max(Y_full, 0)

        # Free up memory
        del X_full
        del Y_full

        # Normalize train, test, and validation controlling variables
        self._X_train_norm = (self._X_train - self._X_min) / (self._X_max - self._X_min)
        self._X_test_norm = (self._X_test - self._X_min) / (self._X_max - self._X_min)
        self._X_val_norm = (self._X_val - self._X_min) / (self._X_max - self._X_min)

        # Normalize train, test, and validation data
        self._Y_train_norm = (self._Y_train - self._Y_min) / (self._Y_max - self._Y_min)
        self._Y_test_norm = (self._Y_test - self._Y_min) / (self._Y_max - self._Y_min)
        self._Y_val_norm = (self._Y_val - self._Y_min) / (self._Y_max - self._Y_min)
        return 
    
    
    def write_SU2_MLP(self, file_out:str):
        """Write the network to ASCII format readable by the MLPCpp module in SU2.

        :param file_out: MLP output path and file name.
        :type file_out: str
        """

        n_layers = len(self._weights)+1

        # Select trimmed weight matrices for output.
        weights_for_output = self._weights
        biases_for_output = self._biases

        # Opening output file
        fid = open(file_out+'.mlp', 'w+')
        fid.write("<header>\n\n")
        

        # Writing number of neurons per layer
        fid.write('[number of layers]\n%i\n\n' % n_layers)
        fid.write('[neurons per layer]\n')
        activation_functions = []

        for iLayer in range(n_layers-1):
            if iLayer == 0:
                activation_functions.append('linear')
            else:
                activation_functions.append(self._activation_function_name)
            n_neurons = np.shape(weights_for_output[iLayer])[0]
            fid.write('%i\n' % n_neurons)
        fid.write('%i\n' % len(self._train_vars))

        activation_functions.append('linear')

        # Writing the activation function for each layer
        fid.write('\n[activation function]\n')
        for iLayer in range(n_layers):
            fid.write(activation_functions[iLayer] + '\n')

        # Writing the input and output names
        fid.write('\n[input names]\n')
        for input in self._controlling_vars:
                fid.write(input + '\n')
        
        fid.write('\n[input normalization]\n')
        for i in range(len(self._controlling_vars)):
            fid.write('%+.16e\t%+.16e\n' % (self._X_min[i], self._X_max[i]))
        
        fid.write('\n[output names]\n')
        for output in self._train_vars:
            fid.write(output+'\n')
            
        fid.write('\n[output normalization]\n')
        for i in range(len(self._train_vars)):
            fid.write('%+.16e\t%+.16e\n' % (self._Y_min[i], self._Y_max[i]))

        fid.write("\n</header>\n")
        # Writing the weights of each layer
        fid.write('\n[weights per layer]\n')
        for W in weights_for_output:
            fid.write("<layer>\n")
            for i in range(np.shape(W)[0]):
                fid.write("\t".join("%+.16e" % float(w) for w in W[i, :]) + "\n")
            fid.write("</layer>\n")
        
        # Writing the biases of each layer
        fid.write('\n[biases per layer]\n')
        
        # Input layer biases are set to zero
        fid.write("\t".join("%+.16e" % 0 for _ in self._controlling_vars) + "\n")

        #for B in self.biases:
        for B in biases_for_output:
            try:
                fid.write("\t".join("%+.16e" % float(b) for b in B.numpy()) + "\n")
            except:
                fid.write("\t".join("%+.16e" % float(B.numpy())) + "\n")

        fid.close()
        return 
    
    def Save_Relevant_Data(self):
        """Save network performance characteristics in text file and write SU2 MLP input file.
        """
        fid = open(self._save_dir + "/Model_"+str(self._model_index)+"/MLP_performance.txt", "w+")
        fid.write("Training time[minutes]: %+.3e\n" % self._train_time)
        fid.write("Validation score: %+.16e\n" % self._test_score)
        fid.write("Total neuron count:  %i\n" % np.sum(np.array(self._hidden_layers)))
        fid.write("Evaluation time[seconds]: %+.3e\n" % (self._test_time))
        fid.write("Evaluation cost parameter: %+.3e\n" % (self._cost_parameter))
        fid.write("Alpha exponent: %+.4e\n" % self._alpha_expo)
        fid.write("Learning rate decay: %+.4e\n" % self._lr_decay)
        fid.write("Batch size exponent: %i\n" % self._batch_expo)
        fid.write("Decay steps: %i\n" % self._decay_steps)
        fid.write("Activation function index: %i\n" % self._i_activation_function)
        fid.write("Number of hidden layers: %i\n" % len(self._hidden_layers))
        fid.write("Architecture: " + " ".join(str(n) for n in self._hidden_layers) + "\n")
        fid.close()

        self.write_SU2_MLP(self._save_dir + "/Model_"+str(self._model_index)+"/MLP_entropy")
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
        fig.savefig(self._save_dir +"/Model_"+str(self._model_index) + "/architecture.png",format='png', bbox_inches='tight')
        plt.close(fig)
        return 
    
    def CustomCallback(self):
        return 
    
    def Plot_and_Save_History(self):
        return
    
class TensorFlowFit(MLPTrainer):
    __model:keras.models.Sequential
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
            self.__model = keras.models.Sequential()
            self.history = None 

            # Add input layer
            self.__model.add(keras.layers.Input([len(self._controlling_vars, )]))

            # Add hidden layersSetTrainFileHeader
            iLayer = 0
            while iLayer < len(self._hidden_layers):
                self.__model.add(keras.layers.Dense(self._hidden_layers[iLayer], activation=self._activation_function_name, kernel_initializer="he_uniform"))
                iLayer += 1
            
            # Add output layer
            self.__model.add(keras.layers.Dense(len(self._train_vars), activation='linear'))

            # Define learning rate schedule and optimizer
            #self.SetDecaySteps()
            self._decay_steps = 1e4
            __lr_schedule = keras.optimizers.schedules.ExponentialDecay(10**self._alpha_expo, decay_steps=self._decay_steps,
                                                                    decay_rate=self._lr_decay, staircase=False)
            opt = keras.optimizers.Adam(learning_rate=__lr_schedule, beta_1=0.9, beta_2=0.999, epsilon=1e-8, amsgrad=False) 

            # Compile model on device
            self.__model.compile(optimizer=opt, loss="mean_squared_error", metrics=["mape"])
        return 
    
    def EvaluateMLP(self,input_data_norm):
        pred_data_norm = self.__model.predict(input_data_norm, verbose=0)
        return pred_data_norm
    
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
        for layer in self.__model.layers:
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
            self.history = self.__model.fit(self._X_train_norm, self._Y_train_norm, \
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
            self._test_score = self.__model.evaluate(self._X_test_norm, self._Y_test_norm, verbose=0)[0]
            t_end = time.time()
            self._test_time = (t_end - t_start)
            self._weights = []
            self._biases = []
            for layer in self.__model.layers:
                self._weights.append(layer.weights[0])
                self._biases.append(layer.weights[1])
            self.SaveWeights()

        self._cost_parameter = 0
        for w in self._trimmed_weights:
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
        fig.savefig(self._save_dir + "/Model_"+str(self._model_index)+ "/History_Plot_Direct.png", format='png', bbox_inches='tight')
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
    _dt = tf.float32 
    __trainable_hyperparams=[]
    _optimizer = None 
    __lr_schedule = None 
    _train_name:str = ""
    _figformat:str="pdf"
    __keep_training:bool = True 

    def __init__(self):
        MLPTrainer.__init__(self)

        return
    
    def SetWeights(self, weights_input):
        self._weights = []
        for W in weights_input:
            self._weights.append(tf.Variable(W, self._dt))
        return 
    
    def SetBiases(self, biases_input):
        self._biases = []
        for b in biases_input:
            self._biases.append(tf.Variable(b, self._dt))
        return 
    
    def InitializeWeights_and_Biases(self):
        self._weights = []
        self._biases = []

        NN = [len(self._controlling_vars)]
        for N in self._hidden_layers:
            NN.append(N)
        NN.append(len(self._train_vars))

        for i in range(len(NN)-1):
            self._weights.append(tf.Variable(tf.random.normal([NN[i], NN[i+1]],
                                            mean=0.0,
                                            stddev=0.5,
                                            name="weights"),self._dt))
            self._biases.append(tf.Variable(tf.random.normal([NN[i+1],],
                                    mean=0.0,
                                    stddev=0.5,
                                    name="bias"),self._dt))
            
        return 
    def LoadWeights(self):
        for i in range(len(self._weights)):
            loaded_W = np.load(self._save_dir + "/Model_"+str(self._model_index) + "/W_"+self._train_name+"_"+str(i)+".npy", allow_pickle=True)
            loaded_b = np.load(self._save_dir + "/Model_"+str(self._model_index) + "/b_"+self._train_name+"_"+str(i)+".npy", allow_pickle=True)
            self._weights[i] = tf.Variable(loaded_W, self._dt)
            self._biases[i]= tf.Variable(loaded_b, self._dt)
        return 
    
    @tf.function
    def CollectVariables(self):
        self.__trainable_hyperparams = []
        for W in self._weights:
            self.__trainable_hyperparams.append(W)
        for b in self._biases:
            self.__trainable_hyperparams.append(b)
    
    def SetOptimizer(self):
        #self.SetDecaySteps()
        self._decay_steps = 3e4
        self.__lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(10**self._alpha_expo, decay_steps=self._decay_steps,
                                                                            decay_rate=self._lr_decay, staircase=False)

        self._optimizer = tf.keras.optimizers.Adam(self.__lr_schedule, beta_1=0.9, beta_2=0.999, epsilon=1e-8, amsgrad=False)
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
        input_data_tf = tf.constant(input_data_norm, tf.float32)
        return self._MLP_Evaluation(input_data_tf).numpy()
    
    @tf.function
    def mean_square_error(self, y_true, y_pred):
        return tf.reduce_mean(tf.pow(y_pred - y_true, 2), axis=0)
    
    @tf.function
    def Compute_Direct_Error(self, x_norm:tf.constant, y_label_norm:tf.constant):
        y_pred_norm = self._MLP_Evaluation(x_norm)
        return self.mean_square_error(y_label_norm, y_pred_norm)
    
    @tf.function
    def ComputeGradients_Direct_Error(self, x_norm:tf.constant, y_label_norm:tf.constant):
        with tf.GradientTape() as tape:
            tape.watch(self.__trainable_hyperparams)
            y_norm_loss = self.Compute_Direct_Error(x_norm, y_label_norm)
            grads_loss = tape.gradient(y_norm_loss, self.__trainable_hyperparams)
            
        return y_norm_loss, grads_loss
    
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
        self._optimizer.apply_gradients(zip(grads_loss, self.__trainable_hyperparams))
        
        return y_norm_loss 
    
    def Train_MLP(self):
        """Commence network training.
        """
        # prepare trainer start
        self.Preprocessing()
        # prepare trainer end 

        # loop epochs start
        self.LoopEpochs()
        # loop epochs end 

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

        self.val_loss_history=[]
        for _ in self._train_vars:
            self.val_loss_history.append([])
        return 
    
    def SetTrainBatches(self):
        indices = tf.range(start=0, limit=self._Np_train, dtype=tf.int32)
        shuffled_indices = tf.random.shuffle(indices)
        X_train_shuffled = tf.gather(self._X_train_norm, indices)
        Y_train_shuffled = tf.gather(self._Y_train_norm, indices)
        train_batches = tf.data.Dataset.from_tensor_slices((X_train_shuffled, Y_train_shuffled)).batch(2**self._batch_expo)
        return train_batches
    
    def LoopEpochs(self):
        t_start = time.time()
        worst_error = 1e32
        i = 0
        while (i < self._n_epochs) and self.__keep_training:
            train_batches_shuffled = self.SetTrainBatches()
            self.LoopBatches(train_batches=train_batches_shuffled)

            val_loss = self.ValidationLoss()
            
            if (i + 1) % self.callback_every == 0:
                self.TestLoss()
                self.CustomCallback()
            
            worst_error = self.__CheckEarlyStopping(val_loss, worst_error)

            self.PrintEpochInfo(i, val_loss)
            
            i += 1
        t_end = time.time()
        self._train_time = (t_end - t_start)/60
        return 
    
    def PrintEpochInfo(self, i_epoch, val_loss):
        if self._verbose > 0:
                print("Epoch: ", str(i_epoch), " Validation loss: ", str(val_loss.numpy()))
        return 
    
    #@tf.function
    def LoopBatches(self, train_batches):
        
        for x_norm_batch, y_norm_batch in train_batches:
            indices = tf.range(start=0,limit=tf.shape(x_norm_batch)[0],dtype=tf.int32)
            shuffled_indices = tf.random.shuffle(indices)
            x_batch_shuffled = tf.gather(x_norm_batch, shuffled_indices)
            y_batch_shuffled = tf.gather(y_norm_batch, shuffled_indices)
            
            self.Train_Step(x_batch_shuffled, y_batch_shuffled)

        return
    
    def ValidationLoss(self):
        val_loss = self.Compute_Direct_Error(tf.constant(self._X_val_norm, self._dt), tf.constant(self._Y_val_norm, self._dt))
        for iVar in range(len(self._train_vars)):
            self.val_loss_history[iVar].append(val_loss[iVar])
        return val_loss
    
    def TestLoss(self):
        t_start = time.time()
        self._test_score = tf.reduce_mean(self.Compute_Direct_Error(tf.constant(self._X_test_norm, self._dt), tf.constant(self._Y_test_norm, self._dt)))
        t_end = time.time()
        self._test_time = (t_end - t_start)/60
        return 
    
    def Plot_and_Save_History(self):
        """Plot the training convergence trends.
        """

        # with open(self._save_dir + "/Model_"+str(self._model_index)+"/TrainingHistory.csv", "w+") as fid:
        #     fid.write("epoch,loss,validation_loss\n")
        #     csvWriter = csv.writer(fid, delimiter=',')
        #     csvWriter.writerows(np.array([self.history_epochs, self.history_val_loss]).T)

        fig = plt.figure(figsize=[10,10])
        ax = plt.axes()
        #ax.plot(np.log10(self.history.history['loss']), 'b', label=r'Training score')
        H = np.array(self.val_loss_history)
        for i in range(len(self._train_vars)):
            ax.plot(H[i,:], label=r"Validation score "+self._train_vars[i])
        #ax.plot([0, np.log10(self.history_val_loss)], [np.log10(self._test_score), np.log10(self._test_score)], 'm--', label=r"Test score")
        ax.grid()
        ax.set_yscale('log')
        ax.legend(fontsize=20)
        ax.set_xlabel(r"Iteration[-]", fontsize=20)
        ax.set_ylabel(r"Training loss function [-]", fontsize=20)
        ax.set_title(r""+self._train_name+r" Training History", fontsize=22)
        ax.tick_params(axis='both', which='major', labelsize=18)
        fig.savefig(self._save_dir + "/Model_"+str(self._model_index)+ "/History_Plot_"+self._train_name+"."+self._figformat, format=self._figformat, bbox_inches='tight')
        plt.close(fig)
        return 
    
    def __CheckEarlyStopping(self, val_loss, worst_error):
        current_error = tf.reduce_max(val_loss)
        if current_error < worst_error - self._stagnation_tolerance:
            self.__keep_training = True 
            self.__stagnation_iter = 0
            
            worst_error = current_error
        else:
            self.__stagnation_iter += 1
            if self.__stagnation_iter > self._stagnation_patience:
                self.__keep_training = False 
                print("Early stopping due to stagnation")
        return worst_error 
    
    def PostProcessing(self):
        self.TestLoss()
        self.CustomCallback()
        return 
    
class PhysicsInformedTrainer(CustomTrainer):

    def __init__(self):
        CustomTrainer.__init__(self)
        return 
    
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
    

class EvaluateArchitecture:
    """Class for training MLP architectures
    """

    _Config:Config = None 
    _trainer_direct:TensorFlowFit = None 

    architecture:list[int] = [DefaultProperties.NN_hidden]   # Hidden layer architecture.
    alpha_expo:float = DefaultProperties.init_learning_rate_expo # Initial learning rate exponent (base 10)
    lr_decay:float = DefaultProperties.learning_rate_decay # Learning rate decay parameter.
    batch_expo:int = DefaultProperties.batch_size_exponent      # Mini-batch exponent (base 2)
    activation_function:str = "exponential"    # Activation function name applied to hidden layers.

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

    def __init__(self, Config_in:Config):
        """Define EvaluateArchitecture instance and prepare MLP trainer with
        default settings.

        :param Config: FlameletAIConfig object describing the flamelet data manifold.
        :type Config: FlameletAIConfig
        :param group_idx: MLP output group index, defaults to 0
        :type group_idx: int, optional
        :raises Exception: if MLP output group index is undefined by flameletAI configuration.
        """

        self._Config=Config_in
        self.main_save_dir = self._Config.GetOutputDir()
        
        # Define MLPTrainer object with default settings (currently only supports TensorFlowFit)
        self._train_file_header = self._Config.GetOutputDir()+"/"+self._Config.GetConcatenationFileHeader()
        self.SynchronizeTrainer()
        pass

    def SynchronizeTrainer(self):
        """Synchronize all MLP trainer settings with locally stored settings.
        """
        
        self._trainer_direct.SetModelIndex(self.current_iter)
        self._trainer_direct.SetSaveDir(self.main_save_dir)

        self._trainer_direct.SetDeviceKind(self.device)
        self._trainer_direct.SetDeviceIndex(self.process_index)

        self._trainer_direct.SetNEpochs(self.n_epochs)
        self._trainer_direct.SetActivationFunction(self.activation_function)
        self._trainer_direct.SetAlphaExpo(self.alpha_expo)
        self._trainer_direct.SetLRDecay(self.lr_decay)
        self._trainer_direct.SetBatchExpo(self.batch_expo)
        self._trainer_direct.SetHiddenLayers(self.architecture)
        self._trainer_direct.SetTrainFileHeader(self._train_file_header)
        self._trainer_direct.SetVerbose(self.verbose)

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
    
    def SetHiddenLayers(self, NN_hidden_layers:list[int]):
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
        self.SynchronizeTrainer()
        return 
    
    def SetBatchExpo(self, batch_expo_in:int):
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
    
    def SetActivationFunction(self, activation_function_name:str):
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
    
    def SetAlphaExpo(self, alpha_expo_in:float):
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
    
    def SetLRDecay(self, lr_decay_in:float):
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
        self.main_save_dir += "/Worker_"+str(worker_idx) + "/"
        self.SynchronizeTrainer()
        return 
    
    def CommenceTraining(self):
        """Initiate the training process.
        """
        self.PrepareOutputDir()
        self._trainer_direct.Train_MLP()
        self.TrainPostprocessing()

        fid = open(self.main_save_dir + "/current_iter.txt", "w+")
        fid.write(str(self.current_iter) + "\n")
        fid.close()

        self._test_score = self._trainer_direct.GetTestScore()
        self.__cost_parameter = self._trainer_direct.GetCostParameter()
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
        return self.__cost_parameter
    
    def GetTestScore(self):
        """Get MLP evaluation test score upon completion of training.

        :return: MLP evaluation test score.
        :rtype: float
        """
        return self._test_score
    
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