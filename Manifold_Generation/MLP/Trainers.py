# Set all seeds to ensure reproducebility for MLP training.
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
from Common.EntropicAIConfig import EntropicAIConfig 

def GetReferenceData(dataset_file, x_vars, train_variables):
    # Open data file and get variable names from the first line
    fid = open(dataset_file, 'r')
    line = fid.readline()
    fid.close()
    line = line.strip()
    line_split = line.split(',')
    if(line_split[0][0] == '"'):
        varnames = [s[1:-1] for s in line_split]
    else:
        varnames = line_split
    
    # Get indices of controlling and train variables
    iVar_x = [varnames.index(v) for v in x_vars]
    iVar_y = [varnames.index(v) for v in train_variables]

    # Retrieve respective data from data set
    D = np.loadtxt(dataset_file, delimiter=',', skiprows=1, dtype=np.float32)
    X_data = D[:, iVar_x]
    Y_data = D[:, iVar_y]

    return X_data, Y_data

class MLPTrainer:
    # Base class for flamelet MLP trainer

    _n_epochs:int = 250      # Number of epochs to train for.
    _alpha_expo:float = -3.0   # Alpha training exponent parameter.
    _lr_decay:float = 1.0      # Learning rate decay parameter.
    _batch_expo:int = 10     # Mini-batch size exponent.

    _i_activation_function:int = 0   # Activation function index.
    _activation_function_name:str = "elu"
    _activation_function = None
    _restart_training:bool = False # Restart training process
    _activation_function_names_options:list[str] = ["linear","elu","relu","tanh","exponential"]
    _activation_function_options = activation_functions = [tf.keras.activations.linear,
                            tf.keras.activations.elu,\
                            tf.keras.activations.relu,\
                            tf.keras.activations.tanh,\
                            tf.keras.activations.exponential]
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

    optuna_trial = None 

    # Intermediate history update window settings.
    history_plot_window = None 
    history_plot_axes = None
    history_epochs = []
    history_loss = []
    history_val_loss = []

    _stagnation_tolerance:float = 1e-11 
    _stagnation_patience:int = 1000 

    def __init__(self):
        """Initiate MLP trainer object.
        """
        
    def SetTrainFileHeader(self, train_filepathname:str):
        """Set the name for a custom set of train and test data.

        :param train_filepathname: file path and name for custom train data set.
        :type train_filepathname: str
        """
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
    
    def SetNEpochs(self, n_input:int):
        """Set number of training epochs

        :param n_input: epoch count.
        :type n_input: int
        :raises Exception: if the specified number of epochs is lower than zero.
        """
        if n_input <= 0:
            raise Exception("Epoch count should be higher than zero.")
        self._n_epochs = n_input
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

    def SetDeviceIndex(self, device_index:int):
        """Define device index on which to train (CPU core or GPU card).

        :param device_index: CPU node or GPU card index to use for training.
        :type device_index: int
        """
        self._device_index = device_index

    def SetControllingVariables(self, x_vars:list[str]):
        """Specify MLP input or controlling variable names.

        :param x_vars: list of controlling variable names on which to train the MLP's.
        :type x_vars: list[str]
        """
        self._controlling_vars = []
        for var in x_vars:
            self._controlling_vars.append(var)

    def SetLRDecay(self, lr_decay:float):
        """Specify learning rate decay parameter for exponential decay scheduler.

        :param lr_decay: learning rate decay factor.
        :type lr_decay: float
        :raises Exception: if specified learning rate decay factor is not between zero and one.
        """
        if lr_decay < 0 or lr_decay > 1.0:
            raise Exception("Learning rate decay factor should be between zero and one, not "+str(lr_decay))
        self._lr_decay = lr_decay

    def SetAlphaExpo(self, alpha_expo:float):
        """Specify exponent of initial learning rate for exponential decay scheduler.

        :param alpha_expo: initial learning rate exponent.
        :type alpha_expo: float
        :raises Exception: if specified exponent is higher than zero.
        """
        if alpha_expo > 0:
            raise Exception("Initial learning rate exponent should be below zero.")
        self._alpha_expo = alpha_expo

    def SetBatchSize(self, batch_expo:int):
        """Specify exponent of mini-batch size.

        :param batch_expo: mini-batch exponent (base 2) to be used during training.
        :type batch_expo: int
        :raises Exception: if the specified exponent is lower than zero.
        """
        if batch_expo < 0:
            raise Exception("Mini-batch exponent should be higher than zero.")
        self._batch_expo = batch_expo

    def SetHiddenLayers(self, layers_input:list[int]):
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
    
    def RestartTraining(self):
        """Restart the training process.
        """
        self._restart_training = True 

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

    def TrimWeights(self):
        """
        Remove neurons which contribute little to the network output from the network.
        """
        n_layers = len(self._weights)+1

        self._trimmed_weights = [self._weights[0]]
        self._trimmed_biases = []
        for iLayer in range(1, n_layers-1):
            NN = np.shape(self._weights[iLayer])[0]
            neuron_influences = np.zeros(NN)
            # Influence is defined as the sum of the weights connecting to the respective neuron. 
            # If it falls below 2% of the average weight between the layers, the neuron is discarded.
            for iNeuron in range(NN):
                neuron_influences[iNeuron] = np.sum(np.abs(self._weights[iLayer-1][:, iNeuron])) +\
                      np.sum(np.abs(self._weights[iLayer][iNeuron, :]))

            avg_ = np.average(neuron_influences)
            possible_eliminations = np.where(neuron_influences < 0.02*avg_)[0]
            self._trimmed_biases.append(np.delete(self._biases[iLayer-1], possible_eliminations))
            self._trimmed_weights[iLayer - 1] = np.delete(self._trimmed_weights[iLayer-1], possible_eliminations, axis=1)
            self._trimmed_weights.append(np.delete(self._weights[iLayer], possible_eliminations, axis=0))
            
        self._trimmed_biases.append(np.delete(self._biases[-1], []))

    def write_SU2_MLP(self, file_out:str):
        """Write the network to ASCII format readable by the MLPCpp module in SU2.

        :param file_out: MLP output path and file name.
        :type file_out: str
        """

        n_layers = len(self._weights)+1

        # Select trimmed weight matrices for output.
        weights_for_output = self._trimmed_weights
        biases_for_output = self._trimmed_biases

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
            fid.write("\t".join("%+.16e" % float(b) for b in B) + "\n")


        fid.close()

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
        fid.write("Activation function index: %i\n" % self._i_activation_function)
        fid.write("Number of hidden layers: %i\n" % len(self._hidden_layers))
        fid.write("Architecture: " + " ".join(str(n) for n in self._hidden_layers) + "\n")
        fid.close()

        self.write_SU2_MLP(self._save_dir + "/Model_"+str(self._model_index)+"/MLP_entropy")

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

    def Plot_and_Save_History(self):
        return

class Train_Entropic_MLP(MLPTrainer):
    # Description:
    # Construct, train, and save an artificial neural network for FGM simulations in SU2
    # This class trains an MLP to predict fluid entropy based on density and internal energy
    # prior to using physics-informed learning.

    __model:keras.models.Sequential # Keras trainable model.

    # Training validation history.
    history_epochs = []
    history_loss = []
    history_val_loss = []


    def __init__(self):
        MLPTrainer.__init__(self)
        # Set train variables based on what kind of network is trained

        self._controlling_vars = ["Density", "Energy"]
        self._train_vars = ["s"]

        # Set the activation function to exponential by default.
        self._activation_function_name="exponential"
        return 
    
    def DefineMLP(self):
        """Construct trainable model for entropic MLP.
        """
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
            lr_schedule = keras.optimizers.schedules.ExponentialDecay(10**self._alpha_expo, decay_steps=10000,
                                                                    decay_rate=self._lr_decay, staircase=False)
            opt = keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999, epsilon=1e-8, amsgrad=False) 

            # Compile model on device
            self.__model.compile(optimizer=opt, loss="mean_squared_error", metrics=["mape"])
        return 
    
    def EvaluateMLP(self, input_data_norm:np.ndarray):
        pred_data_norm = self.__model.predict(input_data_norm, verbose=0)
        return pred_data_norm
    
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
                                                      verbose=0)
            self.history = self.__model.fit(self._X_train_norm, self._Y_train_norm, \
                                          epochs=self._n_epochs, \
                                          batch_size=2**self._batch_expo,\
                                          verbose=2, \
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
            self.SaveWeights()
            self.TrimWeights()

        self._cost_parameter = 0
        for w in self._trimmed_weights:
            self._cost_parameter += np.shape(w)[0] * np.shape(w)[1]
        return 
    
    def Plot_and_Save_History(self):
        """Plot the training convergence trends.
        """
        epochs = self.history.epoch
        val_loss = self.history.history['val_loss']
        loss = self.history.history['loss']

        with open(self._save_dir + "/Model_"+str(self._model_index)+"/TrainingHistory.csv", "w+") as fid:
            fid.write("epoch,loss,validation_loss\n")
            csvWriter = csv.writer(fid, delimiter=',')
            csvWriter.writerows(np.array([epochs, loss, val_loss]).T)

        fig = plt.figure(figsize=[10,10])
        ax = plt.axes()
        ax.plot(np.log10(self.history.history['loss']), 'b', label=r'Training score')
        ax.plot(np.log10(self.history.history['val_loss']), 'r', label=r"Validation score")
        ax.plot([0, len(self.history.history['loss'])], [np.log10(self._test_score), np.log10(self._test_score)], 'm--', label=r"Test score")
        ax.grid()
        ax.legend(fontsize=20)
        ax.set_xlabel(r"Iteration[-]", fontsize=20)
        ax.set_ylabel(r"Training loss function [-]", fontsize=20)
        ax.set_title(r"Entropy Training History", fontsize=22)
        ax.tick_params(axis='both', which='major', labelsize=18)
        fig.savefig(self._save_dir + "/Model_"+str(self._model_index)+ "/History_Plot_Entropy.png", format='png', bbox_inches='tight')
        plt.close(fig)
        return 
    
    class PlotCallback(tf.keras.callbacks.Callback):
        """Callback function called during MLP training.
        """
        FitClass = None
        def __init__(self, TensorFlowFit:MLPTrainer):
            self.FitClass = TensorFlowFit

        def on_epoch_end(self, epoch, logs=None):
            self.FitClass.history_epochs.append(epoch)
            self.FitClass.history_loss.append(logs["loss"])
            self.FitClass.history_val_loss.append(logs["val_loss"])
            if epoch % 10 == 0:
                fig = plt.figure(figsize=[10,10])
                ax = plt.axes()
                ax.plot(self.FitClass.history_epochs, self.FitClass.history_loss, 'b', label=r"Training loss")
                ax.plot(self.FitClass.history_epochs, self.FitClass.history_val_loss, 'r', label=r"Validation loss")
                ax.grid()
                ax.set_yscale('log')
                ax.legend(fontsize=20)
                ax.tick_params(which='both',labelsize=18)
                ax.set_xlabel(r"Iteration[-]", fontsize=20)
                ax.set_ylabel(r"Training loss function [-]", fontsize=20)
                ax.set_title(r"Entropy Training History", fontsize=22)
                fig.savefig(self.FitClass._save_dir + "/Model_"+str(self.FitClass._model_index)+ \
                            "/Intermediate_History_Plot_Entropic.png", format="png", bbox_inches='tight')
                plt.close(fig)

            return super().on_epoch_end(epoch, logs)


class Train_C2_MLP(MLPTrainer):
    # Description:
    # Construct, train, and save a physics-informed neural network for an entropy-based EOS.
    # Network training for thermodynamic quantities based on entropy derivatives.

    # Minimum and maximum density, energy, and entropy in  the data set.
    __rho_min:float = 0   
    __rho_max:float = 0   
    __e_min:float = 0
    __e_max:float = 0
    __s_min:float = 0
    __s_max:float = 0

    # Data indices for density, energy, temperature, pressure, and speed of sound.
    idx_rho:int = 0
    idx_e:int = 1
    idx_T:int = 0
    idx_p:int = 1
    idx_c2:int = 2

    weights:list[tf.Variable] = []
    biases:list[tf.Variable] = []

    dt = tf.float32 
    optimizer:tf.keras.optimizers.Adam

    idx_step:int = 1
    x_var:tf.Variable
    __keep_training:bool = True 
    __stagnation_iter:int = 0 

    def __init__(self):
        MLPTrainer.__init__(self)
        self._controlling_vars = ["Density","Energy"]
        self._train_vars=["T","p","c2"]

        # Set activation function to exponential to make derivatives more cheap to evaluate.
        self._activation_function = tf.keras.activations.exponential
        return 


    def GetTrainData(self):
        print("Reading train, test, and validation data...")
        X_full, Y_full = GetReferenceData(self._filedata_train+"_full.csv", self._controlling_vars, ["s"])
        self.X_train, self.Y_train = GetReferenceData(self._filedata_train+"_train.csv", self._controlling_vars, ["s"])
        self.X_test, self.Y_test = GetReferenceData(self._filedata_train+"_test.csv", self._controlling_vars, ["s"])
        self.X_val, self.Y_val = GetReferenceData(self._filedata_train+"_val.csv", self._controlling_vars, ["s"])
        print("Done!")

        # Calculate normalization bounds of full data set
        self._X_min = np.min(X_full, axis=0)
        self._X_max = np.max(X_full, axis=0)
        self.__rho_min, self.__rho_max = self._X_min[self.idx_rho], self._X_max[self.idx_rho]
        self.__e_min, self.__e_max =self._X_min[self.idx_e], self._X_max[self.idx_e]
        self.__s_min, self.__s_max = min(Y_full[:, 0]), max(Y_full[:, 0])
        self._Y_min= [np.min(Y_full,axis=0)]
        self._Y_max = [np.max(Y_full,axis=0)]
        self.rhoe_train_norm = (self.X_train - self._X_min) / (self._X_max - self._X_min)

        self.S_test = self.Y_test[:, 0]
        self.S_val = self.Y_val[:, 0]
        self.S_train = self.Y_train[:, 0]

        # Free up memory
        del X_full
        del Y_full

        print("Extracting EOS data...")
        X_full, PTC2_full = GetReferenceData(self._filedata_train+"_full.csv", self._controlling_vars, self._train_vars)
        self.Temperature_min, self.Temperature_max = min(PTC2_full[:, self.idx_T]), max(PTC2_full[:, self.idx_T])
        self.Pressure_min, self.Pressure_max = min(PTC2_full[:, self.idx_p]), max(PTC2_full[:, self.idx_p])
        self.C2_min, self.C2_max = min(PTC2_full[:, self.idx_c2]), max(PTC2_full[:, self.idx_c2])

        _, PTC2_train = GetReferenceData(self._filedata_train+"_train.csv", self._controlling_vars, self._train_vars)
        X_test, PTC2_test = GetReferenceData(self._filedata_train+"_test.csv", self._controlling_vars, self._train_vars)
        X_val, PTC2_val = GetReferenceData(self._filedata_train+"_val.csv", self._controlling_vars, self._train_vars)

        self.P_train = PTC2_train[:, self.idx_p]
        self.T_train = PTC2_train[:, self.idx_T]
        self.C2_train = PTC2_train[:, self.idx_c2]

        self.P_test = PTC2_test[:, self.idx_p]
        self.T_test = PTC2_test[:, self.idx_T]
        self.C2_test = PTC2_test[:, self.idx_c2]
        self.rhoe_test_norm = (X_test - self._X_min) / (self._X_max - self._X_min)

        self.P_val = PTC2_val[:, self.idx_p]
        self.T_val = PTC2_val[:, self.idx_T]
        self.C2_val = PTC2_val[:, self.idx_c2]
        self.rhoe_val_norm = (X_val - self._X_min) / (self._X_max - self._X_min)

        print("Done!")
        return 
    
    def SetWeights(self, weights_input:list[np.ndarray]):
        """Set the initial weights for the network.

        :param weights_input: list with trainable weights values.
        :type weights_input: list[np.ndarray]
        """
        self.weights = []
        for W in weights_input:
            self.weights.append(tf.Variable(W, self.dt))
        return 
    
    def SetBiases(self, biases_input:list[np.ndarray]):
        """Set the initial biases for the network.

        :param biases_input: list with trainable biases values.
        :type biases_input: list[np.ndarray]
        """
        self.biases = []
        for b in biases_input:
            self.biases.append(tf.Variable(b, self.dt))
        return 
    
    def __CollectVariables(self):
        """Define weights and biases as trainable hyper-parameters.
        """
        self.__hyper_parameters = []
        for W in self.weights:
            self.__hyper_parameters.append(W)
        for b in self.biases[:-1]:
            self.__hyper_parameters.append(b)
        return 
    
    def __SetOptimizer(self):
        """Prepare optimizer and learning rate scheduler.
        """
        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(10**self._alpha_expo, decay_steps=30000,
                                                                            decay_rate=self._lr_decay, staircase=False)
        self.optimizer = tf.keras.optimizers.Adam(self.lr_schedule) 
        return 
    
    @tf.function
    def __ComputeLayerInput(self, x:tf.Tensor, W:tf.Tensor, b:tf.Tensor):
        """Compute input for activation function for input x for a given weights and biases tensor.

        :param x: output from previous (hidden) layer.
        :type x: tf.Tensor
        :param W: network weights for current layer.
        :type W: tf.Tensor
        :param b: biases vector for current layer
        :type b: tf.Tensor
        :return: inputs for activation function for the current layer.
        :rtype: tf.Tensor
        """
        X = tf.matmul(x, W) + b 
        return X 
    
    @tf.function
    def EvaluateMLP(self, rhoe_norm:tf.Tensor):
        """Compute (normalized) MLP output based on normalized inputs.

        :param rhoe_norm: normalized tensor with density, energy values.
        :type rhoe_norm: tf.Tensor
        :return: tensor with normalized entropy output.
        :rtype: tf.Tensor
        """
        Y = rhoe_norm 
        for iLayer in range(len(self.weights)-1):
            Y = self._activation_function(self.__ComputeLayerInput(Y, self.weights[iLayer], self.biases[iLayer]))
        Y = self.__ComputeLayerInput(Y, self.weights[-1], self.biases[-1])
        return Y 
    
    @tf.function
    def __Gradients_2(self, x_var:tf.Variable):
        """Compute the normalized first and second derivatives of the MLP output w.r.t. to the normalized inputs.

        :param x_var: normalized input tensor.
        :type x_var: tf.Variable
        :return: normalized dsdrho_e, dsde_rho, d2sdrho2, d2sdedrho, d2sde2 
        :rtype: tf.Tensor
        """

        # Evaluate density derivatives.
        with tf.GradientTape() as tape_1:
            with tf.GradientTape() as tape_2:
                s_norm = self.EvaluateMLP(x_var)
                ds_norm = tape_2.gradient(s_norm, x_var)
                ds_norm_rho = tf.gather(ds_norm, indices=self.idx_rho, axis=1)
                d2s_norm_rho = tape_1.gradient(ds_norm_rho, x_var)

        # Evaluate energy derivatives.
        with tf.GradientTape() as tape_1:
            with tf.GradientTape() as tape_2:
                s_norm = self.EvaluateMLP(x_var)
                ds_norm = tape_2.gradient(s_norm, x_var)
                ds_norm_e = tf.gather(ds_norm, indices=self.idx_e, axis=1)
                d2s_norm_e = tape_1.gradient(ds_norm_e, x_var)

        # Extract derivative information.
        dsdrho_e_norm = tf.gather(ds_norm, self.idx_rho, axis=1)
        dsde_rho_norm = tf.gather(ds_norm, self.idx_e, axis=1)
        d2sdrho2_norm = tf.gather(d2s_norm_rho, self.idx_rho, axis=1)
        d2sdedrho_norm = tf.gather(d2s_norm_e, self.idx_rho, axis=1)
        d2sde2_norm = tf.gather(d2s_norm_e, self.idx_e, axis=1)

        return dsdrho_e_norm, dsde_rho_norm, d2sdrho2_norm, d2sdedrho_norm, d2sde2_norm, s_norm

    @tf.function
    def __EntropicEOS(self, rho:tf.Tensor, ds_dim:list[tf.Tensor], d2s_dim:list[tf.Tensor]):
        """Compute thermodynamic quantities based on entropy derivatives w.r.t. energy and density.

        :param rho: density [kg m^-3]
        :type rho: tf.Tensor
        :param ds_dim: dsdrho_e, dsde_rho
        :type ds_dim: list[tf.Tensor]
        :param d2s_dim: d2sdrho2, d2sdedrho, d2sde2
        :type d2s_dim: list[tf.Tensor]
        :return: temperature, pressure, and squared speed of sound
        :rtype: tf.Tensor
        """

        # Retrieve derivative information
        dsdrho_e = ds_dim[0]
        dsde_rho = ds_dim[1]
        d2sdrho2 = d2s_dim[0]
        d2sdedrho = d2s_dim[1]
        d2sde2 = d2s_dim[2]

        # Entropic EOS
        T = 1.0 / dsde_rho
        P = -tf.pow(rho, 2) * T * dsdrho_e
        blue_term = (dsdrho_e * (2 - rho * tf.pow(dsde_rho, -1) * d2sdedrho) + rho*d2sdrho2)
        green_term = (-tf.pow(dsde_rho, -1) * d2sde2 * dsdrho_e + d2sdedrho)
        c2 = -rho * tf.pow(dsde_rho, -1) * (blue_term - rho * green_term * (dsdrho_e / dsde_rho))
  
        return T, P, c2
    
    @tf.function
    def __DeNormalize_Gradients(self, x_var:tf.Variable):
        """Dimensionalize entropy derivatives.

        :param x_var: normalized density, energy tensor
        :type x_var: tf.Variable
        :return: density, first entropy derivatives, second entropy derivatives
        :rtype: tf.Tensor
        """

        # Compute normalized gradients.
        dsdrho_e_norm, \
        dsde_rho_norm, \
        d2sdrho2_norm, \
        d2sdedrho_norm, \
        d2sde2_norm, \
        s_norm = self.__Gradients_2(x_var)

        # Compute scales for density, energy and entropy.
        rho_scale = self.__rho_max - self.__rho_min
        e_scale = self.__e_max - self.__e_min 
        s_scale = self.__s_max - self.__s_min 
        
        # Scale normalized density.
        rho_norm = tf.gather(x_var, self.idx_rho,axis=1)
        rho_dim = rho_scale * rho_norm + self.__rho_min

        # Scale first derivatives.
        dsdrho_e = tf.math.multiply(s_scale / rho_scale, dsdrho_e_norm)
        dsde_rho = tf.math.multiply(s_scale / e_scale, dsde_rho_norm)

        # Scale second derivatives
        d2sdrho2 = tf.math.multiply(s_scale / (rho_scale * rho_scale), d2sdrho2_norm)
        d2sdedrho = tf.math.multiply(s_scale / (rho_scale * e_scale), d2sdedrho_norm)
        d2sde2 = tf.math.multiply(s_scale / (e_scale * e_scale), d2sde2_norm)
       
        s_dim = s_norm * (s_scale) + self.__s_min
        return rho_dim, [dsdrho_e, dsde_rho], [d2sdrho2, d2sdedrho, d2sde2], s_dim
    
    
    @tf.function
    def __TD_Evaluation(self, x_var:tf.Variable):
        """Compute the thermodynamic state based on normalized density and energy.

        :param x_var: normalized density, energy tensor
        :type x_var: tf.Variable
        :return: temperature, pressure, squared speed of sound
        :rtype: tf.Tensor
        """

        # Compute entropy derivatives w.r.t. to density and energy.
        rho_dim, ds_dim, d2s_dim, s_dim = self.__DeNormalize_Gradients(x_var)
        
        # Convert entropy derivatives into temperature, pressure, and speed of sound.
        T, P, c2 = self.__EntropicEOS(rho_dim, ds_dim, d2s_dim)

        return T, P, c2, s_dim

    @tf.function
    def __mean_squared_error(self, y_true:tf.Tensor, y_pred:tf.Tensor):
        """Loss function definition.

        :param y_true: labeled data.
        :type y_true: tf.Tensor
        :param y_pred: predicted data.
        :type y_pred: tf.Tensor
        :return: mean, squared difference between y_pred and y_true.
        :rtype: tf.Tensor
        """
        return tf.reduce_mean(tf.pow(y_pred - y_true, 2))
    
    @tf.function
    def __Compute_T_error(self, T_label:tf.Tensor, x_var:tf.Variable):
        """Compute the temperature prediction error.

        :param T_label: reference temperature data
        :type T_label: tf.Tensor
        :param x_var: normalized density and energy tensor.
        :type x_var: tf.Variable
        :return: mean squared error between predicted and reference temperature.
        :rtype: tf.Tensor
        """

        # Evaluate thermodynamic state.
        T, _, _, _ = self.__TD_Evaluation(x_var)

        # Normalize reference and predicted temperature.
        T_pred_norm = (T - self.Temperature_min) / (self.Temperature_max - self.Temperature_min)
        T_label_norm = (T_label - self.Temperature_min) / (self.Temperature_max - self.Temperature_min)

        # Apply loss function.
        T_error = self.__mean_squared_error(y_true=T_label_norm, y_pred=T_pred_norm)

        return T_error
    
    @tf.function
    def __Compute_P_error(self, P_label:tf.Tensor,x_var:tf.Variable):
        """Compute the pressure prediction error.

        :param P_label: reference pressure data
        :type P_label: tf.Tensor
        :param x_var: normalized density and energy tensor.
        :type x_var: tf.Variable
        :return: mean squared error between predicted and reference pressure.
        :rtype: tf.Tensor
        """

        # Evaluate thermodynamic state.
        _, P, _, _ = self.__TD_Evaluation(x_var)

        # Normalize reference and predicted pressure.
        P_pred_norm = (P - self.Pressure_min) / (self.Pressure_max - self.Pressure_min)
        P_label_norm = (P_label - self.Pressure_min) / (self.Pressure_max - self.Pressure_min)

        # Apply loss function.
        P_error = self.__mean_squared_error(y_true=P_label_norm, y_pred=P_pred_norm)

        return P_error 
    
    @tf.function
    def __Compute_C2_error(self, C2_label,x_var:tf.Variable):
        """Compute the prediction error for squared speed of sound (SoS).

        :param C2_label: reference pressure data
        :type C2_label: tf.Tensor
        :param x_var: normalized density and energy tensor.
        :type x_var: tf.Variable
        :return: mean squared error between predicted and reference squared speed of sound.
        :rtype: tf.Tensor
        """

        # Evaluate thermodynamic state.
        _, _, C2, _ = self.__TD_Evaluation(x_var)
        
        # Normalize reference and predicted squared SoS.
        C2_pred_norm = (C2 - self.C2_min) / (self.C2_max - self.C2_min)
        C2_label_norm = (C2_label - self.C2_min) / (self.C2_max - self.C2_min)

        # Apply loss function.
        C2_error = self.__mean_squared_error(y_true=C2_label_norm, y_pred=C2_pred_norm)

        return C2_error
    
    @tf.function 
    def __Compute_S_error(self, S_label, x_var:tf.Variable):
        # Evaluate thermodynamic state.
        _, _, _, S_pred = self.__TD_Evaluation(x_var)
        
        # Normalize reference and predicted squared SoS.
        S_pred_norm = (S_pred - self.__s_min)/(self.__s_max - self.__s_min)
        S_label_norm = (S_label -  self.__s_min)/(self.__s_max - self.__s_min)
        
        # Apply loss function.
        S_error = self.__mean_squared_error(y_true=S_label_norm, y_pred=S_pred_norm)

        return S_error 
    
    @tf.function
    def __ComputeGradients_T_error(self, T_label:tf.Tensor, rhoe_norm:tf.Variable):
        """Compute temperature prediction error and respective MLP weight sensitivities.

        :param T_label: reference temperature data
        :type T_label: tf.Tensor
        :param rhoe_norm: normalized density and energy tensor.
        :type rhoe_norm: tf.Variable
        :return: temperature prediction loss value and MLP weight sensitivities.
        :rtype: tf.Tensor
        """

        with tf.GradientTape() as tape:
            # Evaluate temperature loss value.
            T_loss = self.__Compute_T_error(T_label, rhoe_norm)
            
            # Compute MLP weight sensitvities.
            grads_T = tape.gradient(T_loss, self.__hyper_parameters)
        
        return T_loss, grads_T

    @tf.function
    def __ComputeGradients_P_error(self, P_label:tf.Tensor, rhoe_norm:tf.Variable):
        """Compute pressure prediction error and respective MLP weight sensitivities.

        :param P_label: reference pressure data
        :type P_label: tf.Tensor
        :param rhoe_norm: normalized density and energy tensor.
        :type rhoe_norm: tf.Variable
        :return: pressure prediction loss value and MLP weight sensitivities.
        :rtype: tf.Tensor
        """
        with tf.GradientTape() as tape:
            # Evaluate pressure loss value.
            P_loss = self.__Compute_P_error(P_label, rhoe_norm)

            # Compute MLP weight sensitvities.
            grads_P = tape.gradient(P_loss, self.__hyper_parameters)

        return P_loss, grads_P
    
    @tf.function
    def __ComputeGradients_C2_error(self, C2_label:tf.Tensor, rhoe_norm:tf.Variable):
        """Compute SoS prediction error and respective MLP weight sensitivities.

        :param C2_label: reference squared SoS data
        :type C2_label: tf.Tensor
        :param rhoe_norm: normalized density and energy tensor.
        :type rhoe_norm: tf.Variable
        :return: SoS prediction loss value and MLP weight sensitivities.
        :rtype: tf.Tensor
        """
        with tf.GradientTape() as tape:
            C2_loss = self.__Compute_C2_error(C2_label, rhoe_norm)
            grads_C2 = tape.gradient(C2_loss, self.__hyper_parameters)
        return C2_loss, grads_C2
    
    @tf.function
    def __ComputeGradients_S_error(self, S_label:tf.Tensor, rhoe_norm:tf.Variable):
        """Compute SoS prediction error and respective MLP weight sensitivities.

        :param C2_label: reference squared SoS data
        :type C2_label: tf.Tensor
        :param rhoe_norm: normalized density and energy tensor.
        :type rhoe_norm: tf.Variable
        :return: SoS prediction loss value and MLP weight sensitivities.
        :rtype: tf.Tensor
        """
        with tf.GradientTape() as tape:
            S_loss = self.__Compute_S_error(S_label, rhoe_norm)
            grads_S = tape.gradient(S_loss, self.__hyper_parameters)
        return S_loss, grads_S
    
    @tf.function
    def __Triple_Variable_Train_Step(self, T_batch:tf.Tensor, P_batch:tf.Tensor, C2_batch:tf.Tensor, S_batch:tf.Tensor, rhoe_norm_batch:tf.Variable):
        """Apply MLP weight updates based on thermodynamic quantity losses.

        :param T_batch: temperature batch data
        :type T_batch: tf.Tensor
        :param P_batch: pressure batch data
        :type P_batch: tf.Tensor
        :param C2_batch: squared speed of sound batch data
        :type C2_batch: tf.Tensor
        :param rhoe_norm_batch: normalized density and energy batch data
        :type rhoe_norm_batch: tf.Variable
        :return: loss values for temperature, pressure, and speed of sound
        :rtype: tf.Tensor
        """


        # Weight update for temperature prediction.
        T_loss, grads_T = self.__ComputeGradients_T_error(T_batch, rhoe_norm_batch)
        self.optimizer.apply_gradients(zip(grads_T, self.__hyper_parameters))

        # Weight update for pressure prediction.
        P_loss, grads_P = self.__ComputeGradients_P_error(P_batch,rhoe_norm_batch)
        self.optimizer.apply_gradients(zip(grads_P, self.__hyper_parameters))

        # Weight update for SoS prediction.
        C2_loss, grads_C2 = self.__ComputeGradients_C2_error(C2_batch,rhoe_norm_batch)
        self.optimizer.apply_gradients(zip(grads_C2, self.__hyper_parameters))
        
        # # Weight update for temperature prediction.
        # S_loss, grads_S = self.__ComputeGradients_S_error(S_batch, rhoe_norm_batch)
        # self.optimizer.apply_gradients(zip(grads_S, self.__hyper_parameters))

        return T_loss, P_loss, C2_loss
    

    def Train_MLP(self):
        """Commence training process.
        """

        # ORCHID_data_file = "/home/ecbunschoten/NICFD/NICFD_MLP_Optimization/Data_Generation/ORCHID_dataset.csv"
        # with open(ORCHID_data_file,'r') as fid:
        #     vars_ORCHID = fid.readline().strip().split(',')
        # ORCHID_data = np.loadtxt(ORCHID_data_file,delimiter=',',skiprows=1,dtype=np.float32)[::10,:]

        # self.rho_ORCHID = ORCHID_data[:, vars_ORCHID.index("Density")]
        # self.e_ORCHID = ORCHID_data[:, vars_ORCHID.index("Energy")]
        

        # Prepare output directory
        if not os.path.isdir(self._save_dir + "/Model_"+str(self._model_index)):
            os.mkdir(self._save_dir + "/Model_"+str(self._model_index))

        self.T_val_errors = []
        self.P_val_errors = []
        self.C2_val_errors = []

        # Load training data and pre-process thermo-dynamic quantities.
        self.GetTrainData()

        # Define network weights for optimization and learning rate schedule.
        self.__CollectVariables()
        self.__SetOptimizer()

        # rho_ORCHID_norm = (self.rho_ORCHID - self.__rho_min) / (self.__rho_max - self.__rho_min)
        # e_ORCHID_norm = (self.e_ORCHID - self.__e_min) / (self.__e_max - self.__e_min)
        
        # self.rhoe_ORCHID_norm = tf.Variable(np.hstack((rho_ORCHID_norm[:,np.newaxis],e_ORCHID_norm[:,np.newaxis])),self.dt)
        # self.T_ORCHID_ref = ORCHID_data[:, vars_ORCHID.index("T")]
        # self.P_ORCHID_ref = ORCHID_data[:, vars_ORCHID.index("p")]
        # self.C_ORCHID_ref = np.sqrt(ORCHID_data[:, vars_ORCHID.index("c2")])
        
        # Split train data into batches.
        train_batches_TPC2 = tf.data.Dataset.from_tensor_slices((self.rhoe_train_norm, self.T_train, self.P_train, self.C2_train, self.S_train)).batch(2**self._batch_expo)
        
        # Prepare validation and test set data.
        rhoe_val = tf.Variable(self.rhoe_val_norm,tf.float32)
        rhoe_test = tf.Variable(self.rhoe_test_norm,tf.float32)

        # Evaluate test set and generate plots every 5 epochs.
        callback_every = 5

        # Initiate training loop.
        i = 0
        worst_error = 1e32 

        while i < self._n_epochs and self.__keep_training:

            # Loop over train batches.
            j = 0 
            for rhoe_norm_batch, T_train_batch, P_train_batch, C2_train_batch, S_train_batch in train_batches_TPC2:
                j += 1

                # Update weights.
                rhoe_batch_var = tf.Variable(rhoe_norm_batch, tf.float32)
                T_loss, P_loss, C2_loss = self.__Triple_Variable_Train_Step(T_train_batch, P_train_batch, C2_train_batch, S_train_batch, rhoe_batch_var)

                # Display trainign loss information.
                T_loss = T_loss.numpy()
                P_loss = P_loss.numpy()
                C2_loss = C2_loss.numpy()
                if j % (2**self._batch_expo) == 0:
                    print("Epoch ", str(i), "batch", str(j), " P loss", str(P_loss), " T loss", str(T_loss), " C2 loss", str(C2_loss))
                self.idx_step = 1
            
            # Compute loss on validation set.
            T_val_loss, P_val_loss, C2_val_loss = self.__TestSetLoss(rhoe_val, self.T_val, self.P_val, self.C2_val)
            self.T_val_errors.append(T_val_loss.numpy())
            self.P_val_errors.append(P_val_loss.numpy())
            self.C2_val_errors.append(C2_val_loss.numpy())

            worst_error_current = max([T_val_loss.numpy(), P_val_loss.numpy(), C2_val_loss.numpy()])
            worst_error = self.__CheckEarlyStopping(worst_error_current, worst_error)

            # Compute loss on test set and plot predictions.
            if i % callback_every == 0:

                self.T_test_loss, self.P_test_loss, self.C2_test_loss,\
                T_test, P_test, C2_test, S_test = self.__TestSetLoss(rhoe_test, self.T_test, self.P_test, self.C2_test, True)
                self.T_test_pred = T_test.numpy()
                self.P_test_pred = P_test.numpy()
                self.C2_test_pred = C2_test.numpy()
                self.S_test_pred = S_test.numpy()
                self.CallbackFunction()

            i += 1 

        # Display test set loss information upon completion of training.
        self._test_score = max([self.T_test_loss, self.P_test_loss, self.C2_test_loss])
        return 
    
    def __CheckEarlyStopping(self, current_error:tf.Tensor, worst_error:tf.Tensor):
        """Check for training stagnation.

        :param current_error: maximum error of current epoch.
        :type current_error: tf.Tensor
        :param worst_error: last error value to compare with.
        :type worst_error: tf.Tensor
        :return: minimum between current and worst evaluation error.
        :rtype: tf.Tensor
        """
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
    
    def __TestSetLoss(self, rhoe:tf.Variable, T_test, P_test, C2_test, return_predictions=False):
        T_val_pred, P_val_pred, C2_val_pred, S_val_pred = self.__TD_Evaluation(rhoe)
        T_val_loss = self.__mean_squared_error(y_true=(T_test/(self.Temperature_max - self.Temperature_min)), \
                                                y_pred=(T_val_pred/(self.Temperature_max - self.Temperature_min)))
        P_val_loss = self.__mean_squared_error(y_true=(P_test/(self.Pressure_max - self.Pressure_min)), \
                                                y_pred=(P_val_pred/(self.Pressure_max - self.Pressure_min)))
        C2_val_loss = self.__mean_squared_error(y_true=(C2_test/(self.C2_max - self.C2_min)), \
                                                y_pred=(C2_val_pred/(self.C2_max - self.C2_min)))
        if return_predictions:
            return T_val_loss, P_val_loss, C2_val_loss, T_val_pred, P_val_pred, C2_val_pred, S_val_pred
        else:
            return T_val_loss, P_val_loss, C2_val_loss
    
    def CallbackFunction(self):
        """Callback function called during the training process.
        """

        # Update current weights and biases.
        self._weights = []
        self._biases = []
        for W in self.weights:
            self._weights.append(W.numpy())
        for b in self.biases:
            self._biases.append(b.numpy())   

        # Save SU2 MLP and performance metrics.
        self.Save_Relevant_Data()

        # Plot intermediate history trends.
        self.Plot_Intermediate_History()

        # Error scatter plots for thermodynamic properties.
        self.Generate_Error_Plots()

        # 3D plots of predicted vs true thermodynamic properties.
        self.Generate_3D_Error_Plots()

        return 
    
    def Save_Relevant_Data(self):
        """Write loss values for thermodynamic properties.
        """
        self.TrimWeights()
        
        fid = open(self._save_dir + "/Model_"+str(self._model_index)+"/MLP_C2_performance.txt", "w+")
        fid.write("T test loss: %+.16e\n" % self.T_test_loss)
        fid.write("P test loss: %+.16e\n" % self.P_test_loss)
        fid.write("C2 test loss: %+.16e\n" % self.C2_test_loss)
        fid.close()

        self.write_SU2_MLP(self._save_dir + "/Model_"+str(self._model_index)+"/MLP_TPC2")
        return 
    

    def Plot_and_Save_History(self):
        fig = plt.figure(figsize=[10,10])
        ax = plt.axes()
        ax.plot(np.log10(np.array(self.T_val_errors)), 'r', label='Temperature')
        ax.plot(np.log10(np.array(self.P_val_errors)), 'b', label='Pressure')
        ax.plot(np.log10(np.array(self.C2_val_errors)), 'm', label='Speed of sound')
        ax.plot([0, self._n_epochs], [np.log10(self.T_test_loss), np.log10(self.T_test_loss)], 'r--')
        ax.plot([0, self._n_epochs], [np.log10(self.P_test_loss), np.log10(self.P_test_loss)], 'b--')
        ax.plot([0, self._n_epochs], [np.log10(self.C2_test_loss), np.log10(self.C2_test_loss)], 'm--')
        ax.tick_params(which='both',labelsize=18)
        ax.set_xlabel("Epoch",fontsize=20)
        ax.set_ylabel("Loss value",fontsize=20)
        ax.grid()
        ax.legend(fontsize=20)
        fig.savefig(self.main_save_dir + "/Model_"+str(self.model_index) + "/C2_history.pdf",format='pdf',bbox_inches='tight')
        plt.close(fig)
        return 

    def Plot_Intermediate_History(self):
        fig = plt.figure(figsize=[10,10])
        ax = plt.axes()
        ax.plot(np.log10(np.array(self.T_val_errors)), 'r', label='Temperature')
        ax.plot(np.log10(np.array(self.P_val_errors)), 'b', label='Pressure')
        ax.plot(np.log10(np.array(self.C2_val_errors)), 'm', label='Speed of sound')
        ax.set_xlabel("Epoch",fontsize=20)
        ax.set_ylabel("Loss value",fontsize=20)
        ax.grid()
        ax.legend(fontsize=20)
        ax.tick_params(which='both',labelsize=18)

        fig.savefig(self._save_dir + "/Model_"+str(self._model_index) + "/C2_history.pdf",format='pdf',bbox_inches='tight')
        plt.close(fig)

    def PlotR2Data(self):
        """Plot true vs predicted thermodynamic properties in R2 plot format.
        """

        figformat = "png"
        fig = plt.figure(figsize=[10,10])
        ax = plt.axes() 
        ax.plot([min(self.T_test),max(self.T_test)],[min(self.T_test),max(self.T_test)],'r')
        ax.plot(self.T_test_pred, self.T_test, 'k.')
        ax.grid()
        ax.set_xlabel("Predicted temperature[K]",fontsize=20)
        ax.set_ylabel("Reference temperature[K]",fontsize=20)
        ax.tick_params(which='both',labelsize=18)
        ax.set_title("Temperature R2 plot",fontsize=20)
        fig.savefig(self._save_dir + "/Model_"+str(self._model_index)+"/R2_Temperature."+figformat,format=figformat,bbox_inches='tight')
        plt.close(fig)

        fig = plt.figure(figsize=[10,10])
        ax = plt.axes() 
        ax.plot([min(self.P_test),max(self.P_test)],[min(self.P_test),max(self.P_test)],'r')
        ax.plot(self.P_test_pred, self.P_test, 'k.')
        ax.grid()
        ax.set_xlabel("Predicted pressure [Pa]",fontsize=20)
        ax.set_ylabel("Reference pressure [Pa]",fontsize=20)
        ax.tick_params(which='both',labelsize=18)
        ax.set_title("Pressure R2 plot",fontsize=20)
        fig.savefig(self._save_dir + "/Model_"+str(self._model_index)+"/R2_Pressure."+figformat,format=figformat,bbox_inches='tight')
        plt.close(fig)

        fig = plt.figure(figsize=[10,10])
        ax = plt.axes() 
        ax.plot([min(self.C2_test),max(self.C2_test)],[min(self.C2_test),max(self.C2_test)],'r')
        ax.plot(self.C2_test_pred, self.C2_test, 'k.')
        ax.grid()
        ax.set_xlabel("Predicted squared sos [m2s-2]",fontsize=20)
        ax.set_ylabel("Reference squared sos [m2s-2]",fontsize=20)
        ax.tick_params(which='both',labelsize=18)
        ax.set_title("Squared sos R2 plot",fontsize=20)
        fig.savefig(self._save_dir + "/Model_"+str(self._model_index)+"/R2_sos."+figformat,format=figformat,bbox_inches='tight')
        plt.close(fig)
    
    def Generate_Error_Plots(self):

        self.PlotR2Data()

        figformat = "png"
        plot_fontsize = 20
        label_fontsize=18

        fig = plt.figure(figsize=[10,10])
        ax = plt.axes() 
        cax = ax.scatter(self.X_test[:, self.idx_rho], self.X_test[:, self.idx_e], c=100*np.abs((self.T_test_pred - self.T_test)/self.T_test))
        cbar = plt.colorbar(cax, ax=ax)
        ax.set_xlabel("Density",fontsize=plot_fontsize)
        ax.set_ylabel("Energy",fontsize=plot_fontsize)
        ax.set_title("Temperature prediction error",fontsize=plot_fontsize)
        ax.tick_params(which='both',labelsize=label_fontsize)
        fig.savefig(self._save_dir + "/Model_"+str(self._model_index)+"/Temperature_prediction_error."+figformat,format=figformat,bbox_inches='tight')
        plt.close(fig)

        fig = plt.figure(figsize=[10,10])
        ax = plt.axes() 
        cax = ax.scatter(self.X_test[:, self.idx_rho], self.X_test[:, self.idx_e], c=100*np.abs((self.P_test_pred - self.P_test)/self.P_test))
        cbar = plt.colorbar(cax, ax=ax)
        cbar.set_label(r'Pressure prediction error $(\epsilon_p)[\%]$', rotation=270, fontsize=label_fontsize)
        ax.set_xlabel("Density",fontsize=plot_fontsize)
        ax.set_ylabel("Energy",fontsize=plot_fontsize)
        ax.set_title("Pressure prediction error",fontsize=plot_fontsize)
        ax.tick_params(which='both',labelsize=label_fontsize)
        fig.savefig(self._save_dir + "/Model_"+str(self._model_index)+"/Pressure_prediction_error."+figformat,format=figformat,bbox_inches='tight')
        plt.close(fig)

        fig = plt.figure(figsize=[10,10])
        ax = plt.axes() 
        cax = ax.scatter(self.X_test[:, self.idx_rho], self.X_test[:, self.idx_e], c=100*np.abs((self.C2_test_pred - self.C2_test)/self.C2_test))
        cbar = plt.colorbar(cax, ax=ax)
        ax.set_xlabel("Density",fontsize=plot_fontsize)
        ax.set_ylabel("Energy",fontsize=plot_fontsize)
        ax.set_title("SoS prediction error",fontsize=plot_fontsize)
        ax.tick_params(which='both',labelsize=label_fontsize)
        fig.savefig(self._save_dir + "/Model_"+str(self._model_index)+"/SoS_prediction_error."+figformat,format=figformat,bbox_inches='tight')
        plt.close(fig)

        return

    def Generate_3D_Error_Plots(self):

        figformat = "png"
        plot_fontsize = 20
        label_fontsize=18

        fig = plt.figure(figsize=[10,10])
        ax = plt.axes(projection='3d') 
        ax.plot3D(self.X_test[:, self.idx_rho], self.X_test[:, self.idx_e], self.T_test,'ko',label='Reference')
        ax.plot3D(self.X_test[:, self.idx_rho], self.X_test[:, self.idx_e], self.T_test_pred,'ro',label='Predicted')
        
        ax.set_xlabel("Density",fontsize=plot_fontsize)
        ax.set_ylabel("Energy",fontsize=plot_fontsize)
        ax.set_zlabel("Temperature",fontsize=plot_fontsize)
        ax.set_title("Temperature prediction error",fontsize=plot_fontsize)
        ax.tick_params(which='both',labelsize=label_fontsize)
        ax.legend(fontsize=18)
        fig.savefig(self._save_dir + "/Model_"+str(self._model_index)+"/Temperature_3D_plot."+figformat,format=figformat,bbox_inches='tight')
        plt.close(fig)

        fig = plt.figure(figsize=[10,10])
        ax = plt.axes(projection='3d') 
        ax.plot3D(self.X_test[:, self.idx_rho], self.X_test[:, self.idx_e], self.P_test,'ko',label='Reference')
        ax.plot3D(self.X_test[:, self.idx_rho], self.X_test[:, self.idx_e], self.P_test_pred,'ro',label='Predicted')
        ax.set_xlabel("Density",fontsize=plot_fontsize)
        ax.set_ylabel("Energy",fontsize=plot_fontsize)
        ax.set_zlabel("Pressure",fontsize=plot_fontsize)
        ax.set_title("Pressure prediction",fontsize=plot_fontsize)
        ax.tick_params(which='both',labelsize=label_fontsize)
        ax.legend(fontsize=18)
        fig.savefig(self._save_dir + "/Model_"+str(self._model_index)+"/Pressure_3D_plot."+figformat,format=figformat,bbox_inches='tight')
        plt.close(fig)

        fig = plt.figure(figsize=[10,10])
        ax = plt.axes(projection='3d') 
        ax.plot3D(self.X_test[:, self.idx_rho], self.X_test[:, self.idx_e], np.sqrt(self.C2_test),'ko',label='Reference')
        ax.plot3D(self.X_test[:, self.idx_rho], self.X_test[:, self.idx_e], np.sqrt(self.C2_test_pred),'ro',label='Predicted')
        ax.set_xlabel("Density",fontsize=plot_fontsize)
        ax.set_ylabel("Energy",fontsize=plot_fontsize)
        ax.set_zlabel("Speed of sound",fontsize=plot_fontsize)
        ax.set_title("SoS prediction",fontsize=plot_fontsize)
        ax.tick_params(which='both',labelsize=label_fontsize)
        ax.legend(fontsize=18)
        fig.savefig(self._save_dir + "/Model_"+str(self._model_index)+"/SoS_3D_plot."+figformat,format=figformat,bbox_inches='tight')
        plt.close(fig)

        fig = plt.figure(figsize=[10,10])
        ax = plt.axes(projection='3d') 
        ax.plot3D(self.X_test[:, self.idx_rho], self.X_test[:, self.idx_e], self.S_test,'ko',label='Reference')
        ax.plot3D(self.X_test[:, self.idx_rho], self.X_test[:, self.idx_e], self.S_test_pred[:,0],'ro',label='Predicted')
        ax.set_xlabel("Density",fontsize=plot_fontsize)
        ax.set_ylabel("Energy",fontsize=plot_fontsize)
        ax.set_zlabel("Fluid entropy",fontsize=plot_fontsize)
        ax.tick_params(which='both',labelsize=label_fontsize)
        ax.legend(fontsize=18)
        fig.savefig(self._save_dir + "/Model_"+str(self._model_index)+"/Entropy_3D_plot."+figformat,format=figformat,bbox_inches='tight')
        plt.close(fig)
        return
    
    def write_SU2_MLP(self, file_out:str):
        """Write the network to ASCII format readable by the MLPCpp module in SU2.

        :param file_out: MLP output path and file name.
        :type file_out: str
        """

        n_layers = len(self._weights)+1

        # Select trimmed weight matrices for output.
        weights_for_output = self.weights#self._trimmed_weights
        biases_for_output = self.biases#self._trimmed_biases

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
        fid.write('%i\n' % 1)

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
        fid.write('s\n')
            
        fid.write('\n[output normalization]\n')
        fid.write('%+.16e\t%+.16e\n' % (self._Y_min[0], self._Y_max[0]))

        fid.write("\n</header>\n")
        # Writing the weights of each layer
        fid.write('\n[weights per layer]\n')
        for W in weights_for_output:
            fid.write("<layer>\n")
            for i in range(np.shape(W.numpy())[0]):
                fid.write("\t".join("%+.16e" % float(w) for w in W.numpy()[i, :]) + "\n")
            fid.write("</layer>\n")
        
        # Writing the biases of each layer
        fid.write('\n[biases per layer]\n')
        
        # Input layer biases are set to zero
        fid.write("\t".join("%+.16e" % 0 for _ in self._controlling_vars) + "\n")

        #for B in self.biases:
        for B in biases_for_output:
            fid.write("\t".join("%+.16e" % float(b) for b in B.numpy()) + "\n")

        fid.close()

class EvaluateArchitecture:
    """Class for training MLP architectures
    """

    __Config:EntropicAIConfig
    __trainer_entropy:Train_Entropic_MLP
    __trainer_tpc2:Train_C2_MLP      # MLP trainer object responsible for training itself.

    architecture:list[int] = [30]   # Hidden layer architecture.
    alpha_expo:float = -2.6 # Initial learning rate exponent (base 10)
    lr_decay:float = 0.9985 # Learning rate decay parameter.
    batch_expo:int = 5      # Mini-batch exponent (base 2)
    activation_function:str = "exponential"    # Activation function name applied to hidden layers.

    n_epochs:int = 1000 # Number of epochs to train for.
    save_dir:str        # Directory to save trained networks in.

    device:str = "CPU"      # Hardware to train on.
    process_index:int = 0   # Hardware index.

    current_iter:int=0
    test_score:float        # MLP evaluation score on test set.
    cost_parameter:float    # MLP evaluation cost parameter.

    activation_function_options:list[str] = ["linear","elu","relu","gelu","sigmoid","tanh","swish","exponential"]

    main_save_dir:str = "./"

    def __init__(self, Config_in:EntropicAIConfig):
        """Define EvaluateArchitecture instance and prepare MLP trainer with
        default settings.

        :param Config: FlameletAIConfig object describing the flamelet data manifold.
        :type Config: FlameletAIConfig
        :param group_idx: MLP output group index, defaults to 0
        :type group_idx: int, optional
        :raises Exception: if MLP output group index is undefined by flameletAI configuration.
        """

        self.__Config=Config_in
        self.main_save_dir = self.__Config.GetOutputDir()
        
        # Define MLPTrainer object with default settings (currently only supports TensorFlowFit)
        self.__trainer_tpc2 = Train_C2_MLP()
        self.__trainer_entropy = Train_Entropic_MLP()

        self.__trainer_entropy.SetNEpochs(self.n_epochs)
        self.__trainer_tpc2.SetNEpochs(self.n_epochs)

        
        self.__SynchronizeTrainer()

        self.SetTrainFileHeader(self.__Config.GetOutputDir()+"/"+self.__Config.GetConcatenationFileHeader())
        pass

    def __SynchronizeTrainer(self):
        """Synchronize all MLP trainer settings with locally stored settings.
        """
        self.__trainer_entropy.SetAlphaExpo(self.alpha_expo)
        self.__trainer_entropy.SetLRDecay(self.lr_decay)
        self.__trainer_entropy.SetBatchSize(self.batch_expo)
        self.__trainer_entropy.SetHiddenLayers(self.architecture)

        self.__trainer_tpc2.SetAlphaExpo(self.alpha_expo)
        self.__trainer_tpc2.SetLRDecay(self.lr_decay)
        self.__trainer_tpc2.SetBatchSize(self.batch_expo)
        self.__trainer_tpc2.SetHiddenLayers(self.architecture)
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
    
    def SetNEpochs(self, n_epochs:int=250):
        """Set the number of epochs to train for.

        :param n_epochs: Number of training epoch, defaults to 250.
        :type n_epochs: int
        :raises Exception: provided number of epochs is negative.
        """
        if n_epochs <= 0:
            raise Exception("Number of epochs should be at least one.")
        self.n_epochs = n_epochs
        self.__trainer_entropy.SetNEpochs(self.n_epochs)
        self.__trainer_tpc2.SetNEpochs(self.n_epochs)
        return 
    
    def SetArchitecture(self, NN_hidden_layers:list[int]):
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
        self.__trainer_tpc2.SetHiddenLayers(self.architecture)
        self.__trainer_entropy.SetHiddenLayers(self.architecture)
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
        self.__trainer_tpc2.SetBatchSize(self.batch_expo)
        self.__trainer_entropy.SetBatchSize(self.batch_expo)
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
        self.__trainer_entropy.SetAlphaExpo(alpha_expo_in)
        self.__trainer_tpc2.SetAlphaExpo(alpha_expo_in)
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
        self.__trainer_entropy.SetLRDecay(lr_decay_in)
        self.__trainer_tpc2.SetLRDecay(lr_decay_in)
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
        self.__trainer_entropy.SetDeviceKind(self.device)
        self.__trainer_entropy.SetDeviceIndex(self.process_index)
        self.__trainer_tpc2.SetDeviceKind(self.device)
        self.__trainer_tpc2.SetDeviceIndex(self.process_index)
        return
     
    def __PrepareOutputDir(self):
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
        self.__trainer_entropy.SetModelIndex(self.current_iter)
        self.__trainer_entropy.SetSaveDir(self.main_save_dir)
        self.__trainer_tpc2.SetModelIndex(self.current_iter)
        self.__trainer_tpc2.SetSaveDir(self.main_save_dir)

    def CommenceTraining(self):
        """Initiate the training process.
        """
        self.__PrepareOutputDir()

        # Direct training for entropy prediction as a good first estimate.
        self.__trainer_entropy.Train_MLP()
        self.__trainer_entropy.Save_Relevant_Data()
        self.__trainer_entropy.Plot_and_Save_History()

        # Transfer weights and biases from entropic MLP to physics-informed MLP.
        weights_entropy = self.__trainer_entropy.GetWeights()
        biases_entropy = self.__trainer_entropy.GetBiases()
        self.__trainer_tpc2.SetWeights(weights_entropy)
        self.__trainer_tpc2.SetBiases(biases_entropy)

        # Commence physics-informed learning process.
        self.__trainer_tpc2.Train_MLP()

        # Write all relevant output data.
        fid = open(self.main_save_dir + "/current_iter.txt", "w+")
        fid.write(str(self.current_iter) + "\n")
        fid.close()
        self.test_score = self.__trainer_tpc2.GetTestScore()
        self.cost_parameter = self.__trainer_tpc2.GetCostParameter()
        self.__trainer_tpc2.Save_Relevant_Data()
        return 
    
    def CommenceTraining_OnlyEntropy(self):
        """Only perform training on direct entropy prediction.
        """
        self.__PrepareOutputDir()
        self.__trainer_entropy.Train_MLP()
        self.__trainer_entropy.Save_Relevant_Data()
        self.test_score = self.__trainer_entropy.GetTestScore()
        self.cost_parameter = self.__trainer_entropy.GetCostParameter()
        fid = open(self.main_save_dir + "/current_iter.txt", "w+")
        fid.write(str(self.current_iter) + "\n")
        fid.close()
        return 
    
    def TrainPostprocessing(self):
        """Post-process MLP training by saving all relevant data/figures.
        """
        self.__trainer_entropy.Save_Relevant_Data()
        self.__trainer_entropy.Plot_and_Save_History()
        return 
    
    def GetCostParameter(self):
        """Get MLP evaluation cost parameter.

        :return: MLP cost parameter
        :rtype: float
        """
        return self.__trainer_entropy.GetCostParameter()
    
    def GetTestScore(self):
        """Get MLP evaluation test score upon completion of training.

        :return: MLP evaluation test score.
        :rtype: float
        """
        return self.test_score
    
    def SetTrainFileHeader(self, fileheader:str):
        """Set file path for custom training data file.
        """
        self.__trainer_entropy.SetTrainFileHeader(fileheader)
        self.__trainer_tpc2.SetTrainFileHeader(fileheader)
        return 
    