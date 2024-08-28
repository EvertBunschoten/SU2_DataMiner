###############################################################################################
#       #      _____ __  _____      ____        __        __  ____                   #        #
#       #     / ___// / / /__ \    / __ \____ _/ /_____ _/  |/  (_)___  ___  _____   #        #
#       #     \__ \/ / / /__/ /   / / / / __ `/ __/ __ `/ /|_/ / / __ \/ _ \/ ___/   #        #
#       #    ___/ / /_/ // __/   / /_/ / /_/ / /_/ /_/ / /  / / / / / /  __/ /       #        #
#       #   /____/\____//____/  /_____/\__,_/\__/\__,_/_/  /_/_/_/ /_/\___/_/        #        #
#       #                                                                            #        #
###############################################################################################

############################### FILE NAME: optimizeHP.py ######################################
#=============================================================================================#
# author: Evert Bunschoten                                                                    |
#    :PhD Candidate ,                                                                         |
#    :Flight Power and Propulsion                                                             |
#    :TU Delft,                                                                               |
#    :The Netherlands                                                                         |
#                                                                                             |
#                                                                                             |
# Description:                                                                                |
#   Optimize the hyper-parameters describing the MLP architecture and learning parameters for |
#   accuracy and the number of weights in the network.                                        |
#                                                                                             |
# Version: 1.0.0                                                                              |
#                                                                                             |
#=============================================================================================#

import pygad
import os 
import pickle 
import csv
import numpy as np
import matplotlib.pyplot as plt 
from scipy.optimize import minimize, Bounds, differential_evolution
from multiprocessing import current_process

from Common.Properties import DefaultProperties, DefaultSettings_NICFD
from Common.Config_base import Config 
from Common.DataDrivenConfig import FlameletAIConfig, EntropicAIConfig
from Manifold_Generation.MLP.Trainer_Base import EvaluateArchitecture
from Manifold_Generation.MLP.Trainers_NICFD.Trainers import EvaluateArchitecture_NICFD
from Manifold_Generation.MLP.Trainers_FGM.Trainers import EvaluateArchitecture_FGM

from Common.Properties import ActivationFunctionOptions

class MLPOptimizer:
    """Class for hyper-parameter optimization of entropic fluid model multi-layer perceptrons.
    """

    _Config:Config = None     # EntropicAI configuration.
    __optimizer:pygad.GA = None         # PyGaD optimization instance.
    _n_workers:int = 1                 # Number of CPU cores used for distributing the work per generation.
    _n_epochs:int=DefaultProperties.N_epochs
    # Hyper-parameter default settings and bounds.

    # Mini-batch exponent (base 2) for training.
    _optimizebatch:bool = True 
    __batch_expo:int = DefaultProperties.batch_size_exponent
    __batch_expo_min:int=3
    __batch_expo_max:int=7
    
    # Optimize learning rate decay parameters.
    _optimizeLR:bool = True 

    # Initial learning rate exponent (base 10).
    __alpha_expo:float = DefaultProperties.init_learning_rate_expo
    __alpha_expo_min:float = -4.0
    __alpha_expo_max:float = -1.0

    # Learning rate decay parameter for exponential learning rate decay schedule.
    __lr_decay:float=DefaultProperties.learning_rate_decay
    __lr_decay_min:float = 0.9
    __lr_decay_max:float = 1.0 

    # Optimize hidden layer architecture.
    _optimizeNN:bool = True

    _optimizephi:bool = True 

    # Number of perceptrons applied to the hidden layer of the network.
    __NN_min:int = 10
    __NN_max:int = 100 
    __architecture:list[int]=[30]

    __activation_function:str = "exponential"
    __activation_function_options:list[str] = [a for a in ActivationFunctionOptions.keys()]

    # Optimization history.
    __population_history:list = []
    __fitness_history:list = []
    __generation_number:int = 0

    __x_optim:np.ndarray = None 
    
    _history_extenstion:str = ""

    def __init__(self, Config_in:Config=None, load_file:str=None):
        """Class constructor
        """
        # Store configuration
        self._Config = Config_in 

        self.__alpha_expo = self._Config.GetAlphaExpo()
        self.__lr_decay = self._Config.GetLRDecay()
        self.__batch_expo = self._Config.GetBatchExpo()
        self.__activation_function = self._Config.GetActivationFunction()
        self.SetArchitecture(self._Config.GetHiddenLayerArchitecture())

        if load_file:
            print("Loading optimizer configuration")
            with open(load_file, "rb") as fid:
                loaded_config = pickle.load(fid)
            print(loaded_config.__dict__)
            self.__dict__ = loaded_config.__dict__.copy()
            print("Loaded optimizer file")
        
        return 

    def SetNWorkers(self, n_workers:int=1):
        """Set the number of workers used for work distribution of training each generation.

        :param n_workers: number of processors used, defaults to 1
        :type n_workers: int, optional
        :raises Exception: if number of workers is lower than one.
        """
        if n_workers < 1:
            raise Exception("Number of workers should be at least one.")
        self._n_workers = n_workers
        return 
    
    def SetNEpochs(self, n_epochs:int=DefaultProperties.N_epochs):
        if n_epochs <=0:
            raise Exception("Training should occur for at least one epoch.")
        self._n_epochs = n_epochs
        return 
    
    def Optimize_LearningRate_HP(self, optimize_LR:bool=True):
        """Consider learning-rate hyper-parameters in optimization.

        :param optimize_LR: consider learning rate parameters(True, default), or not (False)
        :type optimize_LR: bool, optional
        """
        self._optimizeLR = optimize_LR
        return 
    
    def Optimize_ActivationFunction(self, optimize_phi:bool=True):
        """Consider the hidden layer activation function in the optimization.

        :param optimize_phi: consider the hidden layer activation function (True, default), or not (False)
        :type optimize_phi: bool, optional
        """
        self._optimizephi = optimize_phi
        return
    
    def SetAlpha_Expo(self, val_alpha_expo:float=DefaultSettings_NICFD.init_learning_rate_expo):
        """Set initial learning rate exponent (base 10).

        :param val_alpha_expo: _description_, defaults to -2.8
        :type val_alpha_expo: float, optional
        :raises Exception: if initial learning rate exponent value is positive.
        """

        if val_alpha_expo >= 0:
            raise Exception("Initial learing rate exponent should be negative.")
        self.__alpha_expo = val_alpha_expo

        return 
    
    def SetLR_Decay(self, val_lr_decay:float=DefaultSettings_NICFD.learning_rate_decay):
        """Set the learning rate decay parameter value.

        :param val_lr_decay: learning rate decay parameter, defaults to 0.996
        :type val_lr_decay: float, optional
        :raises Exception: if learning rate decay parameter is not between 0 and 1.
        """
        if val_lr_decay > 1.0 or val_lr_decay < 0.0:
            raise Exception("Learning rate decay parameter should be between 0 and 1")
        self.__lr_decay = val_lr_decay 

        return 
    
    def SetBounds_Alpha_Expo(self, alpha_expo_min:float=-3.0, alpha_expo_max:float=-1.0):
        """Set minimum and maximum values for the initial learning rate exponent (base 10) during optimization.

        :param alpha_expo_min: minimum initial learning rate exponent, defaults to -3.0
        :type alpha_expo_min: float, optional
        :param alpha_expo_max: maximum initial learning rate exponent, defaults to -1.0
        :type alpha_expo_max: float, optional
        :raises Exception: if lower bound exceeds upper bound.
        :raises Exception: if positive values are provided.
        """
        if alpha_expo_max <= alpha_expo_min:
            raise Exception("Upper bound value should exceed lower bound value.")
        if alpha_expo_min > 0 or alpha_expo_max > 0:
            raise Exception("Initial learning rate exponent value should be negative.")
        self.__alpha_expo_min = alpha_expo_min
        self.__alpha_expo_max = alpha_expo_max
        return 
    
    def Optimize_Batch_HP(self, optimize_batch:bool=True):
        """Consider the mini-batch size exponent as a hyper-parameter during optimization.

        :param optimize_batch: consider mini-batch size exponent (True, default), or not (False)
        :type optimize_batch: bool, optional
        """
        self._optimizebatch = optimize_batch 
        return 
    
    def SetBatch_Expo(self, batch_expo:int=DefaultSettings_NICFD.batch_size_exponent):
        """Set training batch exponent value (base 2).

        :param batch_expo: training batch exponent value, defaults to 6
        :type batch_expo: int, optional
        :raises Exception: if training batch exponent value is lower than 1.
        """

        if batch_expo < 1:
            raise Exception("Batch size exponent should be at least 1.")
        self.__batch_expo = batch_expo 
        
        return 
    
    def SetActivationFunction(self, activation_function:str=DefaultSettings_NICFD.activation_function):
        """Set the hiden layer activation function name.

        :param activation_function: hidden layer activation function name, defaults to DefaultSettings_NICFD.activation_function
        :type activation_function: str, optional
        """

        self.__activation_function = activation_function
        return
    
    def SetBounds_Batch_Expo(self, batch_expo_min:int=3, batch_expo_max:int=7):
        """Set minimum and maximum values for the training batch exponent (base 2) during optimization.

        :param batch_expo_min: minimum batch exponent, defaults to 3
        :type batch_expo_min: int, optional
        :param batch_expo_max: maximum batch exponent, defaults to 7
        :type batch_expo_max: int, optional
        :raises Exception: if lower bound exceeds upper bound.
        :raises Exception: if batch exponent is lower than one.
        """

        if batch_expo_max <= batch_expo_min:
            raise Exception("Upper bound value should exceed lower bound value.")
        if batch_expo_min < 1 or batch_expo_max < 1:
            raise Exception("Training batch exponent value should exceed 1")
        self.__batch_expo_min = batch_expo_min
        self.__batch_expo_max = batch_expo_max

        return
    
    def Optimize_Architecture_HP(self, optimize_architecture:bool=True):
        """Consider the hidden layer perceptron count as a hyper-parameter during optimization.

        :param optimize_architecture: consider hidden layer perceptron count (True, default) or not (False)
        :type optimize_architecture: bool, optional
        """
        self._optimizeNN = optimize_architecture 
        return 
    
    def SetArchitecture(self, architecture:list[int]=DefaultSettings_NICFD.hidden_layer_architecture):
        """Set MLP hidden layer architecture.

        :param architecture: list with perceptron count per hidden layer, defaults to [40]
        :type architecture: list[int], optional
        :raises Exception: if any of the layers has fewer than one perceptron.
        """

        if any(tuple(NN<1 for NN in architecture)):
            raise Exception("At least one perceptron should be applied per hidden layer.")
        self.__architecture = []
        for NN in architecture:
            self.__architecture.append(NN)

        return 
    
    def SetBounds_Architecture(self, NN_min:int=10, NN_max:int=100):
        """Set the minimum and maximum values for the perceptron count in the hidden layer.

        :param NN_min: minimum number of perceptrons, defaults to 10
        :type NN_min: int, optional
        :param NN_max: maximum number of perceptrons, defaults to 100
        :type NN_max: int, optional
        :raises Exception: if lower value exceeds upper value.
        :raises Exception: if perceptron count is lower than one.
        """

        if NN_min >= NN_max:
            raise Exception("Upper bound value should exceed lower bound value.")
        if NN_min <= 1 or NN_max <= 1:
            raise Exception("At least one hidden layer perceptron should be used.")
        self.__NN_min = NN_min 
        self.__NN_max = NN_max 

        return 
    
    def __prepareBounds(self):
        """Prepare hyper-parameter bounds prior to optimization
        """

        # Store hyper-parameter data type, lower, and upper bound.
        gene_trainparams = []
        lowerbound = []
        upperbound = []
        if self._optimizebatch:
            gene_trainparams.append(int)
            lowerbound.append(self.__batch_expo_min)
            upperbound.append(self.__batch_expo_max)
        if self._optimizeLR:
            gene_trainparams.append(float)
            gene_trainparams.append(float)
            
            lowerbound.append(self.__alpha_expo_min)
            lowerbound.append(self.__lr_decay_min)
            upperbound.append(self.__alpha_expo_max)
            upperbound.append(self.__lr_decay_max)
        if self._optimizeNN:
            gene_trainparams.append(int)
            lowerbound.append(self.__NN_min)
            upperbound.append(self.__NN_max)
        if self._optimizephi:
            gene_trainparams.append(int)
            lowerbound.append(0)
            upperbound.append(len(self.__activation_function_options))

        return gene_trainparams, lowerbound, upperbound 
    
    def __prepareBounds_DE(self):
        lowerbound = []
        upperbound = []
        integrality = []
        if self._optimizebatch:
            integrality.append(True)
            lowerbound.append(self.__batch_expo_min)
            upperbound.append(self.__batch_expo_max)
        if self._optimizeLR:
            integrality.append(False)
            integrality.append(False)
            lowerbound.append(self.__alpha_expo_min)
            lowerbound.append(self.__lr_decay_min)
            upperbound.append(self.__alpha_expo_max)
            upperbound.append(self.__lr_decay_max)
        if self._optimizeNN:
            integrality.append(True)
            lowerbound.append(int(self.__NN_min))
            upperbound.append(int(self.__NN_max))
        if self._optimizephi:
            integrality.append(True)
            lowerbound.append(0)
            upperbound.append(len(self.__activation_function_options))

        return np.array(integrality), np.array(lowerbound), np.array(upperbound)
    
    def __setOptimizer(self):
        run_DE = False
        if self._optimizebatch or self._optimizephi or self._optimizeNN:
            run_DE = True 

        print("Initializing hyper-parameter optimization for:")
        if self._optimizebatch:
            print("- mini-batch size exponent")
        if self._optimizeLR:
            print("- learning rate parameters")
        if self._optimizeNN:
            print("- hidden layer neuron count")
        if self._optimizephi:
            print("- hidden layer activation function")

        if run_DE:
            print("Initializing differential evolution algorithm")
            self.__setOptimizer_DE()
        else:
            print("Initializing simplex optimizer")
            self.__setSimplexOptimizer()

        return
    
    def __setOptimizer_DE(self):
        integrality, lowerbound, upperbound = self.__prepareBounds_DE()
        N_genes = len(integrality)
        bounds = Bounds(lb=lowerbound, ub=upperbound)
        # Determine update strategy based on whether parallel processing is used.
        if self._n_workers > 1:
            update_strategy = "deferred"
        else:
            update_strategy = "immediate"
        self.__generation_number = 0
        result = differential_evolution(func = self.fitnessFunction,\
                                        callback=self.saveGenerationInfo_DE,\
                                        maxiter=10*N_genes,\
                                        popsize=10,\
                                        bounds=bounds,\
                                        workers=self._n_workers,\
                                        updating=update_strategy,\
                                        integrality=integrality,\
                                        strategy='best1exp',\
                                        seed=1,\
                                        tol=1e-6)
        self.__x_optim = result.x 

        return 

    
    
    def __setSimplexOptimizer(self):
        gene_trainparams, lowerbound, upperbound = self.__prepareBounds()
        N_genes = len(gene_trainparams)
        x0 = np.zeros(len(lowerbound))
        x0[0] = self.__alpha_expo
        x0[1] = self.__lr_decay
        bounds = Bounds(lb=lowerbound, ub=upperbound)
        options = {"maxiter":1,\
                   "disp":True}
        res = minimize(self.fitnessFunction, x0=x0, method='Nelder-Mead',bounds=bounds,options=options, callback=self.saveGenerationInfo_DE)

        self.__x_optim = res.x 

        return 
    
    def _PostProcessing(self, x):

        idx_x = 0
        print("Optimized hyper-parameters:")
        if self._optimizebatch:
            batch_expo = int(x[idx_x])
            self._Config.SetBatchExpo(batch_expo)
            idx_x += 1 
            print("- mini-batch exponent: %i" % batch_expo)
        if self._optimizeLR:
            alpha_expo = x[idx_x]
            idx_x += 1 
            print("- initial learning rate exponent: %.5e" % (alpha_expo))
            self._Config.SetAlphaExpo(alpha_expo)
            lr_decay = x[idx_x]
            print("- learning rate decay parameter: %.5e" % lr_decay)
            self._Config.SetLRDecay(lr_decay)
            idx_x += 1 
        if self._optimizeNN:
            architecture = [int(x[idx_x])]
            idx_x += 1 
            print("- hidden layer architecture: "+ " ".join(("%i" % n) for n in architecture))
            self._Config.SetHiddenLayerArchitecture(architecture)
        if self._optimizephi:
            phi = self.__activation_function_options[int(x[idx_x])]
            idx_x += 1
            print("- hidden layer activation function: %s" % phi)
            self._Config.SetActivationFunction(phi)
        
        self._Config.SaveConfig()
        return
    
    def PreprocessOptimization(self):
        if not any((self._optimizebatch, self._optimizeLR, self._optimizeNN, self._optimizephi)):
            raise Exception("At least one of the hyper-parameter options should be considered for optimization.")
        # Prepare optimization history output file.
        self._history_extension = ""
        n_params = 0
        if self._optimizebatch:
            self._history_extension += "B"
            n_params += 1
        if self._optimizeLR:
            self._history_extension += "LR"
            n_params += 1
        if self._optimizeNN:
            self._history_extension += "A"
            n_params += 1
        if self._optimizephi:
            self._history_extension += "Phi"

        self.CreateOutputs()

        return 
    
    def CreateOutputs(self):
        self.opt_history_filepath = self._Config.GetOutputDir() + "/history_opt_"+self._history_extension+".csv"
        with open(self.opt_history_filepath, "w+") as fid:
            if self._optimizeNN:
                fid.write("Iteration,mini-batch exp, activation function, alpha expo, lr decay, architecture,fitness,cost\n")
            else:
                fid.write("Iteration,mini-batch exp, activation function, alpha expo, lr decay, architecture,solution,fitness\n")
        # Prepare optimization output directory.
        self.save_dir = self._Config.GetOutputDir()+"/Architectures_Optim"+self._history_extension
        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)
        return 
    
    def optimizeHP(self):
        """Initate hyper-parameter optimization routine.

        :raises Exception: if neither of the available sets of hyper-parameters are considered for optimization.
        """
        self.PreprocessOptimization()
        
        # Prepare bounds and commence hyper-parameter optimization.
        self.__setOptimizer()

        # Extract optimized hyper-parameters and update configuration.
        self._PostProcessing(self.__x_optim)
        return 
    
    def saveGenerationInfo_DE(self, x, convergence=False):
        #x = ga_result.x 
        f_best = convergence 
        idx_x = 0
        batch_expo = self.__batch_expo
        alpha_expo = self.__alpha_expo
        lr_decay = self.__lr_decay
        architecture = self.__architecture
        phi = self.__activation_function
        if self._optimizebatch:
            batch_expo = int(x[idx_x])
            idx_x += 1 
        if self._optimizeLR:
            alpha_expo = x[idx_x]
            idx_x += 1 
            lr_decay = x[idx_x]
            idx_x += 1 
        if self._optimizeNN:
            architecture = [int(x[idx_x])]
            idx_x += 1 
        if self._optimizephi:
            phi = self.__activation_function_options[x[idx_x]]
            idx_x += 1

        line_to_write = ("%i,%i,%s,%+.5e,%+.5e," % (self.__generation_number, batch_expo, phi, alpha_expo, lr_decay))
        line_to_write += ",".join(("%i" % n) for n in architecture)
        line_to_write += ",%+.6e" % f_best
        line_to_write += "\n"
        with open(self.opt_history_filepath, 'a+') as fid:
            fid.write(line_to_write)
        self.__generation_number += 1
        return
    
    def saveGenerationInfo(self, ga_instance:pygad.GA):
        """Save population information per completed generation.
        """

        # Collect population parameters and fitness
        population = ga_instance.population
        pop_fitness = ga_instance.last_generation_fitness

        # Scale fitness to test set evaluation score.
        test_score = np.power(10, -pop_fitness)

        # Update history.
        self.__population_history.append(population)
        self.__fitness_history.append(pop_fitness)
        generation = ga_instance.generations_completed

        # Write population data to history file.
        pop_and_fitness = np.hstack((generation*np.ones([len(population),1]), population, test_score[:, np.newaxis]))
        with open(self.opt_history_filepath, "a+") as fid:
            csvWriter =csv.writer(fid, delimiter=',')
            csvWriter.writerows(pop_and_fitness)

        return 
    
    def _translateGene(self, x:np.ndarray[float], Evaluator:EvaluateArchitecture):
        """Translate gene to hyper-parameters

        :param x: gene as passed from genetic algorithm.
        :type x: np.ndarray[float]
        :param Evaluator: MLP evaluation class instance.
        :type Evaluator: EvaluateArchitecture
        """

        # Set default hyper-parameters.
        Evaluator.SetBatchExpo(self.__batch_expo)
        Evaluator.SetAlphaExpo(self.__alpha_expo)
        Evaluator.SetLRDecay(self.__lr_decay)
        Evaluator.SetHiddenLayers(self.__architecture)
        Evaluator.SetActivationFunction(self.__activation_function)

        print(x)
        # Set hyper-parameter according to gene.
        idx_x = 0
        if self._optimizebatch:
            batch_expo = int(x[idx_x])
            Evaluator.SetBatchExpo(batch_expo)
            idx_x += 1 
        if self._optimizeLR:
            alpha_expo = x[idx_x]
            Evaluator.SetAlphaExpo(alpha_expo)
            idx_x += 1 
            lr_decay = x[idx_x]
            Evaluator.SetLRDecay(lr_decay)
            idx_x += 1 
        if self._optimizeNN:
            architecture = [int(x[idx_x])]
            Evaluator.SetHiddenLayers(architecture)
            idx_x += 1 
        if self._optimizephi:
            phi = self.__activation_function_options[int(x[idx_x])]
            Evaluator.SetActivationFunction(phi)
            idx_x += 1

        return

    def fitnessFunction(self, x:np.ndarray):
        if self._n_workers > 1:
            p = current_process()
            worker_idx = p._identity[0]
        else:
            worker_idx = 0
        Evaluator:EvaluateArchitecture = EvaluateArchitecture_NICFD(self._Config)

        # Set CPU index.
        Evaluator.SetTrainHardware("CPU", worker_idx)

        Evaluator.SetVerbose(0)

        # Translate gene and update hyper-parameters.
        self._translateGene(x, Evaluator=Evaluator)

        # Set output directory for MLP training callbacks.
        Evaluator.SetSaveDir(self.save_dir)

        # Train for 1000 epochs direct and physics-informed.
        Evaluator.SetNEpochs(1000)
        Evaluator.CommenceTraining()

        # Extract test set evaluation score.
        Evaluator.TrainPostprocessing()
        test_score = Evaluator.GetTestScore() 

        # Scale test set loss to fitness.
        fitness_test_score = np.log10(test_score)

        # Free up memory
        del Evaluator 

        return fitness_test_score

    def inv_fitnessFunction(self, x):
        return -self.fitnessFunction(x=x)
    
    
    def SaveOptimizer(self, file_name:str):
        """Save optimizer instance.

        :param file_name: file name under which to save the optimizer instance.
        :type file_name: str
        """

        file = open(self._Config.GetOutputDir()+"/"+file_name+'.hpo','wb')
        pickle.dump(self, file)
        file.close()
        return 
    

class MLPOptimizer_FGM(MLPOptimizer):
    __output_group:int = 0 
    def __init__(self, Config_in:FlameletAIConfig):
        MLPOptimizer.__init__(self, Config_in)
        return
    
    def SetOutputGroup(self, output_group:int=0):
        self.__output_group = output_group
        self.SetAlpha_Expo(self._Config.GetAlphaExpo(output_group))
        self.SetLR_Decay(self._Config.GetLRDecay(output_group))
        return 
    
    def CreateOutputs(self):
        self.opt_history_filepath = self._Config.GetOutputDir() + "/history_optim_Group"+str(self.__output_group+1)+"_"+self._history_extension+".csv"
        with open(self.opt_history_filepath, "w+") as fid:
            if self._optimizeNN:
                fid.write("Iteration,mini-batch exp, activation function, alpha expo, lr decay, architecture,fitness,cost\n")
            else:
                fid.write("Iteration,mini-batch exp, activation function, alpha expo, lr decay, architecture,solution,fitness\n")
        # Prepare optimization output directory.
        self.save_dir = self._Config.GetOutputDir()+"/Architectures_Group"+str(self.__output_group+1)+"_Optim"+self._history_extension
        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)
        return 
    
    def fitnessFunction(self, x:np.ndarray):
        if self._n_workers > 1:
            p = current_process()
            worker_idx = p._identity[0]
        else:
            worker_idx = 0
        Evaluator:EvaluateArchitecture = EvaluateArchitecture_FGM(self._Config, self.__output_group)

        # Set CPU index.
        Evaluator.SetTrainHardware("CPU", worker_idx)

        Evaluator.SetVerbose(0)

        # Translate gene and update hyper-parameters.
        self._translateGene(x, Evaluator=Evaluator)

        # Set output directory for MLP training callbacks.
        Evaluator.SetSaveDir(self.save_dir)

        # Train for 1000 epochs direct and physics-informed.
        Evaluator.SetNEpochs(self._n_epochs)
        Evaluator.CommenceTraining()

        # Extract test set evaluation score.
        Evaluator.TrainPostprocessing()
        test_score = Evaluator.GetTestScore() 

        # Scale test set loss to fitness.
        fitness_test_score = np.log10(test_score)

        # Free up memory
        del Evaluator 

        return fitness_test_score
    
    def saveGenerationInfo_DE(self, x, convergence=False):
        self._PostProcessing(x)
        
        return super().saveGenerationInfo_DE(x, convergence)

    def _PostProcessing(self, x):

        Config:FlameletAIConfig = FlameletAIConfig(self._Config.GetConfigName() + ".cfg")
        idx_x = 0
        print("Optimized hyper-parameters:")
        if self._optimizebatch:
            batch_expo = int(x[idx_x])
            Config.SetBatchExpo(batch_expo, self.__output_group)
            idx_x += 1 
            print("- mini-batch exponent: %i" % batch_expo)
        if self._optimizeLR:
            alpha_expo = x[idx_x]
            idx_x += 1 
            print("- initial learning rate exponent: %.5e" % (alpha_expo))
            Config.SetAlphaExpo(alpha_expo, self.__output_group)
            lr_decay = x[idx_x]
            print("- learning rate decay parameter: %.5e" % lr_decay)
            self._Config.SetLRDecay(lr_decay, self.__output_group)
            idx_x += 1 
        if self._optimizeNN:
            architecture = [int(x[idx_x])]
            idx_x += 1 
            print("- hidden layer architecture: "+ " ".join(("%i" % n) for n in architecture))
            Config.SetHiddenLayerArchitecture(architecture, self.__output_group)
        if self._optimizephi:
            phi = self.__activation_function_options[int(x[idx_x])]
            idx_x += 1
            print("- hidden layer activation function: %s" % phi)
            Config.SetActivationFunction(phi, self.__output_group)
        
        Config.SaveConfig()
        return
    
class MLPOptimizer_NICFD(MLPOptimizer):

    def __init__(self, Config_in:EntropicAIConfig):
        MLPOptimizer.__init__(self, Config_in)
        return
    
    def CreateOutputs(self):
        self.opt_history_filepath = self._Config.GetOutputDir() + "/history_entropy_"+self._history_extension+".csv"
        with open(self.opt_history_filepath, "w+") as fid:
            if self._optimizeNN:
                fid.write("Iteration,mini-batch exp, activation function, alpha expo, lr decay, architecture,fitness,cost\n")
            else:
                fid.write("Iteration,mini-batch exp, activation function, alpha expo, lr decay, architecture,solution,fitness\n")
        # Prepare optimization output directory.
        self.save_dir = self._Config.GetOutputDir()+"/Architectures_Optim"+self._history_extension
        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)
        return 
    
    def fitnessFunction(self, x:np.ndarray):
        if self._n_workers > 1:
            p = current_process()
            worker_idx = p._identity[0]
        else:
            worker_idx = 0
        Evaluator:EvaluateArchitecture = EvaluateArchitecture_NICFD(self._Config)

        # Set CPU index.
        Evaluator.SetTrainHardware("CPU", worker_idx)

        Evaluator.SetVerbose(0)

        # Translate gene and update hyper-parameters.
        self._translateGene(x, Evaluator=Evaluator)

        # Set output directory for MLP training callbacks.
        Evaluator.SetSaveDir(self.save_dir)

        # Train for 1000 epochs direct and physics-informed.
        Evaluator.SetNEpochs(self._n_epochs)
        Evaluator.CommenceTraining()

        # Extract test set evaluation score.
        Evaluator.TrainPostprocessing()
        test_score = Evaluator.GetTestScore() 

        # Scale test set loss to fitness.
        fitness_test_score = np.log10(test_score)

        # Free up memory
        del Evaluator 

        return fitness_test_score
    
    
class PlotHPOResults:
    __Config:Config = None 
    _optimizelearningrate:bool = True 
    _optimizebatch:bool = True 
    _optimizearchitecture:bool = True 
    _optimizephi:bool = True 

    __completed_models:list[str] = []
    __C2_test_set_scores:list[float] = []
    __P_test_set_scores:list[float] = []
    __T_test_set_scores:list[float] = []
    __hidden_layer_neurons:list[int] = []
    __lr_decay:list[float] = []
    __alpha_expo:list[float] = []
    __batch_expo:list[int] = [] 
    __phi_idx:list[int] = []

    __completed_workers:list[int] = []
    __completed_models:list[int] = [] 

    def __init__(self, Config_in:Config):
        
        self.__Config = Config_in 
        return 
    
    
    def Optimize_LearningRate_HP(self, optimize_LR:bool=True):
        """Consider learning-rate hyper-parameters in optimization.

        :param optimize_LR: consider learning rate parameters(True, default), or not (False)
        :type optimize_LR: bool, optional
        """
        self._optimizelearningrate = optimize_LR
        return 
    
    def Optimize_Batch_HP(self, optimize_batch:bool=True):
        """Consider the mini-batch size exponent as a hyper-parameter during optimization.

        :param optimize_batch: consider mini-batch size exponent (True, default), or not (False)
        :type optimize_batch: bool, optional
        """
        self._optimizebatch = optimize_batch 
        return 
    
    def Optimize_Architecture_HP(self, optimize_architecture:bool=True):
        """Consider the hidden layer perceptron count as a hyper-parameter during optimization.

        :param optimize_architecture: consider hidden layer perceptron count (True, default) or not (False)
        :type optimize_architecture: bool, optional
        """
        self._optimizearchitecture = optimize_architecture 
        return 
    
    def Optimize_Activation_HP(self, optimize_activation_function:bool=True):
        self._optimizephi = optimize_activation_function
        return 
    
    def ReadArchitectures(self):
        optim_header = "Architectures_Optim"
        if self._optimizebatch:
            optim_header += "B"
        if self._optimizelearningrate:
            optim_header += "LR"
        if self._optimizearchitecture:
            optim_header += "A"
        if self._optimizephi:
            optim_header += "Phi"

        optim_directory = self.__Config.GetOutputDir()+"/"+optim_header

        population_indices = os.listdir(optim_directory)
        self.__completed_models = []
        for p in population_indices:
            if "Worker" in p:
                models = os.listdir(optim_directory + "/" + p)
                for m in models:
                    if "Model" in m:
                        if os.path.isfile(optim_directory + "/" + p + "/" + m + "/MLP_NICFD_PINN_performance.txt"):

                            model_idx = int(m.split("_")[-1])
                            worker_idx = int(p.split("_")[-1])
                            with open(optim_directory + "/" + p + "/" + m + "/MLP_NICFD_PINN_performance.txt",'r') as fid:
                                lines = fid.readlines()
                                T_loss = float(lines[1].strip().split(':')[-1])
                                P_loss = float(lines[2].strip().split(':')[-1])
                                C2_loss = float(lines[3].strip().split(':')[-1])
                                self.__T_test_set_scores.append(T_loss)
                                self.__P_test_set_scores.append(P_loss)
                                self.__C2_test_set_scores.append(C2_loss)

                            with open(optim_directory + "/" + p + "/" + m + "/MLP_performance.txt",'r') as fid:
                                lines = fid.readlines()
                                nN = int(lines[2].strip().split(":")[-1])
                                alpha_expo = float(lines[5].strip().split(":")[-1])
                                lr_decay = float(lines[6].strip().split(":")[-1])
                                batch_expo = int(lines[7].strip().split(":")[-1])
                                phi_idx = int(lines[9].strip().split(":")[-1])
                                self.__alpha_expo.append(alpha_expo)
                                self.__lr_decay.append(lr_decay)
                                self.__batch_expo.append(batch_expo)
                                self.__hidden_layer_neurons.append(nN)
                                self.__phi_idx.append(phi_idx)

                            self.__completed_models.append(model_idx)
                            self.__completed_workers.append(worker_idx)

        return 
    
    def PlotLossAlphaExpo(self):
        fig = plt.figure(figsize=[27, 9])
        ax0 = fig.add_subplot(1,3,1)
        ax0.plot(self.__T_test_set_scores, self.__alpha_expo, 'ro')
        ax0.grid()
        ax0.set_xscale('log')
        ax0.set_xlabel('Temperature test set evaluation loss',fontsize=20)
        ax0.set_ylabel('Initial learning rate exponent', fontsize=20)
        ax0.tick_params(which='both',labelsize=18)

        ax1 = fig.add_subplot(1,3,2)
        ax1.plot(self.__P_test_set_scores, self.__alpha_expo, 'bo')
        ax1.grid()
        ax1.set_xscale('log')
        ax1.set_xlabel('Pressure test set evaluation loss',fontsize=20)
        ax1.tick_params(which='both',labelsize=18)

        ax2 = fig.add_subplot(1,3,3)
        ax2.plot(self.__C2_test_set_scores, self.__alpha_expo, 'mo')
        ax2.grid()
        ax2.set_xscale('log')
        ax2.set_xlabel('Speed of sound test set evaluation loss',fontsize=20)
        ax2.tick_params(which='both',labelsize=18)

        for nn, sT, sP, sC2, w, m in zip(self.__alpha_expo, self.__T_test_set_scores, self.__P_test_set_scores, self.__C2_test_set_scores, self.__completed_workers, self.__completed_models):
            ax0.text(sT, nn, "W"+str(w)+"M"+str(m),color='r')
            ax1.text(sP, nn, "W"+str(w)+"M"+str(m),color='b')
            ax2.text(sC2, nn, "W"+str(w)+"M"+str(m),color='m')
        plt.show()
        return 
    
    def PlotLossLRDecay(self):
        fig = plt.figure(figsize=[27, 9])
        ax0 = fig.add_subplot(1,3,1)
        ax0.plot(self.__T_test_set_scores, self.__lr_decay, 'ro')
        ax0.grid()
        ax0.set_xscale('log')
        ax0.set_xlabel('Temperature test set evaluation loss',fontsize=20)
        ax0.set_ylabel('Learning rate decay parameter', fontsize=20)
        ax0.tick_params(which='both',labelsize=18)

        ax1 = fig.add_subplot(1,3,2)
        ax1.plot(self.__P_test_set_scores, self.__lr_decay, 'bo')
        ax1.grid()
        ax1.set_xscale('log')
        ax1.set_xlabel('Pressure test set evaluation loss',fontsize=20)
        ax1.tick_params(which='both',labelsize=18)

        ax2 = fig.add_subplot(1,3,3)
        ax2.plot(self.__C2_test_set_scores, self.__lr_decay, 'mo')
        ax2.grid()
        ax2.set_xscale('log')
        ax2.set_xlabel('Speed of sound test set evaluation loss',fontsize=20)
        ax2.tick_params(which='both',labelsize=18)

        for nn, sT, sP, sC2, w, m in zip(self.__lr_decay, self.__T_test_set_scores, self.__P_test_set_scores, self.__C2_test_set_scores, self.__completed_workers, self.__completed_models):
            ax0.text(sT, nn, "W"+str(w)+"M"+str(m),color='r')
            ax1.text(sP, nn, "W"+str(w)+"M"+str(m),color='b')
            ax2.text(sC2, nn, "W"+str(w)+"M"+str(m),color='m')
        plt.show()
        return 
    
    def PlotLossBatchSize(self):
        fig = plt.figure(figsize=[27, 9])
        ax0 = fig.add_subplot(1,3,1)
        ax0.plot(self.__T_test_set_scores, self.__batch_expo, 'ro')
        ax0.grid()
        ax0.set_xscale('log')
        ax0.set_xlabel('Temperature test set evaluation loss',fontsize=20)
        ax0.set_ylabel('Training batch size exponent', fontsize=20)
        ax0.tick_params(which='both',labelsize=18)

        ax1 = fig.add_subplot(1,3,2)
        ax1.plot(self.__P_test_set_scores, self.__batch_expo, 'bo')
        ax1.grid()
        ax1.set_xscale('log')
        ax1.set_xlabel('Pressure test set evaluation loss',fontsize=20)
        ax1.tick_params(which='both',labelsize=18)

        ax2 = fig.add_subplot(1,3,3)
        ax2.plot(self.__C2_test_set_scores, self.__batch_expo, 'mo')
        ax2.grid()
        ax2.set_xscale('log')
        ax2.set_xlabel('Speed of sound test set evaluation loss',fontsize=20)
        ax2.tick_params(which='both',labelsize=18)

        for nn, sT, sP, sC2, w, m in zip(self.__batch_expo, self.__T_test_set_scores, self.__P_test_set_scores, self.__C2_test_set_scores, self.__completed_workers, self.__completed_models):
            ax0.text(sT, nn, "W"+str(w)+"M"+str(m),color='r')
            ax1.text(sP, nn, "W"+str(w)+"M"+str(m),color='b')
            ax2.text(sC2, nn, "W"+str(w)+"M"+str(m),color='m')
        plt.show()
        return 
    
    def PlotLossSize(self):
        fig = plt.figure(figsize=[27, 9])
        ax0 = fig.add_subplot(1,3,1)
        ax0.plot(self.__T_test_set_scores, self.__hidden_layer_neurons, 'ro')
        ax0.grid()
        ax0.set_xscale('log')
        ax0.set_xlabel('Temperature test set evaluation loss',fontsize=20)
        ax0.set_ylabel('Number of hidden layer neurons', fontsize=20)
        ax0.tick_params(which='both',labelsize=18)

        ax1 = fig.add_subplot(1,3,2)
        ax1.plot(self.__P_test_set_scores, self.__hidden_layer_neurons, 'bo')
        ax1.grid()
        ax1.set_xscale('log')
        ax1.set_xlabel('Pressure test set evaluation loss',fontsize=20)
        ax1.tick_params(which='both',labelsize=18)

        ax2 = fig.add_subplot(1,3,3)
        ax2.plot(self.__C2_test_set_scores, self.__hidden_layer_neurons, 'mo')
        ax2.grid()
        ax2.set_xscale('log')
        ax2.set_xlabel('Speed of sound test set evaluation loss',fontsize=20)
        ax2.tick_params(which='both',labelsize=18)

        for nn, sT, sP, sC2, w, m in zip(self.__hidden_layer_neurons, self.__T_test_set_scores, self.__P_test_set_scores, self.__C2_test_set_scores, self.__completed_workers, self.__completed_models):
            ax0.text(sT, nn, "W"+str(w)+"M"+str(m),color='r')
            ax1.text(sP, nn, "W"+str(w)+"M"+str(m),color='b')
            ax2.text(sC2, nn, "W"+str(w)+"M"+str(m),color='m')
            
        plt.show()
        return 
    
    def PlotLossPhi(self):
        fig = plt.figure(figsize=[27, 9])
        ax0 = fig.add_subplot(1,3,1)
        ax0.plot(self.__T_test_set_scores, self.__phi_idx, 'ro')
        ax0.grid()
        ax0.set_xscale('log')
        ax0.set_xlabel('Temperature test set evaluation loss',fontsize=20)
        ax0.set_yticks([i for i in range(len(ActivationFunctionOptions.keys()))])
        ax0.set_yticklabels([s for s in ActivationFunctionOptions.keys()])

        ax0.set_ylabel('Number of hidden layer neurons', fontsize=20)
        ax0.tick_params(which='both',labelsize=18)

        ax1 = fig.add_subplot(1,3,2)
        ax1.plot(self.__P_test_set_scores, self.__phi_idx, 'bo')
        ax1.grid()
        ax1.set_xscale('log')
        ax1.set_yticks([i for i in range(len(ActivationFunctionOptions.keys()))])
        ax1.set_xlabel('Pressure test set evaluation loss',fontsize=20)
        ax1.tick_params(which='both',labelsize=18)

        ax2 = fig.add_subplot(1,3,3)
        ax2.plot(self.__C2_test_set_scores, self.__phi_idx, 'mo')
        ax2.grid()
        ax2.set_xscale('log')
        ax2.set_xlabel('Speed of sound test set evaluation loss',fontsize=20)
        ax2.set_yticks([i for i in range(len(ActivationFunctionOptions.keys()))])
        ax2.tick_params(which='both',labelsize=18)

        for nn, sT, sP, sC2, w, m in zip(self.__phi_idx, self.__T_test_set_scores, self.__P_test_set_scores, self.__C2_test_set_scores, self.__completed_workers, self.__completed_models):
            ax0.text(sT, nn, "W"+str(w)+"M"+str(m),color='r')
            ax1.text(sP, nn, "W"+str(w)+"M"+str(m),color='b')
            ax2.text(sC2, nn, "W"+str(w)+"M"+str(m),color='m')
            
        plt.show()