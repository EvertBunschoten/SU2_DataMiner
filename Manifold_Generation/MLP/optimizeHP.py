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
# Version: 2.0.0                                                                              |
#                                                                                             |
#=============================================================================================#

import pygad
import os 
import pickle 
import csv
import numpy as np
import matplotlib.pyplot as plt 
from paretoset import paretoset
from multiprocessing import current_process
from pymoo.indicators.gd_plus import GDPlus
from pymoo.indicators.hv import HV

from Common.Properties import DefaultProperties, DefaultSettings_NICFD
from Common.Config_base import Config 
from Common.DataDrivenConfig import Config_FGM, Config_NICFD
from Manifold_Generation.MLP.Trainer_Base import TrainMLP
from Manifold_Generation.MLP.Trainers_NICFD.Trainers import TrainMLP_NICFD
from Manifold_Generation.MLP.Trainers_FGM.Trainers import TrainMLP_FGM

from Common.Properties import ActivationFunctionOptions

class MLPOptimizer:
    """Class for hyper-parameter optimization of entropic fluid model multi-layer perceptrons.
    """

    _Config:Config = None     # EntropicAI configuration.
    __optimizer:pygad.GA = None         # PyGaD optimization instance.
    _n_workers:int = 1                 # Number of CPU cores used for distributing the work per generation.
    _n_epochs:int=DefaultProperties.N_epochs


    # Hyper-parameter default settings and bounds.

    # Optimize for both cost and loss
    __run_multiobj:bool = False

    # Mini-batch exponent (base 2) for training.
    _optimizebatch:bool = True 
    _batch_expo:int = DefaultProperties.batch_size_exponent
    __batch_expo_min:int=4
    __batch_expo_max:int=7
    
    # Optimize learning rate decay parameters.
    _optimizeLR:bool = True 

    # Initial learning rate exponent (base 10).
    _alpha_expo:float = DefaultProperties.init_learning_rate_expo
    __alpha_expo_min:float = -4.0
    __alpha_expo_max:float = -1.0

    # Learning rate decay parameter for exponential learning rate decay schedule.
    _lr_decay:float=DefaultProperties.learning_rate_decay
    __lr_decay_min:float = 0.8
    __lr_decay_max:float = 1.0 

    # Optimize hidden layer architecture.
    _optimizeNN:bool = True
    __NN_min:int = 6
    __NN_max:int = 40 
    NLayers_min:int = 2
    NLayers_max:int = 10

    _architecture:list[int]=[30]

    # Hidden layer activation function
    _optimizephi:bool = True 
    _activation_function:str = "gelu"
    __activation_function_options:list[str] = [a for a in ActivationFunctionOptions.keys()]

    # Genetic algorithm settings
    __set_custom_generation:bool = False 
    __n_generations_custom:int = 10

    __set_custom_popsize:bool = False 
    __custom_popsize:int = 100

    # Optimization history.
    __population_history:list = []
    __fitness_history:list = []
    _history_extension:str = ""

    # Optimized solution
    __x_optim:np.ndarray = None 
    
    # Restart optimization from previous instance
    __restart_optim:bool=False

    def __init__(self, Config_in:Config=None, load_file:str=None):
        """Class constructor
        """
        # Store configuration
        self._Config = Config_in 

        self._alpha_expo = self._Config.GetAlphaExpo()
        self._lr_decay = self._Config.GetLRDecay()
        self._batch_expo = self._Config.GetBatchExpo()
        self._activation_function = self._Config.GetActivationFunction()
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
        """Set the number of epochs for which networks are trained.

        :param n_epochs: number of epochs, defaults to 1000
        :type n_epochs: int, optional
        :raises Exception: if provided number is lower than 0
        """
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
    
    def SetNGenerations(self, n_generations:int):
        """Set the number of generations for which to run the genetic algorithm.

        :param n_generations: number of generations, defaults to the population size.
        :type n_generations: int
        :raises Exception: if a number lower than one is provided.
        """
        if n_generations < 1:
            raise Exception("Number of generations should be greater than one.")
        self.__set_custom_generation = True 
        self.__n_generations_custom = n_generations
        return 
    
    def SetPopSize(self, pop_size:int):
        """Define the population size considered for the genetic optimizer.

        :param pop_size: number of individuals populating each generation. Defaults to 10x the number of hyper-parameters.
        :type pop_size: int
        :raises Exception: if the provided number is lower than 2.
        """
        if pop_size < 2:
            raise Exception("Population should consist of at least two individuals.")
        self.__set_custom_popsize = True 
        self.__custom_popsize = pop_size
        return 
    
    def SetAlpha_Expo(self, val_alpha_expo:float=DefaultSettings_NICFD.init_learning_rate_expo):
        """Set initial learning rate exponent (base 10).

        :param val_alpha_expo: _description_, defaults to -2.8
        :type val_alpha_expo: float, optional
        :raises Exception: if initial learning rate exponent value is positive.
        """

        if val_alpha_expo >= 0:
            raise Exception("Initial learing rate exponent should be negative.")
        self._alpha_expo = val_alpha_expo

        return 
    
    def SetLR_Decay(self, val_lr_decay:float=DefaultSettings_NICFD.learning_rate_decay):
        """Set the learning rate decay parameter value.

        :param val_lr_decay: learning rate decay parameter, defaults to 0.996
        :type val_lr_decay: float, optional
        :raises Exception: if learning rate decay parameter is not between 0 and 1.
        """
        if val_lr_decay > 1.0 or val_lr_decay < 0.0:
            raise Exception("Learning rate decay parameter should be between 0 and 1")
        self._lr_decay = val_lr_decay 

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
    
    def SetBounds_LR_Decay(self, lr_decay_min:float=0.8, lr_decay_max:float=1.0):
        self.__lr_decay_max = lr_decay_max
        self.__lr_decay_min = lr_decay_min
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
        self._batch_expo = batch_expo 
        
        return 
    
    def Optimize_Pareto(self, optimize_multiobj:bool=False):
        """Enable multi-objective optimization where a Pareto front is formed for evaluation cost and validation loss.

        :param optimize_multiobj: enable multi-objective optimization, defaults to False
        :type optimize_multiobj: bool, optional
        """
        self.__run_multiobj = optimize_multiobj
        return
    
    def SetActivationFunction(self, activation_function:str=DefaultSettings_NICFD.activation_function):
        """Set the hiden layer activation function name.

        :param activation_function: hidden layer activation function name, defaults to DefaultSettings_NICFD.activation_function
        :type activation_function: str, optional
        """

        self._activation_function = activation_function
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
    
    def SetBounds_NLayers(self, Nlayers_min:int=2, Nlayers_max:int=10):
        if Nlayers_max <=Nlayers_min:
            raise Exception("Upper bound value should exceed lower bound value.")
        if Nlayers_min < 1 or Nlayers_max < 1:
            raise Exception("Number of hidden layers should be higher than one.")
        self.NLayers_max = Nlayers_max
        self.NLayers_min = Nlayers_min
        return 
    
    def SetBounds_NNeurons(self, NN_min:int=6, NN_max:int=40):
        self.__NN_min = NN_min
        self.__NN_max = NN_max 
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
        self._architecture = []
        for NN in architecture:
            self._architecture.append(NN)

        return 
    
    def SetBounds_Architecture(self, NN_min:int=10, NN_max:int=100, NL_min:int=2, NL_max:int=10):
        """Set the minimum and maximum values for the perceptron count in the hidden layer.

        :param NN_min: minimum number of perceptrons, defaults to 10
        :type NN_min: int, optional
        :param NN_max: maximum number of perceptrons, defaults to 100
        :type NN_max: int, optional
        :raises Exception: if lower value exceeds upper value.
        :raises Exception: if perceptron count is lower than one.
        """

        if NN_min >= NN_max or NL_min >= NL_max:
            raise Exception("Upper bound value should exceed lower bound value.")
        if NN_min <= 1 or NN_max <= 1:
            raise Exception("At least one hidden layer perceptron should be used.")
        
        if NL_min < 1:
            raise Exception("At least one hidden layer should be used.")
        
        self.__NN_min = NN_min 
        self.__NN_max = NN_max 
        self.NLayers_min = NL_min
        self.NLayers_max = NL_max
        return 
    
    def __prepareBounds(self):
        bounds = []

        if self._optimizeLR:
            bounds += [{"low":self.__alpha_expo_min, "high":self.__alpha_expo_max},\
                       {"low":self.__lr_decay_min, "high":self.__lr_decay_max}]
            
        if self._optimizebatch:
            bounds += [{"low":self.__batch_expo_min, "high":self.__batch_expo_max}]
        
        if self._optimizephi:
            bounds += [{"low":0, "high":len(self.__activation_function_options)}]
        
            
        if self._optimizeNN:
            bounds += self.NLayers_min * [{"low":self.__NN_min, "high":self.__NN_max}]
            bounds += (self.NLayers_max - self.NLayers_min) * [{"low":0, "high":2},\
                                                               {"low":self.__NN_min, "high":self.__NN_max}]
        return bounds 

    
    def __setOptimizer(self):

        print("Initializing hyper-parameter optimization for:")
        if self._optimizebatch:
            print("- mini-batch size exponent")
        if self._optimizeLR:
            print("- learning rate parameters")
        if self._optimizeNN:
            print("- hidden layer neuron count")
        if self._optimizephi:
            print("- hidden layer activation function")
        
        self.__setOptimizer_PyGad()
        return
    
    
    def _postprocess_optimization(self, x):

        print("Optimized hyper-parameters:")
        idx_x = 0
        if self._optimizeLR:
            alpha_expo = x[idx_x]
            idx_x += 1 
            print("- initial learning rate exponent: %.5e" % (alpha_expo))
            self._Config.SetAlphaExpo(alpha_expo)
            lr_decay = x[idx_x]
            print("- learning rate decay parameter: %.5e" % lr_decay)
            self._Config.SetLRDecay(lr_decay)
            idx_x += 1 
        if self._optimizebatch:
            batch_expo = int(x[idx_x])
            self._Config.SetBatchExpo(batch_expo)
            idx_x += 1 
            print("- mini-batch exponent: %i" % batch_expo)
        if self._optimizephi:
            phi = self.__activation_function_options[int(x[idx_x])]
            idx_x += 1
            print("- hidden layer activation function: %s" % phi)
            self._Config.SetActivationFunction(phi)
        if self._optimizeNN:
            architecture = [int(x[idx_x])]
            idx_x += 1 
            print("- hidden layer architecture: "+ " ".join(("%i" % n) for n in architecture))
            self._Config.SetHiddenLayerArchitecture(architecture)
        
        self._Config.SaveConfig()
        return
    
    def _get_optim_extension(self):
        optim_extension = ""
        if self._optimizebatch:
            optim_extension += "B"
        if self._optimizeLR:
            optim_extension += "LR"
        if self._optimizeNN:
            optim_extension += "A"
        if self._optimizephi:
            optim_extension += "Phi"
        return optim_extension
    
    def _preprocess_optimization(self):
        if not any((self._optimizebatch, self._optimizeLR, self._optimizeNN, self._optimizephi)):
            raise Exception("At least one of the hyper-parameter options should be considered for optimization.")
        # Prepare optimization history output file.
        self._history_extension = self._get_optim_extension()
        n_params = sum([self._optimizebatch,self._optimizeLR, self._optimizeNN,self._optimizephi])

        self.CreateOutputs()

        return 
    
    def SetOutputFolder(self):
        self.save_dir = self._Config.GetOutputDir()+"/Architectures_Optim"+self._history_extension + "/"
        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)
        return 
    
    def _initialize_history_file(self):
        self.opt_history_filepath = self.save_dir + "/history_opt_"+self._history_extension+".csv"
        return 
    
    def _set_history_header(self):
        if not self.__restart_optim:
            with open(self.opt_history_filepath, "w+") as fid:
                if self.__run_multiobj:
                    fid.write("Iteration,mini-batch exp, activation function, alpha expo, lr decay, architecture,fitness,cost\n")
                else:
                    fid.write("Iteration,mini-batch exp, activation function, alpha expo, lr decay, architecture,fitness\n")
        return 
    
    def CreateOutputs(self):
        self.SetOutputFolder()

        self._initialize_history_file()

        self._set_history_header()
        
        return 
    
    def optimizeHP(self):
        """Initate hyper-parameter optimization routine.

        :raises Exception: if neither of the available sets of hyper-parameters are considered for optimization.
        """
        self._preprocess_optimization()
        
        # Prepare bounds and commence hyper-parameter optimization.
        self.__setOptimizer()

        # Extract optimized hyper-parameters and update configuration.
        self._postprocess_optimization(self.__x_optim)
        return 
    
    def RestartOptimizer(self, restart_from_prev:bool=False):
        """_summary_

        :param restart_from_prev: _description_, defaults to False
        :type restart_from_prev: bool, optional
        """
        self.__restart_optim = restart_from_prev
        return 
    
    def __setOptimizer_PyGad(self):
        """Commence hyper-parameter optimization.
        """

        #self._preprocess_optimization()

        # Collect gene types and bounds.
        gene_type = self.__prepareGeneType()
        bounds = self.__prepareBounds()
        n_genes = len(gene_type)

        # Size population
        if self.__set_custom_popsize:
            popsize = self.__custom_popsize
        else:
            popsize = 10*n_genes
        
        # Set generation count
        if self.__set_custom_generation:
            n_gens = self.__n_generations_custom
        else:
            n_gens = popsize 

        # Half of the parents in the population mate
        num_parents_mating = int(0.5 * popsize)
        
        # Construct initial population
        initial_pop = self.__GenerateInitialPopulation(popsize)

        # Set parent selection algorithm
        if self.__run_multiobj:
            parent_selector = "nsga2"
        else:
            parent_selector = "sss"
        if self.__restart_optim:
            self.__optimizer = self.LoadOptimizer()
            self.__optimizer.parallel_processing=["process",self._n_workers]
        else:
            # Initiate pyGAD genetic algorithm
            self.__optimizer = pygad.GA(num_generations=n_gens,\
                        fitness_func=self.fitnessGA,\
                        gene_type=gene_type,\
                        num_genes=n_genes,\
                        gene_space=bounds,\
                        sol_per_pop=popsize,\
                        initial_population=initial_pop,\
                        num_parents_mating=num_parents_mating,\
                        parallel_processing=["process",self._n_workers],\
                        random_seed=1,\
                        parent_selection_type=parent_selector,\
                        on_generation=self.saveGenerationInfo)
            

        self.__optimizer.run()

        self.__x_optim, _, _ = self.__optimizer.best_solution()
        return 
    
    def __prepareGeneType(self):
        gene_type = []
        if self._optimizeLR:
            gene_type += [float, float]
        if self._optimizebatch:
            gene_type += [int]
        if self._optimizephi:
            gene_type += [int]
        if self._optimizeNN:
            gene_type += self.NLayers_min * [int] + 2*(self.NLayers_max - self.NLayers_min) * [int]

        return gene_type 
    
    def saveGenerationInfo(self, ga_instance:pygad.GA):
        """Save population information per completed generation.
        """

        # Collect population parameters and fitness
        population = ga_instance.population.copy()
        pop_fitness = ga_instance.last_generation_fitness.copy()
        # Scale fitness to test set evaluation score.
        if self.__run_multiobj:
            test_score = self.inv_transformTestScore(pop_fitness[:, -2])
            cost_parameter = self.inv_transformCostParam(pop_fitness[:, -1])
            pop_fitness = np.hstack((test_score[:,np.newaxis], cost_parameter[:,np.newaxis]))
        else:
            test_score = self.inv_transformTestScore(pop_fitness)
            pop_fitness = test_score[:,np.newaxis]

        # Update history.
        self.__population_history.append(population)
        self.__fitness_history.append(pop_fitness)
        generation = ga_instance.generations_completed

        # Write population data to history file.
        pop_and_fitness = np.hstack((generation*np.ones([len(population),1]), population, pop_fitness))
        with open(self.opt_history_filepath, "a+") as fid:
            csvWriter =csv.writer(fid, delimiter=',')
            csvWriter.writerows(pop_and_fitness)

        self.SaveOptimizer(ga_instance)

        return 
    
    def SaveOptimizer(self, ga_instance:pygad.GA):
        ga_instance.save(self.save_dir+"/optimizer_instance_"+self._history_extension)
        return
    
    def LoadOptimizer(self):
        ga_instance = pygad.load(self.save_dir+"/optimizer_instance_"+self._history_extension)
        return ga_instance
    
    def _translateGene(self, x:np.ndarray[float], Evaluator:TrainMLP):
        """Translate gene to hyper-parameters

        :param x: gene as passed from genetic algorithm.
        :type x: np.ndarray[float]
        :param Evaluator: MLP evaluation class instance.
        :type Evaluator: TrainMLP
        """

        # Set default hyper-parameters.
        Evaluator.SetBatchExpo(self._batch_expo)
        Evaluator.SetAlphaExpo(self._alpha_expo)
        Evaluator.SetLRDecay(self._lr_decay)
        Evaluator.SetHiddenLayers(self._architecture)
        Evaluator.SetActivationFunction(self._activation_function)

        # Set hyper-parameter according to gene.
        idx_x = 0
        if self._optimizeLR:
            alpha_expo = x[idx_x]
            Evaluator.SetAlphaExpo(alpha_expo)
            idx_x += 1 
            lr_decay = x[idx_x]
            Evaluator.SetLRDecay(lr_decay)
            idx_x += 1 
        if self._optimizebatch:
            batch_expo = int(x[idx_x])
            Evaluator.SetBatchExpo(batch_expo)
            idx_x += 1 
        if self._optimizephi:
            phi = self.__activation_function_options[int(x[idx_x])]
            Evaluator.SetActivationFunction(phi)
            idx_x += 1
        if self._optimizeNN:
            architecture = [int(x[idx_x])]
            architecture = []
            for i in range(idx_x, idx_x + self.NLayers_min):
                architecture.append(x[i])
            for i in range(idx_x + self.NLayers_min, len(x)-1, 2):
                if x[i] > 0:
                    architecture.append(x[i+1])
            Evaluator.SetHiddenLayers(architecture)
            idx_x += 1 

        return

    def __GenerateInitialPopulation(self, popsize:int):
        """Generate the initial population for the GA optimization.

        :param popsize: number of individuals
        :type popsize: int
        :return: array with initial population hyper-parameters
        :rtype: np.ndarray
        """
        initial_pop = []
        if self._optimizeLR:
            initial_learningrates = self.__GenerateInitialLearningRates(popsize)
            initial_pop.append(initial_learningrates)
        if self._optimizebatch:
            batchphi = self.__GenerateInitialBatch(popsize)
            initial_pop.append(batchphi)
        if self._optimizephi:
            phi = self.__GenerateInitialPhi(popsize)
            initial_pop.append(phi)
        if self._optimizeNN:
            initial_architectures = self.__GenerateInitialArchitectures(popsize)
            initial_pop.append(initial_architectures)
        initial_pop = np.hstack(tuple(s for s in initial_pop))
        return initial_pop 
    
    def __GenerateInitialLearningRates(self,N_individuals:int):
        """Generate initial set of learning rate hyper-parameters.

        :param N_individuals: number of individuals in the initial population.
        :type N_individuals: int
        :return: array of initial learning rate parameters.
        :rtype: np.ndarray
        """
        lr_array = np.ones([N_individuals, 2])
        lr_array[:, 0] = self._alpha_expo
        lr_array[:, 0] += 0.1*(self.__alpha_expo_max - self.__alpha_expo_min)*(np.random.rand(N_individuals) - 0.5)
        lr_array[:, 1] = self._lr_decay
        lr_array[:, 1] += 0.1*(self.__lr_decay_max - self.__lr_decay_min)*(np.random.rand(N_individuals) - 0.5)
        
        return lr_array
    
    def __GenerateInitialBatch(self, N_individuals:int):
        """Generate initial set of batch size exponents and activation function indices.

        :param N_individuals: number of individuals in the initial population.
        :type N_individuals: int
        :return: array of initial batch expos and activation functions.
        :rtype: np.ndarray
        """
        batch_size = np.random.randint(size=N_individuals, low=self.__batch_expo_min,high=self.__batch_expo_max+1) 
        return batch_size[:,np.newaxis]

    def __GenerateInitialPhi(self, N_individuals:int):
        """Generate initial set of batch size exponents and activation function indices.

        :param N_individuals: number of individuals in the initial population.
        :type N_individuals: int
        :return: array of initial batch expos and activation functions.
        :rtype: np.ndarray
        """
        phi = np.random.randint(size=N_individuals, low=0,high=len(self.__activation_function_options))
        return phi[:,np.newaxis]
    
    def __GenerateInitialArchitectures(self, N_individuals:int):
        """Generate (random) initial set of hidden layer architectures. These
        architectures are unimodal, which should perform relatively well.

        :param N_individuals: number of individuals in the initial population.
        :type N_individuals: int
        :return: array of initial hidden layer architectures.
        :rtype: np.ndarray
        """
        architecture_array = np.zeros([N_individuals, self.NLayers_min + 2 * (self.NLayers_max - self.NLayers_min)],dtype=int)
        
        for i in range(N_individuals):
            NLayers = np.random.randint(self.NLayers_min, self.NLayers_max)
            NN = np.random.randint(self.__NN_min, self.__NN_max, NLayers)
            if NLayers > 1:
                idx_max = np.random.randint(0, NLayers-1)
                architecture_mon = np.hstack((np.sort(NN[:idx_max]), np.sort(NN[idx_max:])[::-1]))
            else:
                architecture_mon = np.array([NN])
            for iLayer in range(self.NLayers_min):
                    architecture_array[i, iLayer] = architecture_mon[iLayer]
            for jLayer in range(NLayers - self.NLayers_min):
                architecture_array[i, self.NLayers_min + 2*jLayer] = 1
                architecture_array[i, self.NLayers_min + 2*jLayer + 1] = architecture_mon[jLayer + self.NLayers_min]
            for jLayer in range(NLayers - self.NLayers_min, self.NLayers_max - self.NLayers_min):
                architecture_array[i, self.NLayers_min + 2 * jLayer] = 0
                architecture_array[i, self.NLayers_min + 2* jLayer + 1] = np.random.randint(self.__NN_min, self.__NN_max)
        return architecture_array
    
    def fitnessGA(self, ga_instance:pygad.GA, x:np.ndarray, x_idx:int):
        return self.fitnessFunction(x, worker_idx=x_idx)
    
    def transformTestScore(self, val_test_score:float):
        return -np.log10(val_test_score)
    
    def transformCostParameter(self, val_cost_param:float):
        return 1000 / val_cost_param
    
    def inv_transformTestScore(self, val_norm_test_score:float):
        return np.power(10, -val_norm_test_score)
    
    def inv_transformCostParam(self, val_norm_cost_param:float):
        return self.transformCostParameter(val_norm_cost_param)
    
    def fitnessFunction(self, x:np.ndarray, worker_idx:int=None):
        if worker_idx == None:
            if self._n_workers > 1:
                p = current_process()
                worker_idx = p._identity[0]
            else:
                worker_idx = 0
        Evaluator:TrainMLP = self._prepare_evaluator()

        # Set CPU index.
        Evaluator.SetTrainHardware("CPU", worker_idx)

        self._evaluate_MLP_performance(x, Evaluator)

        objective_function = self._extract_objective_function(Evaluator)

        # Free up memory
        del Evaluator 
        return objective_function

    def _prepare_evaluator(self):
        return TrainMLP(self._Config)
    
    def _evaluate_MLP_performance(self, x:np.ndarray, MLP_evaluator:TrainMLP):

        self._translateGene(x, Evaluator=MLP_evaluator)

        MLP_evaluator.SetVerbose(0)

        # Set output directory for MLP training callbacks.
        MLP_evaluator.SetSaveDir(self.save_dir)

        # Train for 1000 epochs direct and physics-informed.
        MLP_evaluator.SetNEpochs(self._n_epochs)

        # Train MLP
        MLP_evaluator.CommenceTraining()

        # Extract test set evaluation score.
        MLP_evaluator.TrainPostprocessing()
        return 
    
    def _extract_objective_function(self, MLP_evaluator:TrainMLP):
        test_score = MLP_evaluator.GetTestScore() 
        cost_parameter = MLP_evaluator.GetCostParameter()

        # Convert performance metrics (to be minimized) into fitness values (to be maximized).
        fitness_test_score = self.transformTestScore(test_score)
        fitness_cost = self.transformCostParameter(cost_parameter)
        if self.__run_multiobj:
            return [fitness_test_score, fitness_cost]
        else:
            return fitness_test_score

    
    def inv_fitnessFunction(self, x):
        return -self.fitnessFunction(x=x)
    
    
    
class MLPOptimizer_NICFD(MLPOptimizer):
    _activation_function:str = "exponential"

    def _prepare_evaluator(self):
        return TrainMLP_NICFD(self._Config)
    
    def _postprocess_optimization(self, x):

        Config:Config_NICFD = Config_NICFD(self._Config.GetConfigName() + ".cfg")
        idx_x = 0
        if self._optimizeLR:
            alpha_expo = x[idx_x]
            Config.SetAlphaExpo(alpha_expo)
            print("- initial learning rate exponent: %.5e" % (alpha_expo))
            idx_x += 1 
            lr_decay = x[idx_x]
            Config.SetLRDecay(lr_decay)
            print("- learning rate decay parameter: %.5e" % lr_decay)
            idx_x += 1 
        if self._optimizebatch:
            batch_expo = int(x[idx_x])
            Config.SetBatchExpo(batch_expo)
            idx_x += 1 
            print("- mini-batch exponent: %i" % batch_expo)
        if self._optimizephi:
            phi = self.__activation_function_options[int(x[idx_x])]
            Config.SetActivationFunction(phi)
            print("- hidden layer activation function: %s" % phi)
            idx_x += 1
        if self._optimizeNN:
            architecture = [int(x[idx_x])]
            architecture = []
            for i in range(idx_x, idx_x + self.NLayers_min):
                architecture.append(x[i])
            for i in range(idx_x + self.NLayers_min, len(x)-1, 2):
                if x[i] > 0:
                    architecture.append(x[i+1])
            print("- hidden layer architecture: "+ " ".join(("%i" % n) for n in architecture))
            Config.SetHiddenLayerArchitecture(architecture)
            idx_x += 1 
        Config.SaveConfig()
        return
    
class MLPOptimizer_FGM(MLPOptimizer):
    __output_group:int = 0 
    _activation_function:str="gelu"
    def __init__(self, Config_in:Config_FGM):
        MLPOptimizer.__init__(self, Config_in)
        return
    
    def SetOutputGroup(self, output_group:int=0):
        self.__output_group = output_group
        self.SetAlpha_Expo(self._Config.GetAlphaExpo(output_group))
        self.SetLR_Decay(self._Config.GetLRDecay(output_group))
        self.SetBatch_Expo(self._Config.GetBatchExpo(output_group))
        self.SetArchitecture(self._Config.GetHiddenLayerArchitecture(output_group))
        return 
    
    def _initialize_history_file(self):
        self.opt_history_filepath = "%s/history_optim_Group%i_%s.csv" % (self.save_dir, (self.__output_group+1),self._history_extension)
        return 
    
    def SetOutputFolder(self):
        self.save_dir = "%s/Architectures_Group%i_Optim%s" % (self._Config.GetOutputDir(), (self.__output_group+1), self._history_extension)
        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)
        return 
    
    def _prepare_evaluator(self):
        return TrainMLP_FGM(self._Config, self.__output_group)
    
    def _postprocess_optimization(self, x):

        Config:Config_FGM = Config_FGM(self._Config.GetConfigName() + ".cfg")

        print("Optimized hyper-parameters:")
        idx_x = 0
        if self._optimizeLR:
            alpha_expo = x[idx_x]
            Config.SetAlphaExpo(alpha_expo, self.__output_group)
            print("- initial learning rate exponent: %.5e" % (alpha_expo))
            idx_x += 1 
            lr_decay = x[idx_x]
            Config.SetLRDecay(lr_decay, self.__output_group)
            print("- learning rate decay parameter: %.5e" % lr_decay)
            idx_x += 1 
        if self._optimizebatch:
            batch_expo = int(x[idx_x])
            Config.SetBatchExpo(batch_expo, self.__output_group)
            idx_x += 1 
            print("- mini-batch exponent: %i" % batch_expo)
        if self._optimizephi:
            phi = self.__activation_function_options[int(x[idx_x])]
            Config.SetActivationFunction(phi, self.__output_group)
            print("- hidden layer activation function: %s" % phi)
            idx_x += 1
        if self._optimizeNN:
            architecture = [int(x[idx_x])]
            architecture = []
            for i in range(idx_x, idx_x + self.NLayers_min):
                architecture.append(x[i])
            for i in range(idx_x + self.NLayers_min, len(x)-1, 2):
                if x[i] > 0:
                    architecture.append(x[i+1])
            print("- hidden layer architecture: "+ " ".join(("%i" % n) for n in architecture))
            Config.SetHiddenLayerArchitecture(architecture, self.__output_group)
            idx_x += 1 
        Config.SaveConfig()
        return
    
    def SaveOptimizer(self, ga_instance:pygad.GA):
        ga_instance.save("%s/optimizer_instance_Group%i_%s" % (self.save_dir, self.__output_group+1, self._history_extension))
        return
    
    def LoadOptimizer(self):
        ga_instance = pygad.load("%s/optimizer_instance_Group%i_%s" % (self.save_dir, self.__output_group+1, self._history_extension))
        return ga_instance
    
class MLPOptimizer_NICFD(MLPOptimizer):

    def __init__(self, Config_in:Config_NICFD):
        MLPOptimizer.__init__(self, Config_in)
        return
    
    def _prepare_evaluator(self):
        return TrainMLP_NICFD(self._Config)
    
class PlotHPOResults:
    _Config:Config = None 
    _optimizelearningrate:bool = True 
    _optimizebatch:bool = True 
    _optimizearchitecture:bool = True 
    _optimizephi:bool = True 

    _hidden_layer_neurons:list[int] = []
    _val_score:list[float] = []
    _lr_decay:list[float] = []
    _alpha_expo:list[float] = []
    _batch_expo:list[int] = [] 
    _idx_phi:list[int] = [] 

    _completed_workers:list[int] = []
    _completed_models:list[int] = [] 
    _optimize_pareto:bool = True 

    def __init__(self, Config_in:Config):
        
        self._Config = Config_in 
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
    
    
    def SetFolderHeader(self):
        optim_header = "Architectures_Optim"
        optim_header += self._get_optim_extension()
        return optim_header
    
    def ReadModelPerformance(self, filedirpath:str, model_idx:int, worker_idx:int):
        if os.path.isfile(filedirpath):
            with open(filedirpath, 'r') as fid:
                lines = fid.readlines()

            validation_loss = float(lines[1].strip().split(":")[-1])
            if np.isnan(validation_loss):
                validation_loss = 1e1
            cost_param = float(lines[4].strip().split(':')[-1])
            alpha_expo = float(lines[5].strip().split(':')[-1])
            lr_decay = float(lines[6].strip().split(':')[-1])
            batch_expo = int(lines[7].strip().split(':')[-1])
            ix_phi = int(lines[9].strip().split(':')[-1])
            self._alpha_expo.append(alpha_expo)
            self._lr_decay.append(lr_decay)
            self._batch_expo.append(batch_expo)
            self._val_score.append(validation_loss)
            self._hidden_layer_neurons.append(cost_param)
            self._idx_phi.append(ix_phi)

            self._completed_models.append(model_idx)
            self._completed_workers.append(worker_idx)
        return 
    
    def _get_optim_extension(self):
        optim_extension = ""
        if self._optimizebatch:
            optim_extension += "B"
        if self._optimizelearningrate:
            optim_extension += "LR"
        if self._optimizearchitecture:
            optim_extension += "A"
        if self._optimizephi:
            optim_extension += "Phi"
        return optim_extension
    
    def ReadArchitectures(self):
        optim_header = self.SetFolderHeader()

        optim_directory = self._Config.GetOutputDir()+"/"+optim_header

        population_indices = os.listdir(optim_directory)
        self._completed_models = []
        for p in population_indices:
            if "Worker" in p:
                models = os.listdir(optim_directory + "/" + p)
                for m in models:
                    if "Model" in m:
                        model_idx = int(m.split("_")[-1])
                        worker_idx = int(p.split("_")[-1])
                        self.ReadModelPerformance(optim_directory + "/" + p + "/" + m + "/MLP_performance.txt", model_idx, worker_idx)

        return 
    
    def GetOptimHistory(self):
        history_filename = "history_optim_" + self._get_optim_extension()
        history_filepath = self._Config.GetOutputDir()+"/"+history_filename + ".csv"
        return history_filepath
    
    def PlotParetoFront(self):
        
        history_filepath = self.GetOptimHistory()

        H = np.loadtxt(history_filepath, delimiter=',',skiprows=1)
        N_gen = int(H[-1,0])
        Np_per_gen = np.sum(H[:,0] == N_gen)
        
        plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.jet(np.linspace(0,1,N_gen)))
        total_data = None 
        init = True

        plt.figure(figsize=[10,10])
        ax = plt.axes()
        
        for iGen in range(N_gen):
            gen_data = H[iGen*Np_per_gen:(iGen+1)*Np_per_gen, [-2, -1]]
            if init:
                total_data = gen_data 
                init = False
            else:
                total_data = np.vstack((total_data, gen_data))

            mask = paretoset(total_data, sense=["min","min"])
            pareto_scores = total_data[mask, 0]
            pareto_sizes = total_data[mask, 1]
            sorted_indices = np.argsort(pareto_scores)

            ax.plot(pareto_scores[sorted_indices], pareto_sizes[sorted_indices], 'o-',label=("Generation %i"% (iGen+1)))
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid()
        ax.set_xlabel(r"Validation loss $(\mathcal{L})[-]$",fontsize=20)
        ax.set_ylabel(r"Evaluation cost $(\mathcal{C})[-]$",fontsize=20)
        ax.set_title(r"Cost-accuracy Pareto front",fontsize=20)
        #ax.legend(fontsize=20)
        ax.tick_params(which='both',labelsize=18)
        plt.tight_layout()
        plt.show()
        return 
    
    def PlotLossPhi(self):

        plt.figure(figsize=[9,9])
        ax = plt.axes()

        ax.plot(self._val_score, self._idx_phi, 'ko')
        for nn, sc, w, m in zip(self._idx_phi, self._val_score, self._completed_workers, self._completed_models):
            ax.text(sc, nn, "W"+str(w)+"M"+str(m),color='k')
        
        plot_data = np.hstack((np.array(self._hidden_layer_neurons)[:,np.newaxis],np.array(self._val_score)[:,np.newaxis]))
        mask = paretoset(plot_data, sense=["min","min"])
        score_pareto, ix_phi_pareto, workers_pareto, models_pareto = np.array(self._val_score)[mask], np.array(self._idx_phi)[mask], np.array(self._completed_workers)[mask], np.array(self._completed_models)[mask]
        ax.plot(score_pareto, ix_phi_pareto, 'ro')
        for nn, sc, w, m in zip(score_pareto,ix_phi_pareto, workers_pareto, models_pareto):
            ax.text(nn,sc, "W"+str(w)+"M"+str(m),color='r')
        ax.grid()
        ax.set_xscale('log')
        ax.set_xlabel(r"Validation loss $(\mathcal{L})[-]$",fontsize=20)
        ax.set_ylabel(r"Activation function index",fontsize=20)
        ax.tick_params(which='both',labelsize=18)
        plt.tight_layout()
        plt.show()
        return 
    
    def PlotLossAlphaExpo(self):

        plt.figure(figsize=[9,9])
        ax = plt.axes()

        ax.plot(self._val_score, self._alpha_expo, 'ko')
        for nn, sc, w, m in zip(self._alpha_expo, self._val_score, self._completed_workers, self._completed_models):
            ax.text(sc, nn, "W"+str(w)+"M"+str(m),color='k')
        
        plot_data = np.hstack((np.array(self._hidden_layer_neurons)[:,np.newaxis],np.array(self._val_score)[:,np.newaxis]))
        mask = paretoset(plot_data, sense=["min","min"])
        score_pareto, alpha_expo_pareto, workers_pareto, models_pareto = np.array(self._val_score)[mask], np.array(self._alpha_expo)[mask], np.array(self._completed_workers)[mask], np.array(self._completed_models)[mask]
        ax.plot(score_pareto, alpha_expo_pareto, 'ro')
        for nn, sc, w, m in zip(score_pareto,alpha_expo_pareto, workers_pareto, models_pareto):
            ax.text(nn,sc, "W"+str(w)+"M"+str(m),color='r')
        ax.grid()
        ax.set_xscale('log')
        ax.set_xlabel(r"Validation loss $(\mathcal{L})[-]$",fontsize=20)
        ax.set_ylabel(r"Initial learning rate exponent $(\log_{10}(r_{l,0}))[-]$",fontsize=20)
        ax.tick_params(which='both',labelsize=18)
        plt.tight_layout()
        plt.show()
        return 
    
    def PlotLossLRDecay(self):
        
        plt.figure(figsize=[9,9])
        ax = plt.axes()

        ax.plot(self._val_score, self._lr_decay, 'ko')
        for nn, sc, w, m in zip(self._lr_decay, self._val_score, self._completed_workers, self._completed_models):
            ax.text(sc, nn, "W"+str(w)+"M"+str(m),color='k')
        
        plot_data = np.hstack((np.array(self._hidden_layer_neurons)[:,np.newaxis],np.array(self._val_score)[:,np.newaxis]))
        mask = paretoset(plot_data, sense=["min","min"])
        score_pareto, lr_decay_pareto, workers_pareto, models_pareto = np.array(self._val_score)[mask], np.array(self._lr_decay)[mask], np.array(self._completed_workers)[mask], np.array(self._completed_models)[mask]
        ax.plot(score_pareto, lr_decay_pareto, 'ro')
        for nn, sc, w, m in zip(score_pareto, lr_decay_pareto, workers_pareto, models_pareto):
            ax.text(nn,sc, "W"+str(w)+"M"+str(m),color='r')

        ax.grid()
        ax.set_xscale('log')
        ax.set_xlabel(r"Validation loss $(\mathcal{L})[-]$",fontsize=20)
        ax.set_ylabel(r"Learning rate decay parameter $(d)[-]$",fontsize=20)
        ax.tick_params(which='both',labelsize=18)
        plt.tight_layout()
        plt.show()
        return 
    
    def PlotLossBatchSize(self):
        
        plt.figure(figsize=[9,9])
        ax = plt.axes()

        ax.plot(self._val_score, self._batch_expo, 'ko')
        for nn, sc, w, m in zip(self._batch_expo, self._val_score, self._completed_workers, self._completed_models):
            ax.text(sc, nn, "W"+str(w)+"M"+str(m),color='k')
        
        plot_data = np.hstack((np.array(self._hidden_layer_neurons)[:,np.newaxis],np.array(self._val_score)[:,np.newaxis]))
        mask = paretoset(plot_data, sense=["min","min"])
        score_pareto, batch_expo_pareto, workers_pareto, models_pareto = np.array(self._val_score)[mask], np.array(self._batch_expo)[mask], np.array(self._completed_workers)[mask], np.array(self._completed_models)[mask]
        ax.plot(score_pareto, batch_expo_pareto, 'ro')
        for nn, sc, w, m in zip(score_pareto, batch_expo_pareto, workers_pareto, models_pareto):
            ax.text(nn,sc, "W"+str(w)+"M"+str(m),color='r')

        ax.grid()
        ax.set_xscale('log')
        ax.set_xlabel(r"Validation loss $(\mathcal{L})[-]$",fontsize=20)
        ax.set_ylabel(r"Mini batch size exponent $(\log_2(b))[-]$",fontsize=20)
        ax.tick_params(which='both',labelsize=18)
        plt.tight_layout()
        plt.show()
        return 
    
    def PlotLossSize(self):

        plot_data = np.hstack((np.array(self._hidden_layer_neurons)[:,np.newaxis],np.array(self._val_score)[:,np.newaxis]))
        mask = paretoset(plot_data, sense=["min","min"])

        plt.figure(figsize=[9,9])
        ax = plt.axes()

        frq = 1
        ax.plot(self._val_score[::frq], self._hidden_layer_neurons[::frq], 'ko')
        for nn, sc, w, m in zip(self._hidden_layer_neurons[::frq], self._val_score[::frq], self._completed_workers[::frq], self._completed_models[::frq]):
            ax.text(sc, nn, "W"+str(w)+"M"+str(m),color='k')
        
        ax.plot(np.array(self._val_score)[mask], np.array(self._hidden_layer_neurons)[mask], 'ro')

        ax.grid()
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r"Validation loss $(\mathcal{L})[-]$",fontsize=20)
        ax.set_ylabel(r"Evaluation cost parameter $(\mathcal{C})[-]$",fontsize=20)
        ax.tick_params(which='both',labelsize=18)
        plt.tight_layout()
        plt.show()
        return 
    
    def GetParetoIndices(self):
        plot_data = np.hstack((np.array(self._hidden_layer_neurons)[:,np.newaxis],np.array(self._val_score)[:,np.newaxis]))
        mask = paretoset(plot_data, sense=["min","min"])
        optim_header = self.SetFolderHeader()

        optim_directory = self._Config.GetOutputDir()+"/"+optim_header

        pareto_indices = []
        for w, m in zip(np.array(self._completed_workers)[mask], np.array(self._completed_models)[mask]):
            pareto_indices.append(optim_directory + ("/Worker_%i/Model_%i/" % (w, m)))
        
        
        return pareto_indices
    
    def PlotParetoArchitectures(self):

        plot_data = np.hstack((np.array(self._hidden_layer_neurons)[:,np.newaxis],np.array(self._val_score)[:,np.newaxis]))
        mask = paretoset(plot_data, sense=["min","min"])

        pareto_costs = np.array(self._hidden_layer_neurons)[mask]
        pareto_scores = np.array(self._val_score)[mask]
        pareto_workers = np.array(self._completed_workers)[mask]
        pareto_models = np.array(self._completed_models)[mask]

        sorted_score_indices = np.argsort(pareto_scores)

        pareto_costs = pareto_costs[sorted_score_indices]
        pareto_scores = pareto_scores[sorted_score_indices]
        pareto_workers = pareto_workers[sorted_score_indices]
        pareto_models = pareto_models[sorted_score_indices]

        optim_header = self.SetFolderHeader()

        optim_directory = self._Config.GetOutputDir()+"/"+optim_header

        hidden_layer_architectures = []
        fig, axs = plt.subplots(nrows=len(pareto_scores), ncols=2, figsize=[6, 6*len(pareto_scores)])

        for i in range(len(pareto_scores)):
            worker_idx = int(pareto_workers[i])
            model_idx = int(pareto_models[i])
            filedirpath = optim_directory + ("/Worker_%i/Model_%i/MLP_performance.txt" % (worker_idx, model_idx))
            with open(filedirpath, 'r') as fid:
                lines = fid.readlines()
            NN = [int(s) for s in lines[-1].strip().split(":")[-1].split()]
            hidden_layer_architectures.append(NN)
        
            axs[i, 0].plot(pareto_scores, pareto_costs, 'ko-',markersize=10)
            axs[i, 0].plot(pareto_scores[i],pareto_costs[i], 'ro',markersize=12)
            axs[i, 0].text(pareto_scores[i],pareto_costs[i], ("Worker_%i/Model_%i" % (worker_idx, model_idx)), color='r')
            axs[i, 0].set_xscale('log')
            axs[i, 0].set_yscale('log')
            axs[i, 0].grid()
            axs[i, 0].set_xlabel(r"Validation score $(\mathcal{L})[-]$")
            axs[i, 0].set_ylabel(r"Cost parameter $(\mathcal{C})[-]$")
            for j in range(len(hidden_layer_architectures[i])):
                axs[i, 1].plot((j+1)*np.ones(int(hidden_layer_architectures[i][j])), np.arange(int(hidden_layer_architectures[i][j])) - 0.5*hidden_layer_architectures[i][j], 'ko')
            
            axs[i, 1].set_ylim([-20, 20])
            axs[i, 1].set_xlim([0, 10])
            axs[i, 1].grid()
            axs[i, 1].set_title(r"$\mathcal{L}$: %.4e, $\mathcal{C}$: %.3e" % (pareto_scores[i], pareto_costs[i]))

        fig.savefig(optim_directory + "/pareto_analysis.pdf",format='pdf',bbox_inches='tight')
        plt.close(fig)
        return 
    
    def PlotParetoConvergence(self):
        history_file = self.GetOptimHistory()

        optim_header = self.SetFolderHeader()

        optim_directory = self._Config.GetOutputDir()+"/"+optim_header

        last_gen_groups = []
        with open(history_file,'r') as fid:
            lines = fid.readlines()[1:]
        history_score = []
        history_size= []
        history_gens = []
        for line in lines:
            history_score.append(np.log10(float(line.strip().split(',')[-2])))
            history_size.append(float(line.strip().split(',')[-1]))
            history_gens.append(int(float(line.strip().split(',')[0])))
        history_score = np.array(history_score)
        history_size = np.array(history_size)
        history_gens = np.array(history_gens)
        last_gen = history_gens[-1]
        ix_last_gen = (history_gens == last_gen)
        popsize = np.sum(ix_last_gen)

        last_gen_size = history_size[ix_last_gen]
        last_gen_score = history_score[ix_last_gen]
        
        min_score,max_score = np.min(last_gen_score),np.max(last_gen_score)
        min_size,max_size = np.min(last_gen_size),np.max(last_gen_size)
        
        last_gen_size_scaled = (last_gen_size - min_size)/(max_size - min_size)
        last_gen_score_scaled = (last_gen_score - min_score)/(max_score - min_score)
        
        P_last_scaled = np.hstack((last_gen_size_scaled[:,np.newaxis],last_gen_score_scaled[:,np.newaxis]))
        last_gen_groups.append(np.hstack((last_gen_size[:,np.newaxis],last_gen_score[:,np.newaxis])))

        max_score_scaled = (np.max(history_score) - min_score)/(max_score - min_score)
        max_size_scaled = (np.max(history_size) - min_size)/(max_size - min_size)

        #ref_point_scaled = np.array([max_size_scaled, max_score_scaled])
        ref_point_scaled = 0.5 * (np.min(P_last_scaled,axis=0) + np.max(P_last_scaled,axis=0))
        ind_GD = GDPlus(P_last_scaled)
        ind_HV = HV(ref_point=ref_point_scaled)

        history_score_scaled = (np.array(history_score) - min_score)/(max_score - min_score)
        history_size_scaled = (np.array(history_size) - min_size)/(max_size - min_size)
        
        hypervolume = []
        general_distance = []

        for iGen in range(0, len(history_score), popsize):
            P_gen = np.hstack((history_size_scaled[iGen:iGen+popsize][:,np.newaxis], history_score_scaled[iGen:iGen+popsize][:,np.newaxis]))
            dist_general = ind_GD(P_gen)
            HV_gen = ind_HV(P_gen)
            general_distance.append(dist_general)
            hypervolume.append(HV_gen)

        
        HV_last = hypervolume[-1]
        HV_pref = hypervolume[np.argwhere(np.array(hypervolume) < HV_last)[-1][0]]
        HV_change = 100*(HV_pref - HV_last)/HV_last
        fig, axs = plt.subplots(nrows=1,ncols=2,figsize=[16,7])
        ax = axs[0]
        ax.plot(general_distance, 'o-')
        #ax.legend(fontsize=20)
        ax.set_title(r"General Distance Evolution w.r.t. Last Generation",fontsize=20)
        ax.set_xlabel(r"Generation $[-]$",fontsize=20)
        ax.set_ylabel(r"General Distance Indicator $(GD)[-]$",fontsize=20)
        ax.grid()
        ax.tick_params(which='both',labelsize=20)

        ax = axs[1]
        ax.plot(hypervolume, 'o-')
        #ax.legend(fontsize=20)
        ax.set_title(r"Pareto front hyper-volume evolution",fontsize=20)
        ax.set_xlabel(r"Generation $[-]$",fontsize=20)
        ax.set_ylabel(r"Pareto hyper-volume $(HV)[-]$",fontsize=20)
        ax.grid()
        ax.tick_params(which='both',labelsize=20)
        
        fig.savefig(optim_directory + "/pareto_history_plot.pdf",format='pdf',bbox_inches='tight')
        plt.show()

class PlotHPOResults_FGM(PlotHPOResults):
    __group_idx:int = None 

    def __init__(self, Config:Config_FGM, group_idx:int=0):
        PlotHPOResults.__init__(self, Config)
        self.__group_idx = group_idx
        return 
    
    def SetGroupIndex(self, group_idx:int):
        self.__group_idx = group_idx 
        return 
    
    def SetFolderHeader(self):
        optim_header = "Architectures_Group%i_Optim" % (self.__group_idx+1)
        optim_header += self._get_optim_extension()
        return optim_header
    
    def GetOptimHistory(self):
        history_filename = "history_optim_Group%i" % (self.__group_idx+1) + "_"
        history_filename += self._get_optim_extension()
        history_filepath = self._Config.GetOutputDir()+("/Architectures_Group%i_Optim%s/" % (self.__group_idx+1, self._get_optim_extension()))+history_filename + ".csv"
        return history_filepath
    
    
