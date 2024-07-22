import pygad
import os 
import pickle 
import csv
import numpy as np
from Common.EntropicAIConfig import EntropicAIConfig 
from Manifold_Generation.MLP.Trainers import EvaluateArchitecture

class MLPOptimizer:
    """Class for hyper-parameter optimization of entropic fluid model multi-layer perceptrons.
    """

    _Config:EntropicAIConfig = None     # EntropicAI configuration.
    __optimizer:pygad.GA = None         # PyGaD optimization instance.
    __n_workers:int = 1                 # Number of CPU cores used for distributing the work per generation.

    # Hyper-parameter default settings and bounds.

    # Mini-batch exponent (base 2) for training.
    __optimize_batch:bool = True 
    __batch_expo:int =6
    __batch_expo_min:int=3
    __batch_expo_max:int=7
    
    # Optimize learning rate decay parameters.
    __optimize_LR:bool = True 

    # Initial learning rate exponent (base 10).
    __alpha_expo:float = -2.8
    __alpha_expo_min:float = -3.0
    __alpha_expo_max:float = -1.0

    # Learning rate decay parameter for exponential learning rate decay schedule.
    __lr_decay:float=0.996
    __lr_decay_min:float = 0.85
    __lr_decay_max:float = 1.0 

    # Optimize hidden layer architecture.
    __optimize_NN:bool = True

    # Number of perceptrons applied to the hidden layer of the network.
    __NN_min:int = 10
    __NN_max:int = 100 
    __architecture:list[int]=[40]

    # Optimization history.
    __population_history:list = []
    __fitness_history:list = []

    def __init__(self, Config_in:EntropicAIConfig=None, load_file:str=None):
        """Class constructor
        """

        # Store configuration
        self._Config = Config_in 
    
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
        self.__n_workers = n_workers
        return 
    
    def Optimize_LearningRate_HP(self, optimize_LR:bool=True):
        """Consider learning-rate hyper-parameters in optimization.

        :param optimize_LR: consider learning rate parameters(True, default), or not (False)
        :type optimize_LR: bool, optional
        """
        self.__optimize_LR = optimize_LR
        return 
    
    def SetAlpha_Expo(self, val_alpha_expo:float=-2.8):
        """Set initial learning rate exponent (base 10).

        :param val_alpha_expo: _description_, defaults to -2.8
        :type val_alpha_expo: float, optional
        :raises Exception: if initial learning rate exponent value is positive.
        """

        if val_alpha_expo >= 0:
            raise Exception("Initial learing rate exponent should be negative.")
        self.__alpha_expo = val_alpha_expo

        return 
    
    def SetLR_Decay(self, val_lr_decay:float=0.996):
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
        self.__optimize_batch = optimize_batch 
        return 
    
    def SetBatch_Expo(self, batch_expo:int=6):
        """Set training batch exponent value (base 2).

        :param batch_expo: training batch exponent value, defaults to 6
        :type batch_expo: int, optional
        :raises Exception: if training batch exponent value is lower than 1.
        """

        if batch_expo < 1:
            raise Exception("Batch size exponent should be at least 1.")
        self.__batch_expo = batch_expo 
        
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
        self.__optimize_NN = optimize_architecture 
        return 
    
    def SetArchitecture(self, architecture:list[int]=[40]):
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
        if self.__optimize_batch:
            gene_trainparams += [int]
            lowerbound.append(self.__batch_expo_min)
            upperbound.append(self.__batch_expo_max)
        if self.__optimize_LR:
            gene_trainparams += [float, float]
            lowerbound.append(self.__alpha_expo_min)
            lowerbound.append(self.__lr_decay_min)
            upperbound.append(self.__alpha_expo_max)
            upperbound.append(self.__lr_decay_max)
        if self.__optimize_NN:
            gene_trainparams += [int]
            lowerbound.append(self.__NN_min)
            upperbound.append(self.__NN_max)

        return gene_trainparams, lowerbound, upperbound 
    
    def __setOptimizer(self):
        """Prepare PyGaD optimization routine.
        """

        # Set gene types and bounds
        gene_trainparams, lowerbound, upperbound = self.__prepareBounds()
        N_genes = len(gene_trainparams)
        gene_space = []
        for lb, ub in zip(lowerbound, upperbound):
            gene_space.append({'low':lb,'high':ub})

        # Initiate PyGaD instance with 
        self.__optimizer = pygad.GA(num_generations=10*N_genes,\
                     fitness_func=self.__fitnessFunction,\
                     gene_type=gene_trainparams,\
                     num_genes=N_genes,\
                     init_range_low=lowerbound,\
                     init_range_high=upperbound,\
                     gene_space=gene_space,\
                     sol_per_pop=10*N_genes,\
                     num_parents_mating=6,\
                     parallel_processing=["process",self.__n_workers],\
                     random_seed=1,\
                     on_generation=self.__saveGenerationInfo)
        
        return 

    def optimizeHP(self):
        """Initate hyper-parameter optimization routine.

        :raises Exception: if neither of the available sets of hyper-parameters are considered for optimization.
        """
        if not any((self.__optimize_batch, self.__optimize_LR, self.__optimize_NN)):
            raise Exception("At least one of the hyper-parameter options should be considered for optimization.")
        
        # Prepare optimization history output file.
        history_extension = ""
        if self.__optimize_batch:
            history_extension += "B"
        if self.__optimize_LR:
            history_extension += "LR"
        if self.__optimize_NN:
            history_extension += "A"
        self.opt_history_filepath = self._Config.GetOutputDir() + "/history_entropy_"+history_extension+".csv"
        with open(self.opt_history_filepath, "w+") as fid:
            if self.__optimize_NN:
                fid.write("Generation nr,solution,fitness,cost\n")
            else:
                fid.write("Generation nr,solution,fitness\n")
        
        # Prepare optimization output directory.
        self.save_dir = self._Config.GetOutputDir()+"/Architectures_Optim"+history_extension+"/"
        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)
        
        # Prepare bounds and set optimizer.
        self.__setOptimizer()
        
        # Initiate HP optimization.
        self.__optimizer.run()

        return 
    
    def __saveGenerationInfo(self, ga_instance:pygad.GA):
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
    
    def __translateGene(self, x:np.ndarray[float], Evaluator:EvaluateArchitecture):
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
        Evaluator.SetArchitecture(self.__architecture)

        # Set hyper-parameter according to gene.
        idx_x = 0
        if self.__optimize_batch:
            batch_expo = x[idx_x]
            Evaluator.SetBatchExpo(batch_expo)
            idx_x += 1 
        if self.__optimize_LR:
            alpha_expo = x[idx_x]
            Evaluator.SetAlphaExpo(alpha_expo)
            idx_x += 1 
            lr_decay = x[idx_x]
            Evaluator.SetLRDecay(lr_decay)
            idx_x += 1 
        if self.__optimize_NN:
            architecture = [x[idx_x]]
            Evaluator.SetArchitecture(architecture)
            idx_x += 1 

        return

    def __fitnessFunction(self, ga_instance:pygad.GA, x:np.ndarray, x_idx:int):
        """ Fitness function evaluated during GA routine.
        """

        # Initate MLP evaluation class.
        Evaluator:EvaluateArchitecture = EvaluateArchitecture(self._Config)

        # Set CPU index.
        Evaluator.SetTrainHardware("CPU", x_idx)

        # Translate gene and update hyper-parameters.
        self.__translateGene(x, Evaluator=Evaluator)

        # Set output directory for MLP training callbacks.
        Evaluator.SetSaveDir(self.save_dir)

        # Train for 1000 epochs direct and physics-informed.
        Evaluator.SetNEpochs(1000)
        Evaluator.CommenceTraining()

        # Extract test set evaluation score.
        Evaluator.TrainPostprocessing()
        test_score = Evaluator.GetTestScore() 

        # Scale test set loss to fitness.
        fitness_test_score = -np.log10(test_score)

        # Free up memory
        del Evaluator 

        return fitness_test_score
    
    def SaveOptimizer(self, file_name:str):
        """Save optimizer instance.

        :param file_name: file name under which to save the optimizer instance.
        :type file_name: str
        """

        file = open(self._Config.GetOutputDir()+"/"+file_name+'.hpo','wb')
        pickle.dump(self, file)
        file.close()
        return 
    