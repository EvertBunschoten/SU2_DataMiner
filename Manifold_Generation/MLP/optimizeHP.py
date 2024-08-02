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
from scipy.optimize import minimize, Bounds

from Common.Properties import DefaultProperties
from Common.Config_base import Config 
from Manifold_Generation.MLP.Trainer_Base import EvaluateArchitecture
from Manifold_Generation.MLP.Trainers_NICFD.Trainers import EvaluateArchitecture_NICFD

class MLPOptimizer:
    """Class for hyper-parameter optimization of entropic fluid model multi-layer perceptrons.
    """

    _Config:Config = None     # EntropicAI configuration.
    __optimizer:pygad.GA = None         # PyGaD optimization instance.
    __n_workers:int = 1                 # Number of CPU cores used for distributing the work per generation.

    # Hyper-parameter default settings and bounds.

    # Mini-batch exponent (base 2) for training.
    __optimize_batch:bool = True 
    __batch_expo:int = DefaultProperties.batch_size_exponent
    __batch_expo_min:int=3
    __batch_expo_max:int=7
    
    # Optimize learning rate decay parameters.
    __optimize_LR:bool = True 

    # Initial learning rate exponent (base 10).
    __alpha_expo:float = DefaultProperties.init_learning_rate_expo
    __alpha_expo_min:float = -3.0
    __alpha_expo_max:float = -1.0

    # Learning rate decay parameter for exponential learning rate decay schedule.
    __lr_decay:float=DefaultProperties.learning_rate_decay
    __lr_decay_min:float = 0.9
    __lr_decay_max:float = 1.0 

    # Optimize hidden layer architecture.
    __optimize_NN:bool = True

    # Number of perceptrons applied to the hidden layer of the network.
    __NN_min:int = 10
    __NN_max:int = 100 
    __architecture:list[int]=[30]

    # Optimization history.
    __population_history:list = []
    __fitness_history:list = []

    def __init__(self, Config_in:Config=None, load_file:str=None):
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
            gene_trainparams.append(int)
            lowerbound.append(self.__batch_expo_min)
            upperbound.append(self.__batch_expo_max)
        if self.__optimize_LR:
            gene_trainparams.append(float)
            gene_trainparams.append(float)
            
            lowerbound.append(self.__alpha_expo_min)
            lowerbound.append(self.__lr_decay_min)
            upperbound.append(self.__alpha_expo_max)
            upperbound.append(self.__lr_decay_max)
        if self.__optimize_NN:
            gene_trainparams.append(int)
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
                     fitness_func=self.fitnessFunction,\
                     gene_type=gene_trainparams,\
                     num_genes=N_genes,\
                     init_range_low=lowerbound,\
                     init_range_high=upperbound,\
                     gene_space=gene_space,\
                     sol_per_pop=10*N_genes,\
                     num_parents_mating=6,\
                     parallel_processing=["process",self.__n_workers],\
                     random_seed=1,\
                     on_generation=self.saveGenerationInfo)
        
        return 

    def inv_fitnessFunction(self, x):
        return -self.fitnessFunction(ga_instance=None, x=x, x_idx=0)
    
    def __setSimplexOptimizer(self):
        gene_trainparams, lowerbound, upperbound = self.__prepareBounds()
        N_genes = len(gene_trainparams)
        x0 = np.zeros(len(lowerbound))
        x0[0] = self.__alpha_expo
        x0[1] = self.__lr_decay
        print(x0)
        bounds = Bounds(lb=lowerbound, ub=upperbound)
        print(bounds)
        options = {"maxiter":1000,\
                   "disp":True}
        res = minimize(self.inv_fitnessFunction, x0=x0, method='Nelder-Mead',bounds=bounds,options=options)

        x_optim = res.x 

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
        self.save_dir = self._Config.GetOutputDir()+"/Architectures_Optim"+history_extension
        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)
        
        # Prepare bounds and set optimizer.
        if self.__optimize_NN or self.__optimize_batch:
            self.__setOptimizer()
            
            # Initiate HP optimization.
            self.__optimizer.run()
        else:
            self.__setSimplexOptimizer()

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
        Evaluator.SetHiddenLayers(self.__architecture)

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
            Evaluator.SetHiddenLayers(architecture)
            idx_x += 1 

        return

    def fitnessFunction(self, ga_instance:pygad.GA, x:np.ndarray, x_idx:int):
        """ Fitness function evaluated during GA routine.
        """

        # Initate MLP evaluation class.
        Evaluator:EvaluateArchitecture = EvaluateArchitecture_NICFD(self._Config)

        # Set CPU index.
        Evaluator.SetTrainHardware("CPU", x_idx)

        Evaluator.SetVerbose(0)

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
    

class PlotHPOResults:
    __Config:Config = None 
    __optimize_learningrate:bool = True 
    __optimize_batch:bool = True 
    __optimize_architecture:bool = True 
    __completed_models:list[str] = []
    __C2_test_set_scores:list[float] = []
    __P_test_set_scores:list[float] = []
    __T_test_set_scores:list[float] = []
    __hidden_layer_neurons:list[int] = []
    __lr_decay:list[float] = []
    __alpha_expo:list[float] = []
    __batch_expo:list[int] = [] 

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
        self.__optimize_learningrate = optimize_LR
        return 
    
    def Optimize_Batch_HP(self, optimize_batch:bool=True):
        """Consider the mini-batch size exponent as a hyper-parameter during optimization.

        :param optimize_batch: consider mini-batch size exponent (True, default), or not (False)
        :type optimize_batch: bool, optional
        """
        self.__optimize_batch = optimize_batch 
        return 
    
    def Optimize_Architecture_HP(self, optimize_architecture:bool=True):
        """Consider the hidden layer perceptron count as a hyper-parameter during optimization.

        :param optimize_architecture: consider hidden layer perceptron count (True, default) or not (False)
        :type optimize_architecture: bool, optional
        """
        self.__optimize_architecture = optimize_architecture 
        return 
    
    def ReadArchitectures(self):
        optim_header = "Architectures_Optim"
        if self.__optimize_batch:
            optim_header += "B"
        if self.__optimize_learningrate:
            optim_header += "LR"
        if self.__optimize_architecture:
            optim_header += "A"
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
                                self.__alpha_expo.append(alpha_expo)
                                self.__lr_decay.append(lr_decay)
                                self.__batch_expo.append(batch_expo)
                                self.__hidden_layer_neurons.append(nN)
                        
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