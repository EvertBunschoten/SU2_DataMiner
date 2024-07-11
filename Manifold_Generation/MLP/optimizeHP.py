import pygad
from Trainers import EvaluateArchitecture
import os 
import pickle 
from joblib import Parallel, delayed
import sys
import csv
import numpy as np
from scipy.optimize import minimize,Bounds
    
class MLPOptimizer:
    group_idx:int = 0
    batch_size_bounds:list[int] = [3, 6]
    optimizer:pygad.GA
    batch_expo:int =6
    activation_function:str = "exponential"
    NLayers_max:int = 10
    NLayers_min:int = 1
    alpha_expo:float = -2.8
    alpha_expo_min:float = -3.0
    lr_decay_min:float = 0.85
    alpha_expo_max:float = -1.0
    lr_decay_max:float = 1.0 

    NN_min:int = 10
    NN_max:int = 100 
    n_workers:int = 1 

    NN_current:int = 36
    __population_history:list = []
    __fitness_history:list = []
    def __init__(self, load_file:str=None):
        """Class constructor
        """
        if load_file:
            print("Loading optimizer configuration")
            with open(load_file, "rb") as fid:
                loaded_config = pickle.load(fid)
            print(loaded_config.__dict__)
            self.__dict__ = loaded_config.__dict__.copy()
            print("Loaded optimizer file")


    def SetNWorkers(self, n_workers:int):
        self.n_workers = n_workers
    def optimizeBatch_and_Activation_Function(self):
        self.optimizer = pygad.GA(num_generations=20,\
                     fitness_func=self.fitness,\
                     gene_type=[int, int],\
                     num_genes=2,\
                     gene_space=[range(7),range(self.batch_size_bounds[0],self.batch_size_bounds[1]+1)],\
                     sol_per_pop=20,\
                     num_parents_mating=6,\
                     parallel_processing=["process",self.n_workers],\
                     random_seed=1,\
                     on_generation=self.saveGenerationInfo)
        self.optimizer.run()
        solution, solution_fitness, solution_idx = self.optimizer.best_solution()
        self.batch_expo = solution[0]
        self.activation_function_index = solution[1]
        print("Best activation function index: %i" % solution[0])
        print("Best batch size exponenet: %i" % solution[1])
        self.optimizer.plot_fitness()

    def optimizeLearningRate_simplex(self):
        x0 = np.array([-1.8654e+00, +9.8629e-01])
        bounds = Bounds(lb=np.array([self.alpha_expo_min, self.lr_decay_min]), ub=np.array([self.alpha_expo_max,self.lr_decay_max]))

        result = minimize(self.fitness_thingy, x0=x0, bounds=bounds, method="Nelder-Mead")
        print(result.x)

    def optimizeArchitectures_and_LearningRate(self):
        gene_architecture = [int]# + 2*(self.NLayers_max - self.NLayers_min)*[int]
        gene_trainparams = 2*[float]

        gene_types = gene_trainparams + gene_architecture
        lowerbound = [self.alpha_expo_min, self.lr_decay_min] + [self.NN_min]# * self.NLayers_min + [0, self.NN_min]*(self.NLayers_max - self.NLayers_min)
        upperbound = [self.alpha_expo_max, self.lr_decay_max] + [self.NN_max]# * self.NLayers_min + [2, self.NN_max]*(self.NLayers_max - self.NLayers_min)
        
        self.opt_history_filepath = "/home/ecbunschoten/NICFD/NICFD_MLP_Optimization/Architecture_Optimization/history_entropy.csv"
        with open(self.opt_history_filepath, "w+") as fid:
            fid.write("Generation nr,solution,fitness\n")

        self.optimizer = pygad.GA(num_generations=10*len(gene_types),\
                     fitness_func=self.fitness_ScoreOnly,\
                     gene_type=gene_types,\
                     num_genes=len(gene_types),\
                     init_range_low=lowerbound,\
                     init_range_high=upperbound,\
                     gene_space=[{'low':self.alpha_expo_min, 'high':self.alpha_expo_max},\
                                 {'low':self.lr_decay_min, 'high':self.lr_decay_max},\
                                 {'low':self.NN_min, 'high':self.NN_max}],\
                     sol_per_pop=10*len(gene_types),\
                     num_parents_mating=6,\
                     parallel_processing=["process",self.n_workers],\
                     random_seed=1,\
                     on_generation=self.saveGenerationInfo)
        self.optimizer.run()

    def optimizeLearningRate(self):
        gene_trainparams = 2*[float]

        gene_types = gene_trainparams 
        lowerbound = [self.alpha_expo_min, self.lr_decay_min] 
        upperbound = [self.alpha_expo_max, self.lr_decay_max] 
        
        self.opt_history_filepath = "/home/ecbunschoten/NICFD/NICFD_MLP_Optimization/Architecture_Optimization/history_entropy_LR.csv"
        with open(self.opt_history_filepath, "w+") as fid:
            fid.write("Generation nr,solution,fitness\n")

        self.optimizer = pygad.GA(num_generations=10*len(gene_types),\
                     fitness_func=self.fitness_ScoreOnly,\
                     gene_type=gene_types,\
                     num_genes=len(gene_types),\
                     init_range_low=lowerbound,\
                     init_range_high=upperbound,\
                     gene_space=[{'low':self.alpha_expo_min, 'high':self.alpha_expo_max},\
                                 {'low':self.lr_decay_min, 'high':self.lr_decay_max}],\
                     sol_per_pop=10*len(gene_types),\
                     num_parents_mating=6,\
                     parallel_processing=["process",self.n_workers],\
                     random_seed=1,\
                     on_generation=self.saveGenerationInfo)
        self.optimizer.run()

    def saveGenerationInfo(self, ga_instance:pygad.GA):
        population = ga_instance.population
        pop_fitness = ga_instance.last_generation_fitness

        # test_score = np.power(10, -pop_fitness[:, -2])
        # cost_parameter = pop_fitness[:, -1]/1000

        test_score = np.power(10, -pop_fitness)
        self.__population_history.append(population)
        self.__fitness_history.append(pop_fitness)
        generation = ga_instance.generations_completed
        #pop_and_fitness = np.hstack((generation*np.ones([len(population),1]), population, test_score[:, np.newaxis], cost_parameter[:, np.newaxis]))
        pop_and_fitness = np.hstack((generation*np.ones([len(population),1]), population, test_score[:, np.newaxis]))
        
        with open(self.opt_history_filepath, "a+") as fid:
            csvWriter =csv.writer(fid, delimiter=',')
            csvWriter.writerows(pop_and_fitness)

    def translateGene(self, x, Evaluator:EvaluateArchitecture):
        alpha_expo = None 
        lr_decay = None 
        batch_expo = None 
        activation_idx = None 
        architecture = None 
        # if self.optimize_batchphi:
        #     activation_idx = x[0]
        #     batch_expo = x[1]
        #     Evaluator.SetActivationFunctionIndex(activation_idx)
        #     Evaluator.SetBatchExpo(batch_expo)
        # elif self.optimize_learningrate and self.optimize_architectures:
        alpha_expo = x[0]
        lr_decay = x[1]
        architecture = [self.NN_current]
        if len(x) > 2:
            architecture = [x[2]]
        # for i in range(2, 2+self.NLayers_min):
        #     architecture.append(x[i])
        # for i in range(2+self.NLayers_min,len(x)-1,2):
        #     if x[i] > 0:
        #         architecture.append(x[i+1])

        Evaluator.SetBatchExpo(self.batch_expo)
        Evaluator.SetActivationFunction(self.activation_function)
        Evaluator.SetAlphaExpo(alpha_expo)
        Evaluator.SetLRDecay(lr_decay)
        Evaluator.SetArchitecture(architecture)
        # elif self.optimize_architectures:
        #     architecture = []
        #     for i in range(self.NLayers_min):
        #         architecture.append(x[i])
        #     for i in range(2+self.NLayers_min,len(x)-1,2):
        #         if x[i] > 0:
        #             architecture.append(x[i+1])

        #     Evaluator.SetBatchExpo(self.batch_expo)
        #     Evaluator.SetActivationFunction(self.activation_function)
        #     Evaluator.SetArchitecture(architecture)
        return
    def SetArchitecture(self, NN:int):
        self.NN_current = NN 

    # def fitness_trainparams_only(self, ga_instance:pygad.GA, x, x_idx):
        
    #     Evaluator = EvaluateArchitecture(Config=self.Config,group_idx=self.group_idx)
    #     Evaluator.SetTrainHardware("CPU", os.getpid() % self.n_workers)
        
    #     self.translateGene(x, Evaluator=Evaluator)

    #     Evaluator.CommenceTraining()
    #     Evaluator.TrainPostprocessing()
    #     return -np.log10(Evaluator.GetTestScore())

    def fitness(self, ga_instance:pygad.GA, x, x_idx):

        Evaluator:EvaluateArchitecture = EvaluateArchitecture()
        Evaluator.SetTrainHardware("CPU", x_idx)
        self.translateGene(x, Evaluator=Evaluator)
        Evaluator.SetSaveDir("/home/ecbunschoten/NICFD/NICFD_MLP_Optimization/Architecture_Optimization/Architectures_TPC2/")
        Evaluator.SetTrainFileHeader("/home/ecbunschoten/NICFD/NICFD_MLP_Optimization/single_dataset")
        Evaluator.SetNEpochs(1000)
        Evaluator.CommenceTraining()
        Evaluator.TrainPostprocessing()
        test_score = Evaluator.GetTestScore() 
        cost_parameter = Evaluator.GetCostParameter()
        fitness_test_score = -np.log10(test_score)
        fitness_cost = 1000 / cost_parameter
        del Evaluator 
        return [fitness_test_score, fitness_cost]

    def fitness_ScoreOnly(self, ga_instance:pygad.GA, x, x_idx):
        Evaluator:EvaluateArchitecture = EvaluateArchitecture()
        Evaluator.SetTrainHardware("CPU", x_idx)
        self.translateGene(x, Evaluator=Evaluator)
        Evaluator.SetSaveDir("/home/ecbunschoten/NICFD/NICFD_MLP_Optimization/Architecture_Optimization/Architectures_LR/")
        Evaluator.SetTrainFileHeader("/home/ecbunschoten/NICFD/NICFD_MLP_Optimization/single_dataset")
        Evaluator.SetNEpochs(1000)
        Evaluator.CommenceTraining()
        Evaluator.TrainPostprocessing()
        test_score = Evaluator.GetTestScore() 
        fitness_test_score = -np.log10(test_score)
        del Evaluator 
        return fitness_test_score
    
    def fitness_thingy(self, x):
        alpha_expo = x[0]
        lr_decay = x[1]
        Evaluator:EvaluateArchitecture = EvaluateArchitecture()
        Evaluator.SetTrainHardware("CPU", 0)
        Evaluator.SetAlphaExpo(alpha_expo)
        Evaluator.SetLRDecay(lr_decay)
        Evaluator.SetArchitecture([self.NN_current])
        Evaluator.SetBatchExpo(6)
        Evaluator.SetSaveDir("/home/ecbunschoten/NICFD/NICFD_MLP_Optimization/Architecture_Optimization/Architectures_LR/")
        Evaluator.SetTrainFileHeader("/home/ecbunschoten/NICFD/NICFD_MLP_Optimization/single_dataset")
        Evaluator.SetNEpochs(1000)
        Evaluator.CommenceTraining()
        Evaluator.TrainPostprocessing()
        test_score = Evaluator.GetTestScore() 
        fitness_test_score = np.log10(test_score)
        del Evaluator 
        return fitness_test_score
    
    def SaveOptimizer(self, file_name:str):
        file = open(file_name+'.cfg','wb')
        pickle.dump(self, file)
        file.close()

thingy = MLPOptimizer()
nWorkers = int(sys.argv[-1])
thingy.SetNWorkers(nWorkers)
thingy.SetArchitecture(40)
thingy.optimizeLearningRate_simplex()


# batch_range = [7,6,5,4,3]
# phi_range = ["linear","elu","relu","gelu","sigmoid","tanh","swish"]
# best_batch = batch_range[0]
# betst_phi = phi_range[0]
# L_min = 1e5

# n_workers = 10
# def EvaluateStuff(b, phi, iWorker):
#     Evaluator = EvaluateArchitecture(Config=C, group_idx=0)
#     Evaluator.SetTrainHardware("CPU", iWorker)
#     Evaluator.SetBatchExpo(b)
#     Evaluator.SetActivationFunction(phi)
#     Evaluator.CommenceTraining()
#     Evaluator.TrainPostprocessing()
#     return Evaluator.GetTestScore()

# iWorker = int(sys.argv[-1])
# for phi in phi_range:
#     b = int(sys.argv[-2])
#     score = EvaluateStuff(b,phi,iWorker)
#     print(b, score)

# for b in batch_range:
#     for p in phi_range:
#         Evaluator = EvaluateArchitecture(Config=C, group_idx=0)
#         Evaluator.SetSaveDir(C.GetOutputDir() + "/architectures_Group1/")
#         Evaluator.SetBatchExpo(b)
#         Evaluator.SetActivationFunction(p)
#         Evaluator.CommenceTraining()
#         Evaluator.TrainPostprocessing()
#         L = Evaluator.GetTestScore()
#         if L < L_min:
#             best_batch = b 
#             betst_phi = p 
#             L_min = L 
#         print("Best batch size: %i" % best_batch)
#         print("Best activation function: "+ betst_phi)
# ArchitectureOptimizer = MLPOptimizer(Config_in=C)
# ArchitectureOptimizer.SetNWorkers(1)
# ArchitectureOptimizer.batch_expo = 6
# ArchitectureOptimizer.activation_function_index = 5
# ArchitectureOptimizer.optimizeArchitectures_and_LearningRate()

# batch_expo_min = 3
# batch_expo_max = 6 

# # optimizer = pygad.GA(num_generations=1,\
# #                      fitness_func=fitness_trainparams_only,\
# #                      gene_type=[int, int],\
# #                      num_genes=2,\
# #                      gene_space=[range(len(activation_function_options)),range(batch_expo_min,batch_expo_max+1)],\
# #                      sol_per_pop=20,\
# #                      num_parents_mating=6,\
# #                      parallel_processing=["process",n_workers],\
# #                      random_seed=1)
# # optimizer.run()

# N_layers_max = 10
# N_layers_min = 1 

# gene_architecture = N_layers_min * [int] + 2*(N_layers_max - N_layers_min)*[int]
# gene_trainparams = 2*[float]
# gene_types = gene_trainparams + gene_architecture 

# alpha_expo_low = -3.0
# lr_decay_low = 0.85
# alpha_expo_high = -1.0
# lr_decay_high = 1.0 

# NN_min = 3 
# NN_max = 40 

# lowerbound = [alpha_expo_low, lr_decay_low] + [NN_min] * N_layers_min + [0, NN_min]*(N_layers_max - N_layers_min)
# upperbound = [alpha_expo_high, lr_decay_high] + [NN_max] * N_layers_min + [2, NN_max]*(N_layers_max - N_layers_min)

# optimizer_architectures = pygad.GA(num_generations=1,\
#                      fitness_func=fitness_trainparams_only,\
#                      gene_type=gene_types,\
#                      num_genes=len(gene_types),\
#                      init_range_low=lowerbound,\
#                      init_range_high=upperbound,\
#                      sol_per_pop=10*len(gene_types),\
#                      num_parents_mating=6,\
#                      parallel_processing=["process",n_workers],\
#                      random_seed=1)
# optimizer_architectures.run()
# solution, solution_fitness, solution_idx = optimizer.best_solution()
# print("Parameters of the best solution : {solution}".format(solution=solution))
# print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

