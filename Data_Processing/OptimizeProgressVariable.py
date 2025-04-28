###############################################################################################
#       #      _____ __  _____      ____        __        __  ____                   #        #
#       #     / ___// / / /__ \    / __ \____ _/ /_____ _/  |/  (_)___  ___  _____   #        #
#       #     \__ \/ / / /__/ /   / / / / __ `/ __/ __ `/ /|_/ / / __ \/ _ \/ ___/   #        #
#       #    ___/ / /_/ // __/   / /_/ / /_/ / /_/ /_/ / /  / / / / / /  __/ /       #        #
#       #   /____/\____//____/  /_____/\__,_/\__/\__,_/_/  /_/_/_/ /_/\___/_/        #        #
#       #                                                                            #        #
###############################################################################################

######################### FILE NAME: OptimizeProgressVariable.py ##############################
#=============================================================================================#
# author: Evert Bunschoten                                                                    |
#    :PhD Candidate ,                                                                         |
#    :Flight Power and Propulsion                                                             |
#    :TU Delft,                                                                               |
#    :The Netherlands                                                                         |
#                                                                                             |
#                                                                                             |
# Description:                                                                                |
#  Class for optimizing the definition for the progress variable for a given set of flamelet  |
#  data.                                                                                      |
#                                                                                             |
# Version: 2.0.0                                                                              |
#                                                                                             |
#=============================================================================================#

import numpy as np 
import sys
import os
from numpy.core.multiarray import array as array
from tqdm import tqdm
import matplotlib.pyplot as plt 
from scipy.optimize import differential_evolution, Bounds, LinearConstraint, minimize
from sklearn.decomposition import PCA

from Common.Properties import FGMVars
from Common.DataDrivenConfig import Config_FGM

class PVOptimizer:
    """Optimize the progress variable weights for all flamelets in the manifold given relevant species in the mixture.
    """
    # Class for optimizing the progress variable weights for monotonicity

    _Config:Config_FGM = None  # Config_FGM class.

    _pv_definition_optim = None # Optimized progress variable definition.
    _pv_weights_optim = None    # Optimized progress variable weights.

    __valid_filepathnames:list[str] = []
    __flamelet_variables:list[str] = None  # Variable names in flamelet data files.

    _Y_flamelets = None   # Flamelet mass fraction data
    _idx_relevant_species:np.ndarray[float] = None  # Indices of species with a high enough range to be considered as pv species.
    __additional_variables:list[str] = []   # Additional variable names to consider in the definition of the progress vector.
    _idx_additional_vars:list[int] = []    # Indices of additional variables.

    _delta_Y_flamelets:np.ndarray[float] = None # Flamelet species mass fraction increment vector.
    _progress_vector:np.ndarray[float] = None   # Progress vector values of flamelet data set.


    __custom_generation_set:bool = False    # User-defined number of generations.
    _N_generations:int    # Number of generations during the genetic algorithm optimization.
    __custom_population_set:bool = False    # User-defined population size.
    _population_size:int     # Genetic algorithm population size.

    _convergence = None # Convergence of optimization process.
    _min_fitness = 1.0  # Current minimum fitness value.

    _n_workers:int=1   # Number of CPUs used to compute the population merit in parallel.

    __CurveStepTolerance:float = 1e-4       # Progress vector increment threshold.
    __SpeciesRangeTolerance:float = 1e-5    # Species range threshold.
    _current_generation:int = 0

    __Tu_min:float = -1e32
    __Tu_max:float = 1e32
    __phi_min:float = 0.0
    __phi_max:float = 1e32 

    _species_bounds_set:list[str] = []
    _species_custom_lb:list[float] = []
    _species_custom_ub:list[float] = []

    _output_dir:str = None 

    def __init__(self, Config:Config_FGM):
        """Progress variable optimization class constructor
        :param Config: Config_FGM class containing manifold info.
        :type Config: Config_FGM
        """

        self._Config = Config 
        print("Loading flameletAI configuration with name " + self._Config.GetConfigName())
        self._output_dir = self._Config.GetOutputDir()
        
        for sp in self._Config.GetFuelDefinition():
            self.SetSpeciesBounds(sp, ub=0.0)
        for sp in self._Config.GetOxidizerDefinition():
            self.SetSpeciesBounds(sp, ub=0.0)    

        sp_major_product = self.GetMajorProduct()
        self.SetSpeciesBounds(sp_major_product,lb=0)

        T_u_bounds = self._Config.GetUnbTempBounds()
        self.__Tu_max = T_u_bounds[1]
        self.__Tu_min = T_u_bounds[0]
        
        if not os.path.isdir(self._Config.GetOutputDir()+"/PV_Optimization"):
            os.mkdir(self._Config.GetOutputDir()+"/PV_Optimization")
        self._output_dir = self._Config.GetOutputDir()+"/PV_Optimization/"

        self.SetAdditionalProgressVariables([])
        return 
    
    def SetOutputDir(self, output_dir:str):
        """Specify custom directory where to store progress variable optimization history.

        :param output_dir: output directory.
        :type output_dir: str
        :raises Exception: if specified directory is not present or inaccessible.
        """
        if not os.path.isdir(output_dir):
            raise Exception("Specified output directory is inaccessible on current hardware.")
        
        self._output_dir = output_dir 
        return
    
    def SetMixtureBounds(self, phi_min:float=0.0, phi_max:float=1e32):
        """Specify the mixture status bounds between which flamelet data is considered for progress variable optimization.

        :param phi_min: lower mixture status bound, defaults to 0.0
        :type phi_min: float, optional
        :param phi_max: upper mixture status bound, defaults to 1e32
        :type phi_max: float, optional
        :raises Exception: if lower bound exceeds upper bound or negative values are provided.
        """
        if phi_min >= phi_max:
            raise Exception("Minimum mixture status should not exceed maximum mixture status.")
        if phi_min < 0 or phi_max < 0:
            raise Exception("Mixture status should be positive.")
        
        self.__phi_min = phi_min
        self.__phi_max = phi_max 
        return 
    
    def SetTemperatureBounds(self, Tu_min:float=250, Tu_max:float=2000):
        """Specify the unburnt temperature bounds between which flamelet data is considered for progress variable optimization.

        :param Tu_min: unburnt temperature lower bound, defaults to 250
        :type Tu_min: float, optional
        :param Tu_max: unburnt temperature upper bound, defaults to 2000
        :type Tu_max: float, optional
        :raises Exception: if lower bound exceeds upper bound or negative values are provided.
        """
        if Tu_min >= Tu_max:
            raise Exception("Minimum temperature should not exceed maximum temperature.")
        if Tu_min < 0 or Tu_max < 0:
            raise Exception("Temperature should be positive.")
        self.__Tu_min = Tu_min 
        self.__Tu_max = Tu_max
        return
    
    def SetNWorkers(self, n_workers:int):
        """Define the number of parallel workers to be used during the population merit and constraint computation.

        :param n_workers: number of CPU cores to be used in parallel.
        :type n_workers: int
        :raises Exception: if the given number of workers is lower than one.
        """
        if n_workers < 1:
            raise Exception("The number of workers used during the optimization process should be at least one.")
        self._n_workers = n_workers
        return
    
    def SetAdditionalProgressVariables(self, additional_vars:list[str]):
        """Add additional variables to the progress vector.

        :param additional_vars: list of variable names to include in progress vector.
        :type additional_vars: list[str]
        """
        self.__additional_variables = []
        for var in additional_vars:
            self.__additional_variables.append(var)
        return
    
    def SetSpeciesBounds(self, sp_name:str, lb:float=-1,ub:float=1):
        if ub <= lb:
            raise Exception("Lower bound should be higher than upper bound.")
        if sp_name in self._species_bounds_set:
            idx_sp = self._species_bounds_set.index(sp_name)
            self._species_custom_lb[idx_sp] = lb 
            self._species_custom_ub[idx_sp] = ub 
        else:
            self._species_bounds_set.append(sp_name)
            self._species_custom_lb.append(lb)
            self._species_custom_ub.append(ub)
        return
    
    def OptimizePV(self):
        """Run the progress variable species selection and weights optimization process."""

        # Load data from flamelet manifold for progress variable computation.
        self._CollectFlameletData()
        print("Initiating Progress Variable Optimization")
        print("Relevant species in definition: " + ", ".join(s for s in self._pv_definition_optim))
        if any(self.__additional_variables):
            print("Additional variables in progress vector: " + ", ".join(v for v in self.__additional_variables))

        self._current_generation = 0
        # Initiate progress variable optimization.
        self.RunOptimizer()

        # Scale optimized progress variable weights for normalization sake.
        self.ScalePV()

        # Visualize weights and save convergence trend.
        self.VisualizeWeights(self._pv_weights_optim, self._min_fitness)

        # Display optimized progress variable definition.
        print("Optimized progress variable:")
        print("[" + ",".join("\"" + pv + "\"" for pv in self._pv_definition_optim) + "]")
        print("[" + ",".join("%+.16e" % (pv) for pv in self._pv_weights_optim) + "]")
        np.save(self._output_dir + "/" + self._Config.GetConfigName() + "_PV_Def_optim.npy", self._pv_definition_optim, allow_pickle=True)
        np.save(self._output_dir + "/" + self._Config.GetConfigName() + "_Weights_optim.npy", self._pv_weights_optim, allow_pickle=True)
        return
    
    def CheckMonotonicity(self):
        monotonicity_full = LinearConstraint(self._delta_Y_flamelets, lb=0.0,keep_feasible=True)
        
        res_full = min(monotonicity_full.residual(self._pv_weights_optim)[0])
        if res_full > 0:
            print("PV definition is monotonic")
        else:
            print("PV definition is not monotonic!")
        return 
    
    def _CollectFlameletData(self):
        """Load flamelet manifold data from storage.
        """

        # Collect flamelet data between specified Tu and phi bounds.
        self.__FilterFlamelets()

        # Determine which species are most relevant for PV definition.
        self.__SelectRelevantSpecies()

        # Generate monotonic progress and mass fraction incement vectors.
        self._FilterFlameletData()
        return
    
    def VisualizeWeights(self, pv_weights_input:np.array, fitness_val:float):
        """Visualize progress variable weights via bar chart.

        :param pv_weights_input: array containing progress variable weights values.
        :type pv_weights_input: np.array
        :param fitness_val: fitness function value of current progress variable definition.
        :type fitness_val: float
        """
        fig = plt.figure(figsize=[10,10])
        ax = plt.axes()
        ax.clear()
        ax.bar(x=range(len(self._pv_definition_optim)), height=pv_weights_input,zorder=3)
        ax.set_xticks(range(len(self._pv_definition_optim)))
        ax.set_xticklabels(self._pv_definition_optim, rotation='vertical')
        ax.tick_params(which='both', labelsize=18)
        ax.set_ylabel(r"Progress Variable Weight", fontsize=20)
        ax.set_title(r"Fitness value: %+.3e" % fitness_val, fontsize=20)
        ax.grid(zorder=0)
        fig.savefig(self._output_dir + "/Weights_Visualization_"+self._Config.GetConfigName() + ".pdf", format='pdf', bbox_inches='tight')
        plt.close(fig)
        return
    
    def PlotConvergence(self):
        """Plot the convergence trend of the progress variable optimization process.
        """
        fig = plt.figure(figsize=[10, 10])
        ax = plt.axes()
        ax.plot(self._convergence, 'k',linewidth=2)
        ax.set_xlabel(r"Generation", fontsize=20)
        ax.set_ylabel(r"Penalty function value", fontsize=20)
        ax.set_title(r"Progress Variable Optimization Convergence", fontsize=20)
        ax.grid()
        ax.tick_params(which='both', labelsize=18)
        ax.set_yscale('log')
        fig.savefig(self._output_dir+"/PV_Optimization_History_"+self._Config.GetConfigName()+".pdf", format='pdf', bbox_inches='tight')
        plt.close(fig)
        return
    

    def penalty_function(self, x:np.array):
        """Optimization penalty function computing the derivative of the 
        progress vector w.r.t. that of the progress variable increment.

        :param x: progress variable weights vector.
        :type x: np.array[float]
        :return: maximum, absolute derivative (finite-differences) of the progress vector w.r.t. the progress variable increment.
        :rtype: float
        """
        pv_w = x / np.linalg.norm(x)
        pv_term = np.dot(self._delta_Y_flamelets, pv_w)
        y_term = self._progress_vector[:,0]
        fac = y_term / pv_term
        abs_fac = np.abs(fac)

        # Varience function:
        #avg_fac = np.average(abs_fac)
        #fac_variance = np.average(np.power(abs_fac - avg_fac, 2))
        #result = fac_variance + np.sum(np.power(fac[pv_term < 0.0],2))

        # Average function:
        #result = avg_fac + np.sum(np.power(fac[pv_term < 0.0],2))

        # Max grad:
        result = np.max(abs_fac) + np.sum(np.power(fac[pv_term < 0.0],2))
        return result 
    
    def monotonicity_penalty(self, x):
        pv_w = x / np.linalg.norm(x)
        pv_term = np.dot(self._delta_Y_flamelets, pv_w)
        y_term = self._progress_vector[:,0]
        fac = y_term / pv_term
        return np.sum(np.power(fac[pv_term < 0.0],2))
    
    def _Optimization_Callback(self, x:np.array,convergence:bool=False):
        """Callback function during progress variable optimization.

        :param x: progress variable weights.
        :type x: np.array[float]
        :param convergence: whether solution has converged, defaults to False
        :type convergence: bool, optional
        """
        # Normalize and scale pv weights at equivalence ratio of 1.0.
        pv_weights = x / np.linalg.norm(x)
        self._pv_weights_optim = pv_weights
        self.ScalePV()

        # Compute penalty function of current best solution.
        y = self.penalty_function(x)

        # Generate bar chart of pv weights
        self.VisualizeWeights(self._pv_weights_optim, y)

        # Update convergence terminal output and plot.
        self._min_fitness = y
        self._convergence.append(y)
        with open(self._convergence_history_filename, "a+") as fid:
            fid.write(str(y) + "," + ",".join("%+.6e" % alpha for alpha in self._pv_weights_optim) + "\n")
        print(("%i,%+.3e," % (self._current_generation, y)) + ",".join(("%+.4e" % w) for w in self._pv_weights_optim)) 
        self.PlotConvergence()

        # Plot relevant species trends along progress variable.
        #self.PlotPV(x)

        # Save current best progress variable definition.
        np.save(self._output_dir + "/" + self._Config.GetConfigName() + "_PV_Def_optim.npy", self._pv_definition_optim, allow_pickle=True)
        np.save(self._output_dir + "/" + self._Config.GetConfigName() + "_Weights_optim.npy", self._pv_weights_optim, allow_pickle=True)
        self._current_generation += 1

        return
    
    def SetNGenerations(self, n_generations:int):
        """Define the number of generations for which to run the genetic algorithm.

        :param n_generations: number of generations to run for.
        :type n_generations: int
        :raises Exception: if the number of generations is lower than one.
        """
        if n_generations < 1:
            raise Exception("Number of generations should be higher than one.")
        self.__custom_generation_set = True
        self._N_generations = n_generations
        return 
    
    def SetPopulationSize(self, popsize:int):
        """Define the number of individuals per generation.

        :param popsize: number of individuals per generation.
        :type popsize: int
        :raises Exception: if the population size is lower than one.
        """
        if popsize <= 1:
            raise Exception("Population size should be higher than one.")
        self.__custom_population_set = True
        self._population_size = popsize

    def SetCurveStepThreshold(self, val_tol:float=1e-4):
        """Set the threshold value for monotonizing flamelet data.

        :param val_tol: threshold value below which flamelet data is ommitted.
        :type val_tol: float
        :raises Exception: if the specified value is negative.
        """
        if val_tol <= 0:
            raise Exception("Curve step tolerance value should be positive.")
        self.__CurveStepTolerance = val_tol

    def SetSpeciesRangeTolerance(self, val_tol:float=1e-5):
        """Specify the minimum change in species mass fraction for species to be 
        considered in progress variable optimization.

        :param val_tol: species mass fraction range threshold. 
        :type val_tol: float
        :raises Exception: if threshold value is negative.
        """
        if val_tol <= 0:
            raise Exception("Curve step tolerance value should be positive.")
        self.__SpeciesRangeTolerance = val_tol
    
    def ConstraintFunction(self, x):
        pv_w = x / np.linalg.norm(x)
        pv_term = np.dot(self._delta_Y_flamelets, pv_w)
        return min(pv_term)
    
    def RunOptimizer(self):
        """Initiate progress variable optimization process.
        """

        # Generate upper and lower bounds. 
        nVar = len(self._pv_definition_optim)
        lb = -1*np.ones(nVar)
        ub = 1*np.ones(nVar)

        # Implement user-defined bounds if specified.
        for isp, sp in enumerate(self._species_bounds_set):
            try:
                idx = self._pv_definition_optim.index(sp)
                lb[idx] = self._species_custom_lb[isp]
                ub[idx] = self._species_custom_ub[isp]
            except:
                pass
        
        self._Config.gas.set_equivalence_ratio(1.0, self._Config.GetFuelString(), self._Config.GetOxidizerString())
        self._Config.gas.TP = self._Config.GetUnbTempBounds()[0], 101325
        self.__Y_stoch_unb = self._Config.gas.Y
        self._Config.gas.equilibrate("TP")
        self.__Y_stoch_b = self._Config.gas.Y 

        bounds = Bounds(lb=lb, ub=ub)
        # Generate convergence output file.
        self._convergence = []
        self._convergence_history_filename = self._output_dir + "/" + self._Config.GetConfigName() + "_PV_convergence_history.csv"
        with open(self._convergence_history_filename, "w") as fid:
            fid.write("y value," + ",".join("alpha-"+Sp for Sp in self._pv_definition_optim) + "\n")

        # Determine update strategy based on whether parallel processing is used.
        if self._n_workers > 1:
            update_strategy = "deferred"
        else:
            update_strategy = "immediate"

        # Initiate evolutionary algorithm.
        print("Generation,Penalty," + ",".join(s for s in self._pv_definition_optim))
        result = differential_evolution(func = self.penalty_function,\
                                        callback=self._Optimization_Callback,\
                                        maxiter=self._N_generations,\
                                        popsize=self._population_size,\
                                        bounds=bounds,\
                                        workers=self._n_workers,\
                                        updating=update_strategy,\
                                        strategy='best1exp',\
                                        seed=1,\
                                        tol=1e-3)
        
        # Initiate simplex search algorithm.
        result = minimize(self.penalty_function, \
                          result.x, \
                          method='Nelder-Mead', \
                          bounds=bounds,\
                          callback=self._Optimization_Callback)

        # Post-processing for best solution.
        self._Optimization_Callback(result.x)

        # Check if optimized progress variable is monotonic.
        self.CheckMonotonicity()

        print("Progress variable definition upon completion:")
        print("".join("%+.4e %s" % (self._pv_weights_optim[i], self._pv_definition_optim[i]) for i in range(len(self._pv_weights_optim))))
        return
    
    def ScalePV(self):
        """Scale progress variable weights based on stochiometric conditions.
        """

        # Compute reactant progress variable at stochiometry.
        self._Config.gas.set_equivalence_ratio(1.0, self._Config.GetFuelString(),\
                                                    self._Config.GetOxidizerString())
        self._Config.gas.TP=self._Config.GetUnbTempBounds()[0], 101325
        pv_unburnt = 0
        for iPv, pv in enumerate(self._pv_definition_optim):
            pv_unburnt += self._pv_weights_optim[iPv] * self._Config.gas.Y[self._Config.gas.species_index(pv)]

        # Compute product progress variable at stochiometry.
        self._Config.gas.equilibrate('TP')
        pv_burnt = 0
        for iPv, pv in enumerate(self._pv_definition_optim):
            pv_burnt += self._pv_weights_optim[iPv] * self._Config.gas.Y[self._Config.gas.species_index(pv)]

        # Scale progress variable.
        scale = 1 / (pv_burnt - pv_unburnt)
        for i in range(len(self._pv_weights_optim)):
            self._pv_weights_optim[i] *= scale 
        return
    
    def GetMajorProduct(self):
        self._Config.gas.set_equivalence_ratio(1.0, self._Config.GetFuelString(),\
                                                    self._Config.GetOxidizerString())
        self._Config.gas.TP=self._Config.GetUnbTempBounds()[0], 101325
        self._Config.gas.equilibrate('TP')
        sp_to_consider = self._Config.gas.species_names.copy()
        if "N2" in sp_to_consider:
            sp_to_consider.remove("N2")
        Y_to_consider = [self._Config.gas.Y[self._Config.gas.species_index(s)] for s in sp_to_consider]

        i_sp_max = np.argmax(Y_to_consider)
        sp_major = sp_to_consider[i_sp_max]
        return sp_major
    
    def GetOptimizedWeights(self):
        """Get the optimized progress variable mass fraction weights.

        :return: array containing the optimized progress variable weights.
        :rtype: np.ndarray[float]
        """
        return self._pv_weights_optim
    
    def GetOptimizedSpecies(self):
        """Get the optimized progress variable species.

        :return: array containing the optimized progress variable species names.
        :rtype: np.ndarray[str]
        """
        return self._pv_definition_optim
    
        
    def __FilterFlamelets(self):
        flamelet_dir = self._Config.GetOutputDir()

        # Only adiabatic free-flame data and burner-stabilized flame data are considered in progress variable optimization.
        freeflame_subdirs = os.listdir(flamelet_dir + "/freeflame_data/")

        # First step is to size the data array containing the species mass fraction increments throughout each flamelet.
        reaction_zone_length = 0.15
        self.__valid_filepathnames = []
        j_flamelet = 0
        if self._Config.GenerateFreeFlames():
            for phi in freeflame_subdirs:
                val_phi = float(phi.split('_')[1])
                if (val_phi >= self.__phi_min) and (val_phi <= self.__phi_max):
                    flamelets = os.listdir(flamelet_dir + "/freeflame_data/" + phi)
                    for f in flamelets:
                        Tu = float(f.split("_")[2][2:-4])
                        if (Tu >= self.__Tu_min) and (Tu <= self.__Tu_max):
                            self.__valid_filepathnames.append(flamelet_dir + "/freeflame_data/"+ phi + "/" + f)
                            j_flamelet += 1

        if self._Config.GenerateBurnerFlames():
            freeflame_subdirs = os.listdir(flamelet_dir + "/burnerflame_data/")
            for phi in freeflame_subdirs:
                val_phi = float(phi.split('_')[1])
                if (val_phi >= self.__phi_min) and (val_phi <= self.__phi_max):
                    flamelets = os.listdir(flamelet_dir + "/burnerflame_data/" + phi)
                    for f in flamelets:
                        self.__valid_filepathnames.append(flamelet_dir + "/burnerflame_data/"+ phi + "/" + f)
                        j_flamelet += 1
                    
        with open(self.__valid_filepathnames[0],'r') as fid:
            self.__flamelet_variables = fid.readline().strip().split(',')

        NFlamelets = len(self.__valid_filepathnames)
        valid_filepathnames_new = [f for f in self.__valid_filepathnames]
        print("Reading flamelet data...")
        for f in tqdm(self.__valid_filepathnames):
            data_flamelet = np.loadtxt(f,delimiter=',',skiprows=1)
            max_flamewidth = data_flamelet[-1, self.__flamelet_variables.index("Distance")]
            if max_flamewidth > reaction_zone_length:
                valid_filepathnames_new.remove(f)
                NFlamelets -= 1
        self.__valid_filepathnames = [f for f in valid_filepathnames_new]

        return 
    
    def __SelectRelevantSpecies(self):
        """Determine species which contribute most to the progress vector.

        :raises Exception: if any of the user-defined, additional variables are not present in flamelet data.
        """

        # Check if additional variables are present in flamelet data.
        iX_Y = []
        species_relevant = []
        vars_not_in_data = []
        for var in self.__additional_variables:
            if ("Beta" not in var) and (var not in self.__flamelet_variables):
                vars_not_in_data.append(var)
        if any(vars_not_in_data):
            raise Exception("Additional progress variables not in data set: " + ", ".join(var for var in vars_not_in_data))
        
        #self._idx_additional_vars = [self.__flamelet_variables.index(v) for v in self.__additional_variables]

        # Find relevant species, nitrogen is excluded.
        for iVar, var in enumerate(self.__flamelet_variables):
            if var[:2] == "Y-":
                specie = var[2:]
                if (specie != "N2"):
                    iX_Y.append(iVar)
                    species_relevant.append(var)

        minY, maxY = 1000*np.ones(len(iX_Y)), -1000*np.ones(len(iX_Y))

        self._Y_flamelets = [None] * len(self.__valid_filepathnames)
        self.__AddedD_flamelets = [None] * len(self.__valid_filepathnames)

        # Determine range of species mass fraction between flamelet solutions.
        iFlamelet = 0 
        for iFlamelet, flamelet_file in enumerate(self.__valid_filepathnames):
            D = np.loadtxt(flamelet_file,delimiter=',',skiprows=1)

            Y_relevant = D[:, iX_Y]
            self._Y_flamelets[iFlamelet] = Y_relevant 
            
            if any(self.__additional_variables):
                _, beta_h1, beta_h2, beta_z = self._Config.ComputeBetaTerms(self.__flamelet_variables, D)
                addedD_flamelet = np.zeros([np.shape(D)[0], len(self.__additional_variables)])
                for iVar, var in enumerate(self.__additional_variables):
                    if var == FGMVars.Beta_Enth_Thermal.name:
                        addedD_flamelet[:, iVar] = beta_h1
                    elif var == FGMVars.Beta_Enth.name:
                        addedD_flamelet[:, iVar] = beta_h2
                    elif var == FGMVars.Beta_MixFrac.name:
                        addedD_flamelet[:, iVar] = beta_z 
                    else:
                        addedD_flamelet[:, iVar] = D[:, self.__flamelet_variables.index(var)]

            #if any(self._idx_additional_vars):
                self.__AddedD_flamelets[iFlamelet] = addedD_flamelet#D[:, self._idx_additional_vars]

            minY_flamelet, maxY_flamelet = np.min(Y_relevant, 0), np.max(Y_relevant, 0)
            for iSp in range(len(iX_Y)):
                minY[iSp] = min([minY_flamelet[iSp], minY[iSp]])
                maxY[iSp] = max([maxY_flamelet[iSp], maxY[iSp]])
        range_Y = maxY - minY

        # Filter species based on whether the mass fraction range exceeds threshold.
        iY_sig = np.argwhere(range_Y > self.__SpeciesRangeTolerance)[:,0]

        self._idx_relevant_species = iY_sig
        self._pv_definition_optim = [s[2:] for s in [species_relevant[i] for i in iY_sig]]

        if not self.__custom_population_set:
            self._population_size = 20 * len(self._pv_definition_optim)
        
        if not self.__custom_generation_set:
            self._N_generations = 5 * self._population_size
        return 

    def _FilterFlameletData(self):
        """Generate monotonic species mass fraction increment vector and progress vector.
        """

        NFlamelets = len(self._Y_flamelets)
        deltaY_arrays = [None] * NFlamelets
        progress_vector = [None] * NFlamelets

        print("Retrieving progress vector data...")
        for iFlamelet in tqdm(range(NFlamelets)):
            otherdata = None 
            if any(self.__additional_variables):
                otherdata = self.__AddedD_flamelets[iFlamelet]
            idx_mon, pv_famelet, deltaY_flamelet = self.GetFlameletProgressVector(self._Y_flamelets[iFlamelet], otherdata)
            if any(idx_mon):
                progress_vector[iFlamelet] = pv_famelet
                deltaY_arrays[iFlamelet] = deltaY_flamelet

        deltaY_arrays = [x for x in deltaY_arrays if x is not None]
        progress_vector = [x for x in progress_vector if x is not None]
        self._delta_Y_flamelets = np.vstack(tuple((b for b in deltaY_arrays)))
        self._progress_vector = np.vstack(tuple((b for b in progress_vector)))
        self._delta_Y_flamelets_constraints = np.vstack(tuple((deltaY_arrays[b] for b in np.random.choice(NFlamelets, 10))))
        return 
    
    def GetFlameletProgressVector(self, Y_flamelet, otherdata=None):
        """Retrieve the progress vector for a given set of monotonic flamelet data.

        :param Y_flamelet: monotonic flamelet mass fraction data
        :type Y_flamelet: _type_
        :param otherdata: _description_, defaults to None
        :type otherdata: _type_, optional
        :return: _description_
        :rtype: _type_
        """
        Y_filtered = Y_flamelet[:, self._idx_relevant_species]

        # Generate monotonic mass fraction increment vector for the current flamelet.
        idx_mon = self._MakeMonotonic(Y_filtered, otherdata)
        if any(idx_mon):
            Y_mon = Y_filtered[idx_mon,:]
            delta_Y_mon = Y_mon[1:, :] - Y_mon[:-1, :]

            # Normalize mass fraction increment vector to generate progress vector components.
            range_Y = np.max(Y_filtered,axis=0) - np.min(Y_filtered,axis=0) + 1e-32
            delta_Y_norm = delta_Y_mon / range_Y

            # Add additional data as progress vector components if defined.
            if any(self.__additional_variables):
                range_otherdata = np.max(otherdata,axis=0) - np.min(otherdata,axis=0) + 1e-32
                otherdata_mon = otherdata[idx_mon, :]
                delta_otherdata_mon = otherdata_mon[1:, :] - otherdata_mon[:-1,:]
                delta_Y_norm = np.hstack((delta_Y_norm, delta_otherdata_mon / range_otherdata))

            # Compute progress vector as modulus of mass fraction and additional data increments.
            progress_vector_flamelet = np.linalg.norm(delta_Y_norm, axis=1)[:, np.newaxis]
            deltaY_flamelet_mon = delta_Y_mon
            
        return idx_mon, progress_vector_flamelet, deltaY_flamelet_mon

    def _MakeMonotonic(self, Y_flamelet:np.ndarray, AdditionalData:np.ndarray=None):
        """Generate monotonic flamelet solution.

        :param Y_flamelet: flamelet mass fraction data.
        :type Y_flamelet: np.ndarray
        :param AdditionalData: flamelet additional data array, defaults to None
        :type AdditionalData: np.ndarray, optional
        :return: Monotonic flamelet data indices.
        :rtype: np.ndarray[bool]
        """

        # Create single vector of mass fraction and additional data.
        if AdditionalData is not None:
            flamedata = np.hstack((Y_flamelet, AdditionalData))
        else:
            flamedata = Y_flamelet
        
        range_Y = np.max(flamedata,axis=0) - np.min(flamedata,axis=0)
    
        # Compute normalized flamelet data increment vector.
        delta_Y_scaled = (flamedata[1:, :]-flamedata[:-1, :]) / (range_Y[np.newaxis, :]+1e-32)

        # Compute flamelet progress vector and accumulation.
        abs_delta_Y_scaled = np.linalg.norm(delta_Y_scaled, axis=1)
        CurvLength = np.cumsum(abs_delta_Y_scaled, axis=0)

        pv = np.vstack((np.zeros(1), CurvLength[:, np.newaxis]))
        pv_norm = pv
        maxcv= np.max(pv_norm)
        mincv= np.min(pv_norm)

        deltacv = self.__CurveStepTolerance * (maxcv - mincv)

        # Find instances where progress incement exceeds threshold.
        I = np.zeros(np.shape(pv)[0], dtype=bool)
        I[-1] = True 
        for k in range(len(pv)-2, 0, -1):
            if np.all(pv[k, :] < maxcv - deltacv):
                maxcv = pv[k, :]
                I[k] = True 
        I[0] = True 
        
        k = np.argwhere(pv > (pv[0] + deltacv))[0,0]
        I[1:k-1] = False
        return I

class PVOptimizer_Niu(PVOptimizer):
    def __init__(self, Config:Config_FGM):
        super().__init__(Config)
        return 
    
    def _FilterFlameletData(self):
        """Generate monotonic species mass fraction increment vector and progress vector.
        """

        NFlamelets = len(self._Y_flamelets)
        deltaY_arrays = [None] * NFlamelets
        progress_vector = [None] * NFlamelets

        for iFlamelet in tqdm(range(NFlamelets)):
            # Extract species mass fraction and additional data.
            Y_filtered = self._Y_flamelets[iFlamelet][:, self._idx_relevant_species]

            delta_Y = Y_filtered[1:, :] - Y_filtered[:-1, :]
            
            # Compute progress vector as modulus of mass fraction and additional data increments.
            progress_vector[iFlamelet] = np.max(np.abs(delta_Y), axis=1)[:, np.newaxis]
            deltaY_arrays[iFlamelet] = delta_Y 
        deltaY_arrays = [x for x in deltaY_arrays if x is not None]
        progress_vector = [x for x in progress_vector if x is not None]
        self._delta_Y_flamelets = np.vstack(tuple((b for b in deltaY_arrays)))
        self._progress_vector = np.vstack(tuple((b for b in progress_vector)))
        self._delta_Y_flamelets_constraints = np.vstack(tuple((deltaY_arrays[b] for b in np.random.choice(NFlamelets, 10))))
        return
    
    def penalty_function(self, x: np.array):
        x = x.copy()
        val_y = x[0]

        penalty = 1.0 / val_y
        x[1:] /= np.linalg.norm(x[1:])
        A_constr = np.hstack((-self._progress_vector, self._delta_Y_flamelets))
        Ax = np.dot(A_constr, x)
        ix_invalid = Ax < 0.0
        penalty += np.sum(np.abs(Ax[ix_invalid]))**2
        return penalty 
    
    def _Optimization_Callback(self, x:np.array,convergence:bool=False):
        """Callback function during progress variable optimization.

        :param x: progress variable weights.
        :type x: np.array[float]
        :param convergence: whether solution has converged, defaults to False
        :type convergence: bool, optional
        """
        # Normalize and scale pv weights at equivalence ratio of 1.0.
        pv_weights = x[1:] / np.linalg.norm(x[1:])
        self._pv_weights_optim = pv_weights
        self.ScalePV()

        # Compute penalty function of current best solution.
        y = self.penalty_function(x)
        # Generate bar chart of pv weights
        self.VisualizeWeights(self._pv_weights_optim, y)

        # Update convergence terminal output and plot.
        self._min_fitness = y
        self._convergence.append(y)
        with open(self._convergence_history_filename, "a+") as fid:
            fid.write(str(y) + "," + ",".join("%+.6e" % alpha for alpha in self._pv_weights_optim) + "\n")
        print(("%i,%+.3e," % (self._current_generation, y)) + ",".join(("%+.4e" % w) for w in self._pv_weights_optim)) 
        self.PlotConvergence()

        # Plot relevant species trends along progress variable.
        #self.PlotPV(x)

        # Save current best progress variable definition.
        np.save(self._output_dir + "/" + self._Config.GetConfigName() + "_PV_Def_optim.npy", self._pv_definition_optim, allow_pickle=True)
        np.save(self._output_dir + "/" + self._Config.GetConfigName() + "_Weights_optim.npy", self._pv_weights_optim, allow_pickle=True)
        self._current_generation += 1

        return
    
    def VisualizeWeights(self, pv_weights_input:np.array, fitness_val:float):
        """Visualize progress variable weights via bar chart.

        :param pv_weights_input: array containing progress variable weights values.
        :type pv_weights_input: np.array
        :param fitness_val: fitness function value of current progress variable definition.
        :type fitness_val: float
        """
        fig = plt.figure(figsize=[10,10])
        ax = plt.axes()
        ax.clear()
        ax.bar(x=range(len(self._pv_definition_optim)), height=pv_weights_input,zorder=3)
        ax.set_xticks(range(len(self._pv_definition_optim)))
        ax.set_xticklabels(self._pv_definition_optim, rotation='vertical')
        ax.tick_params(which='both', labelsize=18)
        ax.set_ylabel(r"Progress Variable Weight", fontsize=20)
        ax.set_title(r"Fitness value: %+.3e" % fitness_val, fontsize=20)
        ax.grid(zorder=0)
        fig.savefig(self._output_dir + "/Weights_Visualization_"+self._Config.GetConfigName() + "_Niu.pdf", format='pdf', bbox_inches='tight')
        plt.close(fig)
        return
    
    def RunOptimizer(self):
        """Initiate progress variable optimization process.
        """

        # Generate upper and lower bounds. 
        nVar = len(self._pv_definition_optim) + 1
        lb = -1*np.ones(nVar)
        ub = 1*np.ones(nVar)

        # Implement user-defined bounds if specified.
        for isp, sp in enumerate(self._species_bounds_set):
            try:
                idx = self._pv_definition_optim.index(sp)
                lb[idx+1] = self._species_custom_lb[isp]
                ub[idx+1] = self._species_custom_ub[isp]
            except:
                pass
        
        lb[0] = 1e-1
        ub[0] = 1e4

        bounds = Bounds(lb=lb, ub=ub)
        # Generate convergence output file.
        self._convergence = []
        self._convergence_history_filename = self._output_dir + "/" + self._Config.GetConfigName() + "_PV_convergence_history_Niu.csv"
        with open(self._convergence_history_filename, "w") as fid:
            fid.write("y value," + ",".join("alpha-"+Sp for Sp in self._pv_definition_optim) + "\n")

        # Determine update strategy based on whether parallel processing is used.
        if self._n_workers > 1:
            update_strategy = "deferred"
        else:
            update_strategy = "immediate"

        A_constr = np.hstack((-self._progress_vector, self._delta_Y_flamelets))
        #self._monotonicity_full = LinearConstraint(A_constr, lb=-1e-4,keep_feasible=True)
        # Initiate evolutionary algorithm.
        print("Generation,Penalty," + ",".join(s for s in self._pv_definition_optim))
        result = differential_evolution(func = self.penalty_function,\
                                        callback=self._Optimization_Callback,\
                                        maxiter=self._N_generations,\
                                        popsize=self._population_size,\
                                        bounds=bounds,\
                                        workers=self._n_workers,\
                                        updating=update_strategy,\
                                        strategy='best1exp',\
                                        seed=1,\
                                        tol=1e-3)
        
        # Initiate simplex search algorithm.
        result = minimize(self.penalty_function, \
                          result.x, \
                          method='Nelder-Mead', \
                          bounds=bounds,\
                          callback=self._Optimization_Callback)
        
        # Post-processing for best solution.
        self._Optimization_Callback(result.x)

        self.CheckMonotonicity()

        print("Progress variable definition upon completion:")
        print("".join("%+.4e %s" % (self._pv_weights_optim[i], self._pv_definition_optim[i]) for i in range(len(self._pv_weights_optim))))
        return
    
    def OptimizePV(self):
        """Run the progress variable species selection and weights optimization process."""

        # Load data from flamelet manifold for progress variable computation.
        self._CollectFlameletData()
        print("Initiating Progress Variable Optimization")
        print("Relevant species in definition: " + ", ".join(s for s in self._pv_definition_optim))
        

        self._current_generation = 0
        # Initiate progress variable optimization.
        self.RunOptimizer()

        # Scale optimized progress variable weights for normalization sake.
        self.ScalePV()

        # Visualize weights and save convergence trend.
        self.VisualizeWeights(self._pv_weights_optim, self._min_fitness)

        # Display optimized progress variable definition.
        print("Optimized progress variable:")
        print("[" + ",".join("\"" + pv + "\"" for pv in self._pv_definition_optim) + "]")
        print("[" + ",".join("%+.16e" % (pv) for pv in self._pv_weights_optim) + "]")
        np.save(self._output_dir + "/" + self._Config.GetConfigName() + "_PV_Def_optim_Niu.npy", self._pv_definition_optim, allow_pickle=True)
        np.save(self._output_dir + "/" + self._Config.GetConfigName() + "_Weights_optim_Niu.npy", self._pv_weights_optim, allow_pickle=True)
        return
    
class PVOptimizer_Prufert(PVOptimizer):
    def __init__(self, Config:Config_FGM):
        super().__init__(Config)
        return 
    
    def _FilterFlameletData(self):
        """Generate monotonic species mass fraction increment vector and progress vector.
        """

        NFlamelets = len(self._Y_flamelets)
        deltaY_arrays = [None] * NFlamelets
        progress_vector = [None] * NFlamelets

        for iFlamelet in tqdm(range(NFlamelets)):
            # Extract species mass fraction and additional data.
            Y_filtered = self._Y_flamelets[iFlamelet][:, self._idx_relevant_species]
        
            # Generate monotonic mass fraction increment vector for the current flamelet.
            idx_mon = self._MakeMonotonic(Y_filtered, None)
            if any(idx_mon):
                Y_mon = Y_filtered[idx_mon,:]
                delta_Y_mon = Y_mon[1:, :] - Y_mon[:-1, :]

                # Normalize mass fraction increment vector to generate progress vector components.
                # range_Y = np.max(Y_filtered,axis=0) - np.min(Y_filtered,axis=0) + 1e-32
                # delta_Y_norm = delta_Y_mon / range_Y

                # Compute progress vector as modulus of mass fraction and additional data increments.
                progress_vector[iFlamelet] = np.max(np.abs(delta_Y_mon), axis=1)[:, np.newaxis]
                deltaY_arrays[iFlamelet] = delta_Y_mon 
        deltaY_arrays = [x for x in deltaY_arrays if x is not None]
        progress_vector = [x for x in progress_vector if x is not None]
        self._delta_Y_flamelets = np.vstack(tuple((b for b in deltaY_arrays)))
        self._progress_vector = np.vstack(tuple((b for b in progress_vector)))
        self._delta_Y_flamelets_constraints = np.vstack(tuple((deltaY_arrays[b] for b in np.random.choice(NFlamelets, 10))))
        return

class PVOptimizer_PCA(PVOptimizer):
    def __init__(self, Config:Config_FGM):
        super().__init__(Config)
        return 
    def RunOptimizer(self):
        """Initiate progress variable optimization process.
        """

        
        pca_transformer = PCA(n_components=1)
        pca_transformer.fit_transform(self._delta_Y_flamelets)
        self._pv_weights_optim = pca_transformer.components_[0]
        self.ScalePV()

        self.CheckMonotonicity()
        # Post-processing for best solution.
        #self._Optimization_Callback(result.x)
        print("Progress variable definition upon completion:")
        print("".join("%+.4e %s" % (self._pv_weights_optim[i], self._pv_definition_optim[i]) for i in range(len(self._pv_weights_optim))))
        return
    
if __name__ == "__main__":
    config_input_file = sys.argv[-1]
    Config = Config_FGM(config_input_file)
    PVO = PVOptimizer_Niu(Config)
    PVO.OptimizePV()