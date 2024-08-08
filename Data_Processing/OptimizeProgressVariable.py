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
# Version: 1.0.0                                                                              |
#                                                                                             |
#=============================================================================================#

import numpy as np 
import sys
import os
from tqdm import tqdm
import matplotlib.pyplot as plt 
from scipy.optimize import differential_evolution, Bounds, LinearConstraint

from Common.DataDrivenConfig import FlameletAIConfig

class PVOptimizer:
    """Optimize the progress variable weights for all flamelets in the manifold given relevant species in the mixture.
    """
    # Class for optimizing the progress variable weights for monotonicity

    __Config:FlameletAIConfig = None  # FlameletAIConfig class.

    __pv_definition_optim = None # Optimized progress variable definition.
    __pv_weights_optim = None    # Optimized progress variable weights.

    __valid_filepathnames:list[str] = []
    __flamelet_variables:list[str] = None  # Variable names in flamelet data files.
    __Y_flamelets = None 
    __idx_relevant_species:np.ndarray[float] = None 
    __additional_variables:list[str] = []
    __idx_additional_vars = [] 

    __delta_Y_flamelets:np.ndarray[float] = None
    __progress_vector:np.ndarray[float] = None


    __custom_generation_set:bool = False
    __N_generations:int    # Number of generations during the genetic algorithm optimization.
    __custom_population_set:bool = False
    __population_size:int     # Genetic algorithm population size.

    __consider_freeflames = None 
    __consider_burnerflames = None 
    __convergence = None # Convergence of optimization process.
    __min_fitness = 1.0  # Current minimum fitness value.

    __n_workers:int=1   # Number of CPUs used to compute the population merit in parallel.

    __CurveStepTolerance:float = 1e-4 
    __SpeciesRangeTolerance:float = 1e-5 
    __current_generation:int = 0

    __Tu_min:float = 300 
    __Tu_max:float = 500 
    __phi_min:float = 0.0
    __phi_max:float = 600 

    __species_bounds_set:list[str] = []
    __species_custom_lb:list[float] = []
    __species_custom_ub:list[float] = []

    __output_dir:str = None 

    def __init__(self, Config:FlameletAIConfig):
        """Progress variable optimization class constructor
        :param Config: FlameletAIConfig class containing manifold info.
        :type Config: FlameletAIConfig
        """

        self.__Config = Config 
        self.__consider_burnerflames = self.__Config.GenerateBurnerFlames()
        self.__consider_freeflames = self.__Config.GenerateFreeFlames()
        self.__output_dir = self.__Config.GetOutputDir()
        
        for sp in self.__Config.GetFuelDefinition():
            self.SetSpeciesBounds(sp, ub=0.0)
            
        print("Loading flameletAI configuration with name " + self.__Config.GetConfigName())
        return 
    
    def SetOutputDir(self, output_dir:str):
        self.__output_dir = output_dir 

    def SetNWorkers(self, n_workers:int):
        """Define the number of parallel workers to be used during the population merit and constraint computation.

        :param n_workers: number of CPU cores to be used in parallel.
        :type n_workers: int
        :raises Exception: if the given number of workers is lower than one.
        """
        if n_workers < 1:
            raise Exception("The number of workers used during the optimization process should be at least one.")
        self.__n_workers = n_workers

    def SetAdditionalProgressVariables(self, additional_vars:list[str]):
        """Add additional variables to the progress vector.

        :param additional_vars: list of variable names to include in progress vector.
        :type additional_vars: list[str]
        """
        self.__additional_variables = []
        for var in additional_vars:
            self.__additional_variables.append(var)

    def SetSpeciesBounds(self, sp_name:str, lb:float=-1,ub:float=1):
        if ub <= lb:
            raise Exception("Lower bound should be higher than upper bound.")
        if sp_name in self.__species_bounds_set:
            idx_sp = self.__species_bounds_set.index(sp_name)
            self.__species_custom_lb[idx_sp] = lb 
            self.__species_custom_ub[idx_sp] = ub 
        else:
            self.__species_bounds_set.append(sp_name)
            self.__species_custom_lb.append(lb)
            self.__species_custom_ub.append(ub)

    def OptimizePV(self):
        """Run the progress variable species selection and weights optimization process."""

        # Load data from flamelet manifold for progress variable computation.
        self.__CollectFlameletData()
        print("Initiating Progress Variable Optimization")
        print("Relevant species in definition: " + ", ".join(s for s in self.__pv_definition_optim))
        if any(self.__idx_additional_vars):
            print("Additional variables in progress vector: " + ", ".join(v for v in self.__additional_variables))

        self.__current_generation = 0
        # Initiate progress variable optimization.
        self.RunOptimizer()

        # Scale optimized progress variable weights for normalization sake.
        self.ScalePV()

        # Visualize weights and save convergence trend.
        self.VisualizeWeights(self.__pv_weights_optim, self.__min_fitness)

        # Display optimized progress variable definition.
        print("Optimized progress variable:")
        print("[" + ",".join("\"" + pv + "\"" for pv in self.__pv_definition_optim) + "]")
        print("[" + ",".join("%+.16e" % (pv) for pv in self.__pv_weights_optim) + "]")
        np.save(self.__output_dir + "/" + self.__Config.GetConfigName() + "_PV_Def_optim.npy", self.__pv_definition_optim, allow_pickle=True)
        np.save(self.__output_dir + "/" + self.__Config.GetConfigName() + "_Weights_optim.npy", self.__pv_weights_optim, allow_pickle=True)
    
    def ConsiderFreeFlames(self, input:bool):
        """Use adiabatic free-flamelet data set during progress variable optimization.

        :param input: include adiabatic flamelets during pv optimization (True) or not (False)
        :type input: bool
        """
        self.__consider_freeflames=input

    def ConsiderBurnerFlames(self, input:bool):
        """Use burner-stabilized data set during progress variable optimization.

        :param input: include burner-stabilized during pv optimization (True) or not (False)
        :type input: bool
        """
        self.__consider_burnerflames=input

    def __CollectFlameletData(self):
        """Load flamelet manifold data from storage.
        """

        # Collect flamelet data between specified Tu and phi bounds.
        self.__FilterFlamelets()

        # Determine which species are most relevant for PV definition.
        self.__SelectRelevantSpecies()

        # Generate monotonic progress and mass fraction incement vectors.
        self.__FilterFlameletData()
    
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
        ax.bar(x=range(len(self.__pv_definition_optim)), height=pv_weights_input,zorder=3)
        ax.set_xticks(range(len(self.__pv_definition_optim)))
        ax.set_xticklabels(self.__pv_definition_optim, rotation='vertical')
        ax.tick_params(which='both', labelsize=18)
        ax.set_ylabel(r"Progress Variable Weight", fontsize=20)
        ax.set_title(r"Fitness value: %+.3e" % fitness_val, fontsize=20)
        ax.grid(zorder=0)
        fig.savefig(self.__output_dir + "/Weights_Visualization_"+self.__Config.GetConfigName() + ".pdf", format='pdf', bbox_inches='tight')
        plt.close(fig)

    def PlotConvergence(self):
        """Plot the convergence trend of the progress variable optimization process.
        """
        fig = plt.figure(figsize=[10, 10])
        ax = plt.axes()
        ax.plot(self.__convergence, 'k',linewidth=2)
        ax.set_xlabel(r"Generation", fontsize=20)
        ax.set_ylabel(r"Penalty function value", fontsize=20)
        ax.set_title(r"Progress Variable Optimization Convergence", fontsize=20)
        ax.grid()
        ax.tick_params(which='both', labelsize=18)
        ax.set_yscale('log')
        fig.savefig(self.__output_dir+"/PV_Optimization_History_"+self.__Config.GetConfigName()+".pdf", format='pdf', bbox_inches='tight')
        plt.close(fig)
    
    def PlotPV(self,x):
        """Plot the normalized progress variable alongside the normalized species mass fractions for a sample flamelet.

        :param x: progress variable weight vector.
        :type x: np.array
        """

        # fig = plt.figure(figsize=[10,10])
        # ax = plt.axes()
        # self.__Config.SetProgressVariableDefinition(self.__pv_definition_optim, x)

        # # Find flamelet file according to specified equivalence ratio and unburnt temperature.
        # idx_to_plot = self.__valid_filepathnames.index(self.__Config.GetOutputDir() + "/freeflame_data/phi_"+str(round(self.__phi_to_plot_for,6)) + "/freeflamelet_phi"+str(round(self.__phi_to_plot_for,6)) + "_Tu"+str(round(self.__Tu_to_plot_for,4)) + ".csv")
        # filename = self.__valid_filepathnames[idx_to_plot].split("/")[-1]

        # val_phi = float(filename.split('_')[1][3:])
        # Tu = float(filename.split("_")[2][2:-4])

        # # Extract flamelet data
        # D_flamelet = np.loadtxt(self.__valid_filepathnames[idx_to_plot], delimiter=',',skiprows=1)

        # # Compute and normalize progress variable.
        # pv_flamelet = self.__Config.ComputeProgressVariable(self.__flamelet_variables, D_flamelet)
        # pv_flamelet_norm = (pv_flamelet - min(pv_flamelet))/(max(pv_flamelet) - min(pv_flamelet))
        # if self.__initial:
        #     self.initial_pv_norm = pv_flamelet_norm
        #     self.initial_sp_norm = []
        # for isp, sp in enumerate(self.__species_plot):
        #     sp_data = D_flamelet[:, self.__flamelet_variables.index("Y-"+sp)]
        #     sp_data_norm = (sp_data - min(sp_data))/(max(sp_data) - min(sp_data))
        #     if self.__initial:
        #         self.initial_sp_norm.append(sp_data_norm)
        #     ax.plot(pv_flamelet_norm, sp_data_norm,label=sp,color=colors[isp])
        #     if not self.__initial:
        #         ax.plot(self.initial_pv_norm, self.initial_sp_norm[isp],"--",color=colors[isp])
        
        # ax.grid()
        # ax.set_xlabel("Normalized progress variable[-]",fontsize=20)
        # ax.set_ylabel("Normalized mass fraction[-]",fontsize=20)
        # ax.tick_params(which='both',labelsize=18)
        # ax.legend(fontsize=20)
        # ax.set_title(("Generation %i: Species trends at Tu %.3e K, eq ratio: %.3e" % (self.__current_generation, Tu, val_phi)), fontsize=20)
        # fig.savefig(self.__output_dir+"/species_of_interest.png",format='png',bbox_inches='tight')
        # plt.close(fig)
        # self.__initial = False

    def penalty_function(self, x:np.array):
        """Optimization penalty function computing the derivative of the 
        progress vector w.r.t. that of the progress variable increment.

        :param x: progress variable weights vector.
        :type x: np.array[float]
        :return: maximum, absolute derivative (finite-differences) of the progress vector w.r.t. the progress variable increment.
        :rtype: float
        """
        pv_w = x / np.linalg.norm(x)
        pv_term = np.dot(self.__delta_Y_flamelets, pv_w)
        y_term = self.__progress_vector[:,0]
        fac = y_term / pv_term
        result = np.max(np.abs(fac))# + np.sum(np.power(fac[pv_term < 0.0],2))
        return result
        
    def __Optimization_Callback(self, x:np.array,convergence:bool=False):
        """Callback function during progress variable optimization.

        :param x: progress variable weights.
        :type x: np.array[float]
        :param convergence: whether solution has converged, defaults to False
        :type convergence: bool, optional
        """
        # Normalize and scale pv weights at equivalence ratio of 1.0.
        pv_weights = x / np.linalg.norm(x)
        self.__pv_weights_optim = pv_weights
        self.ScalePV()

        # Compute penalty function of current best solution.
        y = self.penalty_function(x)

        # Generate bar chart of pv weights
        self.VisualizeWeights(self.__pv_weights_optim, y)

        # Update convergence terminal output and plot.
        self.__min_fitness = y
        self.__convergence.append(y)
        with open(self.__convergence_history_filename, "a+") as fid:
            fid.write(str(y) + "," + ",".join("%+.6e" % alpha for alpha in self.__pv_weights_optim) + "\n")
        print(("%i,%+.3e," % (self.__current_generation, y)) + ",".join(("%+.4e" % w) for w in self.__pv_weights_optim)) 
        self.PlotConvergence()

        # Plot relevant species trends along progress variable.
        self.PlotPV(x)

        # Save current best progress variable definition.
        np.save(self.__output_dir + "/" + self.__Config.GetConfigName() + "_PV_Def_optim.npy", self.__pv_definition_optim, allow_pickle=True)
        np.save(self.__output_dir + "/" + self.__Config.GetConfigName() + "_Weights_optim.npy", self.__pv_weights_optim, allow_pickle=True)
        self.__current_generation += 1

    def SetNGenerations(self, n_generations:int):
        """Define the number of generations for which to run the genetic algorithm.

        :param n_generations: number of generations to run for.
        :type n_generations: int
        :raises Exception: if the number of generations is lower than one.
        """
        if n_generations < 1:
            raise Exception("Number of generations should be higher than one.")
        self.__custom_generation_set = True
        self.__N_generations = n_generations

    def SetPopulationSize(self, popsize:int):
        """Define the number of individuals per generation.

        :param popsize: number of individuals per generation.
        :type popsize: int
        :raises Exception: if the population size is lower than one.
        """
        if popsize <= 1:
            raise Exception("Population size should be higher than one.")
        self.__custom_population_set = True
        self.__population_size = popsize

    def SetCurveStepThreshold(self, val_tol:float):
        """Set the threshold value for monotonizing flamelet data.

        :param val_tol: threshold value below which flamelet data is ommitted.
        :type val_tol: float
        :raises Exception: if the specified value is negative.
        """
        if val_tol <= 0:
            raise Exception("Curve step tolerance value should be positive.")
        self.__CurveStepTolerance = val_tol

    def SetSpeciesRangeTolerance(self, val_tol:float):
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
        pv_term = np.dot(self.__delta_Y_flamelets, pv_w)
        return min(pv_term)
    
    def RunOptimizer(self):
        """Initiate progress variable optimization process.
        """

        # Generate upper and lower bounds. 
        nVar = len(self.__pv_definition_optim)
        lb = -1*np.ones(nVar)
        ub = 1*np.ones(nVar)

        # Implement user-defined bounds if specified.
        for isp, sp in enumerate(self.__species_bounds_set):
            if sp not in self.__pv_definition_optim:
                raise Warning("Specie " + sp + " not in relevant species, custom bounds are ignored.")
            else:
                idx = self.__pv_definition_optim.index(sp)
                lb[idx] = self.__species_custom_lb[isp]
                ub[idx] = self.__species_custom_ub[isp]

        bounds = Bounds(lb=lb, ub=ub)

        print(bounds)
        # Generate convergence output file.
        self.__convergence = []
        self.__convergence_history_filename = self.__output_dir + "/" + self.__Config.GetConfigName() + "_PV_convergence_history.csv"
        with open(self.__convergence_history_filename, "w") as fid:
            fid.write("y value," + ",".join("alpha-"+Sp for Sp in self.__pv_definition_optim) + "\n")

        # Determine update strategy based on whether parallel processing is used.
        if self.__n_workers > 1:
            update_strategy = "deferred"
        else:
            update_strategy = "immediate"

        monotonicity_full = LinearConstraint(self.__delta_Y_flamelets, lb=0.0,keep_feasible=True)
        # Initiate evolutionary algorithm.
        print("Generation,Penalty," + ",".join(s for s in self.__pv_definition_optim))
        result = differential_evolution(func = self.penalty_function,\
                                        callback=self.__Optimization_Callback,\
                                        maxiter=self.__N_generations,\
                                        popsize=self.__population_size,\
                                        bounds=bounds,\
                                        workers=self.__n_workers,\
                                        updating=update_strategy,\
                                        strategy='best1exp',\
                                        seed=1,\
                                        tol=1e-6)
        
        res_full = min(monotonicity_full.residual(result.x)[0])
        if res_full > 0:
            print("PV definition is monotonic")
        else:
            print("PV definition is not monotonic!")
        # Post-processing for best solution.
        self.__Optimization_Callback(result.x)
        print("Progress variable definition upon completion:")
        print("".join("%+.4e %s" % (self.__pv_weights_optim[i], self.__pv_definition_optim[i]) for i in range(len(self.__pv_weights_optim))))

    def ScalePV(self):
        """Scale progress variable weights based on stochiometric conditions.
        """

        # Compute reactant progress variable at stochiometry.
        self.__Config.gas.set_equivalence_ratio(1.0, "H2:1","O2:1,N2:3.76")
        self.__Config.gas.TP=300, 101325
        pv_unburnt = 0
        for iPv, pv in enumerate(self.__pv_definition_optim):
            pv_unburnt += self.__pv_weights_optim[iPv] * self.__Config.gas.Y[self.__Config.gas.species_index(pv)]

        # Compute product progress variable at stochiometry.
        self.__Config.gas.equilibrate('TP')
        pv_burnt = 0
        for iPv, pv in enumerate(self.__pv_definition_optim):
            pv_burnt += self.__pv_weights_optim[iPv] * self.__Config.gas.Y[self.__Config.gas.species_index(pv)]

        # Scale progress variable.
        scale = 1 / (pv_burnt - pv_unburnt)
        for i in range(len(self.__pv_weights_optim)):
            self.__pv_weights_optim[i] *= scale 

    def GetOptimizedWeights(self):
        """Get the optimized progress variable mass fraction weights.

        :return: array containing the optimized progress variable weights.
        :rtype: np.ndarray[float]
        """
        return self.__pv_weights_optim
    
    def GetOptimizedSpecies(self):
        """Get the optimized progress variable species.

        :return: array containing the optimized progress variable species names.
        :rtype: np.ndarray[str]
        """
        return self.__pv_definition_optim
    
        
    def __FilterFlamelets(self):
        flamelet_dir = self.__Config.GetOutputDir()

        # Only adiabatic free-flame data and burner-stabilized flame data are considered in progress variable optimization.
        if self.__consider_freeflames:
            freeflame_subdirs = os.listdir(flamelet_dir + "/freeflame_data/")
        if self.__consider_burnerflames:
            burnerflame_subdirs = os.listdir(flamelet_dir + "/burnerflame_data/")   

        # First step is to size the data array containing the species mass fraction increments throughout each flamelet.
        # Only 10% of the data is considered in order to reduce computational cost.
        reaction_zone_length = 0.15
        self.__valid_filepathnames = []
        j_flamelet = 0
        if self.__consider_freeflames:
            for phi in freeflame_subdirs:
                val_phi = float(phi.split('_')[1])
                if (val_phi >= self.__phi_min) and (val_phi <= self.__phi_max):
                    flamelets = os.listdir(flamelet_dir + "/freeflame_data/" + phi)
                    for f in flamelets:
                        Tu = float(f.split("_")[2][2:-4])
                        if (Tu >= self.__Tu_min) and (Tu <= self.__Tu_max):
                            self.__valid_filepathnames.append(flamelet_dir + "/freeflame_data/"+ phi + "/" + f)
                            j_flamelet += 1
        if self.__consider_burnerflames:
            for phi in burnerflame_subdirs:
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
            if var not in self.__flamelet_variables:
                vars_not_in_data.append(var)
        if any(vars_not_in_data):
            raise Exception("Additional progress variables not in data set: " + ", ".join(var for var in vars_not_in_data))
        
        self.__idx_additional_vars = [self.__flamelet_variables.index(v) for v in self.__additional_variables]

        # Find relevant species, nitrogen is excluded.
        for iVar, var in enumerate(self.__flamelet_variables):
            if var[:2] == "Y-":
                specie = var[2:]
                if (specie != "N2"):
                    iX_Y.append(iVar)
                    species_relevant.append(var)

        minY, maxY = 1000*np.ones(len(iX_Y)), -1000*np.ones(len(iX_Y))

        self.__Y_flamelets = [None] * len(self.__valid_filepathnames)
        self.__AddedD_flamelets = [None] * len(self.__valid_filepathnames)

        # Determine range of species mass fraction between flamelet solutions.
        iFlamelet = 0 
        for iFlamelet, flamelet_file in enumerate(self.__valid_filepathnames):
            D = np.loadtxt(flamelet_file,delimiter=',',skiprows=1)

            Y_relevant = D[:, iX_Y]
            self.__Y_flamelets[iFlamelet] = Y_relevant 
            if any(self.__idx_additional_vars):
                self.__AddedD_flamelets[iFlamelet] = D[:, self.__idx_additional_vars]

            minY_flamelet, maxY_flamelet = np.min(Y_relevant, 0), np.max(Y_relevant, 0)
            for iSp in range(len(iX_Y)):
                minY[iSp] = min([minY_flamelet[iSp], minY[iSp]])
                maxY[iSp] = max([maxY_flamelet[iSp], maxY[iSp]])
        range_Y = maxY - minY

        # Filter species based on whether the mass fraction range exceeds threshold.
        iY_sig = np.argwhere(range_Y > self.__SpeciesRangeTolerance)[:,0]

        self.__idx_relevant_species = iY_sig
        self.__pv_definition_optim = [s[2:] for s in [species_relevant[i] for i in iY_sig]]

        if not self.__custom_population_set:
            self.__population_size = 20 * len(self.__pv_definition_optim)
        
        if not self.__custom_generation_set:
            self.__N_generations = 10 * self.__population_size
        return 

    def __FilterFlameletData(self):
        """Generate monotonic species mass fraction increment vector and progress vector.
        """

        NFlamelets = len(self.__Y_flamelets)
        deltaY_arrays = [None] * NFlamelets
        progress_vector = [None] * NFlamelets

        for iFlamelet in tqdm(range(NFlamelets)):
            # Extract species mass fraction and additional data.
            Y_filtered = self.__Y_flamelets[iFlamelet][:, self.__idx_relevant_species]
            otherdata = None 
            if any(self.__idx_additional_vars):
                otherdata = self.__AddedD_flamelets[iFlamelet]

            # Generate monotonic mass fraction increment vector for the current flamelet.
            idx_mon = self.__MakeMonotonic(Y_filtered, otherdata)
            if any(idx_mon):
                Y_mon = Y_filtered[idx_mon,:]
                delta_Y_mon = Y_mon[1:, :] - Y_mon[:-1, :]

                # Normalize mass fraction increment vector to generate progress vector components.
                range_Y = np.max(Y_filtered,axis=0) - np.min(Y_filtered,axis=0)
                delta_Y_norm = delta_Y_mon / range_Y

                # Add additional data as progress vector components if defined.
                if any(self.__idx_additional_vars):
                    range_otherdata = np.max(otherdata,axis=0) - np.min(otherdata,axis=0)
                    otherdata_mon = otherdata[idx_mon, :]
                    delta_otherdata_mon = otherdata_mon[1:, :] - otherdata_mon[:-1,:]
                    delta_Y_norm = np.hstack((delta_Y_norm, delta_otherdata_mon / range_otherdata))

                # Compute progress vector as modulus of mass fraction and additional data increments.
                progress_vector[iFlamelet] = np.linalg.norm(delta_Y_norm, axis=1)[:, np.newaxis]
                deltaY_arrays[iFlamelet] = delta_Y_mon 
        deltaY_arrays = [x for x in deltaY_arrays if x is not None]
        progress_vector = [x for x in progress_vector if x is not None]
        self.__delta_Y_flamelets = np.vstack(tuple((b for b in deltaY_arrays)))
        self.__progress_vector = np.vstack(tuple((b for b in progress_vector)))
        self.__delta_Y_flamelets_constraints = np.vstack(tuple((deltaY_arrays[b] for b in np.random.choice(NFlamelets, 10))))

    def __MakeMonotonic(self, Y_flamelet:np.ndarray, AdditionalData:np.ndarray=None):
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
        delta_Y_scaled = (flamedata[1:, :]-flamedata[:-1, :]) / range_Y[np.newaxis, :]

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

if __name__ == "__main__":
    config_input_file = sys.argv[-1]
    Config = FlameletAIConfig(config_input_file)
    PVO = PVOptimizer(Config)
    PVO.ConsiderFreeFlames(True)
    PVO.ConsiderBurnerFlames(False)
    PVO.OptimizePV()