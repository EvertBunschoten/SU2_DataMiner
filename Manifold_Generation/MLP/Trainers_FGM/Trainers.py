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
import matplotlib.pyplot as plt 

# Import 
from Common.DataDrivenConfig import FlameletAIConfig
from Manifold_Generation.MLP.Trainer_Base import TensorFlowFit,PhysicsInformedTrainer,EvaluateArchitecture
from CommonMethods import GetReferenceData

class Train_Flamelet_Direct(TensorFlowFit):
    __Config:FlameletAIConfig
    _train_name:str 

    def __init__(self, Config_in:FlameletAIConfig, group_idx:int=0):
        TensorFlowFit.__init__(self)
        # Set train variables based on what kind of network is trained
        self.__Config = Config_in
        self._filedata_train = self.__Config.GetOutputDir()+"/"+self.__Config.GetConcatenationFileHeader()
        self._controlling_vars = ["ProgressVariable","EnthalpyTot","MixtureFraction"]
        self._train_vars = self.__Config.GetMLPOutputGroup(group_idx)
        self._train_name = "Group"+str(group_idx+1)

        return
    
    def CustomCallback(self):
        """Plot MLP prediction alongside flamelet reference data.
        :file_name_header: file name header for each figure.
        :N_plot: number of equivalence ratio's to plot for in each figure.
        """

        N_plot = 3

        flamelet_dir = self.__Config.GetOutputDir()
        include_freeflames = self.__Config.GenerateFreeFlames()
        include_burnerflames = self.__Config.GenerateBurnerFlames()
        include_eq = self.__Config.GenerateEquilibrium()

        freeflame_phis = os.listdir(flamelet_dir + "/freeflame_data/")
        idx_phi_plot = np.random.randint(0, len(freeflame_phis), N_plot)
        
        freeflamelet_input_files = []
        for phi in idx_phi_plot:
            freeflame_files = os.listdir(flamelet_dir + "/freeflame_data/"+freeflame_phis[phi])
            freeflamelet_input_files.append(flamelet_dir + "/freeflame_data/"+freeflame_phis[phi]+ "/"+freeflame_files[np.random.randint(0, len(freeflame_files))])

        # Prepare a figure window for each output variable.
        figs = []
        axs = []
        for _ in self._train_vars:
            fig, ax = plt.subplots(1,1,figsize=[10,10])
            figs.append(fig)
            axs.append(ax)

        plot_label_ref = "Flamelet data"
        plot_label_MLP = "MLP prediction"

        # Plot flamelet data in respective figure.
        for flamelet_input_file in freeflamelet_input_files:
            with open(flamelet_input_file, "r") as fid:
                line = fid.readline()
                variables_flamelet = line.strip().split(',')
            flameletData = np.loadtxt(flamelet_input_file, delimiter=',',skiprows=1)

            # Collect flamelet controlling variables.
            CV_flamelet = np.zeros([len(flameletData), len(self._controlling_vars)])
            for iCv, Cv in enumerate(self._controlling_vars):
                if Cv == 'ProgressVariable':
                    CV_flamelet[:, iCv] = self.__Config.ComputeProgressVariable(variables_flamelet, flameletData)
                else:
                    CV_flamelet[:, iCv] = flameletData[:, variables_flamelet.index(Cv)]
            
            CV_flamelet_norm = (CV_flamelet - self._X_min)/(self._X_max - self._X_min)
            
            ref_data_flamelet = np.zeros([len(flameletData), len(self._train_vars)])

            # Collect prediction variables from flamelet data. 
            for iVar, Var in enumerate(self._train_vars):
                if "Beta_" in Var:
                    beta_pv, beta_enth_thermal, beta_enth, beta_mixfrac = self.__Config.ComputeBetaTerms(variables_flamelet, flameletData)
                if Var == "Beta_ProgVar":
                    ref_data_flamelet[:, iVar] = beta_pv 
                elif Var == "Beta_Enth_Thermal":
                    ref_data_flamelet[:, iVar] = beta_enth_thermal
                elif Var == "Beta_Enth":
                    ref_data_flamelet[:, iVar] = beta_enth 
                elif Var == "Beta_MixFrac":
                    ref_data_flamelet[:, iVar] = beta_mixfrac 
                elif Var == "ProdRateTot_PV":
                    ref_data_flamelet[:, iVar] = self.__Config.ComputeProgressVariable_Source(variables_flamelet, flameletData)
                elif Var == "DiffusionCoefficient":
                    k = flameletData[:, variables_flamelet.index("Conductivity")]
                    cp = flameletData[:, variables_flamelet.index("Cp")]
                    rho = flameletData[:, variables_flamelet.index("Density")]
                    ref_data_flamelet[:, iVar] = k/(cp*rho)
                elif "NOx" in Var:
                    len_nox = len("NOx")

                    for NOsp in ["NO", "NO2", "N2O"]:
                        ref_data_flamelet[:, iVar]+= flameletData[:, variables_flamelet.index(Var[:-len_nox]+NOsp)]
                else:
                    ref_data_flamelet[:, iVar] = flameletData[:, variables_flamelet.index(Var)]

            # Compute MLP prediction of flamelet data.
            pred_data_norm = self.EvaluateMLP(CV_flamelet_norm)
            pred_data = (self._Y_max - self._Y_min) * pred_data_norm + self._Y_min
            # Plot flamelet data in corresponding figure window.
            for iVar, Var in enumerate(self._train_vars):
                axs[iVar].plot(CV_flamelet[:, 0], ref_data_flamelet[:, iVar], 'bs-', linewidth=2, markevery=10, markerfacecolor='none', markersize=12, label=plot_label_ref)
                axs[iVar].plot(CV_flamelet[:, 0], pred_data[:, iVar], 'ro--', linewidth=1, markevery=10, markersize=10, label=plot_label_MLP)
            plot_label_MLP = ""
            plot_label_ref = ""
        for iVar, Var in enumerate(self._train_vars):
            axs[iVar].set_xlabel(r"Progress Variable $(\mathcal{Y})[-]$", fontsize=20)
            axs[iVar].set_ylabel(r"" + Var, fontsize=20)
            axs[iVar].tick_params(which='both', labelsize=18)
            axs[iVar].legend(fontsize=20)
            axs[iVar].grid()
            figs[iVar].savefig(self._save_dir + "/Model_"+str(self._model_index) + "/flameletdata_"+self._train_name+"_" + Var + "."+self._figformat, format=self._figformat, bbox_inches='tight')
            plt.close(figs[iVar])

        return super().CustomCallback()
   
        
class Train_Source_PINN(PhysicsInformedTrainer):
    __Config:FlameletAIConfig
    __boundary_data_file:str 
    __X_unb_train:np.ndarray
    __X_b_train:np.ndarray
    __Y_unb_train:np.ndarray 
    __Y_b_train:np.ndarray 
    __pv_unb_constraint_factor:float = None 
    __pv_b_constraint_factor_lean:float = None 
    __pv_b_constraint_factor_rich:float = None 

    def __init__(self, Config_in:FlameletAIConfig, group_idx:int=0):
        PhysicsInformedTrainer.__init__(self)
        self.__Config = Config_in 
        self._controlling_vars = ["ProgressVariable","EnthalpyTot","MixtureFraction"]
        self._train_vars = self.__Config.GetMLPOutputGroup(group_idx)

        return 
    
    def SetBoundaryDataFile(self, boundary_data_file:str):
        self.__boundary_data_file = boundary_data_file 
        return 
    
    def GetBoundaryData(self):
        X_boundary, Y_boundary = GetReferenceData(self.__boundary_data_file, x_vars=self._controlling_vars, train_variables=self._train_vars)
        mixfrac_boundary = X_boundary[:, self._controlling_vars.index("MixtureFraction")]
        pv_boundary = X_boundary[:, self._controlling_vars.index("ProgressVariable")]
        h_boundary = X_boundary[:, self._controlling_vars.index("EnthalpyTot")]
        is_unb = np.ones(np.shape(X_boundary)[0],dtype=bool)

        fuel_definition = self.__Config.GetFuelDefinition()
        oxidizer_definition = self.__Config.GetOxidizerDefinition()
        fuel_weights = self.__Config.GetFuelWeights()
        oxidizer_weights = self.__Config.GetOxidizerWeights()
        fuel_string = ",".join((sp+":"+str(w) for sp, w in zip(fuel_definition, fuel_weights)))
        oxidizer_string = ",".join((sp+":"+str(w) for sp, w in zip(oxidizer_definition, oxidizer_weights)))
        
        for iz, z in enumerate(mixfrac_boundary):
            self.__Config.gas.set_mixture_fraction(z, fuel_string, oxidizer_string)
            self.__Config.gas.TP=300,101325
            Y = self.__Config.gas.Y 
            pv_unb = self.__Config.ComputeProgressVariable(variables=None, flamelet_data=None, Y_flamelet=Y[:,np.newaxis])
            if pv_boundary[iz] > pv_unb+1e-3:
                is_unb[iz] = False 
        
        X_boundary_norm = (X_boundary - self._X_min)/(self._X_max-self._X_min)
        Y_boundary_norm = (Y_boundary - self._Y_min)/(self._Y_max-self._Y_min)
        X_unb_n, Y_unb_n = X_boundary_norm[is_unb, :], Y_boundary_norm[is_unb, :]
        X_b_n, Y_b_n = X_boundary_norm[np.invert(is_unb), :], Y_boundary_norm[np.invert(is_unb), :]

        self.__X_unb_train, self.__Y_unb_train = X_unb_n, Y_unb_n 
        self.__X_b_train, self.__Y_b_train = X_b_n, Y_b_n 

        self.__Config.gas.TP = 300, 101325
        self.__Config.gas.set_mixture_fraction(0.0, fuel_string,oxidizer_string)
        Y_ox = self.__Config.gas.Y 
        self.__Config.gas.set_mixture_fraction(1.0,  fuel_string,oxidizer_string)
        Y_f = self.__Config.gas.Y
        pv_ox = self.__Config.ComputeProgressVariable(variables=None, flamelet_data=None, Y_flamelet=Y_ox[:, np.newaxis])
        pv_f = self.__Config.ComputeProgressVariable(variables=None, flamelet_data=None, Y_flamelet=Y_f[:,np.newaxis])
        self.__Config.gas.set_equivalence_ratio(1.0,  fuel_string,oxidizer_string)
        self.__Config.gas.equilibrate("HP")
        Y_b_st = self.__Config.gas.Y 
        pv_st = self.__Config.ComputeProgressVariable(variables=None, flamelet_data=None, Y_flamelet=Y_b_st[:,np.newaxis])
        Z_st = self.__Config.gas.mixture_fraction( fuel_string,oxidizer_string)

        Z_st_norm = (Z_st - self._X_min[2])/(self._X_max[2] - self._X_min[2])
        self.__Z_st_norm = Z_st_norm

        dpv_dz_unb = pv_f - pv_ox 

        dpv_dz_b_rich = (pv_f - pv_st) / (1 - Z_st)
        dpv_dz_b_lean = (pv_st - pv_ox) / (Z_st)
        self.__pv_unb_constraint_factor = dpv_dz_unb / (self._X_max[0] - self._X_min[0])
        self.__pv_b_constraint_factor_lean = dpv_dz_b_lean / (self._X_max[0] - self._X_min[0])
        self.__pv_b_constraint_factor_rich = dpv_dz_b_rich / (self._X_max[0] - self._X_min[0])

        return 
    
    @tf.function
    def ComputeDerivatives(self, x_norm_input:tf.constant,idx_out:int=0):
        x_var = x_norm_input
        with tf.GradientTape(watch_accessed_variables=False) as tape_con:
            tape_con.watch(x_var)
            Y_norm = self._MLP_Evaluation(x_var)
            dY_norm = tape_con.jacobian(Y_norm, x_var)
        return Y_norm, dY_norm

class EvaluateArchitecture_FGM(EvaluateArchitecture):
    """Class for training MLP architectures
    """

    __output_group:int = 0
    __group_name:str = "Group1"

    def __init__(self, Config:FlameletAIConfig, group_idx:int=0):
        """Define EvaluateArchitecture instance and prepare MLP trainer with
        default settings.

        :param Config: FlameletAIConfig object describing the flamelet data manifold.
        :type Config: FlameletAIConfig
        :param group_idx: MLP output group index, defaults to 0
        :type group_idx: int, optional
        :raises Exception: if MLP output group index is undefined by flameletAI configuration.
        """
        self._trainer_direct = Train_Flamelet_Direct(Config_in=Config,group_idx=group_idx)
        self.__output_group=group_idx
        EvaluateArchitecture.__init__(self, Config_in=Config)

        pass

    def SynchronizeTrainer(self):
        """Synchronize all MLP trainer settings with locally stored settings.
        """
        self._trainer_direct.SetTrainVariables(self._Config.GetMLPOutputGroup(self.__output_group))
        super().SynchronizeTrainer()

        return 

    def SetOutputGroup(self, iGroup:int):
        """Define MLP output group.

        :param iGroup: MLP output group index.
        :type iGroup: int
        :raises Exception: if group is not supported by loaded flameletAIConfig class.
        """
        if iGroup < 0 or iGroup >= self._Config.GetNMLPOutputGroups():
            raise Exception("MLP output group index should be between 0 and %i" % self._Config.GetNMLPOutputGroups())
        self.__output_group=iGroup
        self.__group_name = "Group"+str(self.__output_group+1)
        self.main_save_dir = self._Config.GetOutputDir() + "/architectures_"+self.__group_name
        if not os.path.isdir(self.main_save_dir):
            os.mkdir(self.main_save_dir)
        self.SynchronizeTrainer()
        return 