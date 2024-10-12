###############################################################################################
#       #      _____ __  _____      ____        __        __  ____                   #        #
#       #     / ___// / / /__ \    / __ \____ _/ /_____ _/  |/  (_)___  ___  _____   #        #
#       #     \__ \/ / / /__/ /   / / / / __ `/ __/ __ `/ /|_/ / / __ \/ _ \/ ___/   #        #
#       #    ___/ / /_/ // __/   / /_/ / /_/ / /_/ /_/ / /  / / / / / /  __/ /       #        #
#       #   /____/\____//____/  /_____/\__,_/\__/\__,_/_/  /_/_/_/ /_/\___/_/        #        #
#       #                                                                            #        #
###############################################################################################

################################ FILE NAME: Trainers.py #######################################
#=============================================================================================#
# author: Evert Bunschoten                                                                    |
#    :PhD Candidate ,                                                                         |
#    :Flight Power and Propulsion                                                             |
#    :TU Delft,                                                                               |
#    :The Netherlands                                                                         |
#                                                                                             |
#                                                                                             |
# Description:                                                                                |
#  Classes for training multi-layer perceptrons on flamelet data.                             |
#                                                                                             |
# Version: 1.0.0                                                                              |
#                                                                                             |
#=============================================================================================#

# Set seed values.
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
import cantera as ct 

from Common.DataDrivenConfig import FlameletAIConfig
from Manifold_Generation.MLP.Trainer_Base import MLPTrainer, TensorFlowFit,PhysicsInformedTrainer,EvaluateArchitecture, CustomTrainer
from Common.CommonMethods import GetReferenceData
from Common.Properties import DefaultSettings_FGM as DefaultProperties


class Train_Flamelet_Direct(TensorFlowFit):
    __Config:FlameletAIConfig
    _train_name:str 

    def __init__(self, Config_in:FlameletAIConfig, group_idx:int=0):
        TensorFlowFit.__init__(self)
        # Set train variables based on what kind of network is trained
        self.__Config = Config_in
        self._filedata_train = self.__Config.GetOutputDir()+"/"+self.__Config.GetConcatenationFileHeader()
        self._controlling_vars = DefaultProperties.controlling_variables
        self._train_vars = self.__Config.GetMLPOutputGroup(group_idx)
        self._train_name = "Group"+str(group_idx+1)

        return
    
    def CustomCallback(self):
        """Plot MLP prediction alongside flamelet reference data.
        :file_name_header: file name header for each figure.
        :N_plot: number of equivalence ratio's to plot for in each figure.
        """
        PlotFlameletData(self, self.__Config, self._train_name)
        self.write_SU2_MLP(self._save_dir + "/Model_"+str(self._model_index)+"/MLP_"+self._train_name)
        return super().CustomCallback()
   
    def add_additional_header_info(self, fid):
        fid.write("Progress variable definition: " + "+".join(("%+.6e*%s" % (w, s)) for w, s in zip(self.__Config.GetProgressVariableWeights(), self.__Config.GetProgressVariableSpecies())))
        fid.write("\n\n")
        return super().add_additional_header_info(fid)
    
    def SetDecaySteps(self):
        self._decay_steps=0.01157 * self._Np_train
        return 

    def GetTrainData(self):
        super().GetTrainData()
        return 
    
class Train_Source_PINN(CustomTrainer):
    """Physics-informed trainer for training flamelet source terms.
    """
    __Config:FlameletAIConfig
    __boundary_data_file:str =DefaultProperties.boundary_file_header+"_full.csv"
    __X_unb_train:np.ndarray
    __X_b_train:np.ndarray
    __Y_unb_train:np.ndarray 
    __Y_b_train:np.ndarray 

    __pv_unb_constraint_factor:float = None 
    __pv_b_constraint_factor_lean:float = None 
    __pv_b_constraint_factor_rich:float = None 
    __val_lambda:float=0.0
    __current_epoch:int = 0
    def __init__(self, Config_in:FlameletAIConfig, group_idx:int=0):
        """Class constructor. Initialize a source term trainer for a given MLP output group.

        :param Config_in: FlameletAI configuration class.
        :type Config_in: FlameletAIConfig
        :param group_idx: MLP output group index, defaults to 0
        :type group_idx: int, optional
        """
        
        CustomTrainer.__init__(self)
        self.__Config = Config_in 
        self._controlling_vars = DefaultProperties.controlling_variables
        self._train_vars = self.__Config.GetMLPOutputGroup(group_idx)
        self.callback_every = 10
        self.__boundary_data_file = self.__Config.GetOutputDir()+"/"+DefaultProperties.boundary_file_header+"_full.csv"
        
        # Synchronize settings with loaded configuration
        self._alpha_expo = self.__Config.GetAlphaExpo(group_idx)
        self._lr_decay = self.__Config.GetLRDecay(group_idx)
        self._batch_expo = self.__Config.GetBatchExpo(group_idx)
        self._activation_function = self.__Config.GetActivationFunction(group_idx)
        self.SetHiddenLayers(self.__Config.GetHiddenLayerArchitecture(group_idx))

        return 

    def SetBoundaryDataFile(self, boundary_data_file:str):
        self.__boundary_data_file = boundary_data_file 
        return 
    
    def SetDecaySteps(self):

        #self._decay_steps=0.01157 * self._Np_train
        self._decay_steps = int(0.33*self._Np_train / (2**self._batch_expo))
        return 
    
    def GetBoundaryData(self):
        """Load flamelet data equilibrium boundary data.
        """

        # Load controlling and train variables from boundary data.
        X_boundary, Y_boundary = GetReferenceData(self.__boundary_data_file, x_vars=self._controlling_vars, train_variables=self._train_vars)
        mixfrac_boundary = X_boundary[:, self._controlling_vars.index(DefaultProperties.name_mixfrac)]
        pv_boundary = X_boundary[:, self._controlling_vars.index(DefaultProperties.name_pv)]
        is_unb = np.ones(np.shape(X_boundary)[0],dtype=bool)

        # Filter reactant and product data.
        fuel_definition = self.__Config.GetFuelDefinition()
        oxidizer_definition = self.__Config.GetOxidizerDefinition()
        fuel_weights = self.__Config.GetFuelWeights()
        oxidizer_weights = self.__Config.GetOxidizerWeights()
        fuel_string = ",".join((sp+":"+str(w) for sp, w in zip(fuel_definition, fuel_weights)))
        oxidizer_string = ",".join((sp+":"+str(w) for sp, w in zip(oxidizer_definition, oxidizer_weights)))
        
        for iz, z in enumerate(mixfrac_boundary):
            self.__Config.gas.set_mixture_fraction(max(z,0), fuel_string, oxidizer_string)
            self.__Config.gas.TP=300,101325
            Y = self.__Config.gas.Y 
            pv_unb = self.__Config.ComputeProgressVariable(variables=None, flamelet_data=None, Y_flamelet=Y[:,np.newaxis])
            if pv_boundary[iz] > pv_unb+1e-3:
                is_unb[iz] = False 
        
        X_boundary_norm = self.scaler_function_x.transform(X_boundary)#(X_boundary - self._X_min)/(self._X_max-self._X_min)
        Y_boundary_norm = self.scaler_function_y.transform(Y_boundary)#(Y_boundary - self._Y_min)/(self._Y_max-self._Y_min)

        X_unb_n, Y_unb_n = X_boundary_norm[is_unb, :], Y_boundary_norm[is_unb, :]
        X_b_n, Y_b_n = X_boundary_norm[np.invert(is_unb), :], Y_boundary_norm[np.invert(is_unb), :]

        self.__X_unb_train, self.__Y_unb_train = X_unb_n, Y_unb_n 
        self.__X_b_train, self.__Y_b_train = X_b_n, Y_b_n 

        # Compute progress variable derivatives w.r.t. mixture fraction for reactants and products.

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
        h_st = self.__Config.gas.enthalpy_mass
        X_st_norm = self.scaler_function_x.transform(np.array([pv_st, h_st, Z_st]))
        Z_st_norm = X_st_norm[2]#(Z_st - self._X_min[2])/(self._X_max[2] - self._X_min[2])
        self.__Z_st_norm = Z_st_norm

        dpv_dz_unb = pv_f - pv_ox 

        dpv_dz_b_rich = (pv_f - pv_st) / (1 - Z_st)
        dpv_dz_b_lean = (pv_st - pv_ox) / (Z_st)
        self.__pv_unb_constraint_factor = dpv_dz_unb / self._X_scale
        self.__pv_b_constraint_factor_lean = dpv_dz_b_lean / self._X_scale#(self._X_max[0] - self._X_min[0])
        self.__pv_b_constraint_factor_rich = dpv_dz_b_rich / self._X_scale#(self._X_max[0] - self._X_min[0])

        return 
    
    def GetTrainData(self):
        super().GetTrainData()
        if self._verbose > 0:
            print("Extracting boundary data...")
        self.GetBoundaryData()
        if self._verbose > 0:
            print("Done")
        return 
    
    @tf.function
    def ComputeDerivatives(self, x_norm_input:tf.constant,idx_out:int=0):
        x_var = x_norm_input
        with tf.GradientTape(watch_accessed_variables=False) as tape_con:
            tape_con.watch(x_var)
            Y_norm = self._MLP_Evaluation(x_var)
            dY_norm = tape_con.gradient(Y_norm[:, idx_out], x_var)
        return Y_norm, dY_norm

    @tf.function
    def ComputeUnburntSourceConstraint(self, x_boundary_norm:tf.constant, y_boundary_norm:tf.constant):
        """Compute the reactant data physics-informed source term constraint value

        :param x_boundary_norm: normalized reactant controlling variable data.
        :type x_boundary_norm: tf.constant
        :param y_boundary_norm: normalized reactant label data.
        :type y_boundary_norm: tf.constant
        :return: physics-informed source term penalty value.
        :rtype: tf.constant
        """

        penalty=0.0
        # Loop over output variables
        for iVar in range(len(self._train_vars)):

            # Get normalized MLP output Jacobian.
            Y_norm, dY_norm = self.ComputeDerivatives(x_boundary_norm, iVar)
            dY_dpv_norm = dY_norm[:, 0]
            dY_dh_norm = dY_norm[:, 1]
            dY_dz_norm = dY_norm[:, 2]
            
            # Dot product between reactant pv-Z plane and MLP output Jacobian.
            dY_dpvz = dY_dpv_norm * self.__pv_unb_constraint_factor + dY_dz_norm

            # Penalty term for derivative terms.
            term_1 = tf.reduce_mean(tf.pow(dY_dpvz, 2))
            term_2 = tf.reduce_mean(tf.pow(dY_dh_norm, 2))

            # Penalty term for predicted values.
            term_3 = tf.reduce_mean(tf.pow(Y_norm - y_boundary_norm, 2))

            penalty = penalty + term_1 + term_2 + term_3

        return penalty
    
    @tf.function
    def ComputeBurntSourceConstraint(self, x_boundary_norm:tf.constant, y_boundary_norm:tf.constant):
        """Compute the product data physics-informed source term constraint value

        :param x_boundary_norm: normalized product controlling variable data.
        :type x_boundary_norm: tf.constant
        :param y_boundary_norm: normalized product label data.
        :type y_boundary_norm: tf.constant
        :return: physics-informed source term penalty value.
        :rtype: tf.constant
        """

        penalty = 0.0
        # Loop over output variables
        for iVar in range(len(self._train_vars)):

            # Get normalized MLP output Jacobian.
            Y_norm, dY_norm = self.ComputeDerivatives(x_boundary_norm, idx_out=iVar)
            dY_dpv_norm = dY_norm[:, 0]
            dY_dh_norm = dY_norm[:, 1]
            dY_dz_norm = dY_norm[:, 2]
            
            # Identify data points above and below stochiometry.
            dppv_dpv_rich = dY_dpv_norm[x_boundary_norm[:,2]>=self.__Z_st_norm]
            dppv_dz_rich = dY_dz_norm[x_boundary_norm[:,2]>=self.__Z_st_norm]
            dppv_dpv_lean = dY_dpv_norm[x_boundary_norm[:,2]<self.__Z_st_norm]
            dppv_dz_lean = dY_dz_norm[x_boundary_norm[:,2]<self.__Z_st_norm]
            
            # Dot product between reactant pv-Z plane and MLP output Jacobian.
            dY_dpvz_lean = dppv_dpv_lean * self.__pv_b_constraint_factor_lean + dppv_dz_lean
            dY_dpvz_rich = dppv_dpv_rich * self.__pv_b_constraint_factor_rich + dppv_dz_rich

            # Penalty term for derivative terms.
            term_1 = tf.reduce_mean(tf.pow(dY_dpvz_lean,2)) + tf.reduce_mean(tf.pow(dY_dpvz_rich,2))
            term_2 = tf.reduce_mean(tf.pow(dY_dh_norm,2))

            # Penalty term for predicted values.
            term_3 = tf.reduce_mean(tf.pow(Y_norm - y_boundary_norm,2))

            penalty = penalty+ term_1 + term_2 + term_3 
            
        return penalty 

    @tf.function 
    def GetBoundsGrads(self, x_boundary, y_boundary, x_b_boundary, y_b_boundary):
        with tf.GradientTape() as tape:
            tape.watch(self._trainable_hyperparams)
            y_b_loss = self.ComputeUnburntSourceConstraint(x_boundary, y_boundary)
            y_burnt_loss = self.ComputeBurntSourceConstraint(x_b_boundary, y_b_boundary)
            y_loss_total = y_b_loss + y_burnt_loss
            grads_ub = tape.gradient(y_loss_total, self._trainable_hyperparams)
        return grads_ub 
    
    @tf.function
    def ComputeTotalGrads(self, x_domain, y_domain, x_boundary, y_boundary, x_b_boundary, y_b_boundary):
        _, grads_direct = self.ComputeGradients_Direct_Error(x_domain, y_domain)
        
        grads_ub = self.GetBoundsGrads(x_boundary, y_boundary, x_b_boundary, y_b_boundary)
        return grads_direct, grads_ub
    
    @tf.function
    def Train_Step(self, x_domain, y_domain, x_ub_boundary, y_ub_boundary, x_b_boundary, y_b_boundary, val_lambda):
        
        grads_direct, grads_ub = self.ComputeTotalGrads(x_domain, y_domain, x_ub_boundary, y_ub_boundary, x_b_boundary, y_b_boundary)
        
        #if self.__current_epoch >= 0.01*self._n_epochs:
        total_grads = [(g_d + 0.001*val_lambda*g_b) for (g_d, g_b) in zip(grads_direct, grads_ub)]
        # else:
        #     total_grads = grads_direct
        self._optimizer.apply_gradients(zip(total_grads, self._trainable_hyperparams))

        return grads_direct, grads_ub
    
    def FinalAdjustment(self):
        Y_unb_pred = self._MLP_Evaluation(self.__X_unb_train)
        max_Y_pred = tf.reduce_max(Y_unb_pred,axis=0)
        delta_b = -max_Y_pred
        self._biases[-1].assign_add(delta_b)
        return
    
    def PostProcessing(self):
        return super().PostProcessing()
    
    def SetTrainBatches(self):
        train_batches_domain = super().SetTrainBatches()
        domain_batches_list = [b for b in train_batches_domain]

        batch_size_train = 2**self._batch_expo
        
        train_batches_unb = tf.data.Dataset.from_tensor_slices((self.__X_unb_train, self.__Y_unb_train)).batch(batch_size_train)
        train_batches_b = tf.data.Dataset.from_tensor_slices((self.__X_b_train, self.__Y_b_train)).batch(batch_size_train)
        unb_batches_list = [b for b in train_batches_unb]
        b_batches_list = [b for b in train_batches_b]
        Nb_unb = len(unb_batches_list)
        Nb_b = len(b_batches_list)

        unb_batches_list_resized = [unb_batches_list[i % Nb_unb] for i in range(len(domain_batches_list))]
        b_batches_list_resized = [b_batches_list[i % Nb_b] for i in range(len(domain_batches_list))]
        return (domain_batches_list, unb_batches_list_resized, b_batches_list_resized)
    
    # def LoopBatches(self, train_batches):
    #     for XY_domain, XY_unb, XY_b in zip(train_batches[0],train_batches[1],train_batches[2]):
    #         X_domain = XY_domain[0]
    #         Y_domain = XY_domain[1]
    #         X_unb = XY_unb[0]
    #         Y_unb = XY_unb[1]
    #         X_b = XY_b[0]
    #         Y_b = XY_b[1]
    #         grads_domain, grads_boundary = self.Train_Step(X_domain, Y_domain, X_unb, Y_unb, X_b, Y_b,self.__val_lambda)

    #         self.__val_lambda = self.update_lambda(grads_domain, grads_boundary, self.__val_lambda)
    #     self.__current_epoch += 1
    #     return 
    
    @tf.function
    def update_lambda(self, grads_direct, grads_ub, val_lambda_old):
        max_grad_direct = 0.0
        for g in grads_direct:
            max_grad_direct = tf.maximum(max_grad_direct, tf.reduce_max(tf.abs(g)))

        mean_grad_ub = 0.0
        for g_ub in grads_ub:
            mean_grad_ub += tf.reduce_mean(tf.abs(g_ub))
        mean_grad_ub /= len(grads_ub)

        lambda_prime = max_grad_direct / (mean_grad_ub + 1e-32)
        val_lambda_new = 0.1 * val_lambda_old + 0.9 * lambda_prime
        return val_lambda_new
    
    def CustomCallback(self):
        self.Plot_and_Save_History()
        PlotFlameletData(self, self.__Config, self._train_name)
        self.PlotBoundaryData()
        return super().CustomCallback()
    
    def PlotBoundaryData(self):
        
        Y_unb_pred_norm = self._MLP_Evaluation(self.__X_unb_train).numpy()
        Y_b_pred_norm = self._MLP_Evaluation(self.__X_b_train).numpy()

        fig, axs = plt.subplots(nrows=1, ncols=len(self._train_vars),figsize=[6*len(self._train_vars), 6])
        fig_b, axs_b = plt.subplots(nrows=1, ncols=len(self._train_vars),figsize=[6*len(self._train_vars), 6])
       
        Y_unb_pred = self.scaler_function_y.inverse_transform(Y_unb_pred_norm)#(self._Y_max - self._Y_min)*Y_unb_pred_norm + self._Y_min
        Y_b_pred = self.scaler_function_y.inverse_transform(Y_b_pred_norm)#(self._Y_max - self._Y_min)*Y_b_pred_norm + self._Y_min
        X_unb = self.scaler_function_x.inverse_transform(self.__X_unb_train)#(self._X_max - self._X_min)*self.__X_unb_train + self._X_min
        X_b = self.scaler_function_x.inverse_transform(self.__X_b_train)#(self._X_max - self._X_min)*self.__X_b_train + self._X_min
        for iVar in range(len(self._train_vars)):
            if len(self._train_vars) == 1:
                ax_unb = axs 
                ax_b = axs_b 
            else:
                ax_unb = axs[iVar]
                ax_b = axs_b[iVar]
            sc = ax_unb.scatter(X_unb[:, 1], X_unb[:, 2], c=Y_unb_pred[:, iVar])
            sc_b = ax_b.scatter(X_b[:, 1], X_b[:, 2], c=Y_b_pred[:, iVar])
            cbar = fig.colorbar(sc, ax=ax_unb)
            cbar.set_label(self._train_vars[iVar],rotation=270,fontsize=18)
            cbar_b = fig_b.colorbar(sc_b, ax=ax_b)
            cbar_b.set_label("Scaled reaction rate",rotation=270,fontsize=18)
            ax_unb.set_xlabel("Total enthalpy", fontsize=20)
            ax_b.set_xlabel("Total enthalpy", fontsize=20)
            if iVar == 0:
                ax_unb.set_ylabel("Mixture fraction", fontsize=20)
                ax_b.set_ylabel("Mixture fraction", fontsize=20)
        fig.savefig(self._save_dir+"/Model_"+str(self._model_index)+"/Unburnt_Pred."+self._figformat,format=self._figformat,bbox_inches='tight')
        plt.close(fig)

        fig_b.savefig(self._save_dir+"/Model_"+str(self._model_index)+"/Burnt_Pred."+self._figformat,format=self._figformat,bbox_inches='tight')
        plt.close(fig_b)
        return 
    

class Train_Beta_PINN(Train_Source_PINN):
    __beta_pv_unb_fac:float = 0 
    __beta_z_unb_fac:float = 0         
    def __init__(self, Config_in:FlameletAIConfig, group_idx:int=0):
        """Class constructor. Initialize a source term trainer for a given MLP output group.

        :param Config_in: FlameletAI configuration class.
        :type Config_in: FlameletAIConfig
        :param group_idx: MLP output group index, defaults to 0
        :type group_idx: int, optional
        """
        
        Train_Source_PINN.__init__(self, Config_in, group_idx)
        return 
    
    def GetBoundaryData(self):
        super().GetBoundaryData()

        pv_species = self.__Config.GetProgressVariableSpecies()
        pv_weights = self.__Config.GetProgressVariableWeights()

        Le_species = self.__Config.GetConstSpecieLewisNumbers()
        idx_pure_fuel = np.argwhere(self.__X_unb_train[:,2] == 1.0)
        idx_pure_OX = np.argwhere(self.__X_unb_train[:,2] == 0.0)

        eq_file_fuel = self.__Config.GetOutputDir()+"/equilibrium_data/mixfrac_1.0/equilibrium_ub_mixfrac_1.0.csv"
        eq_file_ox = self.__Config.GetOutputDir()+"/equilibrium_data/mixfrac_1.0/equilibrium_ub_mixfrac_0.0.csv"
        
        with open(eq_file_fuel,'r') as fid:
            vars = fid.readline().strip().split(',')
        D_eq_fuel = np.loadtxt(eq_file_fuel,delimiter=',',skiprows=1)
        D_eq_ox = np.loadtxt(eq_file_ox,delimiter=',',skiprows=1)

        beta_pv_fuel, beta_h1_fuel, beta_h2_fuel, beta_z_fuel = self.__Config.ComputeBetaTerms(vars, D_eq_fuel)
        beta_pv_ox, beta_h1_ox, beta_h2_ox, beta_z_ox = self.__Config.ComputeBetaTerms(vars, D_eq_ox)
        self.__beta_pv_unb_fac = (beta_pv_fuel - beta_pv_ox)
        self.__beta_z_unb_fac = (beta_z_fuel - beta_z_ox)
        
        return 
    
    @tf.function
    def ComputeUnburntSourceConstraint(self, x_boundary_norm:tf.constant, y_boundary_norm:tf.constant):
        """Compute the reactant data physics-informed source term constraint value

        :param x_boundary_norm: normalized reactant controlling variable data.
        :type x_boundary_norm: tf.constant
        :param y_boundary_norm: normalized reactant label data.
        :type y_boundary_norm: tf.constant
        :return: physics-informed source term penalty value.
        :rtype: tf.constant
        """

        penalty=0.0
        # Loop over output variables
        for iVar in range(len(self._train_vars)):

            # Get normalized MLP output Jacobian.
            Y_norm, dY_norm = self.ComputeDerivatives(x_boundary_norm, iVar)
            dY_dpv_norm = dY_norm[:, 0]
            dY_dh_norm = dY_norm[:, 1]
            dY_dz_norm = dY_norm[:, 2]
            
            # Dot product between reactant pv-Z plane and MLP output Jacobian.
            dY_dpvz = dY_dpv_norm * self.__pv_unb_constraint_factor + dY_dz_norm

            # Penalty term for derivative terms.
            term_1 = tf.reduce_mean(tf.pow(dY_dpvz, 2))
            term_2 = tf.reduce_mean(tf.pow(dY_dh_norm, 2))

            # Penalty term for predicted values.
            term_3 = tf.reduce_mean(tf.pow(Y_norm - y_boundary_norm, 2))

            penalty = penalty + term_1 + term_2 + term_3

        return penalty

class Train_FGM_PINN(PhysicsInformedTrainer):

    __Config:FlameletAIConfig = None 

    update_lambda_per_batch:bool = True
    update_lambda_every_iter:int = 1000
    lambda_history:list[list[float]] = []
    eta = 1e-3 
    j_gradient_update:int = 0
    def __init__(self, Config_in:FlameletAIConfig, group_idx:int=0):
        """Class constructor. Initialize a physics-informed trainer object for a given output group.

        :param Config_in: FlameletAI configuration class.
        :type Config_in: FlameletAIConfig
        :param group_idx: MLP output group index, defaults to 0
        :type group_idx: int, optional
        """

        # Initiate parent class.
        PhysicsInformedTrainer.__init__(self)
        self.__Config = Config_in 
        self._controlling_vars = DefaultProperties.controlling_variables
        self._train_vars = self.__Config.GetMLPOutputGroup(group_idx)
        self.callback_every = 10
        self.__boundary_data_file = self.__Config.GetOutputDir()+"/"+DefaultProperties.boundary_file_header+"_full.csv"
        
        # Synchronize settings with loaded configuration
        self._alpha_expo = self.__Config.GetAlphaExpo(group_idx)
        self._lr_decay = self.__Config.GetLRDecay(group_idx)
        self._batch_expo = self.__Config.GetBatchExpo(group_idx)
        self._activation_function = self.__Config.GetActivationFunction(group_idx)
        self.SetHiddenLayers(self.__Config.GetHiddenLayerArchitecture(group_idx))
        self._train_name = "Group"+str(group_idx+1)

        return 
    
    def SetDecaySteps(self):
        self._decay_steps=0.01157 * self._Np_train
        return 
    
    def GetTrainData(self):
        """Read domain and boundary training data and pre-process reactant-product matrices for visualization.
        """
        super().GetTrainData()
        self.GetBoundaryData()
        self.CollectPIVars()
        self.GenerateBoundaryMatrices()

        return 
    
    def SetTrainVariables(self, train_vars: list[str]):
        self._state_vars = train_vars.copy()
        return super().SetTrainVariables(train_vars)
    
    def GetBoundaryData(self):
        """Load flamelet data equilibrium boundary data.
        """

        # Load controlling and train variables from boundary data.
        X_boundary, Y_boundary = GetReferenceData(self.__boundary_data_file, x_vars=self._controlling_vars, train_variables=self._train_vars)
        
        
        # Normalize controlling and labeled data with respect to domain data.
        self.X_boundary_norm = self.scaler_function_x.transform(X_boundary)
        self.Y_boundary_norm = self.scaler_function_y.transform(Y_boundary)

        return 
    
    def SetBeta_pv_projection(self):
        """Neumann boundary condition for progress variable preferential diffusion scalar.
        The penalty equates the square of the Jacobian with respect to total enthalpy on the boundaries.

        :return: penalty projection array, target projected gradient
        :rtype: np.ndarray[float],np.ndarray[float]
        """
        projection_array_train = np.zeros(np.shape(self.X_boundary_norm))
        projection_array_train[:, self._controlling_vars.index(DefaultProperties.name_enth)] = 1.0 
        
        target_grad_array = np.zeros([len(projection_array_train)])
        return projection_array_train, target_grad_array
    
    def SetBeta_z_projection(self):
        """Neumann boundary condition for mixture fraction preferential diffusion scalar.
        The penalty equates the square of the Jacobian with respect to total enthalpy on the boundaries.

        :return: penalty projection array, target projected gradient
        :rtype: np.ndarray[float],np.ndarray[float]
        """
        return self.SetBeta_pv_projection()
    
    def LocateUnbBoundaryNodes(self):
        """Locate the reactant and product nodes of the boundary data.

        :return: reactant identification array.
        :rtype: np.ndarray[bool]
        """

        # Initiate default reactant identifier array.
        is_unb = np.ones(np.shape(self.X_boundary_norm)[0],dtype=bool)
        
        fuel_string = self.__Config.GetFuelString()
        oxidizer_string = self.__Config.GetOxidizerString()
        
        # Loop over boundary nodes and locate indices where the progress variable is greater than the reactant progress variable.
        X_boundary = self.scaler_function_x.inverse_transform(self.X_boundary_norm)
        for iz in range(np.shape(self.X_boundary_norm)[0]):
            z = X_boundary[iz,self._controlling_vars.index(DefaultProperties.name_mixfrac)]
            pv_boundary = X_boundary[iz,self._controlling_vars.index(DefaultProperties.name_pv)]
            self.__Config.gas.set_mixture_fraction(min(max(z,0),1.0), fuel_string, oxidizer_string)
            self.__Config.gas.TP=self.__Config.GetUnbTempBounds()[0], DefaultProperties.pressure
            Y = self.__Config.gas.Y 
            pv_unb = self.__Config.ComputeProgressVariable(variables=None, flamelet_data=None, Y_flamelet=Y[:,np.newaxis])
            
            if pv_boundary > (pv_unb+1e-3*self._X_scale[self._controlling_vars.index(DefaultProperties.name_pv)]):
                is_unb[iz] = False 

        return is_unb
    
    def SetSource_projection(self):
        """Neumann boundary condition for source terms.
        Two penalties are applied: for non-zero gradients along the total enthalpy direction and 
        for non-zero projected gradients along the equilibrium progress variable-mixture fraction line.
        """

        is_unb = self.LocateUnbBoundaryNodes()
        
        fuel_string = self.__Config.GetFuelString()
        oxidizer_string = self.__Config.GetOxidizerString()
        mixfrac_boundary_norm = self.X_boundary_norm[:, self._controlling_vars.index(DefaultProperties.name_mixfrac)]
        
        # Compute the progress variable derivative w.r.t. mixture fraction for equilibrium conditions

        self.__Config.gas.TP = self.__Config.GetUnbTempBounds()[0], DefaultProperties.pressure

        # Compute progress variable values for pure oxidizer and pure fuel.
        self.__Config.gas.set_mixture_fraction(0.0, fuel_string,oxidizer_string)
        Y_ox = self.__Config.gas.Y 
        self.__Config.gas.set_mixture_fraction(1.0,  fuel_string,oxidizer_string)
        Y_f = self.__Config.gas.Y
        
        pv_ox = self.__Config.ComputeProgressVariable(variables=None, flamelet_data=None, Y_flamelet=Y_ox[:, np.newaxis])[0]
        pv_f = self.__Config.ComputeProgressVariable(variables=None, flamelet_data=None, Y_flamelet=Y_f[:,np.newaxis])[0]

        # Compute the progress variable and mixture fraction for the stochiometric equilibrium condition.
        self.__Config.gas.set_equivalence_ratio(1.0,  fuel_string,oxidizer_string)
        self.__Config.gas.equilibrate("HP")
        Y_b_st = self.__Config.gas.Y 
        pv_st = self.__Config.ComputeProgressVariable(variables=None, flamelet_data=None, Y_flamelet=Y_b_st[:,np.newaxis])[0]
        Z_st = self.__Config.gas.mixture_fraction( fuel_string,oxidizer_string)
        h_st = self.__Config.gas.enthalpy_mass
        X_st_norm = self.scaler_function_x.transform(np.array([[pv_st, h_st, Z_st]]))[0,:]
        Z_st_norm = X_st_norm[2]

        # Identify rich and lean boundary nodes.
        is_rich = mixfrac_boundary_norm > Z_st_norm 
        is_stoch = mixfrac_boundary_norm == Z_st_norm
        is_lean = np.invert(is_rich)

        # Calculate progress variable-mixture fraction derivative for reactants and products.
        dpv_dz_unb = (pv_f - pv_ox)
        dpv_dz_b_lean = (pv_st - pv_ox) / (Z_st)
        dpv_dz_b_rich = (pv_f - pv_st) / (1 - Z_st)
        dpv_dz_scale = self._X_scale[self._controlling_vars.index(DefaultProperties.name_pv)] \
            / self._X_scale[self._controlling_vars.index(DefaultProperties.name_mixfrac)]
        projection_unb_pvz = dpv_dz_unb/dpv_dz_scale
        projection_b_pvz_lean = dpv_dz_b_lean /dpv_dz_scale
        projection_b_pvz_rich = dpv_dz_b_rich /dpv_dz_scale

        # Populate projection arrays for the boundary condition.
        projection_array_pvz = np.zeros(np.shape(self.X_boundary_norm))

        projection_array_pvz[:, self._controlling_vars.index(DefaultProperties.name_mixfrac)] = 1.0
        projection_array_pvz[:, self._controlling_vars.index(DefaultProperties.name_enth)] = 1.0
        projection_array_pvz[is_unb, self._controlling_vars.index(DefaultProperties.name_pv)] = projection_unb_pvz
        projection_array_pvz[np.logical_and(np.invert(is_unb),is_rich), self._controlling_vars.index(DefaultProperties.name_pv)] = projection_b_pvz_rich
        projection_array_pvz[np.logical_and(np.invert(is_unb),is_lean), self._controlling_vars.index(DefaultProperties.name_pv)] = projection_b_pvz_lean
        projection_array_pvz[np.logical_and(np.invert(is_unb),is_stoch), self._controlling_vars.index(DefaultProperties.name_pv)] = 0.0
        
        target_grad_array_pvz = np.zeros(len(projection_array_pvz))

        return projection_array_pvz, target_grad_array_pvz
        
    def SetT_projection(self):
        """Neumann boundary condition for temperature.
        The temperature derivative w.r.t. total enthalpy should be equal to the inverse of the specific heat.

        :return: penalty projection array, target projected gradient
        :rtype: np.ndarray[float],np.ndarray[float]
        """

        _, Cp_boundary = GetReferenceData(self.__boundary_data_file, x_vars=self._controlling_vars, train_variables=["Cp"])
        T_scale = self._Y_scale[self._train_vars.index("Temperature")]
        h_scale = self._X_scale[self._controlling_vars.index(DefaultProperties.name_enth)]
        projection_array_train = np.zeros(np.shape(self.X_boundary_norm))
        projection_array_train[:, self._controlling_vars.index(DefaultProperties.name_enth)] = 1.0
        
        target_grad_array = (1.0 / Cp_boundary[:,0]) * T_scale / h_scale

        return projection_array_train, target_grad_array
    
    def CollectPIVars(self):
        """Collect the Jacobian projection arrays and projected target arrays for any physics-informed variables in the training data set.
        """

        self.projection_arrays = []
        self.target_arrays = []
        self.vals_lambda = []
        self.idx_PIvar = []
        val_lambda_default = tf.constant(1.0,dtype=self._dt)
        for ivar, var in enumerate(self._train_vars):

            # Apply general invariance for source terms.
            if (var == "ProdRateTot_PV") or (var == "Heat_Release") or ("Y_dot" in var):
                self.idx_PIvar.append(ivar)

                proj_array_pvz, target_grad_pvz = self.SetSource_projection()
                self.projection_arrays.append(proj_array_pvz)
                
                self.target_arrays.append(target_grad_pvz)
                
                self.vals_lambda.append(val_lambda_default)

            # Enforce total enthalpy invariance for pv and z preferential diffusion scalars.
            if (var == "Beta_ProgVar") or (var == "Beta_MixFrac"):
                self.idx_PIvar.append(ivar)
                proj_array, target_grad = self.SetBeta_pv_projection()
                self.projection_arrays.append(proj_array)
                self.target_arrays.append(target_grad)
                self.vals_lambda.append(val_lambda_default)   

            # Enforce consistancy with specific heat for temperature.
            if (var == "Temperature"):
                self.idx_PIvar.append(ivar)
                proj_array, target_grad = self.SetT_projection()
                self.projection_arrays.append(proj_array)
                self.target_arrays.append(target_grad)
                self.vals_lambda.append(val_lambda_default)  
        self._N_bc = len(self.vals_lambda) 
        return 
    
    
    def SetTrainBatches(self):
        """Formulate train batches for domain and boundary data.

        :return: train batches for domain and boundary data.
        :rtype: list
        """

        # Collect domain data train batches
        train_batches_domain = super().SetTrainBatches()
        domain_batches_list = [b for b in train_batches_domain]

        batch_size_train = 2**self._batch_expo

        # Collect boundary labeled data.
        Y_boundary_norm_concat = tf.stack([tf.constant(self.Y_boundary_norm[:, self.idx_PIvar[iVar]], dtype=self._dt) for iVar in range(self._N_bc)],axis=1)
        
        # Collect projection array data.
        p_concatenated = tf.stack([tf.constant(p, dtype=self._dt) for p in self.projection_arrays],axis=2)
        
        # Collect target projection gradient data.
        Y_target_concatenated = tf.stack([tf.constant(t, dtype=self._dt) for t in self.target_arrays], axis=1)

        # Collect boundary controlling variable data.
        X_boundary_tf = tf.constant(self.X_boundary_norm, dtype=self._dt)

        # Forumulate batches.
        batches_concat = tf.data.Dataset.from_tensor_slices((X_boundary_tf, Y_boundary_norm_concat, p_concatenated, Y_target_concatenated)).batch(batch_size_train)
        batches_concat_list = [b for b in batches_concat]

        # Re-size boundary data batches to that of the domain batches such that both data can be evaluated simultaneously during training.
        Nb_boundary = len(batches_concat_list)
        batches_concat_list_resized = [batches_concat_list[i % Nb_boundary] for i in range(len(domain_batches_list))]

        return (domain_batches_list, batches_concat_list_resized)
    
    def LoopEpochs(self):
        self.j_gradient_update = 0
        return super().LoopEpochs()
    
    def LoopBatches(self, train_batches):
        """Loop over domain and boundary batches for each epoch.

        :param train_batches: tuple of domain and boundary data batches.
        :type train_batches: tuple
        """
        domain_batches = train_batches[0]
        boundary_batches = train_batches[1]
        vals_lambda = self.vals_lambda.copy()
        for batch_domain, batch_boundary in zip(domain_batches, boundary_batches):

            # Extract domain batch data.
            X_domain_batch = batch_domain[0]
            Y_domain_batch = batch_domain[1]
            # Extract boundary batch data.
            X_boundary_batch = batch_boundary[0]
            Y_boundary_batch = batch_boundary[1]
            P_boundary_batch = batch_boundary[2]
            Yt_boundary_batch = batch_boundary[3]

            # Run train step and adjust weights.
            self.Train_Step(X_domain_batch, Y_domain_batch, X_boundary_batch, Y_boundary_batch, P_boundary_batch, Yt_boundary_batch, vals_lambda)
            if (self.j_gradient_update + 1)%self.update_lambda_every_iter ==0:
                vals_lambda_updated = self.UpdateLambdas(X_domain_batch, Y_domain_batch, X_boundary_batch, Y_boundary_batch, P_boundary_batch, Yt_boundary_batch, vals_lambda)
                self.vals_lambda = [v for v in vals_lambda_updated]
            self.j_gradient_update += 1
            
        self.lambda_history.append([lamb.numpy() for lamb in self.vals_lambda])

        return 
    
    
    @tf.function
    def ComputeNeumannPenalty(self, x_norm_boundary:tf.constant, y_norm_boundary_target:tf.constant, dy_norm_boundary_target:tf.constant,precon_gradient:tf.constant, iVar:int=0): 
        """Neumann penalty function for projected MLP Jacobians along boundary data.

        :param x_norm_boundary: boundary data controlling variable values.
        :type x_norm_boundary: tf.constant
        :param y_norm_boundary_target: labeled boundary data.
        :type y_norm_boundary_target: tf.constant
        :param dy_norm_boundary_target: target projected gradient.
        :type dy_norm_boundary_target: tf.constant
        :param precon_gradient: MLP Jacobian pre-conditioner.
        :type precon_gradient: tf.constant
        :param iVar: boundary condition index, defaults to 0
        :type iVar: int, optional
        :return: direct evaluation and Neumann penalty values.
        :rtype: tf.constant
        """

        # Evaluate MLP Jacobian on boundary data.
        y_pred_norm, dy_pred_norm = self.ComputeFirstOrderDerivatives(x_norm_boundary, self.idx_PIvar[iVar])

        # Project Jacobian along boundary data according to penalty function.
        project_dy_pred_norm = tf.norm(tf.multiply(precon_gradient[:,:,iVar], dy_pred_norm), axis=1)

        # Compute direct and Neumann penalty values.
        penalty_direct = tf.pow(y_pred_norm[:,self.idx_PIvar[iVar]] - y_norm_boundary_target[:,iVar], 2)
        penalty_gradient = tf.pow(project_dy_pred_norm - dy_norm_boundary_target[:, iVar], 2)
        penalty = tf.reduce_mean(penalty_direct + penalty_gradient)
        return penalty
    
    
    @tf.function
    def Train_Step(self, X_domain_batch:tf.constant, Y_domain_batch:tf.constant, \
                   X_boundary_batch:tf.constant, Y_boundary_batch:tf.constant, \
                    P_boundary_batch:tf.constant, Yt_boundary_batch:tf.constant, vals_lambda:list[tf.constant]):
        """Gradient descend update function for training physics-informed FGM MLP.

        :param X_domain_batch: normalized controlling variable array from the domain data.
        :type X_domain_batch: tf.constant
        :param Y_domain_batch: normalized labeled output array from the domain data.
        :type Y_domain_batch: tf.constant
        :param X_boundary_batch: normalized controlling variable array from the boundary data.
        :type X_boundary_batch: tf.constant
        :param Y_boundary_batch: normalized labeled output array from the boundary data.
        :type Y_boundary_batch: tf.constant
        :param P_boundary_batch: Jacobian projection vector for current training batch.
        :type P_boundary_batch: tf.constant
        :param Yt_boundary_batch: target projection vector for current training batch.
        :type Yt_boundary_batch: tf.constant
        :param vals_lambda: boundary condition weights.
        :type vals_lambda: list[tf.constant]
        :return: training penalty value.
        :rtype: tf.constant
        """

        # Compute training loss for the current batch and extract HP sensitivities.
        batch_loss, sens_batch = self.Train_sensitivity_function(X_domain_batch, Y_domain_batch, X_boundary_batch, Y_boundary_batch, P_boundary_batch, Yt_boundary_batch, vals_lambda)
        
        # Update network weigths and biases.
        self.UpdateWeights(sens_batch)

        return batch_loss
    
    @tf.function 
    def Train_sensitivity_function(self, X_domain_batch, Y_domain_batch, X_boundary_batch, Y_boundary_batch, P_boundary_batch, Yt_boundary_batch, vals_lambda):
        with tf.GradientTape() as tape:
            tape.watch(self._trainable_hyperparams)
            train_loss = self.Train_loss_function(X_domain_batch, Y_domain_batch, X_boundary_batch, Y_boundary_batch, P_boundary_batch, Yt_boundary_batch, vals_lambda)
            grads_loss = tape.gradient(train_loss[0], self._trainable_hyperparams)
        return train_loss, grads_loss 
    
    @tf.function 
    def Train_loss_function(self, X_domain_batch, Y_domain_batch, X_boundary_batch, Y_boundary_batch, P_boundary_batch, Yt_boundary_batch, vals_lambda):
        
        domain_loss = self.TrainingLoss_error(X_domain_batch, Y_domain_batch)

        boundary_loss = 0.0
        for iBc in range(self._N_bc):
            bc_loss = self.ComputeNeumannPenalty(X_boundary_batch, Y_boundary_batch, Yt_boundary_batch, P_boundary_batch,iBc)
            boundary_loss += vals_lambda[iBc] * bc_loss

        total_loss = domain_loss + boundary_loss
        return [total_loss, domain_loss, bc_loss]
    
    @tf.function
    def ComputeGradients_Boundary_Error(self, X_boundary_batch, Y_boundary_batch, Yt_boundary_batch,P_boundary_batch,iVar):
        with tf.GradientTape() as tape:
            tape.watch(self._trainable_hyperparams)
            neumann_penalty_var = self.ComputeNeumannPenalty(X_boundary_batch, Y_boundary_batch, Yt_boundary_batch, P_boundary_batch,iVar)
            grads_neumann = tape.gradient(neumann_penalty_var, self._trainable_hyperparams)
        return neumann_penalty_var, grads_neumann
    
    @tf.function
    def ComputeGradients(self, X_domain_batch, Y_domain_batch, X_boundary_batch, Y_boundary_batch, P_boundary_batch, Yt_boundary_batch):
        y_domain_loss, grads_domain = self.ComputeGradients_Direct_Error(X_domain_batch, Y_domain_batch)
        grads_bc_list = []
        loss_bc_list = []

        for iBC in range(self._N_bc):
            boundary_loss, grads_boundary_loss = self.ComputeGradients_Boundary_Error(X_boundary_batch, Y_boundary_batch, Yt_boundary_batch,P_boundary_batch,iBC)
            grads_bc_list.append(grads_boundary_loss)
            loss_bc_list.append(boundary_loss)
        return y_domain_loss, grads_domain, loss_bc_list, grads_bc_list
    
    @tf.function 
    def UpdateLambdas(self, X_domain_batch, Y_domain_batch, X_boundary_batch, Y_boundary_batch, P_boundary_batch, Yt_boundary_batch, vals_lambda_old):
        _, grads_domain, _, grads_bc_list = self.ComputeGradients(X_domain_batch, Y_domain_batch, X_boundary_batch, Y_boundary_batch, P_boundary_batch, Yt_boundary_batch)
        vals_lambda_new = []
        for iBc, lambda_old in enumerate(vals_lambda_old):
            lambda_new = self.update_lambda(grads_domain, grads_bc_list[iBc], lambda_old)
            vals_lambda_new.append(lambda_new)
        return vals_lambda_new
    
    @tf.function
    def UpdateWeights(self, grads):
        self._optimizer.apply_gradients(zip(grads, self._trainable_hyperparams))
        return
    
    def GenerateBoundaryMatrices(self):
        print("Generating boundary data matrix...")
        mixfrac_range = np.linspace(0, 1, self.__Config.GetNpMix())
        T_range = np.linspace(self.__Config.GetUnbTempBounds()[0], self.__Config.GetUnbTempBounds()[1], self.__Config.GetNpTemp())
        self.pv_unb = np.zeros([len(mixfrac_range), len(T_range)])
        self.h_unb = np.zeros([len(mixfrac_range), len(T_range)])
        self.z_unb = np.zeros([len(mixfrac_range), len(T_range)])
        self.pv_b = np.zeros([len(mixfrac_range), len(T_range)])
        self.h_b = np.zeros([len(mixfrac_range), len(T_range)])
        self.z_b = np.zeros([len(mixfrac_range), len(T_range)])

        gas_unb = ct.Solution(self.__Config.GetReactionMechanism())
        gas_b = ct.Solution(self.__Config.GetReactionMechanism())
        gas_unb.set_equivalence_ratio(1.0, self.__Config.GetFuelString(), self.__Config.GetOxidizerString())
        gas_b.set_equivalence_ratio(1.0, self.__Config.GetFuelString(), self.__Config.GetOxidizerString())
        gas_unb.TP = T_range[0], DefaultProperties.pressure
        gas_b.TP = T_range[0], DefaultProperties.pressure
        
        for iZ, Z in enumerate(mixfrac_range):
            gas_unb.TP = T_range[0], DefaultProperties.pressure
            gas_b.TP = T_range[0], DefaultProperties.pressure
            gas_unb.set_mixture_fraction(Z, self.__Config.GetFuelString(), self.__Config.GetOxidizerString())
            gas_b.set_mixture_fraction(Z, self.__Config.GetFuelString(), self.__Config.GetOxidizerString())
            gas_b.equilibrate("HP")
            for iT, T in enumerate(T_range):
                gas_unb.TP = T, DefaultProperties.pressure
                self.pv_unb[iZ, iT] = self.__Config.ComputeProgressVariable(variables=None, flamelet_data=None, Y_flamelet=gas_unb.Y[:,np.newaxis])[0]
                self.h_unb[iZ, iT] = gas_unb.enthalpy_mass
                self.z_unb[iZ, iT] = Z 

                gas_b.TP = T, DefaultProperties.pressure
                self.pv_b[iZ, iT] = self.__Config.ComputeProgressVariable(variables=None, flamelet_data=None, Y_flamelet=gas_b.Y[:,np.newaxis])[0]
                self.h_b[iZ, iT] = gas_b.enthalpy_mass
                self.z_b[iZ, iT] = Z 
        print("Done!")
        
        return 
    
    def PlotUnbData(self):
        pv_unb = self.pv_unb.flatten()
        h_unb = self.h_unb.flatten()
        z_unb = self.z_unb.flatten()
        
        pv_b = self.pv_b.flatten()
        h_b = self.h_b.flatten()
        z_b = self.z_b.flatten()
        
        X_unb = np.hstack((pv_unb[:, np.newaxis], h_unb[:,np.newaxis], z_unb[:,np.newaxis]))
        X_unb_norm_np = self.scaler_function_x.transform(X_unb)
        X_b = np.hstack((pv_b[:, np.newaxis], h_b[:,np.newaxis], z_b[:,np.newaxis]))
        X_b_norm_np = self.scaler_function_x.transform(X_b)
        
        X_unb_norm = tf.constant(X_unb_norm_np,dtype=self._dt)
        X_b_norm = tf.constant(X_b_norm_np,dtype=self._dt)
        
        Y_unb_pred_norm = self._MLP_Evaluation(X_unb_norm).numpy()
        Y_b_pred_norm = self._MLP_Evaluation(X_b_norm).numpy()

        Y_unb_pred = self.scaler_function_y.inverse_transform(Y_unb_pred_norm)
        Y_b_pred = self.scaler_function_y.inverse_transform(Y_b_pred_norm)

        for iBC in range(self._N_bc):
            iVar = self.idx_PIvar[iBC]
            Y_plot_unb = Y_unb_pred[:, iVar].reshape(np.shape(self.pv_unb))
            fig = plt.figure(figsize=[10,10])
            ax = plt.axes()
            cont = ax.contourf(self.h_unb, self.z_unb, Y_plot_unb, levels=20)
            ax.set_xlabel(r"Total enthalpy $(h)[J/kg]$",fontsize=20)
            ax.set_ylabel(r"Mixture fraction $(Z)[-]$",fontsize=20)
            cbar = plt.colorbar(cont)
            ax.tick_params(which='both',labelsize=18)
            ax.set_title(r""+self._train_vars[iVar] + r" Prediction along reactants",fontsize=20)
            fig.savefig(self._save_dir+"/Model_"+str(self._model_index)+"/Unburnt_Pred_"+self._train_vars[iVar] +"."+self._figformat,format=self._figformat,bbox_inches='tight')
            plt.close(fig)

            Y_plot_b = Y_b_pred[:, iVar].reshape(np.shape(self.pv_unb))
            fig = plt.figure(figsize=[10,10])
            ax = plt.axes()
            cont = ax.contourf(self.h_b, self.z_b, Y_plot_b, levels=20)
            ax.set_xlabel(r"Total enthalpy $(h)[J/kg]$",fontsize=20)
            ax.set_ylabel(r"Mixture fraction $(Z)[-]$",fontsize=20)
            cbar = plt.colorbar(cont)
            ax.tick_params(which='both',labelsize=18)
            ax.set_title(r""+self._train_vars[iVar] + r" Prediction along products",fontsize=20)
            fig.savefig(self._save_dir+"/Model_"+str(self._model_index)+"/Burnt_Pred_"+self._train_vars[iVar] +"."+self._figformat,format=self._figformat,bbox_inches='tight')
            plt.close(fig)
        return 
    
    def PlotLambdaHistory(self):
        fig = plt.figure(figsize=[10,10])
        ax = plt.axes()
        for iBc in range(self._N_bc):
            ax.plot([self.lambda_history[i][iBc] for i in range(len(self.lambda_history))], label=self._train_vars[self.idx_PIvar[iBc]])
        ax.grid()
        ax.set_xlabel(r"Epoch [-]",fontsize=20)
        ax.set_ylabel(r"Lambda value[-]",fontsize=20)
        ax.tick_params(which='both',labelsize=18)
        ax.legend(fontsize=20)
        ax.set_title(r"Neumann penalty modifier history", fontsize=20)
        fig.savefig(self._save_dir+"/Model_"+str(self._model_index)+"/Lambda_history."+self._figformat,format=self._figformat,bbox_inches='tight')
        plt.close(fig)
        return 
    
    def CustomCallback(self):
        self.Plot_and_Save_History()
        PlotFlameletData(self, self.__Config, self._train_name)
        self.PlotUnbData()
        self.PlotLambdaHistory()
        self.write_SU2_MLP(self._save_dir + "/Model_"+str(self._model_index)+"/MLP_"+self._train_name)
        return super().CustomCallback()
    
    def add_additional_header_info(self, fid):
        fid.write("Progress variable definition: " + "+".join(("%+.6e*%s" % (w, s)) for w, s in zip(self.__Config.GetProgressVariableWeights(), self.__Config.GetProgressVariableSpecies())))
        fid.write("\n\n")
        return super().add_additional_header_info(fid)
    
class EvaluateArchitecture_FGM(EvaluateArchitecture):
    """Class for training MLP architectures
    """

    __output_group:int = 0
    __group_name:str = "Group1"
    __trainer_PINN:Train_Source_PINN = None 
    __kind_trainer:str = "direct"

    __PINN_variables:list[str] = ["Temperature", "ProdRateTot_PV", "Heat_Release", "Beta_ProgVar", "Beta_MixFrac"]
    def __init__(self, Config:FlameletAIConfig, group_idx:int=0):
        """Define EvaluateArchitecture instance and prepare MLP trainer with
        default settings.

        :param Config: FlameletAIConfig object describing the flamelet data manifold.
        :type Config: FlameletAIConfig
        :param group_idx: MLP output group index, defaults to 0
        :type group_idx: int, optional
        :raises Exception: if MLP output group index is undefined by flameletAI configuration.
        """
        output_vars = Config.GetMLPOutputGroup(group_idx)
        if any([((v in self.__PINN_variables) or ("Y_dot" in v)) for v in output_vars]):
            self.__kind_trainer = "physicsinformed"
            self.__trainer_PINN = Train_FGM_PINN(Config_in=Config,group_idx=group_idx)
            self._trainer_direct = self.__trainer_PINN
        else:
            self._trainer_direct = Train_Flamelet_Direct(Config_in=Config, group_idx=group_idx)

        #self._trainer_direct = Train_Flamelet_Direct(Config_in=Config, group_idx=group_idx)
        self.__output_group=group_idx
        EvaluateArchitecture.__init__(self, Config_in=Config)
        self.SetOutputGroup(group_idx)
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
    
    def SetBoundaryDataFile(self, boundary_data_file:str):
        if self.__kind_trainer != "physicsinformed" and self.verbose > 0:
            print("Boundary data will be ignored.")
        else:
            self.__trainer_PINN.SetBoundaryDataFile(boundary_data_file)
        return 
    
    def CommenceTraining(self):
        if self.__kind_trainer == "physicsinformed":
            self.__trainer_PINN.InitializeWeights_and_Biases()
        return super().CommenceTraining()
    

def PlotFlameletData(Trainer:MLPTrainer, Config:FlameletAIConfig, train_name:str):
    N_plot = 3

    flamelet_dir = Config.GetOutputDir()

    freeflame_phis = os.listdir(flamelet_dir + "/freeflame_data/")
    idx_phi_plot = np.random.randint(0, len(freeflame_phis), N_plot)
    
    freeflamelet_input_files = []
    for phi in idx_phi_plot:
        freeflame_files = os.listdir(flamelet_dir + "/freeflame_data/"+freeflame_phis[phi])
        freeflamelet_input_files.append(flamelet_dir + "/freeflame_data/"+freeflame_phis[phi]+ "/"+freeflame_files[np.random.randint(0, len(freeflame_files))])

    # Prepare a figure window for each output variable.
    figs = []
    axs = []
    for _ in Trainer._train_vars:
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
        CV_flamelet = np.zeros([len(flameletData), len(Trainer._controlling_vars)])
        for iCv, Cv in enumerate(Trainer._controlling_vars):
            if Cv == DefaultProperties.name_pv:
                CV_flamelet[:, iCv] = Config.ComputeProgressVariable(variables_flamelet, flameletData)
            else:
                CV_flamelet[:, iCv] = flameletData[:, variables_flamelet.index(Cv)]
        
        CV_flamelet_norm = Trainer.scaler_function_x.transform(CV_flamelet)#(CV_flamelet - Trainer._X_min)/(Trainer._X_max - Trainer._X_min)
        
        ref_data_flamelet = np.zeros([len(flameletData), len(Trainer._train_vars)])

        # Collect prediction variables from flamelet data. 
        for iVar, Var in enumerate(Trainer._train_vars):
            if "Beta_" in Var:
                beta_pv, beta_enth_thermal, beta_enth, beta_mixfrac = Config.ComputeBetaTerms(variables_flamelet, flameletData)
            if Var == "Beta_ProgVar":
                ref_data_flamelet[:, iVar] = beta_pv 
            elif Var == "Beta_Enth_Thermal":
                ref_data_flamelet[:, iVar] = beta_enth_thermal
            elif Var == "Beta_Enth":
                ref_data_flamelet[:, iVar] = beta_enth 
            elif Var == "Beta_MixFrac":
                ref_data_flamelet[:, iVar] = beta_mixfrac 
            elif Var == "ProdRateTot_PV":
                ref_data_flamelet[:, iVar] = Config.ComputeProgressVariable_Source(variables_flamelet, flameletData)
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
        pred_data_norm = Trainer.EvaluateMLP(CV_flamelet_norm)
        pred_data = Trainer.scaler_function_y.inverse_transform(pred_data_norm)#(Trainer._Y_max - Trainer._Y_min) * pred_data_norm + Trainer._Y_min
        # Plot flamelet data in corresponding figure window.
        for iVar, Var in enumerate(Trainer._train_vars):
            axs[iVar].plot(CV_flamelet[:, 0], ref_data_flamelet[:, iVar], 'bs-', linewidth=2, markevery=10, markerfacecolor='none', markersize=12, label=plot_label_ref)
            axs[iVar].plot(CV_flamelet[:, 0], pred_data[:, iVar], 'ro--', linewidth=1, markevery=10, markersize=10, label=plot_label_MLP)
        plot_label_MLP = ""
        plot_label_ref = ""
    for iVar, Var in enumerate(Trainer._train_vars):
        axs[iVar].set_xlabel(r"Progress Variable $(\mathcal{Y})[-]$", fontsize=20)
        axs[iVar].set_ylabel(r"" + Var, fontsize=20)
        axs[iVar].tick_params(which='both', labelsize=18)
        axs[iVar].legend(fontsize=20)
        axs[iVar].grid()
        figs[iVar].savefig(Trainer._save_dir + "/Model_"+str(Trainer._model_index) + "/flameletdata_"+train_name+"_" + Var + "."+Trainer._figformat, format=Trainer._figformat, bbox_inches='tight')
        plt.close(figs[iVar])