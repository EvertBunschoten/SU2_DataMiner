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

from Common.EntropicAIConfig import EntropicAIConfig,FlameletAIConfig
from Common.Properties import DefaultProperties 
from Common.CommonMethods import GetReferenceData
from Trainer_Base import MLPTrainer, TensorFlowFit,PhysicsInformedTrainer

class Train_Entropic_Direct(TensorFlowFit):

    def __init__(self):
        TensorFlowFit.__init__(self)
        # Set train variables based on what kind of network is trained

        self._controlling_vars = ["Density", "Energy"]
        self._train_vars = ["s"]
        return

class Train_Entropic_PINN(PhysicsInformedTrainer):

    __idx_rho:int
    __idx_e:int 
    __idx_T:int
    __idx_p:int 
    __idx_c2:int 

    __s_min:float 
    __s_max:float 
    __rho_min:float 
    __rho_max:float 
    __e_min:float 
    __e_max:float 

    T_test_loss:float 
    P_test_loss:float 
    C2_test_loss:float 

    def __init__(self):
        PhysicsInformedTrainer.__init__(self)
        self._controlling_vars = ["Density","Energy"]
        self._train_vars=["T","p","c2"]

        self.__idx_rho = self._controlling_vars.index("Density")
        self.__idx_e = self._controlling_vars.index("Energy")
        self.__idx_T = self._train_vars.index("T")
        self.__idx_p = self._train_vars.index("p")
        self.__idx_c2 = self._train_vars.index("c2")
        
        return 
    
    def GetTrainData(self):
        print("Reading train, test, and validation data...")
        X_full, Y_full = GetReferenceData(self._filedata_train+"_full.csv", self._controlling_vars, ["s"])
        print("Done!")

        self.__s_min, self.__s_max = min(Y_full[:, 0]), max(Y_full[:, 0])

        # Free up memory
        del X_full
        del Y_full

        super().GetTrainData()

        self.__rho_min, self.__rho_max = self._X_min[self.__idx_rho], self._X_max[self.__idx_rho]
        self.__e_min, self.__e_max = self._X_min[self.__idx_e], self._X_max[self.__idx_e]
        return 
    
    @tf.function
    def CollectVariables(self):
        """Define weights and biases as trainable hyper-parameters.
        """
        self.__trainable_hyperparams = []
        for W in self._weights:
            self.__trainable_hyperparams.append(W)
        for b in self._biases[:-1]:
            self.__trainable_hyperparams.append(b)
        return 
    
    @tf.function
    def __ComputeEntropyGradients(self, rhoe_norm:tf.Tensor):
        with tf.GradientTape() as tape_2:
            tape_2.watch(rhoe_norm)
            with tf.GradientTape() as tape_1:
                tape_1.watch(rhoe_norm)
                s_norm = self._MLP_Evaluation(rhoe_norm)
                ds_norm = tape_1.gradient(s_norm, rhoe_norm)
                dsdrho_e_norm = tf.gather(ds_norm, indices=self.__idx_rho,axis=1)
            d2s_norm = tape_2.gradient(dsdrho_e_norm, rhoe_norm)
        d2sdrho2_norm = tf.gather(d2s_norm,indices=self.__idx_rho,axis=1)
        d2sdrhode_norm = tf.gather(d2s_norm, indices=self.__idx_e, axis=1)
        with tf.GradientTape() as tape_2:
            tape_2.watch(rhoe_norm)
            with tf.GradientTape() as tape_1:
                tape_1.watch(rhoe_norm)
                s_norm = self._MLP_Evaluation(rhoe_norm)
                ds_norm = tape_1.gradient(s_norm, rhoe_norm)
                dsde_rho_norm = tf.gather(ds_norm, indices=self.__idx_e,axis=1)
            d2s_norm = tape_2.gradient(dsdrho_e_norm, rhoe_norm)
        d2sde2_norm = tf.gather(d2s_norm, indices=self.__idx_e,axis=1)

        s_scale = self.__s_max - self.__s_min
        rho_scale = self.__rho_max - self.__rho_min
        e_scale = self.__e_max - self.__e_min
        s_dim = s_scale * s_norm + self.__s_min 
        dsdrho_e = (s_scale / rho_scale) * dsdrho_e_norm
        dsde_rho = (s_scale / e_scale) * dsde_rho_norm 
        d2sdrho2 = (s_scale / tf.pow(rho_scale, 2)) * d2sdrho2_norm 
        d2sdedrho = (s_scale / (rho_scale * e_scale)) * d2sdrhode_norm 
        d2sde2 = (s_scale / tf.pow(e_scale, 2)) * d2sde2_norm 
        return s_dim, dsdrho_e, dsde_rho, d2sdrho2, d2sdedrho, d2sde2 
    
    @tf.function 
    def __EntropicEOS(rho, dsdrho_e, dsde_rho, d2sdrho2, d2sdedrho, d2sde2):
        T = 1.0 / dsde_rho
        P = -tf.pow(rho, 2) * T * dsdrho_e
        blue_term = (dsdrho_e * (2 - rho * tf.pow(dsde_rho, -1) * d2sdedrho) + rho*d2sdrho2)
        green_term = (-tf.pow(dsde_rho, -1) * d2sde2 * dsdrho_e + d2sdedrho)
        c2 = -rho * tf.pow(dsde_rho, -1) * (blue_term - rho * green_term * (dsdrho_e / dsde_rho))
  
        return T, P, c2
    
    @tf.function 
    def __TD_Evaluation(self, rhoe_norm:tf.Tensor):
        s, dsdrho_e, dsde_rho, d2sdrho2, d2sdedrho, d2sde2 = self.__ComputeEntropyGradients(rhoe_norm)
        rho_norm = tf.gather(rhoe_norm, indices=self.__idx_rho, axis=1)
        rho = (self.__rho_max - self.__rho_min)*rho_norm + self.__rho_min 
        T, P, c2 = self.__EntropicEOS(rho, dsdrho_e, dsde_rho, d2sdrho2, d2sdedrho, d2sde2)
        return s, T, P, c2 
    
    @tf.function 
    def __Compute_S_error(self, S_label:tf.Tensor, x_var:tf.Variable):
        """Compute the temperature prediction error.

        :param T_label: reference temperature data
        :type T_label: tf.Tensor
        :param x_var: normalized density and energy tensor.
        :type x_var: tf.Variable
        :return: mean squared error between predicted and reference temperature.
        :rtype: tf.Tensor
        """

        # Evaluate thermodynamic state.
        S_norm = self._MLP_Evaluation(x_var)

        S_label_norm = (S_label - self.__s_min) / (self.__s_max - self.__s_min)

        # Apply loss function.
        S_error = self.mean_square_error(y_true=S_label_norm, y_pred=S_norm)

        return S_error
    
    @tf.function
    def __Compute_T_error(self, T_label_norm:tf.Tensor, x_var:tf.Variable):
        """Compute the temperature prediction error.

        :param T_label: reference temperature data
        :type T_label: tf.Tensor
        :param x_var: normalized density and energy tensor.
        :type x_var: tf.Variable
        :return: mean squared error between predicted and reference temperature.
        :rtype: tf.Tensor
        """

        # Evaluate thermodynamic state.
        _, T, _, _ = self.__TD_Evaluation(x_var)

        # Normalize reference and predicted temperature.
        T_pred_norm = (T - self._Y_min[self.__idx_T]) / (self._Y_max[self.__idx_T] - self._Y_min[self.__idx_T])
        
        # Apply loss function.
        T_error = self.mean_square_error(y_true=T_label_norm, y_pred=T_pred_norm)

        return T_error
    
    @tf.function
    def __Compute_P_error(self, P_label_norm:tf.Tensor,x_var:tf.Variable):
        """Compute the pressure prediction error.

        :param P_label: reference pressure data
        :type P_label: tf.Tensor
        :param x_var: normalized density and energy tensor.
        :type x_var: tf.Variable
        :return: mean squared error between predicted and reference pressure.
        :rtype: tf.Tensor
        """

        # Evaluate thermodynamic state.
        _, _, P, _ = self.__TD_Evaluation(x_var)

        # Normalize reference and predicted pressure.
        P_pred_norm = (P - self._Y_min[self.__idx_p]) / (self._Y_max[self.__idx_p] - self._Y_min[self.__idx_p])
        
        # Apply loss function.
        P_error = self.mean_square_error(y_true=P_label_norm, y_pred=P_pred_norm)

        return P_error 
    
    @tf.function
    def __Compute_C2_error(self, C2_label_norm,x_var:tf.Variable):
        """Compute the prediction error for squared speed of sound (SoS).

        :param C2_label: reference pressure data
        :type C2_label: tf.Tensor
        :param x_var: normalized density and energy tensor.
        :type x_var: tf.Variable
        :return: mean squared error between predicted and reference squared speed of sound.
        :rtype: tf.Tensor
        """

        # Evaluate thermodynamic state.
        _, _, _, C2 = self.__TD_Evaluation(x_var)
        
        # Normalize reference and predicted squared SoS.
        C2_pred_norm = (C2 - self._Y_min[self.__idx_c2]) / (self._Y_max[self.__idx_c2] - self._Y_min[self.__idx_c2])

        # Apply loss function.
        C2_error = self.mean_square_error(y_true=C2_label_norm, y_pred=C2_pred_norm)

        return C2_error
    
    @tf.function
    def __ComputeGradients_T_error(self, T_label_norm:tf.Tensor, rhoe_norm:tf.Variable):
        """Compute temperature prediction error and respective MLP weight sensitivities.

        :param T_label: reference temperature data
        :type T_label: tf.Tensor
        :param rhoe_norm: normalized density and energy tensor.
        :type rhoe_norm: tf.Variable
        :return: temperature prediction loss value and MLP weight sensitivities.
        :rtype: tf.Tensor
        """

        with tf.GradientTape() as tape:
            tape.watch(self.__trainable_hyperparams)
            # Evaluate temperature loss value.
            T_loss = self.__Compute_T_error(T_label_norm, rhoe_norm)
            
            # Compute MLP weight sensitvities.
            grads_T = tape.gradient(T_loss, self.__trainable_hyperparams)
        
        return T_loss, grads_T

    @tf.function
    def __ComputeGradients_P_error(self, P_label_norm:tf.Tensor, rhoe_norm:tf.Variable):
        """Compute pressure prediction error and respective MLP weight sensitivities.

        :param P_label: reference pressure data
        :type P_label: tf.Tensor
        :param rhoe_norm: normalized density and energy tensor.
        :type rhoe_norm: tf.Variable
        :return: pressure prediction loss value and MLP weight sensitivities.
        :rtype: tf.Tensor
        """
        with tf.GradientTape() as tape:
            tape.watch(self.__trainable_hyperparams)
            # Evaluate pressure loss value.
            P_loss = self.__Compute_P_error(P_label_norm, rhoe_norm)

            # Compute MLP weight sensitvities.
            grads_P = tape.gradient(P_loss, self.__trainable_hyperparams)

        return P_loss, grads_P
    
    @tf.function
    def __ComputeGradients_C2_error(self, C2_label_norm:tf.Tensor, rhoe_norm:tf.Variable):
        """Compute SoS prediction error and respective MLP weight sensitivities.

        :param C2_label: reference squared SoS data
        :type C2_label: tf.Tensor
        :param rhoe_norm: normalized density and energy tensor.
        :type rhoe_norm: tf.Variable
        :return: SoS prediction loss value and MLP weight sensitivities.
        :rtype: tf.Tensor
        """
        with tf.GradientTape() as tape:
            tape.watch(self.__trainable_hyperparams)
            C2_loss = self.__Compute_C2_error(C2_label_norm, rhoe_norm)
            grads_C2 = tape.gradient(C2_loss, self.__trainable_hyperparams)
        return C2_loss, grads_C2
    
    @tf.function
    def Train_Step(self, X_batch_norm:tf.constant, Y_batch_norm:tf.constant):

        T_batch_norm = tf.gather(Y_batch_norm,indices=self.__idx_T, axis=1)
        P_batch_norm = tf.gather(Y_batch_norm,indices=self.__idx_p, axis=1)
        C2_batch_norm = tf.gather(Y_batch_norm,indices=self.__idx_c2, axis=1)
        
        # Weight update for temperature prediction.
        T_loss, grads_T = self.__ComputeGradients_T_error(T_batch_norm, X_batch_norm)
        self.__optimizer.apply_gradients(zip(grads_T, self.__trainable_hyperparams))

        # Weight update for pressure prediction.
        P_loss, grads_P = self.__ComputeGradients_P_error(P_batch_norm,X_batch_norm)
        self.__optimizer.apply_gradients(zip(grads_P, self.__trainable_hyperparams))

        # Weight update for SoS prediction.
        C2_loss, grads_C2 = self.__ComputeGradients_C2_error(C2_batch_norm,X_batch_norm)
        self.__optimizer.apply_gradients(zip(grads_C2, self.__trainable_hyperparams))
  
        return T_loss, P_loss, C2_loss

    def __ValidationLoss(self):
        rhoe_val_norm = tf.constant(self._X_val_norm, self.__dt)
        T_val_error = self.__Compute_T_error(rhoe_val_norm)
        p_val_error = self.__Compute_P_error(rhoe_val_norm)
        c2_val_error = self.__Compute_C2_error(rhoe_val_norm)

        self.val_loss_history[self.__idx_T].append(T_val_error)
        self.val_loss_history[self.__idx_p].append(p_val_error)
        self.val_loss_history[self.__idx_c2].append(c2_val_error)

        return T_val_error, p_val_error, c2_val_error
    
    def __TestLoss(self):
        
        rhoe_test_norm = tf.constant(self._X_test_norm, self.__dt)
        self.T_test_loss = self.__Compute_T_error(rhoe_test_norm).numpy()
        self.P_test_loss = self.__Compute_P_error(rhoe_test_norm).numpy()
        self.C2_test_loss = self.__Compute_C2_error(rhoe_test_norm).numpy()
        self._test_score = max([self.T_test_loss, self.P_test_loss, self.C2_test_loss])

        return 
    
    def CustomCallback(self):

        # Save SU2 MLP and performance metrics.
        self.Save_Relevant_Data()

        # Plot intermediate history trends.
        self.Plot_and_Save_History()
        # Error scatter plots for thermodynamic properties.
        self.__Generate_Error_Plots()

        return super().CustomCallback()
    
    def __Generate_Error_Plots(self):
        """Make nice plots of the interpolated test data.
        """

        self.PlotR2Data()
        
        s_test_pred, T_test_pred, P_test_pred, C2_test_pred = self.__TD_Evaluation(self._X_test_norm)

        figformat = "png"
        plot_fontsize = 20
        label_fontsize=18

        rho_test = self._X_test[:, self.__idx_rho]#(self._X_max[self.__idx_rho] - self._X_min[self.__idx_rho])*self._X_test_norm[:, self.__idx_rho] + self._X_min[self.__idx_rho]
        e_test = self._X_test[:, self.__idx_e]#(self._X_max[self.__idx_e] - self._X_min[self.__idx_e])*self._X_test_norm[:, self.__idx_e] + self._X_min[self.__idx_e]
        
        T_test = self._Y_test[:, self.__idx_T]
        P_test = self._Y_test[:, self.__idx_p]
        C2_test = self._Y_test[:, self.__idx_c2]

        fig = plt.figure(figsize=[10,10])
        ax = plt.axes() 
        cax = ax.scatter(rho_test, e_test, c=100*np.abs((T_test_pred.numpy() - T_test)/T_test))
        cbar = plt.colorbar(cax, ax=ax)
        #cbar.set_label(r'Temperature prediction error $(\epsilon_T)[\%]$', rotation=270, fontsize=label_fontsize)
        ax.set_xlabel(r"Density $(\rho)[kg m^{-3}]$",fontsize=plot_fontsize)
        ax.set_ylabel(r"Static energy $(e)[J kg^{-1}]$",fontsize=plot_fontsize)
        ax.set_title(r"Temperature prediction error",fontsize=plot_fontsize)
        ax.tick_params(which='both',labelsize=label_fontsize)
        fig.savefig(self._save_dir + "/Model_"+str(self._model_index)+"/Temperature_prediction_error."+figformat,format=figformat,bbox_inches='tight')
        plt.close(fig)

        fig = plt.figure(figsize=[10,10])
        ax = plt.axes() 
        cax = ax.scatter(rho_test, e_test, c=100*np.abs((P_test_pred.numpy() - P_test)/P_test))
        cbar = plt.colorbar(cax, ax=ax)
        cbar.set_label(r'Pressure prediction error $(\epsilon_p)[\%]$', rotation=270, fontsize=label_fontsize)
        ax.set_xlabel("Normalized Density",fontsize=plot_fontsize)
        ax.set_ylabel("Normalized Energy",fontsize=plot_fontsize)
        ax.set_title("Pressure prediction error",fontsize=plot_fontsize)
        ax.tick_params(which='both',labelsize=label_fontsize)
        fig.savefig(self._save_dir + "/Model_"+str(self._model_index)+"/Pressure_prediction_error."+figformat,format=figformat,bbox_inches='tight')
        plt.close(fig)

        fig = plt.figure(figsize=[10,10])
        ax = plt.axes() 
        cax = ax.scatter(rho_test, e_test, c=100*np.abs((C2_test_pred.numpy() - C2_test)/C2_test))
        cbar = plt.colorbar(cax, ax=ax)
        ax.set_xlabel("Normalized Density",fontsize=plot_fontsize)
        ax.set_ylabel("Normalized Energy",fontsize=plot_fontsize)
        ax.set_title("SoS prediction error",fontsize=plot_fontsize)
        ax.tick_params(which='both',labelsize=label_fontsize)
        fig.savefig(self._save_dir + "/Model_"+str(self._model_index)+"/SoS_prediction_error."+figformat,format=figformat,bbox_inches='tight')
        plt.close(fig)

        fig = plt.figure(figsize=[10,10])
        ax = plt.axes() 
        ax.plot(P_test, T_test, 'b.',markersize=3,markerfacecolor='none',label=r'Labeled')
        ax.plot(P_test_pred.numpy(), T_test_pred.numpy(), 'r.',label=r'Predicted')
        ax.set_xscale('log')
        ax.grid()
        ax.legend(fontsize=20)
        ax.set_xlabel(r"Pressure $(p)[Pa]$",fontsize=plot_fontsize)
        ax.set_ylabel(r"Temperature $(T)[K]$",fontsize=plot_fontsize)
        ax.set_title(r"PT diagram",fontsize=plot_fontsize)
        ax.tick_params(which='both',labelsize=label_fontsize)
        fig.savefig(self._save_dir + "/Model_"+str(self._model_index)+"/PT_diagram."+figformat,format=figformat,bbox_inches='tight')
        plt.close(fig)

        return
    
    
    
class Train_Flamelet_Direct(TensorFlowFit):
    __Config:FlameletAIConfig
    _train_name:str 

    def __init__(self, Config_in:FlameletAIConfig, group_idx:int=0):
        TensorFlowFit.__init__(self)
        # Set train variables based on what kind of network is trained
        self.__Config = Config_in
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
    

    
class EvaluateArchitecture:
    """Class for training MLP architectures
    """

    __Config:EntropicAIConfig
    __trainer_entropy:Train_Entropic_MLP
    __trainer_tpc2:Train_C2_MLP      # MLP trainer object responsible for training itself.

    architecture:list[int] = [DefaultProperties.NN_hidden]   # Hidden layer architecture.
    alpha_expo:float = DefaultProperties.init_learning_rate_expo # Initial learning rate exponent (base 10)
    lr_decay:float = DefaultProperties.learning_rate_decay # Learning rate decay parameter.
    batch_expo:int = DefaultProperties.batch_size_exponent      # Mini-batch exponent (base 2)
    activation_function:str = "exponential"    # Activation function name applied to hidden layers.

    n_epochs:int = DefaultProperties.N_epochs # Number of epochs to train for.
    save_dir:str        # Directory to save trained networks in.

    device:str = "CPU"      # Hardware to train on.
    process_index:int = 0   # Hardware index.

    current_iter:int=0
    __test_score:float        # MLP evaluation score on test set.
    __cost_parameter:float    # MLP evaluation cost parameter.

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

        self.__trainer_entropy.SetActivationFunction(self.activation_function)
        self.__trainer_tpc2.SetActivationFunction(self.activation_function)
        
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
        return 
    
    def SetNEpochs(self, n_epochs:int=DefaultProperties.N_epochs):
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
    
    def SetActivationFunction(self, activation_function_name:str):
        """Define hidden layer activation function.

        :param activation_function_name: activation function name.
        :type activation_function_name: str
        :raises Exception: if activation function is not in the list of supported functions.
        """
        if activation_function_name not in self.activation_function_options:
            raise Exception("Activation function name should be one of the following: "\
                             + ",".join(p for p in self.activation_function_options))
        self.activation_function = activation_function_name
        self.__trainer_tpc2.SetActivationFunction(self.activation_function)
        self.__trainer_entropy.SetActivationFunction(self.activation_function)
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
        return 
    
    def CommenceTraining(self):
        """Initiate the training process.
        """
        self.__PrepareOutputDir()
        self.__trainer_entropy.Train_MLP()
        self.__trainer_entropy.Save_Relevant_Data()
        self.__trainer_entropy.Plot_and_Save_History()

        weights_entropy = self.__trainer_entropy.GetWeights()
        biases_entropy = self.__trainer_entropy.GetBiases()
        self.__trainer_tpc2.SetWeights(weights_entropy)
        self.__trainer_tpc2.SetBiases(biases_entropy)
        self.__trainer_tpc2.Train_MLP()

        fid = open(self.main_save_dir + "/current_iter.txt", "w+")
        fid.write(str(self.current_iter) + "\n")
        fid.close()
        self.__test_score = self.__trainer_tpc2.GetTestScore()
        self.__cost_parameter = self.__trainer_tpc2.GetCostParameter()
        self.__trainer_tpc2.Save_Relevant_Data()
        return 
    
    def CommenceTraining_OnlyEntropy(self):
        self.__PrepareOutputDir()
        self.__trainer_entropy.Train_MLP()
        self.__trainer_entropy.Save_Relevant_Data()
        self.__test_score = self.__trainer_entropy.GetTestScore()
        self.__cost_parameter = self.__trainer_entropy.GetCostParameter()
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
        return self.__cost_parameter
    
    def GetTestScore(self):
        """Get MLP evaluation test score upon completion of training.

        :return: MLP evaluation test score.
        :rtype: float
        """
        return self.__test_score
    
    def SetTrainFileHeader(self, fileheader:str):
        """Set a custom training data file header.

        :param fileheader: file path and name
        :type fileheader: str
        """
        self.__trainer_entropy.SetTrainFileHeader(fileheader)
        self.__trainer_tpc2.SetTrainFileHeader(fileheader)
        return 
    
    def SetVerbose(self, val_verbose:int=1):
        """Set verbose level during training. 0 means no information, 1 means minimal information every epoch, 2 means detailed information.

        :param val_verbose: verbose level (0, 1, or 2), defaults to 1
        :type val_verbose: int, optional
        """

        self.__trainer_entropy.SetVerbose(val_verbose)
        self.__trainer_tpc2.SetVerbose(val_verbose)

        return                