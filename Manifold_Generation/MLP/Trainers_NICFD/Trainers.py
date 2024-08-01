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

from Common.DataDrivenConfig import EntropicAIConfig
from Common.Properties import DefaultProperties 
from Common.CommonMethods import GetReferenceData
from Manifold_Generation.MLP.Trainer_Base import TensorFlowFit,PhysicsInformedTrainer,EvaluateArchitecture

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

    __TD_vars:list[str]
    __Entropy_var:list[str]

    def __init__(self):
        PhysicsInformedTrainer.__init__(self)
        self._controlling_vars = ["Density","Energy"]
        self.__TD_vars=["T","p","c2"]
        self.__Entropy_var=["s"]

        self._train_vars = self.__TD_vars
        self.__idx_rho = self._controlling_vars.index("Density")
        self.__idx_e = self._controlling_vars.index("Energy")
        self.__idx_T = self.__TD_vars.index("T")
        self.__idx_p = self.__TD_vars.index("p")
        self.__idx_c2 = self.__TD_vars.index("c2")
        
        self._train_vars = self.__Entropy_var
        return 
    
    def GetTrainData(self):
        print("Reading train, test, and validation data...")
        X_full, Y_full = GetReferenceData(self._filedata_train+"_full.csv", self._controlling_vars, self.__Entropy_var)
        print("Done!")

        self.__s_min, self.__s_max = min(Y_full[:, 0]), max(Y_full[:, 0])

        # Free up memory
        del X_full
        del Y_full

        self._train_vars = self.__TD_vars 
        super().GetTrainData()
        self._train_vars = self.__Entropy_var

        self.__rho_min, self.__rho_max = self._X_min[self.__idx_rho], self._X_max[self.__idx_rho]
        self.__e_min, self.__e_max = self._X_min[self.__idx_e], self._X_max[self.__idx_e]
        
        self.Temperature_min, self.Temperature_max = self._Y_min[self.__idx_T], self._Y_max[self.__idx_T]
        self.Pressure_min, self.Pressure_max = self._Y_min[self.__idx_p], self._Y_max[self.__idx_p]
        self.C2_min, self.C2_max = self._Y_min[self.__idx_c2], self._Y_max[self.__idx_c2]
        
        return 
    
    def Preprocessing(self):
        super().Preprocessing()
        self.val_loss_history=[]
        for _ in self.__TD_vars:
            self.val_loss_history.append([])
        return 
    
    def SetDecaySteps(self):
        super().SetDecaySteps()
        self._decay_steps = 3*self._decay_steps
        return
    
    def Plot_and_Save_History(self):
        self._train_vars = self.__TD_vars
        super().Plot_and_Save_History()
        self._train_vars = self.__Entropy_var
        return 
    
    @tf.function
    def CollectVariables(self):
        """Define weights and biases as trainable hyper-parameters.
        """
        self._trainable_hyperparams = []
        for W in self._weights:
            self._trainable_hyperparams.append(W)
        for b in self._biases[:-1]:
            self._trainable_hyperparams.append(b)
        return 
    
    @tf.function
    def __ComputeEntropyGradients(self, rhoe_norm:tf.Tensor):
        with tf.GradientTape(watch_accessed_variables=False) as tape_2:
            tape_2.watch(rhoe_norm)
            with tf.GradientTape(watch_accessed_variables=False) as tape_1:
                tape_1.watch(rhoe_norm)
                s_norm = self._MLP_Evaluation(rhoe_norm)
                ds_norm = tape_1.gradient(s_norm, rhoe_norm)
                ds_norm_rho = tf.gather(ds_norm, indices=self.__idx_rho,axis=1)
                d2s_norm_rho = tape_2.gradient(ds_norm_rho, rhoe_norm)
        
        with tf.GradientTape(watch_accessed_variables=False) as tape_2:
            tape_2.watch(rhoe_norm)
            with tf.GradientTape(watch_accessed_variables=False) as tape_1:
                tape_1.watch(rhoe_norm)
                s_norm = self._MLP_Evaluation(rhoe_norm)
                ds_norm = tape_1.gradient(s_norm, rhoe_norm)
                ds_norm_e = tf.gather(ds_norm, indices=self.__idx_e,axis=1)
                d2s_norm_e = tape_2.gradient(ds_norm_e, rhoe_norm)

        dsdrho_e_norm = tf.gather(ds_norm, indices=self.__idx_rho,axis=1)
        dsde_rho_norm = tf.gather(ds_norm, indices=self.__idx_e,axis=1)
        d2sde2_norm = tf.gather(d2s_norm_e, indices=self.__idx_e,axis=1)
        d2sdrho2_norm = tf.gather(d2s_norm_rho,indices=self.__idx_rho,axis=1)
        d2sdrhode_norm = tf.gather(d2s_norm_e, indices=self.__idx_rho, axis=1)

        s_scale = self.__s_max - self.__s_min
        rho_scale = self.__rho_max - self.__rho_min
        e_scale = self.__e_max - self.__e_min

        s_dim = s_scale * s_norm + self.__s_min 
        dsdrho_e = tf.math.multiply((s_scale / rho_scale), dsdrho_e_norm)
        dsde_rho = tf.math.multiply((s_scale / e_scale), dsde_rho_norm)
        d2sdrho2 = tf.math.multiply((s_scale / tf.pow(rho_scale, 2)),d2sdrho2_norm)
        d2sdedrho = tf.math.multiply((s_scale / (rho_scale * e_scale)), d2sdrhode_norm)
        d2sde2 = tf.math.multiply((s_scale / tf.pow(e_scale, 2)), d2sde2_norm)

        dsdrhoe = [dsdrho_e, dsde_rho]
        d2sdrho2e2 = [[d2sdrho2, d2sdedrho],[d2sdedrho, d2sde2]]
        return s_dim, dsdrhoe, d2sdrho2e2
    
    @tf.function 
    def __EntropicEOS(self,rho, dsdrhoe, d2sdrho2e2):
        dsdrho_e = dsdrhoe[0]
        dsde_rho = dsdrhoe[1]
        d2sdrho2 = d2sdrho2e2[0][0]
        d2sdedrho = d2sdrho2e2[0][1]
        d2sde2 = d2sdrho2e2[1][1]
        T = 1.0 / dsde_rho
        P = -tf.pow(rho, 2) * T * dsdrho_e
        blue_term = (dsdrho_e * (2 - rho * tf.pow(dsde_rho, -1) * d2sdedrho) + rho*d2sdrho2)
        green_term = (-tf.pow(dsde_rho, -1) * d2sde2 * dsdrho_e + d2sdedrho)
        c2 = -rho * tf.pow(dsde_rho, -1) * (blue_term - rho * green_term * (dsdrho_e / dsde_rho))
  
        return T, P, c2
    
    @tf.function 
    def __TD_Evaluation(self, rhoe_norm:tf.Tensor):
        s, dsdrhoe, d2sdrho2e2 = self.__ComputeEntropyGradients(rhoe_norm)
        rho_norm = tf.gather(rhoe_norm, indices=self.__idx_rho, axis=1)
        rho = (self.__rho_max - self.__rho_min)*rho_norm + self.__rho_min 
        T, P, c2 = self.__EntropicEOS(rho, dsdrhoe, d2sdrho2e2)
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
    def __Compute_T_error(self, T_label_norm:tf.Tensor, x_var:tf.constant):
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
    def __Compute_P_error(self, P_label_norm:tf.Tensor,x_var:tf.constant):
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
    def __Compute_C2_error(self, C2_label_norm,x_var:tf.constant):
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
    def __ComputeGradients_T_error(self, T_label_norm:tf.Tensor, rhoe_norm:tf.constant):
        """Compute temperature prediction error and respective MLP weight sensitivities.

        :param T_label: reference temperature data
        :type T_label: tf.Tensor
        :param rhoe_norm: normalized density and energy tensor.
        :type rhoe_norm: tf.Variable
        :return: temperature prediction loss value and MLP weight sensitivities.
        :rtype: tf.Tensor
        """

        with tf.GradientTape() as tape:
            tape.watch(self._trainable_hyperparams)
            # Evaluate temperature loss value.
            T_loss = self.__Compute_T_error(T_label_norm, rhoe_norm)
            
            # Compute MLP weight sensitvities.
            grads_T = tape.gradient(T_loss, self._trainable_hyperparams)
        
        return T_loss, grads_T

    @tf.function
    def __ComputeGradients_P_error(self, P_label_norm:tf.Tensor, rhoe_norm:tf.constant):
        """Compute pressure prediction error and respective MLP weight sensitivities.

        :param P_label: reference pressure data
        :type P_label: tf.Tensor
        :param rhoe_norm: normalized density and energy tensor.
        :type rhoe_norm: tf.Variable
        :return: pressure prediction loss value and MLP weight sensitivities.
        :rtype: tf.Tensor
        """
        with tf.GradientTape() as tape:
            tape.watch(self._trainable_hyperparams)
            # Evaluate pressure loss value.
            P_loss = self.__Compute_P_error(P_label_norm, rhoe_norm)

            # Compute MLP weight sensitvities.
            grads_P = tape.gradient(P_loss, self._trainable_hyperparams)

        return P_loss, grads_P
    
    @tf.function
    def __ComputeGradients_C2_error(self, C2_label_norm:tf.Tensor, rhoe_norm:tf.constant):
        """Compute SoS prediction error and respective MLP weight sensitivities.

        :param C2_label: reference squared SoS data
        :type C2_label: tf.Tensor
        :param rhoe_norm: normalized density and energy tensor.
        :type rhoe_norm: tf.Variable
        :return: SoS prediction loss value and MLP weight sensitivities.
        :rtype: tf.Tensor
        """
        with tf.GradientTape() as tape:
            tape.watch(self._trainable_hyperparams)
            C2_loss = self.__Compute_C2_error(C2_label_norm, rhoe_norm)
            grads_C2 = tape.gradient(C2_loss, self._trainable_hyperparams)
        return C2_loss, grads_C2
    
    @tf.function
    def Train_Step(self, X_batch_norm:tf.constant, Y_batch_norm:tf.constant):

        T_batch_norm = tf.gather(Y_batch_norm,indices=self.__idx_T, axis=1)
        P_batch_norm = tf.gather(Y_batch_norm,indices=self.__idx_p, axis=1)
        C2_batch_norm = tf.gather(Y_batch_norm,indices=self.__idx_c2, axis=1)
        
        # Weight update for temperature prediction.
        T_loss, grads_T = self.__ComputeGradients_T_error(T_batch_norm, X_batch_norm)
        self._optimizer.apply_gradients(zip(grads_T, self._trainable_hyperparams))

        # Weight update for pressure prediction.
        P_loss, grads_P = self.__ComputeGradients_P_error(P_batch_norm,X_batch_norm)
        self._optimizer.apply_gradients(zip(grads_P, self._trainable_hyperparams))

        # Weight update for SoS prediction.
        C2_loss, grads_C2 = self.__ComputeGradients_C2_error(C2_batch_norm,X_batch_norm)
        self._optimizer.apply_gradients(zip(grads_C2, self._trainable_hyperparams))
  
        return T_loss, P_loss, C2_loss

    def ValidationLoss(self):
        return self.__ValidationLoss()
    
    def PrintEpochInfo(self, i_epoch, val_loss):
        if self._verbose > 0:
            T_val_loss = val_loss[0]
            P_val_loss = val_loss[1]
            C2_val_loss = val_loss[2]
            print("Epoch %i Validation loss Temperature: %.4e, Pressure: %.4e, Speed of sound: %.4e" % (i_epoch, T_val_loss.numpy(), P_val_loss.numpy(), C2_val_loss.numpy()))

        return 
    
    def __ValidationLoss(self):
        rhoe_val_norm = tf.constant(self._X_val_norm, self._dt)
        # T_val_error = self.__Compute_T_error(self._Y_val_norm[:,self.__idx_T], rhoe_val_norm)
        # p_val_error = self.__Compute_P_error(self._Y_val_norm[:,self.__idx_p], rhoe_val_norm)
        # c2_val_error = self.__Compute_C2_error(self._Y_val_norm[:,self.__idx_c2], rhoe_val_norm)
        _, T_pred_val, P_pred_val, C2_pred_val = self.__TD_Evaluation(rhoe_val_norm)
        T_pred_val_norm = (T_pred_val - self.Temperature_min)/(self.Temperature_max - self.Temperature_min)
        P_pred_val_norm = (P_pred_val - self.Pressure_min)/(self.Pressure_max - self.Pressure_min)
        C2_pred_val_norm = (C2_pred_val - self.C2_min)/(self.C2_max - self.C2_min)

        T_val_error = self.mean_square_error(y_true=self._Y_val_norm[:, self.__idx_T], y_pred=T_pred_val_norm).numpy()
        p_val_error = self.mean_square_error(y_true=self._Y_val_norm[:, self.__idx_p], y_pred=P_pred_val_norm).numpy()
        c2_val_error = self.mean_square_error(y_true=self._Y_val_norm[:, self.__idx_c2], y_pred=C2_pred_val_norm).numpy()
        self.val_loss_history[self.__idx_T].append(T_val_error)
        self.val_loss_history[self.__idx_p].append(p_val_error)
        self.val_loss_history[self.__idx_c2].append(c2_val_error)

        return T_val_error, p_val_error, c2_val_error
    
    def TestLoss(self):
        
        rhoe_test_norm = tf.constant(self._X_test_norm, self._dt)

        _, T_pred_test, P_pred_test, C2_pred_test = self.__TD_Evaluation(rhoe_test_norm)
        T_pred_test_norm = (T_pred_test - self.Temperature_min)/(self.Temperature_max - self.Temperature_min)
        P_pred_test_norm = (P_pred_test - self.Pressure_min)/(self.Pressure_max - self.Pressure_min)
        C2_pred_test_norm = (C2_pred_test - self.C2_min)/(self.C2_max - self.C2_min)

        self.T_test_loss = self.mean_square_error(y_true=self._Y_test_norm[:, self.__idx_T], y_pred=T_pred_test_norm).numpy()
        self.P_test_loss = self.mean_square_error(y_true=self._Y_test_norm[:, self.__idx_p], y_pred=P_pred_test_norm).numpy()
        self.C2_test_loss = self.mean_square_error(y_true=self._Y_test_norm[:, self.__idx_c2], y_pred=C2_pred_test_norm).numpy()

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

        s_test_pred, T_test_pred, P_test_pred, C2_test_pred = self.__TD_Evaluation(self._X_test_norm)

        figformat = "png"
        plot_fontsize = 20
        label_fontsize=18

        rho_test = self._X_test[:, self.__idx_rho]#(self._X_max[self.__idx_rho] - self._X_min[self.__idx_rho])*self._X_test_norm[:, self.__idx_rho] + self._X_min[self.__idx_rho]
        e_test = self._X_test[:, self.__idx_e]#(self._X_max[self.__idx_e] - self._X_min[self.__idx_e])*self._X_test_norm[:, self.__idx_e] + self._X_min[self.__idx_e]
        
        T_test = self._Y_test[:, self.__idx_T]
        P_test = self._Y_test[:, self.__idx_p]
        C2_test = self._Y_test[:, self.__idx_c2]

        markevery=10
        fig = plt.figure(figsize=[10,10])
        ax = plt.axes(projection='3d')
        ax.plot3D(rho_test[::markevery], e_test[::markevery], T_test[::markevery], 'ko')
        ax.plot3D(rho_test[::markevery], e_test[::markevery], T_test_pred[::markevery], 'ro')
        ax.set_xlabel(r"Density $(\rho)[kg m^{-3}]$",fontsize=20)
        ax.set_ylabel(r"Static Energy $(e)[J kg^{-1}]$",fontsize=20)
        ax.set_zlabel(r"Temperature $(T)[K]$",fontsize=20)
        ax.tick_params(which='both',labelsize=18)
        fig.savefig(self._save_dir + "/Model_"+str(self._model_index)+"/Temperature_prediction."+figformat,format=figformat,bbox_inches='tight')
        plt.close(fig)

        fig = plt.figure(figsize=[10,10])
        ax = plt.axes(projection='3d')
        ax.plot3D(rho_test[::markevery], e_test[::markevery], P_test[::markevery], 'ko')
        ax.plot3D(rho_test[::markevery], e_test[::markevery], P_test_pred[::markevery], 'ro')
        ax.set_xlabel(r"Density $(\rho)[kg m^{-3}]$",fontsize=20)
        ax.set_ylabel(r"Static Energy $(e)[J kg^{-1}]$",fontsize=20)
        ax.set_zlabel(r"Pressure $(p)[Pa]$",fontsize=20)
        ax.tick_params(which='both',labelsize=18)
        fig.savefig(self._save_dir + "/Model_"+str(self._model_index)+"/Pressure_prediction."+figformat,format=figformat,bbox_inches='tight')
        plt.close(fig)

        fig = plt.figure(figsize=[10,10])
        ax = plt.axes(projection='3d')
        ax.plot3D(rho_test[::markevery], e_test[::markevery], np.sqrt(C2_test[::markevery]), 'ko')
        ax.plot3D(rho_test[::markevery], e_test[::markevery], np.sqrt(C2_test_pred[::markevery]), 'ro')
        ax.set_xlabel(r"Density $(\rho)[kg m^{-3}]$",fontsize=20)
        ax.set_ylabel(r"Static Energy $(e)[J kg^{-1}]$",fontsize=20)
        ax.set_zlabel(r"Speed of sound $(c)[m s^{-1}]$",fontsize=20)
        ax.tick_params(which='both',labelsize=18)
        fig.savefig(self._save_dir + "/Model_"+str(self._model_index)+"/SoS_prediction."+figformat,format=figformat,bbox_inches='tight')
        plt.close(fig)

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
    
class EvaluateArchitecture_NICFD(EvaluateArchitecture):
    """Class for training MLP architectures
    """
    __trainer_PINN:Train_Entropic_PINN      # MLP trainer object responsible for training itself.

    def __init__(self, Config_in:EntropicAIConfig):
        """Define EvaluateArchitecture instance and prepare MLP trainer with
        default settings.

        :param Config: FlameletAIConfig object describing the flamelet data manifold.
        :type Config: FlameletAIConfig
        :param group_idx: MLP output group index, defaults to 0
        :type group_idx: int, optional
        :raises Exception: if MLP output group index is undefined by flameletAI configuration.
        """

        self._trainer_direct = Train_Entropic_Direct()
        self.__trainer_PINN = Train_Entropic_PINN()

        EvaluateArchitecture.__init__(self, Config_in=Config_in)
        pass

    def SynchronizeTrainer(self):
        """Synchronize all MLP trainer settings with locally stored settings.
        """
        super().SynchronizeTrainer()

        self.__trainer_PINN.SetModelIndex(self.current_iter)
        self.__trainer_PINN.SetSaveDir(self.main_save_dir)

        self.__trainer_PINN.SetDeviceKind(self.device)
        self.__trainer_PINN.SetDeviceIndex(self.process_index)

        self.__trainer_PINN.SetNEpochs(self.n_epochs)
        self.__trainer_PINN.SetActivationFunction(self.activation_function)
        self.__trainer_PINN.SetAlphaExpo(self.alpha_expo)
        self.__trainer_PINN.SetLRDecay(self.lr_decay)
        self.__trainer_PINN.SetBatchExpo(self.batch_expo)
        self.__trainer_PINN.SetHiddenLayers(self.architecture)
        self.__trainer_PINN.SetTrainFileHeader(self._train_file_header)
        self.__trainer_PINN.SetVerbose(self.verbose)
        return 
    
    def CommenceTraining(self):
        """Initiate the training process.
        """

        self.PrepareOutputDir()
        self._trainer_direct.SetMLPFileHeader("MLP_direct")
        self._trainer_direct.Train_MLP()
        self.TrainPostprocessing()
        
        self._trainer_direct.Save_Relevant_Data()
        self._trainer_direct.Plot_and_Save_History()

        weights_entropy = self._trainer_direct.GetWeights()
        biases_entropy = self._trainer_direct.GetBiases()
        self.__trainer_PINN.SetMLPFileHeader("MLP_PINN")
        self.__trainer_PINN.SetWeights(weights_entropy)
        self.__trainer_PINN.SetBiases(biases_entropy)
        self.__trainer_PINN.Train_MLP()

        fid = open(self.main_save_dir + "/current_iter.txt", "w+")
        fid.write(str(self.current_iter) + "\n")
        fid.close()
        self._test_score = self.__trainer_PINN.GetTestScore()
        self._cost_parameter = self.__trainer_PINN.GetCostParameter()
        self.__trainer_PINN.Save_Relevant_Data()
        return 
    