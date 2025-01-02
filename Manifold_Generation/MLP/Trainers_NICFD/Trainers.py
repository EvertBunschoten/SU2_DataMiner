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
#  Classes for training multi-layer perceptrons on fluid data.                                |
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
from matplotlib import ticker
from enum import Enum 

from Common.DataDrivenConfig import EntropicAIConfig
from Common.CommonMethods import GetReferenceData
from Common.Properties import DefaultSettings_NICFD, EntropicVars
from Manifold_Generation.MLP.Trainer_Base import MLPTrainer, TensorFlowFit,PhysicsInformedTrainer,EvaluateArchitecture

# class EntropicVars(Enum):
#     T=0
#     P=1
#     C2=2
#     dTdRHO_E=3
#     dTdE_RHO=4
#     dPdRHO_E=5
#     dPdE_RHO=6
#     dHdRHO_E=7
#     dHdE_RHO=8
#     dHdP_RHO=9
#     dHdRHO_P=10
#     dSdP_RHO=11
#     dSdRHO_P=12
#     N_STATE_VARS=13
    

# EntropicVarPairing = {"T":EntropicVars.T.value,\
#                       "p":EntropicVars.P.value,\
#                       "c2":EntropicVars.C2.value,\
#                       "dTdrho_e":EntropicVars.dTdRHO_E.value,\
#                       "dTde_rho":EntropicVars.dTdE_RHO.value,\
#                       "dpdrho_e":EntropicVars.dPdRHO_E.value,\
#                       "dpde_rho":EntropicVars.dPdE_RHO.value,\
#                       "dhdrho_e":EntropicVars.dHdRHO_E.value,\
#                       "dhde_rho":EntropicVars.dHdE_RHO.value,\
#                       "dhdp_rho":EntropicVars.dHdP_RHO.value,\
#                       "dhdrho_p":EntropicVars.dHdRHO_P.value,\
#                       "dsdp_rho":EntropicVars.dSdP_RHO.value,\
#                       "dsdrho_p":EntropicVars.dSdRHO_P.value}

LabelPairing = {EntropicVars.T.name:r"Temperature $(T)[K]$",\
                EntropicVars.p.name:r"Pressure $(p)[Pa]$",\
                EntropicVars.c2.name:r"Squared speed of sound $(c^2)[m/s]$",\
                EntropicVars.dTdrho_e.name:r"Temperature-density derivative $\left(\left.\frac{\partial T}{\partial \rho}\right|_e\right)$",\
                EntropicVars.dTde_rho.name:r"Temperature-energy derivative $\left(\left.\frac{\partial T}{\partial e}\right|_\rho\right)$",\
                EntropicVars.dpdrho_e.name:r"Pressure-density derivative $\left(\left.\frac{\partial p}{\partial \rho}\right|_e\right)$",\
                EntropicVars.dpde_rho.name:r"Pressure-energy derivative $\left(\left.\frac{\partial p}{\partial e}\right|_\rho\right)$"}

class Train_Entropic_Direct(TensorFlowFit):

    def __init__(self):
        TensorFlowFit.__init__(self)
        # Set train variables based on what kind of network is trained

        self._controlling_vars = DefaultSettings_NICFD.controlling_variables
        self._hidden_layers = []
        for NN in DefaultSettings_NICFD.hidden_layer_architecture:
            self._hidden_layers.append(NN)

        self._train_vars = ["s"]
        return                

class Train_Entropic_Derivatives(PhysicsInformedTrainer):
    __s_scale:float=0
    __s_min:float=0
    __s_max:float=0
    __rho_scale:float=0
    __e_scale:float=0
    __rho_max:float=0
    __e_max:float=0
    __rho_min:float=0
    __e_max:float
    __TD_vars:list[float]

    def __init__(self):
        PhysicsInformedTrainer.__init__(self)
        self._mlp_output_file_name = "SU2_MLP_segregated"
        self._controlling_vars = DefaultSettings_NICFD.controlling_variables
        self._train_vars = ["s","dsdrho_e","dsde_rho","d2sdrho2","d2sdedrho","d2sde2"]
        self.__TD_vars = ["s", "T", "p", "c2"]
        for NN in DefaultSettings_NICFD.hidden_layer_architecture:
            self._hidden_layers.append(NN)
        return
    
    def GetTrainData(self):
        super().GetTrainData()
        self.__s_scale = self._Y_max[0] - self._Y_min[0]
        self.__s_min = self._Y_min[0]
        self.__s_max = self._Y_max[0]
        self.__rho_max = self._X_max[0]
        self.__rho_min = self._X_min[0]
        self.__e_max = self._X_max[1]
        self.__e_min = self._X_min[1]
        
        self.__rho_scale = self._X_max[0] - self._X_min[0]
        self.__e_scale = self._X_max[1] - self._X_min[1]
        
        _, TD_data_full = GetReferenceData(self._filedata_train+"_full.csv", self._controlling_vars, self.__TD_vars)
        _, TD_data_train = GetReferenceData(self._filedata_train+"_train.csv", self._controlling_vars, self.__TD_vars)
        _, TD_data_test = GetReferenceData(self._filedata_train+"_test.csv", self._controlling_vars, self.__TD_vars)
        _, TD_data_val = GetReferenceData(self._filedata_train+"_val.csv", self._controlling_vars, self.__TD_vars)
        self.__TD_max, self.__TD_min = np.max(TD_data_full,axis=0),np.min(TD_data_full,axis=0)

        self.__TD_data_norm_train = tf.constant((TD_data_train - self.__TD_min)/(self.__TD_max - self.__TD_min),dtype=self._dt)
        self.__TD_data_norm_test = tf.constant((TD_data_test - self.__TD_min)/(self.__TD_max - self.__TD_min),dtype=self._dt)
        self.__TD_data_norm_val = tf.constant((TD_data_val - self.__TD_min)/(self.__TD_max - self.__TD_min),dtype=self._dt)
        
        return
    
    @tf.function
    def ComputeEntropyGradients(self, rhoe_norm:tf.Tensor):

        Y_norm = self._MLP_Evaluation(rhoe_norm)
        s_norm = tf.gather(Y_norm, indices=0, axis=1)
        dsdrho_e_norm = tf.gather(Y_norm, indices=1, axis=1)
        dsde_rho_norm = tf.gather(Y_norm, indices=2, axis=1)
        d2sdrho2_norm = tf.gather(Y_norm, indices=3, axis=1)
        d2sdedrho_norm = tf.gather(Y_norm, indices=4, axis=1)
        d2sde2_norm = tf.gather(Y_norm, indices=5, axis=1)

        s_dim = self.__s_scale * s_norm + self.__s_min 
        dsdrho_e = tf.math.multiply((self.__s_scale / self.__rho_scale), dsdrho_e_norm)
        dsde_rho = tf.math.multiply((self.__s_scale / self.__e_scale), dsde_rho_norm)
        d2sdrho2 = tf.math.multiply((self.__s_scale / tf.pow(self.__rho_scale, 2)),d2sdrho2_norm)
        d2sdedrho = tf.math.multiply((self.__s_scale / (self.__rho_scale * self.__e_scale)), d2sdedrho_norm)
        d2sde2 = tf.math.multiply((self.__s_scale / tf.pow(self.__e_scale, 2)), d2sde2_norm)

        dsdrhoe = [dsdrho_e, dsde_rho]
        d2sdrho2e2 = [[d2sdrho2, d2sdedrho],[d2sdedrho, d2sde2]]
        return s_dim, dsdrhoe, d2sdrho2e2
    
    @tf.function 
    def TD_Evaluation(self, rhoe_norm:tf.Tensor):
        s, dsdrhoe, d2sdrho2e2 = self.ComputeEntropyGradients(rhoe_norm)
        rho_norm = tf.gather(rhoe_norm, indices=0, axis=1)
        rho = (self.__rho_max - self.__rho_min)*rho_norm + self.__rho_min 
        T, P, c2 = self.EntropicEOS(rho, dsdrhoe, d2sdrho2e2)
        return s, T, P, c2
    
    @tf.function 
    def EntropicEOS(self,rho, dsdrhoe, d2sdrho2e2):
        dsdrho_e = dsdrhoe[0]
        dsde_rho = dsdrhoe[1]
        d2sdrho2 = d2sdrho2e2[0][0]
        d2sdedrho = d2sdrho2e2[0][1]
        d2sde2 = d2sdrho2e2[1][1]
        T = 1.0 / dsde_rho
        P = -tf.pow(rho, 2) * T * dsdrho_e
        blue_term = (dsdrho_e * (2 - rho * T * d2sdedrho) + rho*d2sdrho2)
        green_term = (-T * d2sde2 * dsdrho_e + d2sdedrho)
        c2 = -rho * T * (blue_term - rho * green_term * (dsdrho_e / dsde_rho))
  
        return T, P, c2
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
        _, T, _, _ = self.TD_Evaluation(x_var)

        # Normalize reference and predicted temperature.
        T_pred_norm = (T - self.__TD_min[1]) / (self.__TD_max[1] - self.__TD_min[1])
        
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
        _, _, P, _ = self.TD_Evaluation(x_var)

        # Normalize reference and predicted pressure.
        P_pred_norm = (P - self.__TD_min[2]) / (self.__TD_max[2] - self.__TD_min[2])
        
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
        _, _, _, C2 = self.TD_Evaluation(x_var)
        
        # Normalize reference and predicted squared SoS.
        C2_pred_norm = (C2 - self.__TD_min[3]) / (self.__TD_max[3] - self.__TD_min[3])

        # Apply loss function.
        C2_error = self.mean_square_error(y_true=C2_label_norm, y_pred=C2_pred_norm)

        return C2_error
    @tf.function
    def __ComputeGradients_S_error(self, S_label_norm:tf.Tensor, rhoe_norm:tf.constant):
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
            S_loss = self.__Compute_S_error(S_label_norm, rhoe_norm)
            
            # Compute MLP weight sensitvities.
            grads_S = tape.gradient(S_loss, self._trainable_hyperparams)
        
        return S_loss, grads_S
    
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
        
        s_batch_norm = tf.gather(Y_batch_norm,indices=0,axis=1)
        T_batch_norm = tf.gather(Y_batch_norm,indices=1, axis=1)
        P_batch_norm = tf.gather(Y_batch_norm,indices=2, axis=1)
        C2_batch_norm = tf.gather(Y_batch_norm,indices=3, axis=1)
        
        # Weight update for temperature prediction.
        s_loss, grads_s = self.__ComputeGradients_S_error(s_batch_norm, X_batch_norm)
        self._optimizer.apply_gradients(zip(grads_s, self._trainable_hyperparams))

        # Weight update for temperature prediction.
        T_loss, grads_T = self.__ComputeGradients_T_error(T_batch_norm, X_batch_norm)
        self._optimizer.apply_gradients(zip(grads_T, self._trainable_hyperparams))

        # Weight update for pressure prediction.
        P_loss, grads_P = self.__ComputeGradients_P_error(P_batch_norm,X_batch_norm)
        self._optimizer.apply_gradients(zip(grads_P, self._trainable_hyperparams))

        # Weight update for SoS prediction.
        C2_loss, grads_C2 = self.__ComputeGradients_C2_error(C2_batch_norm,X_batch_norm)
        self._optimizer.apply_gradients(zip(grads_C2, self._trainable_hyperparams))
  
        return s_loss, T_loss, P_loss, C2_loss
    
    def SetTrainBatches(self):
        train_batches = tf.data.Dataset.from_tensor_slices((self._X_train_norm, self.__TD_data_norm_train)).batch(2**self._batch_expo)
        return train_batches
    
    def ValidationLoss(self):
        rhoe_val_norm = tf.constant(self._X_val_norm, self._dt)

        S_pred_val, T_pred_val, P_pred_val, C2_pred_val = self.TD_Evaluation(rhoe_val_norm)

        S_pred_val_norm = (S_pred_val - self.__TD_min[0])/(self.__TD_max[0] - self.__TD_min[0])
        T_pred_val_norm = (T_pred_val - self.__TD_min[1])/(self.__TD_max[1] - self.__TD_min[1])
        P_pred_val_norm = (P_pred_val - self.__TD_min[2])/(self.__TD_max[2] - self.__TD_min[2])
        C2_pred_val_norm = (C2_pred_val - self.__TD_min[3])/(self.__TD_max[3] - self.__TD_min[3])

        s_val_error = self.mean_square_error(y_true=self.__TD_data_norm_val[0], y_pred=S_pred_val_norm).numpy()
        T_val_error = self.mean_square_error(y_true=self.__TD_data_norm_val[1], y_pred=T_pred_val_norm).numpy()
        p_val_error = self.mean_square_error(y_true=self.__TD_data_norm_val[2], y_pred=P_pred_val_norm).numpy()
        c2_val_error = self.mean_square_error(y_true=self.__TD_data_norm_val[3], y_pred=C2_pred_val_norm).numpy()
        self.val_loss_history[0].append(s_val_error)
        self.val_loss_history[1].append(T_val_error)
        self.val_loss_history[2].append(p_val_error)
        self.val_loss_history[3].append(c2_val_error)

        return s_val_error, T_val_error, p_val_error, c2_val_error
    
    def TestLoss(self):
        rhoe_test_norm = tf.constant(self._X_test_norm, self._dt)

        S_pred_test, T_pred_test, P_pred_test, C2_pred_test = self.TD_Evaluation(rhoe_test_norm)

        S_pred_test_norm = (S_pred_test - self.__TD_min[0])/(self.__TD_max[0] - self.__TD_min[0])
        T_pred_test_norm = (T_pred_test - self.__TD_min[1])/(self.__TD_max[1] - self.__TD_min[1])
        P_pred_test_norm = (P_pred_test - self.__TD_min[2])/(self.__TD_max[2] - self.__TD_min[2])
        C2_pred_test_norm = (C2_pred_test - self.__TD_min[3])/(self.__TD_max[3] - self.__TD_min[3])

        s_test_error = self.mean_square_error(y_true=self.__TD_data_norm_test[0], y_pred=S_pred_test_norm).numpy()
        T_test_error = self.mean_square_error(y_true=self.__TD_data_norm_test[1], y_pred=T_pred_test_norm).numpy()
        p_test_error = self.mean_square_error(y_true=self.__TD_data_norm_test[2], y_pred=P_pred_test_norm).numpy()
        c2_test_error = self.mean_square_error(y_true=self.__TD_data_norm_test[3], y_pred=C2_pred_test_norm).numpy()

        self.s_test_loss = s_test_error
        self.T_test_loss = T_test_error
        self.P_test_loss = p_test_error
        self.C2_test_loss = c2_test_error

        return s_test_error, T_test_error, p_test_error, c2_test_error
    
    def PrintEpochInfo(self, i_epoch, val_loss):
        if self._verbose > 0:
            S_val_loss = val_loss[0]
            T_val_loss = val_loss[1]
            P_val_loss = val_loss[2]
            C2_val_loss = val_loss[3]
            print("Epoch %i Validation loss Entropy: %.4e, Temperature: %.4e, Pressure: %.4e, Speed of sound: %.4e" % (i_epoch, S_val_loss, T_val_loss, P_val_loss, C2_val_loss))

        return
    
    def CustomCallback(self):

        # Save SU2 MLP and performance metrics.
        self.Save_Relevant_Data()

        # Plot intermediate history trends.
        self.Plot_and_Save_History()
        # Error scatter plots for thermodynamic properties.
        self.__Generate_Error_Plots()

        return super().CustomCallback()
    
    def Save_Relevant_Data(self):
        """Save network performance characteristics in text file and write SU2 MLP input file.
        """
        fid = open(self._save_dir + "/Model_"+str(self._model_index)+"/MLP_NICFD_PINN_performance.txt", "w+")
        fid.write("Training time[minutes]: %+.3e\n" % self._train_time)
        fid.write("Entropy test loss: %+.16e\n" % self.s_test_loss)
        fid.write("Temperature test loss: %+.16e\n" % self.T_test_loss)
        fid.write("Pressure test loss: %+.16e\n" % self.P_test_loss)
        fid.write("SoS test loss: %+.16e\n" % self.C2_test_loss)
        fid.write("Total neuron count:  %i\n" % np.sum(np.array(self._hidden_layers)))
        fid.write("Evaluation time[seconds]: %+.3e\n" % (self._test_time))
        fid.write("Evaluation cost parameter: %+.3e\n" % (self._cost_parameter))
        fid.write("Alpha exponent: %+.4e\n" % self._alpha_expo)
        fid.write("Learning rate decay: %+.4e\n" % self._lr_decay)
        fid.write("Batch size exponent: %i\n" % self._batch_expo)
        fid.write("Decay steps: %i\n" % self._decay_steps)
        fid.write("Activation function index: %i\n" % self._i_activation_function)
        fid.write("Number of hidden layers: %i\n" % len(self._hidden_layers))
        fid.write("Architecture: " + " ".join(str(n) for n in self._hidden_layers) + "\n")
        fid.close()

        self.write_SU2_MLP(self._save_dir + "/Model_"+str(self._model_index)+"/"+self._mlp_output_file_name)
        return 
    
    def __Generate_Error_Plots(self):
        """Make nice plots of the interpolated test data.
        """

        S_test_pred, T_test_pred, P_test_pred, C2_test_pred = self.TD_Evaluation(self._X_test_norm)

        figformat = "png"
        plot_fontsize = 20
        label_fontsize=18

        rho_test = self._X_test[:, 0]#(self._X_max[self.__idx_rho] - self._X_min[self.__idx_rho])*self._X_test_norm[:, self.__idx_rho] + self._X_min[self.__idx_rho]
        e_test = self._X_test[:, 1]#(self._X_max[self.__idx_e] - self._X_min[self.__idx_e])*self._X_test_norm[:, self.__idx_e] + self._X_min[self.__idx_e]
        
        S_test = self._Y_state_test[:,0]
        T_test = self._Y_state_test[:,1]
        P_test = self._Y_state_test[:,2]
        C2_test = self._Y_state_test[:,3]

        markevery=10
        fig = plt.figure(figsize=[10,10])
        ax = plt.axes(projection='3d')
        ax.plot3D(rho_test[::markevery], e_test[::markevery], S_test[::markevery], 'ko')
        ax.plot3D(rho_test[::markevery], e_test[::markevery], S_test_pred[::markevery], 'ro')
        ax.set_xlabel(r"Density $(\rho)[kg m^{-3}]$",fontsize=20)
        ax.set_ylabel(r"Static Energy $(e)[J kg^{-1}]$",fontsize=20)
        ax.set_zlabel(r"Entropy $(s)[J kg^{-1}]$",fontsize=20)
        ax.tick_params(which='both',labelsize=18)
        fig.savefig(self._save_dir + "/Model_"+str(self._model_index)+"/Entropy_prediction."+figformat,format=figformat,bbox_inches='tight')
        plt.close(fig)

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
        cax = ax.scatter(rho_test, e_test, c=100*np.abs((S_test_pred.numpy() - S_test)/S_test))
        cbar = plt.colorbar(cax, ax=ax)
        #cbar.set_label(r'Temperature prediction error $(\epsilon_T)[\%]$', rotation=270, fontsize=label_fontsize)
        ax.set_xlabel(r"Density $(\rho)[kg m^{-3}]$",fontsize=plot_fontsize)
        ax.set_ylabel(r"Static energy $(e)[J kg^{-1}]$",fontsize=plot_fontsize)
        ax.set_title(r"Entropy prediction error",fontsize=plot_fontsize)
        ax.tick_params(which='both',labelsize=label_fontsize)
        fig.savefig(self._save_dir + "/Model_"+str(self._model_index)+"/Entropy_prediction_error."+figformat,format=figformat,bbox_inches='tight')
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

        fig = plt.figure(figsize=[10,10])
        ax = plt.axes() 
        ax.plot(T_test, S_test, 'b.',markersize=3,markerfacecolor='none',label=r'Labeled')
        ax.plot(T_test_pred.numpy(), S_test_pred.numpy(), 'r.',markersize=2,label=r'Predicted')
        ax.grid()
        ax.legend(fontsize=20)
        ax.set_ylabel(r"Entropy $(s)[J kg^{-1}]$",fontsize=plot_fontsize)
        ax.set_xlabel(r"Temperature $(T)[K]$",fontsize=plot_fontsize)
        ax.set_title(r"T-S diagram",fontsize=plot_fontsize)
        ax.tick_params(which='both',labelsize=label_fontsize)
        fig.savefig(self._save_dir + "/Model_"+str(self._model_index)+"/TS_diagram."+figformat,format=figformat,bbox_inches='tight')
        plt.close(fig)

        return
    
class Train_Entropic_PINN(PhysicsInformedTrainer):

    _s_scale:float = None 
    _rho_scale:float = None 
    _e_scale:float = None 

    def __init__(self):
        PhysicsInformedTrainer.__init__(self)
        self.callback_every = 10
        self._controlling_vars = DefaultSettings_NICFD.controlling_variables
        self._hidden_layers = []
        for n in DefaultSettings_NICFD.hidden_layer_architecture:
            self._hidden_layers.append(n)
            
        self._train_vars = [EntropicVars.s.name]

        self._state_vars = [EntropicVars.T.name,\
                            EntropicVars.p.name,\
                            EntropicVars.c2.name]

        self._train_step_type="Gauss-Seidel"
        self._enable_boundary_loss = False
        self.scaler_function_name="minmax"

        return 
    
    def GetTrainData(self):

        super().GetTrainData()
        self._rho_scale = tf.cast(self._X_scale[0], self._dt)
        self._e_scale = tf.cast(self._X_scale[1], self._dt)
        self._rho_offset = tf.cast(self._X_offset[0], self._dt)
        self._e_offset = tf.cast(self._X_offset[1], self._dt)
        self._s_scale = tf.cast(self._Y_scale[0], self._dt)
        self._s_offset = tf.cast(self._Y_offset[0], self._dt)
        self._Y_state_scale_tf = tf.cast(self._Y_state_scale, self._dt)
        self._Y_state_offset_tf = tf.cast(self._Y_state_offset, self._dt)
        return 
    
    def SetStateVars(self, state_vars_in:list[str]):
        # if any((v not in DefaultSettings_NICFD.supported_state_vars) for v in state_vars_in):
        #     raise Exception("Only the following state variables are supported: "+ ",".join((v for v in DefaultSettings_NICFD.supported_state_vars)))
        self._state_vars = state_vars_in.copy()
        return 

    def CollectPIVars(self):
        val_lambda_default = super().CollectPIVars()
        self.lamba_labels = []
        proj_arrays, target_grads, lambda_labels = self.__SetRhoEProjection()
        for p, t, l in zip(proj_arrays, target_grads, lambda_labels):
            self.projection_arrays.append(p)
            self.target_arrays.append(t)
            self.idx_PIvar.append(0)
            self.vals_lambda.append(val_lambda_default)
            self.lamba_labels.append(l)
        self._N_bc = len(self.vals_lambda)
        return 
    
    def GetBoundaryData(self, y_vars=None):
        X_boundary, Y_boundary = GetReferenceData(self._filedata_train + "_val.csv", self._controlling_vars, self._train_vars)
        self._X_boundary_norm = self.scaler_function_x.transform(X_boundary)
        self._Y_boundary_norm = self.scaler_function_y.transform(Y_boundary)
        
        return 
    
    def __SetRhoEProjection(self):
        
        X_val, Y_val = GetReferenceData(self._filedata_train + "_val.csv", self._controlling_vars, [EntropicVars.p.name, EntropicVars.T.name, EntropicVars.dsdrho_e.name, EntropicVars.dsde_rho.name])

        rho_val = X_val[:, self._controlling_vars.index(EntropicVars.Density.name)]
        p_val = Y_val[:, 0]
        T_val = Y_val[:, 1]
        
        idx_minrho = np.argmin(rho_val)
        Rgas = p_val[idx_minrho] / (rho_val[idx_minrho]*T_val[idx_minrho])

        compress_factor_threshold = 0.0

        compress_factor = p_val / (Rgas * rho_val * T_val)
        
        idx_idealgas = compress_factor > compress_factor_threshold

        proj_rho, proj_e = np.zeros(np.shape(X_val)), np.zeros(np.shape(X_val))
        proj_rho[idx_idealgas, self._controlling_vars.index(EntropicVars.Density.name)] = 1.0
        proj_e[idx_idealgas, self._controlling_vars.index(EntropicVars.Energy.name)] = 1.0
        
        target_grad_rho = Y_val[:, 2]
        target_grad_e = Y_val[:, 3]
        
        target_grad_rho[np.invert(idx_idealgas)] = 0.0
        target_grad_e[np.invert(idx_idealgas)] = 0.0
        
        proj_rho = np.delete(proj_rho, np.invert(idx_idealgas),axis=0)
        proj_e = np.delete(proj_e, np.invert(idx_idealgas),axis=0)
        target_grad_rho = np.delete(target_grad_rho, np.invert(idx_idealgas))
        target_grad_e = np.delete(target_grad_e, np.invert(idx_idealgas))
        
        target_grad_rho /= self._Y_scale[0] / self._X_scale[self._controlling_vars.index(EntropicVars.Density.name)]
        target_grad_e /= self._Y_scale[0] / self._X_scale[self._controlling_vars.index(EntropicVars.Energy.name)]
        
        self._X_boundary_norm = self._X_boundary_norm[idx_idealgas, :]
        lambda_labels = ["dsdrho_e"]
        return [proj_rho], [target_grad_rho], lambda_labels
    
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
    def ComputeEntropyGradients(self, rhoe_norm:tf.Tensor):
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape_2:
            tape_2.watch(rhoe_norm)
            with tf.GradientTape(watch_accessed_variables=False) as tape_1:
                tape_1.watch(rhoe_norm)
                s_norm = self._MLP_Evaluation(rhoe_norm)
            ds_norm = tape_1.gradient(s_norm, rhoe_norm)
            ds_norm_rho = tf.gather(ds_norm, indices=0,axis=1)
            ds_norm_e = tf.gather(ds_norm, indices=1,axis=1)
        d2s_norm_rho = tape_2.gradient(ds_norm_rho, rhoe_norm)
        d2s_norm_e = tape_2.gradient(ds_norm_e, rhoe_norm)

        dsdrho_e_norm = tf.gather(ds_norm, indices=0,axis=1)
        dsde_rho_norm = tf.gather(ds_norm, indices=1,axis=1)
        d2sde2_norm = tf.gather(d2s_norm_e, indices=1,axis=1)
        d2sdrho2_norm = tf.gather(d2s_norm_rho,indices=0,axis=1)
        d2sdrhode_norm = tf.gather(d2s_norm_e, indices=0, axis=1)

        s_dim = self._s_scale * s_norm + self._s_offset
        dsdrho_e = tf.math.multiply((self._s_scale / self._rho_scale), dsdrho_e_norm)
        dsde_rho = tf.math.multiply((self._s_scale / self._e_scale), dsde_rho_norm)
        d2sdrho2 = tf.math.multiply((self._s_scale / tf.pow(self._rho_scale, 2)),d2sdrho2_norm)
        d2sdedrho = tf.math.multiply((self._s_scale / (self._rho_scale * self._e_scale)), d2sdrhode_norm)
        d2sde2 = tf.math.multiply((self._s_scale / tf.pow(self._e_scale, 2)), d2sde2_norm)

        dsdrhoe = [dsdrho_e, dsde_rho]
        d2sdrho2e2 = [[d2sdrho2, d2sdedrho],[d2sdedrho, d2sde2]]
        return s_dim, dsdrhoe, d2sdrho2e2
    
    @tf.function 
    def EntropicEOS(self,rho,e, s, dsdrhoe, d2sdrho2e2):
        dsdrho_e = dsdrhoe[0]
        dsde_rho = dsdrhoe[1]
        d2sdrho2 = d2sdrho2e2[0][0]
        d2sdedrho = d2sdrho2e2[0][1]
        d2sde2 = d2sdrho2e2[1][1]
        T = tf.pow(dsde_rho, -1)
        rho2 = rho*rho
        P = -rho2 * T * dsdrho_e
        blue_term = (dsdrho_e * (2 - rho * T * d2sdedrho) + rho*d2sdrho2)
        green_term = (-T * d2sde2 * dsdrho_e + d2sdedrho)
        c2 = -rho *T * (blue_term - rho * green_term * (dsdrho_e / dsde_rho))

        dTde_rho = -T*T * d2sde2 
        dTdrho_e = -T*T * d2sdedrho 

        dPde_rho = -rho2 * (dTde_rho * dsdrho_e + T * d2sdedrho)
        dPdrho_e = -2 * rho * T * dsdrho_e - rho2 * (dTdrho_e * dsdrho_e + T * d2sdrho2)
        dhdrho_e = -P * (1.0/rho2) + dPdrho_e / rho
        dhde_rho = 1 + dPde_rho / rho

        dhdrho_P = dhdrho_e - dhde_rho * (1 / dPde_rho) * dPdrho_e
        dhdP_rho = dhde_rho * (1 / dPde_rho)
        dsdrho_P = dsdrho_e - dPdrho_e * (1 / dPde_rho) * dsde_rho
        dsdP_rho = dsde_rho / dPde_rho

        drhode_p = -dPde_rho/dPdrho_e
        dTde_p = dTde_rho + dTdrho_e*drhode_p
        dhde_p = dhde_rho + drhode_p*dhdrho_e
        Cp = dhde_p / dTde_p
        
        Y_state = tf.stack((rho, e, T, P, c2, s, dsdrho_e, dsde_rho, d2sdrho2, d2sdedrho, d2sde2, dTdrho_e, dTde_rho, dPdrho_e, dPde_rho, dhdrho_e, dhde_rho, dhdP_rho, dhdrho_P, dsdP_rho, dsdrho_P, Cp),axis=1)
        Y_state_selected = tf.stack(tf.tuple(Y_state[:,EntropicVars[var].value] for var in self._state_vars),axis=1)
        return Y_state_selected
    
    @tf.function 
    def TD_Evaluation(self, rhoe_norm:tf.Tensor):
        s, dsdrhoe, d2sdrho2e2 = self.ComputeEntropyGradients(rhoe_norm)
        rho_norm = tf.gather(rhoe_norm, indices=0, axis=1)
        rho = self._rho_scale*rho_norm + self._rho_offset
        e_norm = tf.gather(rhoe_norm, indices=1, axis=1)
        e = self._e_scale*e_norm + self._e_offset
        return self.EntropicEOS(rho, e, s[:,0], dsdrhoe, d2sdrho2e2)
    
    @tf.function
    def EvaluateState(self, X_norm:tf.Tensor):
        return self.TD_Evaluation(X_norm)
    
    @tf.function 
    def ComputeStateError(self, X_label_norm:tf.constant,Y_state_label_norm:tf.constant):
        Y_state_pred = self.EvaluateState(X_label_norm)
        Y_state_pred_norm = (Y_state_pred - self._Y_state_offset_tf)/self._Y_state_scale_tf
        pred_error = tf.reduce_mean(tf.pow(Y_state_pred_norm - Y_state_label_norm, 2), axis=0)
        return pred_error
    
    
    def PrintEpochInfo(self, i_epoch, val_loss):
        if self._verbose > 0:
            print(("Epoch %i Validation loss " % i_epoch) + ", ".join((" %s: %.4e" % (self._state_vars[iVar], val_loss[iVar])) for iVar in range(len(self._state_vars))))
        return 
    
    def CustomCallback(self):

        # Save SU2 MLP and performance metrics.
        self.Save_Relevant_Data()

        # Plot intermediate history trends.
        self.Plot_and_Save_History()
        # Error scatter plots for thermodynamic properties.
        self.__Generate_Error_Plots()

        return super().CustomCallback()
    
    def Save_Relevant_Data(self):
        """Save network performance characteristics in text file and write SU2 MLP input file.
        """
        fid = open(self._save_dir + "/Model_"+str(self._model_index)+"/MLP_NICFD_PINN_performance.txt", "w+")
        fid.write("Training time[minutes]: %+.3e\n" % self._train_time)
        for ivar, var in enumerate(self._state_vars):
            fid.write("%s test loss: %+.16e\n" % (var, self.state_test_loss[ivar]))
        fid.write("Total neuron count:  %i\n" % np.sum(np.array(self._hidden_layers)))
        fid.write("Evaluation time[seconds]: %+.3e\n" % (self._test_time))
        fid.write("Evaluation cost parameter: %+.3e\n" % (self._cost_parameter))
        fid.write("Alpha exponent: %+.16e\n" % self._alpha_expo)
        fid.write("Learning rate decay: %+.16e\n" % self._lr_decay)
        fid.write("Batch size exponent: %i\n" % self._batch_expo)
        fid.write("Decay steps: %i\n" % self._decay_steps)
        fid.write("Activation function index: %i\n" % self._i_activation_function)
        fid.write("Number of hidden layers: %i\n" % len(self._hidden_layers))
        fid.write("Architecture: " + " ".join(str(n) for n in self._hidden_layers) + "\n")
        fid.close()

        self.write_SU2_MLP(self._save_dir + "/Model_"+str(self._model_index)+"/"+self._mlp_output_file_name)
        return 
    
    def __Generate_Error_Plots(self):
        """Make nice plots of the interpolated test data.
        """
        state_test_pred = self.EvaluateState(self._X_test_norm).numpy()
        plot_fontsize = 20
        label_fontsize=18
        markevery=10
        val_pad = 30

        
        rhoe_test = self.scaler_function_x.inverse_transform(self._X_test_norm)
        rho_test = rhoe_test[:,0]
        e_test = rhoe_test[:,1]

        Y_ref_test = self._scaler_state.inverse_transform(self._Y_state_test_norm)
        
        for var in self._state_vars:
            Y_ref = Y_ref_test[:, self._state_vars.index(var)]
            Y_pred = state_test_pred[:, self._state_vars.index(var)]
            fig = plt.figure(figsize=[10,10])
            ax = plt.axes(projection='3d')
            ax.plot3D(rho_test[::markevery], e_test[::markevery], Y_ref[::markevery], 'ko')
            ax.plot3D(rho_test[::markevery], e_test[::markevery], Y_pred[::markevery], 'ro')
            ax.set_xlabel(r"Density $(\rho)[kg m^{-3}]$",fontsize=20)
            ax.set_ylabel(r"Static Energy $(e)[J kg^{-1}]$",fontsize=20)
            ax.set_zlabel(LabelPairing[var],fontsize=20)
            ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.2e}"))
            ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.2e}"))
            ax.zaxis.set_major_formatter(ticker.StrMethodFormatter("{x:+.2e}"))
            ax.xaxis.labelpad=val_pad
            ax.yaxis.labelpad=val_pad
            ax.zaxis.labelpad=val_pad
            ax.tick_params(which='both',labelsize=18)
            plt.tight_layout()
            fig.savefig(self._save_dir + "/Model_"+str(self._model_index)+"/"+var+"_prediction."+self._figformat,format=self._figformat,bbox_inches='tight')
            plt.close(fig)

            e = 100*np.abs((Y_pred - Y_ref)/(Y_ref+1e-6))
            fig = plt.figure(figsize=[10,10])
            ax = plt.axes() 
            cax = ax.scatter(rho_test, e_test, c=e)
            cbar = fig.colorbar(cax, ax=ax)
            cbar.set_label(r'Interpolation error $(e)[\%]$', rotation=270,fontsize=label_fontsize)
            # cbar.set_ticks([0, int(max(e))])
            # cbar.ax.set_yticklabels([0, max(e)], fontsize=label_fontsize)
            ax.set_xlabel(r"Density $(\rho)[kg m^{-3}]$",fontsize=plot_fontsize)
            ax.set_ylabel(r"Static energy $(e)[J kg^{-1}]$",fontsize=plot_fontsize)
            ax.set_title(("%s prediction error" % var),fontsize=plot_fontsize)
            ax.tick_params(which='both',labelsize=label_fontsize)
            fig.savefig(self._save_dir + "/Model_"+str(self._model_index)+"/"+var+"_prediction_error."+self._figformat,format=self._figformat,bbox_inches='tight')
            plt.close(fig)

        return
    
           
def transform_dsdrho(dsdrho_untransformed):
        return np.log(-dsdrho_untransformed)
def transform_d2sdrho2(d2sdrho2_untransformed):
    return np.log(d2sdrho2_untransformed)

class Train_Entropic_Segregated(TensorFlowFit):
    """Class for training MLP on segregated entropy derivatives using direct training.
    """

    def __init__(self):
        TensorFlowFit.__init__(self)

        self._controlling_vars = DefaultSettings_NICFD.controlling_variables
        self._hidden_layers = []
        for NN in DefaultSettings_NICFD.hidden_layer_architecture:
            self._hidden_layers.append(NN)

        self._train_vars = [EntropicVars.s.name,\
                            EntropicVars.dsdrho_e.name,\
                            EntropicVars.dsde_rho.name,\
                            EntropicVars.d2sdrho2.name,\
                            EntropicVars.d2sdrhode.name,\
                            EntropicVars.d2sde2.name]
        return 
    
    def TransformData(self, Y_untransformed):
        """Transform first and second entropy derivative w.r.t. density through logarithmic scaling.

        :param Y_untransformed: raw training data array.
        :type Y_untransformed: np.ndarray[float]
        :return: transformed training data array.
        :rtype: np.ndarray[float]
        """
        idx_dsdrho_e = self._train_vars.index(EntropicVars.dsdrho_e.name)
        idx_d2sdrho2 = self._train_vars.index(EntropicVars.d2sdrho2.name)
        dsdrho_untransformed = Y_untransformed[:,idx_dsdrho_e]
        d2sdrho2_untransformed = Y_untransformed[:,idx_d2sdrho2]
        dsdrho_transformed = transform_dsdrho(dsdrho_untransformed)
        d2sdrho2_transformed = transform_d2sdrho2(d2sdrho2_untransformed)
        Y_transformed = Y_untransformed
        Y_transformed[:, idx_dsdrho_e] = dsdrho_transformed
        Y_transformed[:, idx_d2sdrho2] = d2sdrho2_transformed

        return Y_transformed
    
    def TransformData_Inv(self, Y_transformed):
        idx_dsdrho_e = self._train_vars.index(EntropicVars.dsdrho_e.name)
        idx_d2sdrho2 = self._train_vars.index(EntropicVars.d2sdrho2.name)
        dsdrho_transformed = Y_transformed[:,idx_dsdrho_e]
        d2sdrho2_transformed = Y_transformed[:,idx_d2sdrho2]
        dsdrho_untransformed = -np.exp(dsdrho_transformed)
        d2sdrho2_untransformed = np.exp(d2sdrho2_transformed)
        Y_untransformed = Y_transformed
        Y_untransformed[:, idx_dsdrho_e] = dsdrho_untransformed
        Y_untransformed[:, idx_d2sdrho2] = d2sdrho2_untransformed
        return Y_untransformed
    
    def add_additional_header_info(self, fid):
        fid.write("Inverse transform dsdrho_e: -exp(dsdrho_e)\nInverse transform d2sdrho2: exp(d2sdrho2)\n")
        return 
    
class EvaluateArchitecture_NICFD(EvaluateArchitecture):
    """Class for training MLP architectures
    """
    __trainer_PINN:Train_Entropic_PINN      # MLP trainer object responsible for training itself.
    _state_vars:list[str] = [EntropicVars.T.name, EntropicVars.p.name, EntropicVars.c2.name]

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
        self.lr_decay = DefaultSettings_NICFD.learning_rate_decay
        self.alpha_expo = DefaultSettings_NICFD.init_learning_rate_expo
        self.activation_function = DefaultSettings_NICFD.activation_function
        self.architecture = []
        for n in DefaultSettings_NICFD.hidden_layer_architecture:
            self.architecture.append(n)

        EvaluateArchitecture.__init__(self, Config_in=Config_in)
        self._scaler = "minmax"
        self._state_vars = Config_in.GetStateVars().copy()

        self.SynchronizeTrainer()

        pass

    def SetStateVars(self, state_vars_in:list[str]):
        """Set the state variables for which the physics-informed neural network is trained.

        :param state_vars_in: list with state variable names.
        :type state_vars_in: list[str]
        :raises Exception: if any of the state variables is not supported.
        """

        # if any((v not in DefaultSettings_NICFD.supported_state_vars) for v in state_vars_in):
        #     raise Exception("Only the following state variables are supported: "+ ",".join((v for v in DefaultSettings_NICFD.supported_state_vars)))
        self._state_vars = state_vars_in.copy()
        self.SynchronizeTrainer()
        
        return 
    
    def SynchronizeTrainer(self):
        """Synchronize all MLP trainer settings with locally stored settings.
        """
        super().SynchronizeTrainer()

        self.worker_dir = self.main_save_dir + ("/Worker_%i/" % self.process_index)
        self.__trainer_PINN.SetModelIndex(self.current_iter)
        self.__trainer_PINN.SetSaveDir(self.worker_dir)
        self.__trainer_PINN.SetStateVars(self._state_vars)

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
        self.__trainer_PINN.SetFigFormat(self._fig_format)
        self.__trainer_PINN.SetScaler(self._scaler)
        return 
    
    def CommenceTraining(self):
        """Initiate the training process.
        """

        self.PrepareOutputDir()
        self._trainer_direct.SetMLPFileHeader("MLP_direct")
        self._trainer_direct.Train_MLP()
        super().TrainPostprocessing()
        
        self._trainer_direct.Save_Relevant_Data()
        self._trainer_direct.Plot_and_Save_History()

        weights_entropy = self._trainer_direct.GetWeights()
        biases_entropy = self._trainer_direct.GetBiases()

        self.__trainer_PINN.SetMLPFileHeader("MLP_PINN")
        self.__trainer_PINN.SetWeights(weights_entropy)
        self.__trainer_PINN.SetBiases(biases_entropy)
        self.__trainer_PINN.Train_MLP()
        self.__trainer_PINN.PostProcessing()

        fid = open(self.worker_dir + "/current_iter.txt", "w+")
        fid.write(str(self.current_iter) + "\n")
        fid.close()
        return 
    
    def TrainPostprocessing(self):
        self._test_score = self.__trainer_PINN.GetTestScore()
        if np.isnan(self._test_score):
            self._test_score = 1e2
        self._cost_parameter = self.__trainer_PINN.GetCostParameter()
        self.__trainer_PINN.Save_Relevant_Data()
        return           
    

class EvaluateArchitecture_NICFD_Segregated(EvaluateArchitecture):
    """Driver class for training a segregated entropic MLP.
    """
    def __init__(self, Config_in:EntropicAIConfig):

        # Use segregated MLP trainer
        self._trainer_direct = Train_Entropic_Segregated()
        self.lr_decay = DefaultSettings_NICFD.learning_rate_decay
        self.alpha_expo = DefaultSettings_NICFD.init_learning_rate_expo
        self.activation_function = DefaultSettings_NICFD.activation_function
        self.architecture = []
        for n in DefaultSettings_NICFD.hidden_layer_architecture:
            self.architecture.append(n)

        EvaluateArchitecture.__init__(self, Config_in=Config_in)
        self.SynchronizeTrainer()
        return 