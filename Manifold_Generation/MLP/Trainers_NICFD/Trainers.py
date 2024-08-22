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

from Common.DataDrivenConfig import EntropicAIConfig
from Common.CommonMethods import GetReferenceData
from Common.Properties import DefaultSettings_NICFD
from Manifold_Generation.MLP.Trainer_Base import MLPTrainer, TensorFlowFit,PhysicsInformedTrainer,EvaluateArchitecture
  
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
    def __ComputeEntropyGradients(self, rhoe_norm:tf.Tensor):

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
        s, dsdrhoe, d2sdrho2e2 = self.__ComputeEntropyGradients(rhoe_norm)
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
        blue_term = (dsdrho_e * (2 - rho * tf.pow(dsde_rho, -1) * d2sdedrho) + rho*d2sdrho2)
        green_term = (-tf.pow(dsde_rho, -1) * d2sde2 * dsdrho_e + d2sdedrho)
        c2 = -rho * tf.pow(dsde_rho, -1) * (blue_term - rho * green_term * (dsdrho_e / dsde_rho))
  
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
        self._controlling_vars = DefaultSettings_NICFD.controlling_variables
        self._hidden_layers = []
        for n in DefaultSettings_NICFD.hidden_layer_architecture:
            self._hidden_layers.append(n)
            
        self._train_vars = ["s"]
        self.__idx_rho = self._controlling_vars.index("Density")
        self.__idx_e = self._controlling_vars.index("Energy")

        self._state_vars = ["T","p","c2"]
        self.__idx_T = self._state_vars.index("T")
        self.__idx_p = self._state_vars.index("p")
        self.__idx_c2 = self._state_vars.index("c2")

        return 
    
    def GetTrainData(self):

        super().GetTrainData()
        self.__s_max, self.__s_min = self._Y_max[0], self._Y_min[0]

        self.__rho_min, self.__rho_max = self._X_min[self.__idx_rho], self._X_max[self.__idx_rho]
        self.__e_min, self.__e_max = self._X_min[self.__idx_e], self._X_max[self.__idx_e]
        
        self.Temperature_min, self.Temperature_max = self._Y_state_min[self.__idx_T], self._Y_state_max[self.__idx_T]
        self.Pressure_min, self.Pressure_max = self._Y_state_min[self.__idx_p], self._Y_state_max[self.__idx_p]
        self.C2_min, self.C2_max = self._Y_state_min[self.__idx_c2], self._Y_state_max[self.__idx_c2]
        
        return 
    
    
    def SetDecaySteps(self):
        super().SetDecaySteps()
        self._decay_steps = 3*self._decay_steps
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
    def EntropicEOS(self,rho, dsdrhoe, d2sdrho2e2):
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
    def TD_Evaluation(self, rhoe_norm:tf.Tensor):
        s, dsdrhoe, d2sdrho2e2 = self.__ComputeEntropyGradients(rhoe_norm)
        rho_norm = tf.gather(rhoe_norm, indices=self.__idx_rho, axis=1)
        rho = (self.__rho_max - self.__rho_min)*rho_norm + self.__rho_min 
        T, P, c2 = self.EntropicEOS(rho, dsdrhoe, d2sdrho2e2)
        return s, T, P, c2 
    
    @tf.function
    def EvaluateState(self, X_norm:tf.Tensor):
        _, dsdrhoe, d2sdrho2e2 = self.__ComputeEntropyGradients(X_norm)
        rho_norm = tf.gather(X_norm, indices=self.__idx_rho, axis=1)
        rho = (self.__rho_max - self.__rho_min)*rho_norm + self.__rho_min 
        T, P, c2 = self.EntropicEOS(rho, dsdrhoe, d2sdrho2e2)
        Y_state_pred = tf.stack((T, P, c2),axis=1)
        return Y_state_pred 
    
    @tf.function 
    def ComputeStateError(self, Y_state_label_norm:tf.constant, X_label_norm:tf.constant):
        Y_state_pred = self.EvaluateState(X_label_norm)
        Y_state_pred_norm = (Y_state_pred - self._Y_state_min)/(self._Y_state_max - self._Y_state_min)

        return tf.reduce_mean(tf.pow(Y_state_pred_norm - Y_state_label_norm, 2),axis=0)
    
    
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
        T_pred_norm = (T - self._Y_state_min[self.__idx_T]) / (self._Y_state_max[self.__idx_T] - self._Y_state_min[self.__idx_T])
        
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
        P_pred_norm = (P - self._Y_state_min[self.__idx_p]) / (self._Y_state_max[self.__idx_p] - self._Y_state_min[self.__idx_p])
        
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
        C2_pred_norm = (C2 - self._Y_state_min[self.__idx_c2]) / (self._Y_state_max[self.__idx_c2] - self._Y_state_min[self.__idx_c2])

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
            print("Epoch %i Validation loss Temperature: %.4e, Pressure: %.4e, Speed of sound: %.4e" % (i_epoch, T_val_loss, P_val_loss, C2_val_loss))

        return 
    
    def __ValidationLoss(self):
        rhoe_val_norm = tf.constant(self._X_val_norm, self._dt)
        T_val_error = self.__Compute_T_error(self._Y_state_val_norm[:,self.__idx_T], rhoe_val_norm)
        p_val_error = self.__Compute_P_error(self._Y_state_val_norm[:,self.__idx_p], rhoe_val_norm)
        c2_val_error = self.__Compute_C2_error(self._Y_state_val_norm[:,self.__idx_c2], rhoe_val_norm)

        self.val_loss_history[self.__idx_T].append(T_val_error)
        self.val_loss_history[self.__idx_p].append(p_val_error)
        self.val_loss_history[self.__idx_c2].append(c2_val_error)

        return T_val_error, p_val_error, c2_val_error
    
    def TestLoss(self):
        
        rhoe_test_norm = tf.constant(self._X_test_norm, self._dt)

        _, T_pred_test, P_pred_test, C2_pred_test = self.TD_Evaluation(rhoe_test_norm)

        T_pred_test_norm = (T_pred_test - self.Temperature_min)/(self.Temperature_max - self.Temperature_min)
        P_pred_test_norm = (P_pred_test - self.Pressure_min)/(self.Pressure_max - self.Pressure_min)
        C2_pred_test_norm = (C2_pred_test - self.C2_min)/(self.C2_max - self.C2_min)

        self.T_test_loss = self.mean_square_error(y_true=self._Y_state_test_norm[:, self.__idx_T], y_pred=T_pred_test_norm).numpy()
        self.P_test_loss = self.mean_square_error(y_true=self._Y_state_test_norm[:, self.__idx_p], y_pred=P_pred_test_norm).numpy()
        self.C2_test_loss = self.mean_square_error(y_true=self._Y_state_test_norm[:, self.__idx_c2], y_pred=C2_pred_test_norm).numpy()

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
    
    def Save_Relevant_Data(self):
        """Save network performance characteristics in text file and write SU2 MLP input file.
        """
        fid = open(self._save_dir + "/Model_"+str(self._model_index)+"/MLP_NICFD_PINN_performance.txt", "w+")
        fid.write("Training time[minutes]: %+.3e\n" % self._train_time)
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
        #Y_state_test_pred = self.EvaluateState(self._X_test_norm)
        #T_test_pred = tf.gather(Y_state_test_pred, indices=self.__idx_T,axis=1)
        #P_test_pred = tf.gather(Y_state_test_pred, indices=self.__idx_p,axis=1)
        #C2_test_pred = tf.gather(Y_state_test_pred, indices=self.__idx_c2,axis=1)
        
        figformat = "png"
        plot_fontsize = 20
        label_fontsize=18

        rho_test = self._X_test[:, self.__idx_rho]#(self._X_max[self.__idx_rho] - self._X_min[self.__idx_rho])*self._X_test_norm[:, self.__idx_rho] + self._X_min[self.__idx_rho]
        e_test = self._X_test[:, self.__idx_e]#(self._X_max[self.__idx_e] - self._X_min[self.__idx_e])*self._X_test_norm[:, self.__idx_e] + self._X_min[self.__idx_e]
        
        S_test = self._Y_test[:, 0]
        T_test = self._Y_state_test[:, self.__idx_T]
        P_test = self._Y_state_test[:, self.__idx_p]
        C2_test = self._Y_state_test[:, self.__idx_c2]

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
        self.lr_decay = DefaultSettings_NICFD.learning_rate_decay
        self.alpha_expo = DefaultSettings_NICFD.init_learning_rate_expo
        self.activation_function = DefaultSettings_NICFD.activation_function
        self.architecture = []
        for n in DefaultSettings_NICFD.hidden_layer_architecture:
            self.architecture.append(n)

        EvaluateArchitecture.__init__(self, Config_in=Config_in)
        
        self.SynchronizeTrainer()

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
        self.__trainer_PINN.PostProcessing()

        fid = open(self.main_save_dir + "/current_iter.txt", "w+")
        fid.write(str(self.current_iter) + "\n")
        fid.close()
        self._test_score = self.__trainer_PINN.GetTestScore()
        self._cost_parameter = self.__trainer_PINN.GetCostParameter()
        self.__trainer_PINN.Save_Relevant_Data()

        return 
    