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
        self._X_scale = self.scaler_function_x.scale_
        if self.scaler_function_name == "standard":
            self._X_offset = self.scaler_function_x.mean_
        elif self.scaler_function_name == "robust":
            self._X_offset = self.scaler_function_x.center_
        elif self.scaler_function_name == "minmax":  
            self._X_scale = 1 / self.scaler_function_x.scale_
            
            self._X_offset = -self.scaler_function_x.min_ / self.scaler_function_x.scale_
        return 
    
class Train_FGM_PINN(PhysicsInformedTrainer):

    __Config:FlameletAIConfig = None    # FlameletAI configuration class to read output variables and hyper-parameters from.

    update_lambda_every_iter:int = 10   # Update boundary condition penalty weights every x weight update iterations.
    lambda_history:list[list[float]] = []   # Boundary condition penalty values history.
    j_gradient_update:int = 0   # Global weight update iteration value.

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
        
        target_grad_array = (1.0 / Cp_boundary[:,0]) * h_scale / T_scale

        return projection_array_train, target_grad_array
    
    def SetBeta_h2_projection(self):
        """Neumann boundary condition for species enthalpy preferential diffusion scalar. 
        In chemical equilibrium conditions, the derivative w.r.t. total enthalpy should be equal to 1 - (beta_h1/cp)

        :return: penalty projection array, target projected gradient
        :rtype: np.ndarray[float],np.ndarray[float]
        """

        # Extract specific heat and beta_h1 from flamelet data.
        _, Y_boundary = GetReferenceData(self.__boundary_data_file, x_vars=self._controlling_vars, train_variables=["Cp", "Beta_Enth_Thermal"])
        Cp_boundary = Y_boundary[:,0]
        Beta_h1_boundary = Y_boundary[:,1]

        # Extract normalization scales.
        Beta_h2_scale = self._Y_scale[self._train_vars.index("Beta_Enth")]
        h_scale = self._X_scale[self._controlling_vars.index(DefaultProperties.name_enth)]

        # Set projection array along total enthalpy direction.
        projection_array_train = np.zeros(np.shape(self.X_boundary_norm))
        projection_array_train[:, self._controlling_vars.index(DefaultProperties.name_enth)] = 1.0

        # Normalized projected target gradient.
        target_grad_array = (1 - (Beta_h1_boundary / Cp_boundary)) * h_scale / Beta_h2_scale

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
            
            if (var == "Beta_Enth"):
                self.idx_PIvar.append(ivar)
                proj_array, target_grad = self.SetBeta_h2_projection()
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
        self.lambda_history.clear()
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

            # Update boundary condition penalty values.
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
        project_dy_pred_norm = tf.reduce_sum(tf.multiply(precon_gradient[:,:,iVar], dy_pred_norm), axis=1)

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
    def UpdateLambdas(self, X_domain_batch:tf.constant, Y_domain_batch:tf.constant,\
                            X_boundary_batch:tf.constant, Y_boundary_batch:tf.constant, \
                            P_boundary_batch:tf.constant, Yt_boundary_batch:tf.constant, vals_lambda_old:list[tf.constant]):
        """Update boundary condition penalty values.

        :return: list of updated penalty values.
        :rtype: list[tf.constant]
        """

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
        """Generate controlling variable matrices for boundary conditions, where predicted quantities are visualized onto during convergence.
        """

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
            pv_unb = self.__Config.ComputeProgressVariable(variables=None, flamelet_data=None, Y_flamelet=gas_unb.Y[:,np.newaxis])[0]
            gas_b.equilibrate("TP")
            pv_b = self.__Config.ComputeProgressVariable(variables=None, flamelet_data=None, Y_flamelet=gas_b.Y[:,np.newaxis])[0]
            for iT, T in enumerate(T_range):
                gas_unb.TP = T, DefaultProperties.pressure
                self.pv_unb[iZ, iT] = pv_unb
                self.h_unb[iZ, iT] = gas_unb.enthalpy_mass
                self.z_unb[iZ, iT] = Z 

                gas_b.TP = T, DefaultProperties.pressure
                self.pv_b[iZ, iT] = pv_b
                self.h_b[iZ, iT] = gas_b.enthalpy_mass
                self.z_b[iZ, iT] = Z 
        print("Done!")
        
        return 
    
    def PlotUnbData(self):
        """Plot predicted quantities on chemical equilibrium data.
        """

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
        
        # Evaluate model predictions in unburnt and burnt conditions.
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
        self.PlotR2Data()
        self.write_SU2_MLP(self._save_dir + "/Model_"+str(self._model_index)+"/MLP_"+self._train_name)
        return super().CustomCallback()
    
    def add_additional_header_info(self, fid):
        fid.write("Progress variable definition: " + "+".join(("%+.6e*%s" % (w, s)) for w, s in zip(self.__Config.GetProgressVariableWeights(), self.__Config.GetProgressVariableSpecies())))
        fid.write("\n\n")
        return super().add_additional_header_info(fid)
    
    def SourcetermCorrection(self):
        """Apply correction to source terms such that pv source term and heat release rate are 
        always interpolated correctly for equilibrium conditions.
        """

        # Evaluate MLP predictions on chemical equilibrium data.
        Y_pred_boundary_norm = self.EvaluateMLP(self.X_boundary_norm)

        # Check maximum predicted source term value.
        Y_pred_boundary = self.scaler_function_y.inverse_transform(Y_pred_boundary_norm)
        Y_pred_boundary_max = np.max(Y_pred_boundary, axis=0)
        Y_pred_boundary_max = np.maximum(Y_pred_boundary_max, 0.0)

        # Normalize predicted and target values.
        Y_target_boundary = np.zeros(np.shape(Y_pred_boundary_max))
        Y_pred_boundary_max_norm = self.scaler_function_y.transform(np.expand_dims(Y_pred_boundary_max, 0))[0,:]
        Y_target_boundary_norm = self.scaler_function_y.transform(np.expand_dims(Y_target_boundary, 0))[0,:]

        # Apply shift to last bias values according to predicted source term values.
        delta_source = Y_target_boundary_norm - Y_pred_boundary_max_norm
        self._biases[-1].assign_add(tf.constant(delta_source, self._dt))
        return
    
    def PostProcessing(self):

        # Apply correction for heat release and/or pv source term.
        if any([(v == "Heat_Release" or v == "ProdRateTot_PV") for v in self._train_vars]):
            self.SourcetermCorrection()
        self.PlotUnbData()

        return super().PostProcessing()
    
class NullMLP(CustomTrainer):
    __Config:FlameletAIConfig = None 
    def __init__(self, Config_in:FlameletAIConfig):
        """Empty MLP to output zero for the NULL variable in SU2 simulations.

        :param Config_in: FlameletAI configuration for the current manifold
        :type Config_in: FlameletAIConfig
        """

        self.__Config = Config_in 

        # Store controlling variables and null variable as output.
        self._controlling_vars = self.__Config.GetControllingVariables()
        self._train_vars = ["NULL"]
        
        # Define simple hidden layer architecture with default activation function.
        self._hidden_layers = [len(self._controlling_vars)]
        self.SetActivationFunction("linear")

        self._train_name = "NULL"

        return 
    
    def GetTrainData(self):
        """Extract controlling variable ranges from manifold data.
        """
        MLPData_filepath = self._filedata_train
        x_vars = self._controlling_vars
        X_full, _ = GetReferenceData(MLPData_filepath + "_full.csv", x_vars, [])
        self.scaler_function_x.fit(X_full)

        self._X_scale = self.scaler_function_x.scale_
        if self.scaler_function_name == "standard":
            self._X_offset = self.scaler_function_x.mean_
        elif self.scaler_function_name == "robust":
            self._X_offset = self.scaler_function_x.center_
        elif self.scaler_function_name == "minmax":  
            self._X_scale = 1 / self.scaler_function_x.scale_
            
            self._X_offset = -self.scaler_function_x.min_ / self.scaler_function_x.scale_

        return 
    
    def add_additional_header_info(self, fid):
        fid.write("Progress variable definition: " + "+".join(("%+.6e*%s" % (w, s)) for w, s in zip(self.__Config.GetProgressVariableWeights(), self.__Config.GetProgressVariableSpecies())))
        fid.write("\n\n")
        return super().add_additional_header_info(fid)
    
    def InitializeWeights_and_Biases(self):
        """Initialize network weights and biases using zeros.
        """

        self._weights = []
        self._biases = []

        NN = [len(self._controlling_vars)]
        for N in self._hidden_layers:
            NN.append(N)
        NN.append(len(self._train_vars))

        for i in range(len(NN)-1):
            self._weights.append(tf.Variable(tf.zeros(shape=(NN[i],NN[i+1]),dtype=self._dt),self._dt))
            self._biases.append(tf.Variable(tf.zeros(shape=(NN[i+1],),dtype=self._dt),self._dt))             
            
        return 

    def Save_Relevant_Data(self):
        """Generate zero MLP for predicting the NULL variable in SU2 simulations.
        """
        self.InitializeWeights_and_Biases()
        self.GetTrainData()
        self._Y_scale = np.ones(1)
        self._Y_offset = np.zeros(1)
        self.write_SU2_MLP(self._save_dir + "/Model_"+str(self._model_index)+"/MLP_"+self._train_name)
        return 
    
class EvaluateArchitecture_FGM(EvaluateArchitecture):
    """Class for training MLP architectures
    """

    __Config:FlameletAIConfig = None  # FlameletAI configuration describing the manifold.
    __output_group:int = 0      # MLP output group index for which to train MLP.
    __group_name:str = "Group1" # MLP output group name.
    __trainer_PINN:Train_FGM_PINN = None # Physics-informed trainer object.
    __kind_trainer:str = "direct"   # Kind of training process (direct of physicsinformed).

    # List of variables that will trigger the use of physics-informed training when any are present in the MLP output group.
    __PINN_variables:list[str] = ["Temperature", "ProdRateTot_PV", "Heat_Release", "Beta_ProgVar", "Beta_MixFrac","Beta_Enth"]

    def __init__(self, Config:FlameletAIConfig, group_idx:int=0):
        """Define EvaluateArchitecture instance and prepare MLP trainer with
        default settings.

        :param Config: FlameletAIConfig object describing the flamelet data manifold.
        :type Config: FlameletAIConfig
        :param group_idx: MLP output group index, defaults to 0
        :type group_idx: int, optional
        :raises Exception: if MLP output group index is undefined by flameletAI configuration.
        """

        self.__Config = Config 
        self.__output_group=group_idx
        self.CheckPINNVars()
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
        self.CheckPINNVars()
        self.SynchronizeTrainer()
        return 
    
    def CheckPINNVars(self):
        """Check if any of the variables in the MLP output group contain physics-informed variables and 
        initiate trainer object accordingly.
        """
        output_vars = self.__Config.GetMLPOutputGroup(self.__output_group)

        # Check for physics-informed variables in the MLP output group.
        if any([((v in self.__PINN_variables) or ("Y_dot" in v)) for v in output_vars]):
            self.__kind_trainer = "physicsinformed"
            self.__trainer_PINN = Train_FGM_PINN(Config_in=self.__Config,group_idx=self.__output_group)
            self._trainer_direct = self.__trainer_PINN
            print("Using physics-informed trainer.")
        else:
            self._trainer_direct = Train_Flamelet_Direct(Config_in=self.__Config, group_idx=self.__output_group)
            self.__kind_trainer = "direct"
            print("Using direct trainer.")
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
    
    def TrainPostprocessing(self):
        """Post-process MLP training by saving all relevant data/figures.
        Generate NULL MLP for predicting the NULL variable in SU2 FGM simulations.
        """

        # Generate NULL MLP.
        nulltrainer = NullMLP(self._Config)
        nulltrainer.SetTrainFileHeader(self._train_file_header)
        nulltrainer.SetSaveDir(self.worker_dir)
        nulltrainer.SetModelIndex(self.current_iter)
        nulltrainer.Save_Relevant_Data()

        return super().TrainPostprocessing()
    
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
        
        CV_flamelet_norm = Trainer.scaler_function_x.transform(CV_flamelet)
        
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
    return 
