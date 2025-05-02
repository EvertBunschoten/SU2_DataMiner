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
# Version: 2.0.0                                                                              |
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

from Common.DataDrivenConfig import Config_FGM
from Manifold_Generation.MLP.Trainer_Base import MLPTrainer, TensorFlowFit,PhysicsInformedTrainer,TrainMLP, CustomTrainer
from Common.CommonMethods import GetReferenceData
from Common.Properties import DefaultSettings_FGM as DefaultProperties
from Common.Properties import FGMVars

class Train_Flamelet_Direct(TensorFlowFit):
    __Config:Config_FGM
    _train_name:str 

    def __init__(self, Config_in:Config_FGM, group_idx:int=0):
        TensorFlowFit.__init__(self)
        # Set train variables based on what kind of network is trained
        self.__Config = Config_in
        self._filedata_train = self.__Config.GetOutputDir()+"/"+self.__Config.GetConcatenationFileHeader()
        self._controlling_vars = Config_in.GetControllingVariables().copy()
        self._train_vars = self.__Config.GetMLPOutputGroup(group_idx)
        self._train_name = "Group"+str(group_idx+1)
        self.SetInitializer("he_uniform")
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
    """Physics-informed trainer class for flamelet manifold applications.
    """

    __Config:Config_FGM = None    # FlameletAI configuration class to read output variables and hyper-parameters from.

    __include_Hradical:bool = False 
    def __init__(self, Config_in:Config_FGM, group_idx:int=0):
        """Class constructor. Initialize a physics-informed trainer object for a given output group.

        :param Config_in: FlameletAI configuration class.
        :type Config_in: Config_FGM
        :param group_idx: MLP output group index, defaults to 0
        :type group_idx: int, optional
        """

        # Initiate parent class.
        PhysicsInformedTrainer.__init__(self)
        self.__Config = Config_in 
        self.__include_Hradical = (FGMVars.Y_H.name in self.__Config.GetControllingVariables())
        self._controlling_vars = self.__Config.GetControllingVariables().copy()
        self._train_vars = self.__Config.GetMLPOutputGroup(group_idx)
        self.callback_every = 10
        self._boundary_data_file = self.__Config.GetOutputDir()+"/"+DefaultProperties.boundary_file_header+"_full.csv"
        
        # Synchronize settings with loaded configuration
        self._alpha_expo = self.__Config.GetAlphaExpo(group_idx)
        self._lr_decay = self.__Config.GetLRDecay(group_idx)
        self._batch_expo = self.__Config.GetBatchExpo(group_idx)
        self._activation_function = self.__Config.GetActivationFunction(group_idx)
        self.SetHiddenLayers(self.__Config.GetHiddenLayerArchitecture(group_idx))
        self._train_name = "Group"+str(group_idx+1)
        self.SetTrainStepType("Jacobi")
        self._enable_boundary_loss=True
        self._boundary_loss_patience=-1

        self.SetInitializer("he_uniform")

        return 
    
    def GetTrainData(self):
        """Read domain and boundary training data and pre-process reactant-product matrices for visualization.
        """
        super().GetTrainData()
        self.__GenerateBoundaryMatrices()
        return 
    
    def SetTrainVariables(self, train_vars: list[str]):
        """Define the dependent variables to train for. For physics-informed FGM training, the state variables are the same 
        as the co-state variables.

        :param train_vars: dependent variables to train for.
        :type train_vars: list[str]
        """
        self._state_vars = train_vars.copy()
        return super().SetTrainVariables(train_vars)
    

    def __SetEnth_projection(self):
        """Get the boundary projection vector in the direction of total enthalpy.

        :return: penalty projection array, target projected gradient
        :rtype: np.ndarray[float],np.ndarray[float]
        """
        projection_array_train = np.zeros(np.shape(self._X_boundary_norm))
        projection_array_train[:, self._controlling_vars.index(DefaultProperties.name_enth)] = 1.0 
        
        target_grad_array = np.zeros([len(projection_array_train)])
        return projection_array_train, target_grad_array
    
    def __SetYH_projection(self):
        projection_array_train = np.zeros(np.shape(self._X_boundary_norm))
        projection_array_train[:, self._controlling_vars.index(FGMVars.Y_H.name)] = 1.0 
        
        target_grad_array = np.ones([len(projection_array_train)])
        return projection_array_train, target_grad_array
    
    def __SetPVZ_projection(self):
        """Get the boundary projection vector along the progress variable-mixture fraction plane.

        :return: penalty projection array, target projected gradient
        :rtype: np.ndarray[float],np.ndarray[float]
        """
        is_unb = self.__LocateUnbBoundaryNodes()
        
        fuel_string = self.__Config.GetFuelString()
        oxidizer_string = self.__Config.GetOxidizerString()
        mixfrac_boundary_norm = self._X_boundary_norm[:, self._controlling_vars.index(DefaultProperties.name_mixfrac)]
        
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
        Z_st = self.__Config.gas.mixture_fraction( fuel_string,oxidizer_string)
        self.__Config.gas.equilibrate("TP")
        Y_b_st = self.__Config.gas.Y 
        pv_st = self.__Config.ComputeProgressVariable(variables=None, flamelet_data=None, Y_flamelet=Y_b_st[:,np.newaxis])[0]
        h_st = self.__Config.gas.enthalpy_mass
        cvs_st = np.zeros([1, len(self._controlling_vars)])
        cvs_st[0, self._controlling_vars.index(FGMVars.ProgressVariable.name)] = pv_st 
        cvs_st[0, self._controlling_vars.index(FGMVars.EnthalpyTot.name)] = h_st 
        cvs_st[0, self._controlling_vars.index(FGMVars.MixtureFraction.name)] = Z_st
        if self.__include_Hradical:
            H_st = self.__Config.gas.Y[self.__Config.gas.species_index("H")]
            cvs_st[0, self._controlling_vars.index(FGMVars.Y_H.name)] = H_st
        
        X_st_norm = self.scaler_function_x.transform(cvs_st)[0,:]
        Z_st_norm = X_st_norm[2]

        # Identify rich and lean boundary nodes.
        is_rich = mixfrac_boundary_norm > Z_st_norm 
        is_stoch = mixfrac_boundary_norm == Z_st_norm
        is_lean = np.invert(is_rich)

        # Calculate progress variable-mixture fraction derivative for reactants and products.
        dpv_dz_unb = (pv_f - pv_ox) / (1.0 - 0.0)
        dpv_dz_b_lean = (pv_st - pv_ox) / (Z_st - 0.0)
        dpv_dz_b_rich = (pv_f - pv_st) / (1 - Z_st)
        
        pv_scale = self._X_scale[self._controlling_vars.index(DefaultProperties.name_pv)]
        z_scale = self._X_scale[self._controlling_vars.index(DefaultProperties.name_mixfrac)]
        projection_unb_pvz = dpv_dz_unb * z_scale / pv_scale
        projection_b_pvz_lean = dpv_dz_b_lean * z_scale / pv_scale
        projection_b_pvz_rich = dpv_dz_b_rich * z_scale /pv_scale

        # Populate projection arrays for the boundary condition.
        projection_array_pvz = np.zeros(np.shape(self._X_boundary_norm))

        projection_array_pvz[:, self._controlling_vars.index(DefaultProperties.name_mixfrac)] = 1.0
        projection_array_pvz[:, self._controlling_vars.index(DefaultProperties.name_enth)] = 0.0
        projection_array_pvz[is_unb, self._controlling_vars.index(DefaultProperties.name_pv)] = projection_unb_pvz
        projection_array_pvz[np.logical_and(np.invert(is_unb),is_rich), self._controlling_vars.index(DefaultProperties.name_pv)] = projection_b_pvz_rich
        projection_array_pvz[np.logical_and(np.invert(is_unb),is_lean), self._controlling_vars.index(DefaultProperties.name_pv)] = projection_b_pvz_lean
        projection_array_pvz[np.logical_and(np.invert(is_unb),is_stoch), self._controlling_vars.index(DefaultProperties.name_pv)] = 0.0
        
        return projection_array_pvz, is_unb, is_lean, is_stoch
    
    def __LocateUnbBoundaryNodes(self):
        """Locate the reactant and product nodes of the boundary data.

        :return: reactant identification array.
        :rtype: np.ndarray[bool]
        """

        # Initiate default reactant identifier array.
        is_unb = np.ones(np.shape(self._X_boundary_norm)[0],dtype=bool)
        
        fuel_string = self.__Config.GetFuelString()
        oxidizer_string = self.__Config.GetOxidizerString()
        
        # Loop over boundary nodes and locate indices where the progress variable is greater than the reactant progress variable.
        X_boundary = self.scaler_function_x.inverse_transform(self._X_boundary_norm)
        for iz in range(np.shape(self._X_boundary_norm)[0]):
            z = X_boundary[iz,self._controlling_vars.index(DefaultProperties.name_mixfrac)]
            pv_boundary = X_boundary[iz,self._controlling_vars.index(DefaultProperties.name_pv)]
            self.__Config.gas.set_mixture_fraction(min(max(z,0),1.0), fuel_string, oxidizer_string)
            self.__Config.gas.TP=self.__Config.GetUnbTempBounds()[0], DefaultProperties.pressure
            Y = self.__Config.gas.Y 
            pv_unb = self.__Config.ComputeProgressVariable(variables=None, flamelet_data=None, Y_flamelet=Y[:,np.newaxis])
            
            if pv_boundary > (pv_unb+1e-3*self._X_scale[self._controlling_vars.index(DefaultProperties.name_pv)]):
                is_unb[iz] = False 

        return is_unb
    def __SetBeta_pv_projection(self):
        """Get boundary penalty gradient projection array and target projected gradient for the progress variable preferential diffusion scalar.

        :return: boundary projection array, target projected gradient, and penalty labels.
        :rtype: list[np.ndarray], list[np.ndarray], list[str]
        """

        # Enthalpy projection: dbeta_pv / dh = 0
        cv_PI = self._X_train_norm 
        projection_array_h = np.zeros(np.shape(cv_PI))
        projection_array_h[:, self._controlling_vars.index(FGMVars.EnthalpyTot.name)] = 1.0
        target_grad_h = np.zeros(len(cv_PI))

        bc_labels = ["Beta_pv : h"]
        projection_arrays = [projection_array_h]
        target_grad_arrays = [target_grad_h]
        cv_boundary_arrays = [cv_PI]

        # pv-Z projection: dpv / dZ * dbeta_pv / dpv + dbeta_pv / dZ = c
        projection_array_pvz, is_unb, is_lean, is_stoch = self.__SetPVZ_projection()
        beta_pv_scale = self._Y_scale[self._train_vars.index(FGMVars.Beta_ProgVar.name)]
        Z_scale = self._X_scale[self._controlling_vars.index(FGMVars.MixtureFraction.name)]
        target_grad_pvz = np.zeros(len(projection_array_pvz))
        
        Le_i = self.__Config.GetConstSpecieLewisNumbers()

        pv_species = self.__Config.GetProgressVariableSpecies()
        pv_weights = self.__Config.GetProgressVariableWeights()

        z_fuel = 1.0
        self.__Config.gas.set_mixture_fraction(z_fuel, self.__Config.GetFuelString(), self.__Config.GetOxidizerString())
        Y_fuel = self.__Config.gas.Y 
        
        z_ox = 0.0
        self.__Config.gas.set_mixture_fraction(z_ox, self.__Config.GetFuelString(), self.__Config.GetOxidizerString())
        Y_ox = self.__Config.gas.Y 
        
        self.__Config.gas.set_equivalence_ratio(1.0, self.__Config.GetFuelString(), self.__Config.GetOxidizerString())
        z_stoch = self.__Config.gas.mixture_fraction(self.__Config.GetFuelString(), self.__Config.GetOxidizerString())
        self.__Config.gas.TP = self.__Config.GetUnbTempBounds()[1], ct.one_atm 
        self.__Config.gas.equilibrate("TP")
        Y_stoch = self.__Config.gas.Y 
        
        beta_pv_fuel = beta_pv_ox = beta_pv_stoch = 0.0
        for sp, w in zip(pv_species, pv_weights):
            i_sp = self.__Config.gas.species_index(sp)
            beta_pv_fuel += (w / Le_i[i_sp]) * Y_fuel[i_sp]
            beta_pv_ox += (w / Le_i[i_sp]) * Y_ox[i_sp]
            beta_pv_stoch += (w / Le_i[i_sp]) * Y_stoch[i_sp]

        target_grad_pvz[is_unb] = (beta_pv_fuel - beta_pv_ox) / (z_fuel - z_ox)
        target_grad_pvz[np.logical_and(np.invert(is_unb), is_lean)] = (beta_pv_stoch - beta_pv_ox) / (z_stoch - z_ox)
        target_grad_pvz[np.logical_and(np.invert(is_unb), np.invert(is_lean))] = (beta_pv_fuel - beta_pv_stoch) / (z_fuel - z_stoch)
        target_grad_pvz[np.logical_and(np.invert(is_unb), is_stoch)] = 0.0

        target_grad_pvz *= (Z_scale / beta_pv_scale)

        projection_arrays.append(projection_array_pvz)
        target_grad_arrays.append(target_grad_pvz)
        bc_labels.append("Beta_pv : pv-Z")
        cv_boundary_arrays.append(self._X_boundary_norm)

        if self.__include_Hradical:

            # Y_H projection: dbeta_pv / dY_H = (alpha_H - alpha_N2) / Le_H
            beta_pv_scale = self._Y_scale[self._train_vars.index(FGMVars.Beta_ProgVar.name)]
            Y_H_scale = self._X_scale[self._controlling_vars.index(FGMVars.Y_H.name)]
            pv_species = self.__Config.GetProgressVariableSpecies()
            pv_weights = self.__Config.GetProgressVariableWeights()
            Le_H = self.__Config.GetConstSpecieLewisNumbers()[self.__Config.gas.species_index("H")]
            alpha_H = 0.0
            alpha_N2 = 0.0
            if "H" in pv_species:
                alpha_H = pv_weights[pv_species.index("H")]
            if ("N2") in pv_species:
                alpha_N2 = 0.0

            cv_PI = self._X_train_norm 
            target_grad_YH = np.ones(len(cv_PI))
            projection_array_YH = np.zeros(np.shape(cv_PI))
            projection_array_YH[:, self._controlling_vars.index(FGMVars.Y_H.name)] = 1.0
            target_grad_YH *= ((alpha_H - alpha_N2) / Le_H) * Y_H_scale / beta_pv_scale
        
            projection_arrays.append(projection_array_YH)
            target_grad_arrays.append(target_grad_YH)
            bc_labels.append("Beta_pv : (a_H - a_N2)/Le_H")
            cv_boundary_arrays.append(cv_PI)


        return cv_boundary_arrays, projection_arrays, target_grad_arrays, bc_labels 
    
    def __SetBeta_Z_projection(self):
        """Get boundary penalty gradient projection array and target projected gradient for the mixture fraction preferential diffusion scalar.

        :return: boundary projection array, target projected gradient, and penalty labels.
        :rtype: list[np.ndarray], list[np.ndarray], list[str]
        """

        # Enthalpy projection: dbeta_z / dh = 0
        cv_PI = self._X_train_norm
        projection_array_h = np.zeros(np.shape(cv_PI))
        projection_array_h[:,self._controlling_vars.index(FGMVars.EnthalpyTot.name)] = 1.0
        target_grad_h = np.zeros(len(cv_PI))
        
        bc_labels = ["Beta_Z : h"]
        projection_arrays = [projection_array_h]
        target_grad_arrays = [target_grad_h]
        cv_boundary_arrays = [cv_PI]

        # pv-Z projection: dpv / dZ * dbeta_z / dpv + dbeta_z / dZ = c
        projection_array_pvz, is_unb, is_lean, is_stoch = self.__SetPVZ_projection()
        beta_Z_scale = self._Y_scale[self._train_vars.index(FGMVars.Beta_MixFrac.name)]
        Z_scale = self._X_scale[self._controlling_vars.index(FGMVars.MixtureFraction.name)]
        target_grad_pvz = np.zeros(len(projection_array_pvz))
        
        Le_i = self.__Config.GetConstSpecieLewisNumbers()
        z_i = self.__Config.GetMixtureFractionCoefficients()
        z_Ns = self.__Config.GetMixtureFractionCoeff_Carrier()

        z_fuel = 1.0
        self.__Config.gas.set_mixture_fraction(z_fuel, self.__Config.GetFuelString(), self.__Config.GetOxidizerString())
        Y_fuel = self.__Config.gas.Y 
        beta_Z_fuel = np.sum((z_i - z_Ns) * Y_fuel / Le_i)

        z_ox = 0.0
        self.__Config.gas.set_mixture_fraction(z_ox, self.__Config.GetFuelString(), self.__Config.GetOxidizerString())
        Y_ox = self.__Config.gas.Y 
        beta_Z_ox = np.sum((z_i - z_Ns) * Y_ox / Le_i)
        
        self.__Config.gas.set_equivalence_ratio(1.0, self.__Config.GetFuelString(), self.__Config.GetOxidizerString())
        z_stoch = self.__Config.gas.mixture_fraction(self.__Config.GetFuelString(), self.__Config.GetOxidizerString())
        self.__Config.gas.TP = self.__Config.GetUnbTempBounds()[1], ct.one_atm 
        self.__Config.gas.equilibrate("TP")
        Y_stoch = self.__Config.gas.Y 
        beta_Z_stoch = np.sum((z_i - z_Ns) * Y_stoch / Le_i)

        target_grad_pvz[is_unb] = (beta_Z_fuel - beta_Z_ox) / (z_fuel - z_ox)
        target_grad_pvz[np.logical_and(np.invert(is_unb), is_lean)] = (beta_Z_stoch - beta_Z_ox) / (z_stoch - z_ox)
        target_grad_pvz[np.logical_and(np.invert(is_unb), np.invert(is_lean))] = (beta_Z_fuel - beta_Z_stoch) / (z_fuel - z_stoch)
        target_grad_pvz[np.logical_and(np.invert(is_unb), is_stoch)] = 0.0

        target_grad_pvz *= (Z_scale / beta_Z_scale)

        projection_arrays.append(projection_array_pvz)
        target_grad_arrays.append(target_grad_pvz)
        bc_labels.append("Beta_Z : pv-Z")
        cv_boundary_arrays.append(self._X_boundary_norm)

        if self.__include_Hradical:

            # Y_H projection: dbeta_z / dY_H = (z_H - z_N2) / Le_H
            beta_Z_scale = self._Y_scale[self._train_vars.index(FGMVars.Beta_MixFrac.name)]
            Y_H_scale = self._X_scale[self._controlling_vars.index(FGMVars.Y_H.name)]
            z_i = self.__Config.GetMixtureFractionCoefficients()
            z_H = z_i[self.__Config.gas.species_index("H")]
            z_Ns = self.__Config.GetMixtureFractionCoeff_Carrier()
            Le_H = self.__Config.GetConstSpecieLewisNumbers()[self.__Config.gas.species_index("H")]
            cv_H = self._X_train_norm
            target_grad_YH = np.ones(len(cv_H))
            projection_array_YH = np.zeros(np.shape(cv_H))
            projection_array_YH[:, self._controlling_vars.index(FGMVars.Y_H.name)] = 1.0
            target_grad_YH *= ((z_H - z_Ns) / Le_H) * Y_H_scale / beta_Z_scale
            
            projection_arrays.append(projection_array_YH)
            target_grad_arrays.append(target_grad_YH)
            bc_labels.append("Beta_Z : (z_H - z_N2)/Le_H")
            cv_boundary_arrays.append(cv_H)
        return cv_boundary_arrays, projection_arrays, target_grad_arrays, bc_labels 
    

    
    def __SetMolarWeight_projection(self):
        """Get boundary penalty gradient projection array and target projected gradient for the mean molecular weight.

        :return: boundary projection array, target projected gradient, and penalty labels.
        :rtype: list[np.ndarray], list[np.ndarray], list[str]
        """

        # Enthalpy projection: dWM / dh = 0

        cv_PI = self._X_train_norm
        projection_array_h = np.zeros(np.shape(cv_PI))
        projection_array_h[:,self._controlling_vars.index(FGMVars.EnthalpyTot.name)] = 1.0
        target_grad_array = np.zeros(len(cv_PI))
        
        bc_name = "W_M : h"
        cv_boundary_arrays = [cv_PI]
        projection_arrays = [projection_array_h]
        target_grad_arrays = [target_grad_array]

        return cv_boundary_arrays, projection_arrays, target_grad_arrays, [bc_name]
    
    def __SetSource_projection(self):
        """Get boundary penalty gradient projection arrays and target projected gradients for source terms.

        :return: boundary projection arrays, target projected gradients, and penalty labels.
        :rtype: list[np.ndarray], list[np.ndarray], list[str]
        """
        cv_boundary_arrays = []
        projection_arrays = []
        target_grad_arrays = []
        bc_labels = []

        # pv-Z projection: dpv/dZ * dsource/dpv + dsource/dZ = 0.0

        cv_PI = self._X_boundary_norm
        projection_array_pvz, _, _, _ = self.__SetPVZ_projection()
        target_grad_array_pvz = np.zeros(len(projection_array_pvz))
        label = "Source : pv-Z"
        cv_boundary_arrays.append(cv_PI)
        projection_arrays.append(projection_array_pvz)
        target_grad_arrays.append(target_grad_array_pvz)
        bc_labels.append(label)

        # Enthalpy projection: dsource / dh = 0.0
        cv_PI = self._X_boundary_norm
        projection_array_h = np.zeros(np.shape(cv_PI))
        projection_array_h[:, self._controlling_vars.index(FGMVars.EnthalpyTot.name)] = 1.0
        target_array_h = np.zeros(len(cv_PI))
        label = "Source : h"
        cv_boundary_arrays.append(cv_PI)
        projection_arrays.append(projection_array_h)
        target_grad_arrays.append(target_array_h)
        bc_labels.append(label)

        return cv_boundary_arrays, projection_arrays, target_grad_arrays, bc_labels
        
    def __SetT_projection(self):
        """Get boundary penalty gradient projection array and target projected gradient for temperature.

        :return: boundary projection array, target projected gradient, and penalty labels.
        :rtype: list[np.ndarray], list[np.ndarray], list[str]
        """

        # Enthalpy projection: dT / dh = 1/c_p 

        cv_PI_unscaled, Cp_boundary = GetReferenceData(self._boundary_data_file, x_vars=self._controlling_vars, train_variables=[FGMVars.Cp.name])
        cv_PI = self.scaler_function_x.transform(cv_PI_unscaled)
        projection_array_train, _ = self.__SetEnth_projection()
        T_scale = self._Y_scale[self._train_vars.index(FGMVars.Temperature.name)]
        h_scale = self._X_scale[self._controlling_vars.index(DefaultProperties.name_enth)]
        
        target_grad_array = (1.0 / Cp_boundary[:,0]) * h_scale / T_scale
        bc_name = "Temperature : 1/cp"

        cv_boundary_arrays = [cv_PI]
        projection_arrays = [projection_array_train]
        target_grads = [target_grad_array]
        bc_names = [bc_name]
        if self.__include_Hradical:

            # Y_H projection: dT / dY_H = 1 / c_p_H
            cv_PI = self._X_boundary_norm
            projection_array_train_YH = np.zeros(np.shape(cv_PI))
            projection_array_train_YH[:, self._controlling_vars.index(FGMVars.Y_H.name)] = 1.0
            target_grad_YH = np.ones(len(cv_PI))
            x_unscaled = self.scaler_function_x.inverse_transform(cv_PI)
            Y_H_scale = self._X_scale[self._controlling_vars.index(FGMVars.Y_H.name)]
            self.__Config.gas.set_equivalence_ratio(1.0, self.__Config.GetFuelString(), self.__Config.GetOxidizerString())
            val_Z_stoch = self.__Config.gas.mixture_fraction(self.__Config.GetFuelString(), self.__Config.GetOxidizerString())

            # Retrieve c_p_H from Cantera
            for i_b in range(len(x_unscaled)):
                x = x_unscaled[i_b, :]
                val_pv = x[self._controlling_vars.index(FGMVars.ProgressVariable.name)]
                val_h = x[self._controlling_vars.index(FGMVars.EnthalpyTot.name)]
                val_Z = x[self._controlling_vars.index(FGMVars.MixtureFraction.name)]
                
                self.__Config.gas.set_mixture_fraction(val_Z, self.__Config.GetFuelString(), self.__Config.GetOxidizerString())
                self.__Config.gas.TP = self.__Config.GetUnbTempBounds()[1], ct.one_atm
                val_pv_unb = self.__Config.ComputeProgressVariable(variables=None, flamelet_data=None, Y_flamelet=self.__Config.gas.Y[:,np.newaxis])[0]
                if val_pv > val_pv_unb+1e-3:
                    if val_Z > val_Z_stoch:
                        self.__Config.gas.equilibrate("HP")
                    else:
                        self.__Config.gas.equilibrate("TP")
                self.__Config.gas.HP = val_h, ct.one_atm

                cp_i = self.__Config.gas.partial_molar_cp/self.__Config.gas.molecular_weights
                cp_H = cp_i[self.__Config.gas.species_index("H")]
                target_grad_YH[i_b] = (1.0/cp_H) * Y_H_scale / T_scale

            cv_boundary_arrays.append(cv_PI)
            projection_arrays.append(projection_array_train_YH)
            target_grads.append(target_grad_YH)
            bc_names.append("Temperature : 1/cp_H")
        return cv_boundary_arrays, projection_arrays, target_grads, bc_names
    
    def __SetCp_projection(self):
        """Get boundary penalty gradient projection array and target projected gradient for the specific heat at constant pressure.

        :return: boundary projection array, target projected gradient, and penalty labels.
        :rtype: list[np.ndarray], list[np.ndarray], list[str]
        """
        Cp_scale = self._Y_scale[self._train_vars.index(FGMVars.Cp.name)]

        # c_p projection: dc_p / dY_H = c_p_H

        cv_PI = self._X_boundary_norm
        projection_array_train_YH = np.zeros(np.shape(cv_PI))
        target_grad_YH = np.ones(len(cv_PI))
        projection_array_train_YH[:, self._controlling_vars.index(FGMVars.Y_H.name)] = 1.0

        x_unscaled = self.scaler_function_x.inverse_transform(cv_PI)
        Y_H_scale = self._X_scale[self._controlling_vars.index(FGMVars.Y_H.name)]
        self.__Config.gas.set_equivalence_ratio(1.0, self.__Config.GetFuelString(), self.__Config.GetOxidizerString())
        val_Z_stoch = self.__Config.gas.mixture_fraction(self.__Config.GetFuelString(), self.__Config.GetOxidizerString())

        for i_b in range(len(x_unscaled)):
            x = x_unscaled[i_b, :]
            val_pv = x[self._controlling_vars.index(FGMVars.ProgressVariable.name)]
            val_h = x[self._controlling_vars.index(FGMVars.EnthalpyTot.name)]
            val_Z = x[self._controlling_vars.index(FGMVars.MixtureFraction.name)]
            
            self.__Config.gas.set_mixture_fraction(val_Z, self.__Config.GetFuelString(), self.__Config.GetOxidizerString())
            self.__Config.gas.TP = self.__Config.GetUnbTempBounds()[1], ct.one_atm
            val_pv_unb = self.__Config.ComputeProgressVariable(variables=None, flamelet_data=None, Y_flamelet=self.__Config.gas.Y[:,np.newaxis])[0]
            if val_pv > val_pv_unb+1e-3:
                if val_Z > val_Z_stoch:
                    self.__Config.gas.equilibrate("HP")
                else:
                    self.__Config.gas.equilibrate("TP")
            self.__Config.gas.HP = val_h, ct.one_atm

            cp_i = self.__Config.gas.partial_molar_cp/self.__Config.gas.molecular_weights
            cp_H = cp_i[self.__Config.gas.species_index("H")]
            target_grad_YH[i_b] = (cp_H) * Y_H_scale / Cp_scale

        projection_arrays = [projection_array_train_YH]
        target_grads = [target_grad_YH]
        bc_names = ["Cp : cp_H"]
        cv_boundary_arrays = [cv_PI]
        return cv_boundary_arrays, projection_arrays, target_grads, bc_names
    
    def __SetBeta_h1_projection(self):
        """Get boundary penalty gradient projection array and target projected gradient for the specific heat preferential diffusion scalar.

        :return: boundary projection array, target projected gradient, and penalty labels.
        :rtype: list[np.ndarray], list[np.ndarray], list[str]
        """
        Beta_h1_scale = self._Y_scale[self._train_vars.index(FGMVars.Beta_Enth_Thermal.name)]

        cv_boundary_arrays = []
        projection_arrays = []
        target_grads = []
        bc_names = []

        # Y_H projection: dbeta_h1/dY_H = (cp_H - cp_N2)/Le_H

        cv_PI = self._X_boundary_norm
        projection_array_train_YH = np.zeros(np.shape(cv_PI))
        projection_array_train_YH[:, self._controlling_vars.index(FGMVars.Y_H.name)] = 1.0
        target_grad_YH = np.ones(len(cv_PI))
        Le_H = self.__Config.GetConstSpecieLewisNumbers()[self.__Config.gas.species_index("H")]
        x_unscaled = self.scaler_function_x.inverse_transform(cv_PI)
        Y_H_scale = self._X_scale[self._controlling_vars.index(FGMVars.Y_H.name)]

        self.__Config.gas.set_equivalence_ratio(1.0, self.__Config.GetFuelString(), self.__Config.GetOxidizerString())
        val_Z_stoch = self.__Config.gas.mixture_fraction(self.__Config.GetFuelString(), self.__Config.GetOxidizerString())

        for i_b in range(len(x_unscaled)):
            x = x_unscaled[i_b, :]
            val_pv = x[self._controlling_vars.index(FGMVars.ProgressVariable.name)]
            val_h = x[self._controlling_vars.index(FGMVars.EnthalpyTot.name)]
            val_Z = x[self._controlling_vars.index(FGMVars.MixtureFraction.name)]
            self.__Config.gas.set_mixture_fraction(val_Z, self.__Config.GetFuelString(), self.__Config.GetOxidizerString())
            val_pv_unb = self.__Config.ComputeProgressVariable(variables=None, flamelet_data=None, Y_flamelet=self.__Config.gas.Y[:,np.newaxis])[0]
            if val_pv > val_pv_unb+1e-3:
                if val_Z > val_Z_stoch:
                    self.__Config.gas.equilibrate("HP")
                else:
                    self.__Config.gas.equilibrate("TP")
            self.__Config.gas.HP = val_h, 101325
            cp_i = self.__Config.gas.partial_cp/self.__Config.gas.molecular_weights
            cp_H = cp_i[self.__Config.gas.species_index("H")]
            cp_carrier = cp_i[self.__Config.gas.species_index("N2")]
            target_grad_YH[i_b] = ((cp_H - cp_carrier) / Le_H) * (Y_H_scale) / Beta_h1_scale
        projection_arrays.append(projection_array_train_YH)
        target_grads.append(target_grad_YH)
        bc_names.append("Beta_h1 : (cp_H - cp_N2)/Le_H")
        cv_boundary_arrays.append(cv_PI)
        return cv_boundary_arrays, projection_arrays, target_grads, bc_names 
    
    def __SetBeta_h2_projection(self):
        """Get boundary penalty gradient projection array and target projected gradient for the formation enthalpy preferential diffusion scalar.

        :return: boundary projection array, target projected gradient, and penalty labels.
        :rtype: list[np.ndarray], list[np.ndarray], list[str]
        """

        # Enthalpy projection: dbeta_h2 / dh = 1 - beta_h1/c_p

        # Extract specific heat and beta_h1 from flamelet data.
        cv_PI_unscaled, Y_boundary = GetReferenceData(self._boundary_data_file, x_vars=self._controlling_vars, train_variables=[FGMVars.Cp.name, FGMVars.Beta_Enth_Thermal.name])
        cv_PI = self.scaler_function_x.transform(cv_PI_unscaled)
        Cp_boundary = Y_boundary[:,0]
        Beta_h1_boundary = Y_boundary[:,1]

        # Extract normalization scales.
        Beta_h2_scale = self._Y_scale[self._train_vars.index(FGMVars.Beta_Enth.name)]
        h_scale = self._X_scale[self._controlling_vars.index(DefaultProperties.name_enth)]

        # Set projection array along total enthalpy direction.
        projection_array_train_enth = np.zeros(np.shape(cv_PI))
        projection_array_train_enth[:, self._controlling_vars.index(FGMVars.EnthalpyTot.name)] = 1.0

        # Normalized projected target gradient.
        target_grad_array_enth = (1 - (Beta_h1_boundary / Cp_boundary)) * h_scale / Beta_h2_scale

        cv_boundary_arrays = [cv_PI]
        projection_arrays = [projection_array_train_enth]
        target_grads = [target_grad_array_enth]
        bc_names = ["Beta_h2 : 1 - Beta_h1/cp"]

        if self.__include_Hradical:
            # Y_H projection: dbeta_h2 / dY_H = (h_H - h_N2)/Le_H

            cv_PI = self._X_train_norm
            projection_array_train_YH = np.zeros(np.shape(cv_PI))
            projection_array_train_YH[:, self._controlling_vars.index(FGMVars.Y_H.name)] = 1.0
            target_grad_YH = np.ones(len(cv_PI))
            
            Le_H = self.__Config.GetConstSpecieLewisNumbers()[self.__Config.gas.species_index("H")]
            x_unscaled = self.scaler_function_x.inverse_transform(cv_PI)
            Y_H_scale = self._X_scale[self._controlling_vars.index(FGMVars.Y_H.name)]
            self.__Config.gas.set_equivalence_ratio(1.0, self.__Config.GetFuelString(), self.__Config.GetOxidizerString())
            val_Z_stoch = self.__Config.gas.mixture_fraction(self.__Config.GetFuelString(), self.__Config.GetOxidizerString())

            for i_b in range(len(x_unscaled)):
                x = x_unscaled[i_b, :]
                val_pv = x[self._controlling_vars.index(FGMVars.ProgressVariable.name)]
                val_h = x[self._controlling_vars.index(FGMVars.EnthalpyTot.name)]
                val_Z = x[self._controlling_vars.index(FGMVars.MixtureFraction.name)]
                self.__Config.gas.set_mixture_fraction(val_Z, self.__Config.GetFuelString(), self.__Config.GetOxidizerString())
                val_pv_unb = self.__Config.ComputeProgressVariable(variables=None, flamelet_data=None, Y_flamelet=self.__Config.gas.Y[:,np.newaxis])[0]
                if val_pv > val_pv_unb+1e-3:
                    if val_Z > val_Z_stoch:
                        self.__Config.gas.equilibrate("HP")
                    else:
                        self.__Config.gas.equilibrate("TP")
                self.__Config.gas.HP = val_h, DefaultProperties.pressure
                h_i = self.__Config.gas.partial_molar_enthalpies/self.__Config.gas.molecular_weights
                h_H = h_i[self.__Config.gas.species_index("H")]
                h_carrier = h_i[self.__Config.gas.species_index("N2")]
                target_grad_YH[i_b] = ((h_H - h_carrier) / Le_H) * (Y_H_scale) / Beta_h2_scale
            cv_boundary_arrays.append(cv_PI)
            projection_arrays.append(projection_array_train_YH)
            target_grads.append(target_grad_YH)
            bc_names.append("Beta_h2 : (h_H - h_N2)/Le_H")
        return cv_boundary_arrays, projection_arrays, target_grads, bc_names 
    
    def CollectPIVars(self):
        """Collect the Jacobian projection arrays and projected target arrays for any physics-informed variables in the training data set.
        """

        val_lambda_default = super().CollectPIVars()
        for ivar, var in enumerate(self._train_vars):
            proj_method = None 

            # Check which projection method should be used.
            if (var == FGMVars.ProdRateTot_PV.name) or (var == FGMVars.Heat_Release.name) or ("Y_dot" in var):
                proj_method = self.__SetSource_projection

            if (var == FGMVars.Beta_ProgVar.name):
                proj_method = self.__SetBeta_pv_projection

            if (var == FGMVars.Beta_MixFrac.name):
                proj_method = self.__SetBeta_Z_projection

            if (var == FGMVars.MolarWeightMix.name):
                proj_method = self.__SetMolarWeight_projection

            if (var == FGMVars.Temperature.name):
                proj_method = self.__SetT_projection

            if (var == FGMVars.Beta_Enth.name):
                proj_method = self.__SetBeta_h2_projection

            if (self.__include_Hradical and (var == FGMVars.Beta_Enth_Thermal.name)):
                proj_method = self.__SetBeta_h1_projection

            if (self.__include_Hradical and (var == FGMVars.Cp.name)):
                proj_method = self.__SetCp_projection
            
            # Collect consistency penalty controlling variable data, projection arrays, and target gradient arrays.
            if proj_method:
                cv_arrays, proj_arrays, target_grads, bc_names = proj_method()
                for cv, p, t, b in zip(cv_arrays, proj_arrays, target_grads, bc_names):
                    self.idx_PIvar.append(ivar)
                    self.projection_arrays.append(p)
                    self.target_arrays.append(t)
                    self.lamba_labels.append(b)
                    self.cv_boundary_arrays.append(cv)
                    self.vals_lambda.append(val_lambda_default)
        self._N_bc = len(self.vals_lambda) 
        return 
    
    @tf.function 
    def EvaluateState(self, X_label_norm:tf.constant):
        Y_pred_norm = self._MLP_Evaluation(X_label_norm)
        Y_pred = Y_pred_norm * self._Y_state_scale_tf + self._Y_state_offset_tf
        return Y_pred 
    
    @tf.function 
    def ComputeStateError(self, X_label_norm:tf.constant,Y_state_label_norm:tf.constant):
        return self.Compute_Direct_Error(X_label_norm, Y_state_label_norm)
    
    
    def __GenerateBoundaryMatrices(self):
        """Generate controlling variable matrices for boundary conditions, where predicted quantities are visualized onto during convergence.
        """

        if self._verbose > 0:
            print("Generating boundary data matrix...")
        mixfrac_range = np.linspace(0, 1, self.__Config.GetNpMix())
        T_range = np.linspace(self.__Config.GetUnbTempBounds()[0], self.__Config.GetUnbTempBounds()[1], self.__Config.GetNpTemp())
        self.pv_unb = np.zeros([len(mixfrac_range), len(T_range)])
        self.h_unb = np.zeros([len(mixfrac_range), len(T_range)])
        self.z_unb = np.zeros([len(mixfrac_range), len(T_range)])
        self.y_h_unb = np.zeros([len(mixfrac_range), len(T_range)])
        self.pv_b = np.zeros([len(mixfrac_range), len(T_range)])
        self.h_b = np.zeros([len(mixfrac_range), len(T_range)])
        self.z_b = np.zeros([len(mixfrac_range), len(T_range)])
        self.y_h_b = np.zeros([len(mixfrac_range), len(T_range)])

        gas_unb = ct.Solution(self.__Config.GetReactionMechanism())
        gas_b = ct.Solution(self.__Config.GetReactionMechanism())
        gas_unb.set_equivalence_ratio(1.0, self.__Config.GetFuelString(), self.__Config.GetOxidizerString())
        gas_b.set_equivalence_ratio(1.0, self.__Config.GetFuelString(), self.__Config.GetOxidizerString())
        gas_unb.TP = T_range[0], DefaultProperties.pressure
        gas_b.TP = T_range[0], DefaultProperties.pressure
        
        for iZ, Z in enumerate(mixfrac_range):
            gas_unb.set_mixture_fraction(Z, self.__Config.GetFuelString(), self.__Config.GetOxidizerString())
            gas_b.set_mixture_fraction(Z, self.__Config.GetFuelString(), self.__Config.GetOxidizerString())
            gas_unb.TP = T_range[0], DefaultProperties.pressure
            gas_b.TP = T_range[-1], DefaultProperties.pressure
            pv_unb = self.__Config.ComputeProgressVariable(variables=None, flamelet_data=None, Y_flamelet=gas_unb.Y[:,np.newaxis])[0]
            gas_unb.TP = T_range[-1], DefaultProperties.pressure
            H_max = gas_unb.enthalpy_mass
            gas_b.TP = T_range[0], DefaultProperties.pressure
            gas_b.equilibrate("HP")
            gas_b.HP = H_max, DefaultProperties.pressure 
            T_range_b = np.linspace(self.__Config.GetUnbTempBounds()[0], gas_b.T, self.__Config.GetNpTemp())
            pv_b = self.__Config.ComputeProgressVariable(variables=None, flamelet_data=None, Y_flamelet=gas_b.Y[:,np.newaxis])[0]
            for iT, T in enumerate(T_range):
                gas_unb.TP = T, DefaultProperties.pressure
                self.pv_unb[iZ, iT] = pv_unb
                self.h_unb[iZ, iT] = gas_unb.enthalpy_mass
                self.z_unb[iZ, iT] = Z 
                self.y_h_unb[iZ, iT] = gas_unb.Y[gas_unb.species_index("H")]

                gas_b.TP = T_range_b[iT], DefaultProperties.pressure
                self.pv_b[iZ, iT] = pv_b
                self.h_b[iZ, iT] = gas_b.enthalpy_mass
                self.z_b[iZ, iT] = Z 
                self.y_h_b[iZ, iT] = gas_b.Y[gas_unb.species_index("H")]
        if self._verbose > 0:
            print("Done!")
        
        return 
    
    def __PlotUnbData(self):
        """Plot predicted quantities on chemical equilibrium data.
        """
        pv_unb = self.pv_unb.flatten()
        h_unb = self.h_unb.flatten()
        z_unb = self.z_unb.flatten()
        
        pv_b = self.pv_b.flatten()
        h_b = self.h_b.flatten()
        z_b = self.z_b.flatten()
        
        X_unb = np.zeros([len(pv_unb), len(self._controlling_vars)])
        X_b = np.zeros([len(pv_unb), len(self._controlling_vars)])

        X_unb[:, self._controlling_vars.index(FGMVars.ProgressVariable.name)] = pv_unb
        X_unb[:, self._controlling_vars.index(FGMVars.EnthalpyTot.name)] = h_unb
        X_unb[:, self._controlling_vars.index(FGMVars.MixtureFraction.name)] = z_unb
        
        X_b[:, self._controlling_vars.index(FGMVars.ProgressVariable.name)] = pv_b
        X_b[:, self._controlling_vars.index(FGMVars.EnthalpyTot.name)] = h_b
        X_b[:, self._controlling_vars.index(FGMVars.MixtureFraction.name)] = z_b

        if self.__include_Hradical:
            y_h_unb = self.y_h_unb.flatten()
            y_h_b = self.y_h_b.flatten()
            X_unb[:, self._controlling_vars.index(FGMVars.Y_H.name)] = y_h_unb
            X_b[:, self._controlling_vars.index(FGMVars.Y_H.name)] = y_h_b
            
        X_unb_norm_np = self.scaler_function_x.transform(X_unb)
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
    
    def CustomCallback(self):
        self.Plot_and_Save_History()
        PlotFlameletData(self, self.__Config, self._train_name)
        self.__PlotUnbData()
        self.PlotR2Data()
        self.write_SU2_MLP(self._save_dir + "/Model_"+str(self._model_index)+"/MLP_"+self._train_name)
        return super().CustomCallback()
    
    def add_additional_header_info(self, fid):
        fid.write("Progress variable definition: " + "+".join(("%+.6e*%s" % (w, s)) for w, s in zip(self.__Config.GetProgressVariableWeights(), self.__Config.GetProgressVariableSpecies())))
        fid.write("\n\n")
        return super().add_additional_header_info(fid)
    
    def __SourcetermCorrection(self):
        """Apply correction to source terms such that pv source term and heat release rate are 
        always interpolated correctly for equilibrium conditions.
        """

        # Evaluate MLP predictions on chemical equilibrium data.
        X_boundary = self.scaler_function_x.inverse_transform(self._X_boundary_norm)
        Y_pred_boundary = self.EvaluateMLP(X_boundary)

        # Check maximum predicted source term value.
        Y_pred_boundary_max = np.max(Y_pred_boundary, axis=0)
        Y_pred_boundary_max = np.maximum(Y_pred_boundary_max, 0.0)


        # Normalize predicted and target values.
        Y_target_boundary = np.zeros(np.shape(Y_pred_boundary_max))
        Y_pred_boundary_max_norm = self.scaler_function_y.transform(np.expand_dims(Y_pred_boundary_max, 0))[0,:]
        Y_target_boundary_norm = self.scaler_function_y.transform(np.expand_dims(Y_target_boundary, 0))[0,:]

        # Apply shift to last bias values according to predicted source term values.
        delta_source = Y_target_boundary_norm - Y_pred_boundary_max_norm

        is_source = np.array([(v==FGMVars.ProdRateTot_PV.name) or (v==FGMVars.Heat_Release.name) or ("Y_dot" in v) for v in self._train_vars])
        delta_source[np.invert(is_source)] = 0.0
        self._biases[-1].assign_add(tf.constant(delta_source, self._dt))
        return
    
    def PostProcessing(self):

        # Apply correction for heat release and/or pv source term.
        if any([(v == FGMVars.Heat_Release.name or v == FGMVars.ProdRateTot_PV.name) for v in self._train_vars]):
            self.__SourcetermCorrection()
        self.__PlotUnbData()

        return super().PostProcessing()
    
class NullMLP(CustomTrainer):
    __Config:Config_FGM = None 
    def __init__(self, Config_in:Config_FGM):
        """Empty MLP to output zero for the NULL variable in SU2 simulations.

        :param Config_in: FlameletAI configuration for the current manifold
        :type Config_in: Config_FGM
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
    
class TrainMLP_FGM(TrainMLP):
    """Class for training MLP architectures
    """

    __Config:Config_FGM = None  # FlameletAI configuration describing the manifold.
    __output_group:int = 0      # MLP output group index for which to train MLP.
    __group_name:str = "Group1" # MLP output group name.
    __trainer_PINN:Train_FGM_PINN = None # Physics-informed trainer object.
    __kind_trainer:str = "direct"   # Kind of training process (direct of physicsinformed).

    # List of variables that will trigger the use of physics-informed training when any are present in the MLP output group.
    __PINN_variables:list[str] = [FGMVars.Temperature.name, \
                                  FGMVars.ProdRateTot_PV.name, \
                                  FGMVars.Heat_Release.name, \
                                  FGMVars.Beta_ProgVar.name,\
                                  FGMVars.Beta_MixFrac.name,\
                                  FGMVars.Beta_Enth.name,\
                                  FGMVars.MolarWeightMix.name]

    def __init__(self, Config:Config_FGM, group_idx:int=0):
        """Define TrainMLP instance and prepare MLP trainer with
        default settings.

        :param Config: Config_FGM object describing the flamelet data manifold.
        :type Config: Config_FGM
        :param group_idx: MLP output group index, defaults to 0
        :type group_idx: int, optional
        :raises Exception: if MLP output group index is undefined by flameletAI configuration.
        """

        self.__Config = Config 
        self.__output_group=group_idx
        self.CheckPINNVars()
        TrainMLP.__init__(self, Config_in=Config)
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
        :raises Exception: if group is not supported by loaded Config_FGM class.
        """
        if iGroup < 0 or iGroup >= self._Config.GetNMLPOutputGroups():
            raise Exception("MLP output group index should be between 0 and %i" % self._Config.GetNMLPOutputGroups())
        self.__output_group=iGroup
        self.__group_name = "Group"+str(self.__output_group+1)
        self.main_save_dir = self._Config.GetOutputDir() + "/architectures_"+self.__group_name
        if not os.path.isdir(self.main_save_dir):
            os.mkdir(self.main_save_dir)
        self.alpha_expo = self._Config.GetAlphaExpo(self.__output_group)
        self.lr_decay = self._Config.GetLRDecay(self.__output_group)
        self.batch_expo = self._Config.GetBatchExpo(self.__output_group)
        self.activation_function = self._Config.GetActivationFunction(self.__output_group)
        self.architecture = []
        for n in self._Config.GetHiddenLayerArchitecture(self.__output_group):
            self.architecture.append(n)
        self.CheckPINNVars()
        self.SynchronizeTrainer()
        return 
    
    def GetOutputGroup(self):
        return self.__output_group
    
    def EnableBCLoss(self, enable_bc_loss:bool=True):
        if self.__kind_trainer == "physicsinformed":
            self.__trainer_PINN.EnableBCLoss(enable_bc_loss)
        return 
    
    def CheckPINNVars(self):
        """Check if any of the variables in the MLP output group contain physics-informed variables and 
        initiate trainer object accordingly.
        """
        output_vars = self.__Config.GetMLPOutputGroup(self.__output_group)
        self.__trainer_PINN = Train_FGM_PINN(Config_in=self.__Config,group_idx=self.__output_group)
        self.__kind_trainer = "physicsinformed"
        self._trainer_direct = self.__trainer_PINN
        # Check for physics-informed variables in the MLP output group.
        if any([((v in self.__PINN_variables) or ("Y_dot" in v)) for v in output_vars]):
            self.__kind_trainer = "physicsinformed"
            self.__trainer_PINN = Train_FGM_PINN(Config_in=self.__Config,group_idx=self.__output_group)
            self._trainer_direct = self.__trainer_PINN
            if self.verbose > 0:
                print("Using physics-informed trainer.")
        else:
            self._trainer_direct = Train_Flamelet_Direct(Config_in=self.__Config, group_idx=self.__output_group)
            self.__kind_trainer = "direct"
            if self.verbose > 0:
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
    
def PlotFlameletData(Trainer:MLPTrainer, Config:Config_FGM, train_name:str):
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
            elif Cv == FGMVars.Y_H.name:
                CV_flamelet[:, iCv] = flameletData[:, variables_flamelet.index("Y-H")]
            else:
                CV_flamelet[:, iCv] = flameletData[:, variables_flamelet.index(Cv)]
        
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
        pred_data = Trainer.EvaluateMLP(CV_flamelet)
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
