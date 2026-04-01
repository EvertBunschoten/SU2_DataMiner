#!/usr/bin/env python3
import numpy as np
from random import sample,choice
import os, sys
from su2dataminer.config import Config_FGM
from su2dataminer.manifold import Train_FGM_PINN
from su2dataminer.generate_data import ComputeFlameletData,ComputeBoundaryData
from su2dataminer.process_data import FlameletConcatenator
from mlpcppwrapper import MLPCppEvaluator 

np.random.seed(1)

config = Config_FGM()
config.SetFuelDefinition(["H2"], [1.0])
config.SetReactionMechanism("h2o2.yaml")
config.SetTransportModel('multicomponent')
config.SetNpConcatenation(200)
config.SetMixtureBounds(0.5, 0.6)
config.SetNpMix(2)
config.SetUnbTempBounds(300, 350)
config.SetNpTemp(6)
config.RunFreeFlames(True)
config.RunBurnerFlames(False)
config.RunEquilibrium(True)
config.AddOutputGroup(['Temperature'])
config.SaveConfig()

ComputeFlameletData(config)
ComputeBoundaryData(config)

F = FlameletConcatenator(config, verbose_level=0)
F.ConcatenateFlameletData()
F.CollectBoundaryData()

flamelet_to_test = choice(os.listdir("freeflame_data/phi_0.5/"))
flamelet_data_file = "freeflame_data/phi_0.5/%s" % flamelet_to_test

with open(flamelet_data_file,'r') as fid:
    vars_flamelet = fid.readline().strip().split(',')
flamelet_data = np.loadtxt(flamelet_data_file,delimiter=',',skiprows=1)
with open(config.GetConcatenationFileHeader()+"_train.csv",'r') as fid:
    train_vars = fid.readline().strip().split(',')[3:]

pv_test = config.ComputeProgressVariable(vars_flamelet,flamelet_data)
h_test = flamelet_data[:, vars_flamelet.index("EnthalpyTot")]
Z_test = flamelet_data[:, vars_flamelet.index("MixtureFraction")]

CV_flamelet_test = np.vstack((pv_test,h_test,Z_test)).T 
controlling_vars = ["ProgressVariable","EnthalpyTot","MixtureFraction"]
activation_functions = ["linear",'relu','elu','tanh','sigmoid']
scalers = ['robust','standard']

def calc_error(MLP_output_Tensorflow:np.ndarray[float], MLP_output_MLPCpp:np.ndarray[float]):
    return(np.sqrt(np.average(np.power((MLP_output_Tensorflow-MLP_output_MLPCpp)/(MLP_output_MLPCpp),2))))

max_error = 0.0
passed = True 
bad_combos = []
for k in range(20):
    query_vars = sample(train_vars, np.random.randint(1, 6))
    activation_function = choice(activation_functions)
    scaler = choice(scalers)

    M_H = np.random.randint(3, 6)
    N_H = []
    for h in range(M_H):
        N_H.append(np.random.randint(3, 10))

    T = Train_FGM_PINN(config, 0)
    T.SetVerbose(0)
    T.SetHiddenLayers(N_H)
    T.SetTrainFileHeader(config.GetConcatenationFileHeader())
    T.SetControllingVariables(controlling_vars)
    T.SetTrainVariables(query_vars)
    T.SetSaveDir(os.getcwd())
    T.SetModelIndex(0)
    T.SetDeviceIndex(0)
    T.SetActivationFunction(activation_function)
    T.SetScaler(scaler)
    T.InitializeWeights_and_Biases()
    T.Preprocessing()
    MLP_file_header = "MLP_test"
    T.write_SU2_MLP(MLP_file_header)
    output_TensorFlow = T.EvaluateMLP(CV_flamelet_test)
    a = MLPCppEvaluator()
    a.AddMLP("%s.mlp" % MLP_file_header)
    a.SetQueryInputs(controlling_vars)
    a.SetQueryOutputs(query_vars)
    a.GenerateMLP()
    output_mlpcpp = np.array(a.EvaluateMLP(CV_flamelet_test))
    diff_TF_MLPCpp = calc_error(output_TensorFlow, output_mlpcpp)
    if diff_TF_MLPCpp > 1e-12:
        passed = False 
        bad_combos.append([activation_function, scaler, query_vars, N_H, diff_TF_MLPCpp])
    max_error = max(max_error, calc_error(output_TensorFlow, output_mlpcpp))

if passed:
    sys.exit(0)
else:
    print("ERROR: TensorFlow and MLPCpp return different outputs for")
    for b in bad_combos:
        print("Activation function: %s\nScaler: %s\nQuery variables: %s\nArchitecture: %s\nAverage error: %.5e\n" % (b[0],b[1], ",".join(s for s in b[2]), ",".join(str(i) for i in b[3]), b[4]))

    sys.exit(1)
    