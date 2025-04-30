#!/usr/bin/env python3

import os 
import sys 
from su2dataminer.manifold import Train_Entropic_PINN

from Common.DataDrivenConfig import Config_NICFD 

C = Config_NICFD(sys.argv[-1])

M = Train_Entropic_PINN()
M.SetTrainFileHeader(os.getcwd()+"/"+C.GetConcatenationFileHeader())
M.SetNEpochs(100)
M.SetHiddenLayers([10])
M.SetBatchExpo(4)
M.SetActivationFunction("exponential")
M.SetVerbose(0)
M.SetSaveDir(os.getcwd())
M.SetDeviceIndex(0)
M.SetModelIndex(0)
M.InitializeWeights_and_Biases()
M.CollectVariables()
M.Train_MLP()