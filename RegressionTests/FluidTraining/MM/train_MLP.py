#!/usr/bin/env python3

import os 
import sys 
from Manifold_Generation.MLP.Trainers_NICFD.Trainers import Train_Entropic_Direct

from Common.DataDrivenConfig import EntropicAIConfig 

C = EntropicAIConfig(sys.argv[-1])

M = Train_Entropic_Direct()
M.SetTrainFileHeader(os.getcwd()+"/"+C.GetConcatenationFileHeader())
M.SetAlphaExpo(-3.0)
M.SetLRDecay(0.85)
M.SetBatchExpo(4)
M.SetHiddenLayers([20])
M.SetNEpochs(100)
M.SetVerbose(0)
M.SetSaveDir(os.getcwd())
M.SetModelIndex(0)
M.SetDeviceIndex(0)
M.Train_MLP()
M.Plot_and_Save_History()