#!/usr/bin/env python3
import sys
from su2dataminer.config import Config_FGM
from su2dataminer.generate_data import ComputeFlameletData,ComputeBoundaryData
from su2dataminer.process_data import FlameletConcatenator
from su2dataminer.manifold import TrainMLP_FGM

alpha_expo = -2.6
lr_decay = 0.999 
activation_function="swish"
N_H = [10,10,10,10]
batch_expo = 2

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
config.SetAlphaExpo(alpha_expo,0)
config.SetLRDecay(lr_decay,0)
config.SetActivationFunction(activation_function,0)
config.SetHiddenLayerArchitecture(N_H, 0)
config.SetBatchExpo(batch_expo,0)
config.SaveConfig()

ComputeFlameletData(config)
ComputeBoundaryData(config)


F = FlameletConcatenator(config, verbose_level=0)
F.ConcatenateFlameletData()
F.CollectBoundaryData()

T = TrainMLP_FGM(config, 0)
T.EnableBCLoss(False)
T.SetTrainHardware("CPU")
consistent_hp_params = (config.GetAlphaExpo(0)==T.alpha_expo==alpha_expo) and \
                 (config.GetLRDecay(0)==T.lr_decay==lr_decay) and \
                 (config.GetActivationFunction(0)==T.activation_function==activation_function) and \
                 (config.GetBatchExpo(0) == T.batch_expo == batch_expo)

T.SetNEpochs(100)
T.CommenceTraining()

config.UpdateMLPHyperParams(T)
config.PrintBanner()

if consistent_hp_params:
    sys.exit(0)
else:
    print("ERROR: network hyperparameters are inconsistent between config and trainer!")
    sys.exit(1)