#!/usr/bin/env python3
import os 
from su2dataminer.config import Config_FGM 
from su2dataminer.generate_data import DataGenerator_Cantera

config = Config_FGM()
config.SetFuelDefinition(["H2"], [1.0])
config.SetReactionMechanism("h2o2.yaml")
config.SetTransportModel("multicomponent")
config.RunFreeFlames(True)
config.RunBurnerFlames(False)
config.RunEquilibrium(False)
config.DefineMixtureStatus(False)
config.SetMixtureBounds(0.8, 1.2)
config.SetOutputDir(os.getcwd())

DG = DataGenerator_Cantera(config)
DG.ComputeFreeFlames(mix_status=1.0, T_ub=300.0)


