#!/usr/bin/env python3
import sys
import os 
from su2dataminer.config import Config_FGM 
from su2dataminer.generate_data import DataGenerator_Cantera

config = Config_FGM(sys.argv[-1])
config.SetOutputDir(os.getcwd())

DG = DataGenerator_Cantera(config)
DG.ComputeFreeFlames(mix_status=1.0, T_ub=300.0)


