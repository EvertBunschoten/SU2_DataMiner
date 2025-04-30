#!/usr/bin/env python3

import os 
import sys

from su2dataminer.config import Config_NICFD 
from su2dataminer.generate_data import DataGenerator_CoolProp 

Config = Config_NICFD(sys.argv[-1])

D = DataGenerator_CoolProp(Config_in=Config)
D.SetOutputDir(os.getcwd())

D.PreprocessData()

D.ComputeData()

D.SaveData()