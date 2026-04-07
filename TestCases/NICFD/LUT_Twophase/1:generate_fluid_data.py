from su2dataminer.config import Config_NICFD 
from su2dataminer.generate_data import DataGenerator_CoolProp

config = Config_NICFD("tabulation_twophase_MM.cfg")

DG = DataGenerator_CoolProp(config)

# Define two-phase settings here

DG.PreprocessData()
DG.ComputeData()
DG.SaveData()
