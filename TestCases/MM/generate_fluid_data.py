from Common.EntropicAIConfig import EntropicAIConfig 
from Data_Generation.DataGenerators import DataGenerator_CoolProp 

Config = EntropicAIConfig("MM_test.cfg")
D = DataGenerator_CoolProp(Config)
D.PreprocessData()
D.VisualizeDataGrid()
D.ComputeFluidData()
D.VisualizeFluidData()
D.SaveFluidData()
