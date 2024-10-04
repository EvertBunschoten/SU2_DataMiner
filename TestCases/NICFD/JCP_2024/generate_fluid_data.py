from Common.DataDrivenConfig import EntropicAIConfig
from Data_Generation.DataGenerator_NICFD import DataGenerator_CoolProp
Config_low = EntropicAIConfig("Segregated_training_low.cfg")
Dgen = DataGenerator_CoolProp(Config_low)
Dgen.PreprocessData()
Dgen.ComputeData()
Dgen.SaveData()

Config_high= EntropicAIConfig("Segregated_training_high.cfg")
Dgen = DataGenerator_CoolProp(Config_high)
Dgen.PreprocessData()
Dgen.ComputeData()
Dgen.SaveData()