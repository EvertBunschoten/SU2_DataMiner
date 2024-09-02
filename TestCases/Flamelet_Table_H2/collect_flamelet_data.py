# Collect flamelet data into data sets for table generation
from Common.DataDrivenConfig import FlameletAIConfig 
from Data_Processing.collectFlameletData import FlameletConcatenator

Config = FlameletAIConfig("TableGeneration.cfg")

Concat = FlameletConcatenator(Config)

# Include NOx reaction rates and heat release in flamelet data set 
Concat.SetAuxilarySpecies(["H2"])
Concat.SetLookUpVars(["Heat_Release"])

# Read and concatenate flamelet data
Concat.ConcatenateFlameletData()
