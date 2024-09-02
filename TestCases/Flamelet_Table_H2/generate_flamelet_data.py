# Generate flamelet data for pre-mixed hydrogen-air problems 

from Common.DataDrivenConfig import FlameletAIConfig 
from Data_Generation.DataGenerator_FGM import ComputeFlameletData

# Load FlameletAI configuration
Config = FlameletAIConfig("TableGeneration.cfg")

# Distribute flamelet data generation process over 20 cores.
ComputeFlameletData(Config, run_parallel=True, N_processors=20)
