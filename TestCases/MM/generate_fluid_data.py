# Test case for MM fluid: generate entropic fluid data.

# Import EntropicAI configuration and data generation module.
from Common.EntropicAIConfig import EntropicAIConfig 
from Data_Generation.DataGenerators import DataGenerator_CoolProp 

# Load configuration.
Config = EntropicAIConfig("MM_test.cfg")
D = DataGenerator_CoolProp(Config)

# Define data grid.
D.PreprocessData()

# Visualize data grid prior to data generation.
D.VisualizeDataGrid()

# For every node in the data grid, compute thermodynamic state.
D.ComputeFluidData()

# Visualize entropic data.
D.VisualizeFluidData()

# Save all fluid data in respective files.
D.SaveFluidData()
