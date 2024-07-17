# Test case for MM fluid: generate manifold generation configuration 

# Import EntropicAI configuration module
from Common.EntropicAIConfig import EntropicAIConfig 

Config = EntropicAIConfig()

# Define fluid name
Config.SetFluid("MM")

# Set bounds and discretization for pressure grid.
Config.SetPressureBounds(P_lower=2e4, P_upper=2e6)
Config.SetNpPressure(Np_P=300)

# Set bounds and discretization for temperature grid.
Config.SetTemperatureBounds(T_lower=300, T_upper=600)
Config.SetNpTemp(Np_Temp=300)

# Define data grid in the density-energy space rather than pressure-temperature.
Config.UsePTGrid(False)

# Save current configuration.
Config.SaveConfig("MM_test")
