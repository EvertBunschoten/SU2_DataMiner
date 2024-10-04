import os
from Common.DataDrivenConfig import EntropicAIConfig 

density_low = 0.2
density_split = 10.0
density_high = 300.0

energy_low = 2.5e5
energy_high = 5.5e5
Np_grid = 500

Config = EntropicAIConfig()
Config.SetFluid("MM")
Config.SetNpDensity(Np_grid)
Config.SetXDistribution("cosine")

Config.SetEnergyBounds(energy_low, energy_high)
Config.SetNpEnergy(Np_grid)
Config.SetYDistribution("linear")

Config.UsePTGrid(False)
if not os.path.isdir(os.getcwd()+"/fluid_data/"):
    os.mkdir(os.getcwd()+"/fluid_data/")
Config.SetOutputDir(os.getcwd()+"/fluid_data/")

# Settings for lower density manifold
Config.SetConfigName("Segregated_training_low")
Config.SetDensityBounds(density_low, density_split)
Config.SetConcatenationFileHeader("fluid_data_low")
Config.SetAlphaExpo(-3.1146)
Config.SetLRDecay(9.8732e-01)
Config.SetBatchExpo(5)
Config.SetActivationFunction("swish")
Config.SetHiddenLayerArchitecture([11, 14])
Config.PrintBanner()
Config.SaveConfig()

# Settings for higher density manifold
Config.SetDensityBounds(density_split, density_high)
Config.SetConfigName("Segregated_training_high")
Config.SetConcatenationFileHeader("fluid_data_high")
Config.SetAlphaExpo(-3.0721)
Config.SetLRDecay(9.9250e-01)
Config.SetBatchExpo(5)
Config.SetActivationFunction("gelu")
Config.SetHiddenLayerArchitecture([13, 15, 12])
Config.PrintBanner()
Config.SaveConfig()