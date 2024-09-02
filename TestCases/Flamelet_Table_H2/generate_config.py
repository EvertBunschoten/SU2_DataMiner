
from Common.DataDrivenConfig import FlameletAIConfig 
import os 

Config = FlameletAIConfig()
Config.SetConfigName("TableGeneration")

# Hydrogen-air flamelets with equivalence ratio between 0.3 and 0.7
Config.SetFuelDefinition(fuel_species=["H2"],fuel_weights=[1.0])
Config.SetReactionMechanism('h2o2.yaml')
Config.SetMixtureBounds(0.3, 0.7)
Config.SetNpMix(80)
Config.SetUnbTempBounds(300, 800)
Config.SetNpTemp(30)

# Enable preferential diffusion through selecting the "multicomponent" transport model.
Config.SetTransportModel('multicomponent')

Config.SetConcatenationFileHeader("LUT_data")

# Setting the Efimov progress variable definition.
Config.SetProgressVariableDefinition(pv_species=['H2', 'H', 'O2', 'O', 'H2O', 'OH', 'H2O2', 'HO2'],\
                                     pv_weights=[-7.36, -23.01, -2.04, -4.8, 1.83, -15.31, -57.02, 24.55])

# Preparing flamelet output directory.
flamelet_data_dir = os.getcwd() + "/flamelet_data/"
if not os.path.isdir(flamelet_data_dir):
    os.mkdir(flamelet_data_dir)
Config.SetOutputDir(flamelet_data_dir) 
 
Config.PrintBanner()
Config.SaveConfig()
