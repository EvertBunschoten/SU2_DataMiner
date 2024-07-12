from Common.EntropicAIConfig import EntropicAIConfig 
import os 

Config = EntropicAIConfig()
Config.SetFluid("MM")
Config.SetPressureBounds(P_lower=2e4, P_upper=2e6)
Config.SetNpPressure(Np_P=700)

Config.SetTemperatureBounds(T_lower=300, T_upper=600)
Config.SetNpTemp(Np_Temp=300)

Config.UsePTGrid(False)
Config.SaveConfig("MM_test")
