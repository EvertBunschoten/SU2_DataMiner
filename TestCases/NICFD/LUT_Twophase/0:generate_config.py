from su2dataminer.config import Config_NICFD 

config = Config_NICFD()
config.SetFluid("MM")
config.SetEquationOfState("HEOS")
config.UseAutoRange(True)
config.UsePTGrid(False)
config.EnableTwophase(True)
config.IncludeTransportProperties(True)
config.SetConfigName("tabulation_twophase_MM")
config.PrintBanner()
config.SaveConfig()

