#!/usr/bin/env python3

from su2dataminer.config import Config_NICFD
from su2dataminer.manifold import SU2TableGenerator_NICFD

config = Config_NICFD()
config.SetFluid("CarbonDioxide")
config.SetEquationOfState("HEOS")
config.SetNpDensity(10)
config.SetNpEnergy(10)
config.UseAutoRange(False)
config.SetDensityBounds(2.0, 500.0)
config.SetEnergyBounds(0,1e6)
config.EnableGasPhase(True)
config.EnableTwophase(True)
config.EnableLiquidPhase(True)
config.EnableSuperCritical(True)
config.IncludeTransportProperties(True)

tablegen = SU2TableGenerator_NICFD(config)
tablegen.SetCellSize_Coarse(8e-2)
tablegen.SetCellSize_Refined(4e-2)
tablegen.SetTableDiscretization("adaptive")
tablegen.GenerateTable()
tablegen.WriteTableFile("LUT_test")