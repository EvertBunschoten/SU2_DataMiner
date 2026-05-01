from su2dataminer.config import Config_NICFD 
from su2dataminer.manifold import SU2TableGenerator_NICFD

# Generate properties of MM with REFPROP library.
config = Config_NICFD()
config.SetFluid("MM")
config.SetEquationOfState("REFPROP")

# Include gas, liquid, two-phase, and supercritical fluid data.
config.EnableTwophase(True)
config.EnableLiquidPhase(True)
config.EnableGasPhase(True)
config.EnableSuperCritical(True)

# Include visosity, conductivity, and vapor quality. 
config.IncludeTransportProperties(True)
config.UseAutoRange(False)
config.SetDensityBounds(0.1, 460)
config.SetEnergyBounds(200e3, 360e3)
config.SetNpDensity(50)
config.SetNpEnergy(50)

# Initiate table generator with adaptive triangulation.
tablegen = SU2TableGenerator_NICFD(config)
tablegen.SetTableDiscretization("adaptive")

# Specify table resolution for coarse and refined sections.
tablegen.SetCellSize_Coarse(2e-2)
tablegen.SetCellSize_Refined(2e-3)

# Relative step size for finite-differences.
tablegen.SetFDStepSize(7e-3)

# Optionally, specify thermophysical variables to be included in the table. By default, all variables are included.
# tablegen.SetTableVars(["Density","Energy","s","p","T", "dsdrho_e","dsde_rho", "d2sdrho2","d2sde2","d2sdedrho","VaporQuality","ViscosityDyn","Conductivity"])

# Specify custom refinement regions (low density, around Trova isentrope)
tablegen.AddRefinementCriterion("Density", 0.0, 10.0)
tablegen.AddRefinementCriterion("s", 729.13-2, 729.13+40)

# Generate table.
tablegen.GenerateTable()

# Write SU2 DRG and vtk table.
tablegen.WriteTableFile("LUT_adaptive.drg")
tablegen.WriteOutParaview("vtktable_adaptive")