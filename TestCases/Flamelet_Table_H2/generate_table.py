from Manifold_Generation.LUT.FlameletTableGeneration import SU2TableGenerator
from Common.DataDrivenConfig import FlameletAIConfig 

# Loading configuration.
Config = FlameletAIConfig("TableGeneration.cfg")

# Initializing table module and pre-process interpolator.
Tgen = SU2TableGenerator(Config, n_near=14, p_fac=3)

# Manually set mixture fraction limits.
Tgen.SetMixtureFractionLimits(mix_frac_max=0.0144703624619207, mix_frac_min=0.00939225575395504)
Tgen.SetNTableLevels(20)

# Visualize the interpolated reaction rate at equivalence ratio 0.5.
cv_target = Config.GetUnburntScalars(equivalence_ratio=0.5, temperature=300.0)
pv_target = cv_target[0]
z_target = cv_target[2]
print("Target unburnt progress variable: ", pv_target)
Tgen.VisualizeTableLevel(z_target, "ProdRateTot_PV")

# Visualize table level connectivity at equivalence ratio 0.5.
Tgen.VisualizeTableLevel(z_target)

# Generate table connectivity and interpolate flamelet data.
Tgen.GenerateTableNodes()

# Write SU2 .drg table file.
Tgen.WriteTableFile()