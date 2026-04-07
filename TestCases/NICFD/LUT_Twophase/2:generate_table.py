from su2dataminer.config import Config_NICFD 
from su2dataminer.manifold import SU2TableGenerator_NICFD

config = Config_NICFD("tabulation_twophase_MM.cfg")

tablegen = SU2TableGenerator_NICFD(config)
tablegen.GenerateTable()
tablegen.WriteTableFile("LUT_test.drg")
