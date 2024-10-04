from Common.DataDrivenConfig import EntropicAIConfig
from Manifold_Generation.MLP.Trainers_NICFD.Trainers import EvaluateArchitecture_NICFD_Segregated
import os 
# Config = EntropicAIConfig("Segregated_training_high.cfg")
# Trainer = EvaluateArchitecture_NICFD_Segregated(Config)
# Trainer.SetNEpochs(300)
# Trainer.SetSaveDir(os.getcwd()+"/Manifolds_high/")
# Trainer.SetVerbose(1)
# Trainer.CommenceTraining()

Config = EntropicAIConfig("Segregated_training_low.cfg")

Trainer = EvaluateArchitecture_NICFD_Segregated(Config)
Trainer.SetNEpochs(300)
Trainer.SetSaveDir(os.getcwd()+"/Manifolds_low/")
Trainer.SetVerbose(1)
Trainer.CommenceTraining()