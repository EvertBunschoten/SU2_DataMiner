from Manifold_Generation.MLP.Trainers import EvaluateArchitecture 
from Common.EntropicAIConfig import EntropicAIConfig 

Config = EntropicAIConfig("MM_test.cfg")

Eval = EvaluateArchitecture(Config)
Eval.SetNEpochs(1000)
Eval.CommenceTraining()