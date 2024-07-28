# Test case for MM fluid: Train physics-informed neural network (PINN) on fluid data.

# Import EntropicAI configuration and MLP training module.
from Manifold_Generation.MLP.Trainers import EvaluateArchitecture, Train_Entropic_Direct
from Common.EntropicAIConfig import EntropicAIConfig 

# Load test case configuration.
Config = EntropicAIConfig("MM_test.cfg")
T = Train_Entropic_Direct()
T.Train_MLP()

# Eval = EvaluateArchitecture(Config)

# # Train for 1000 epochs.
# Eval.SetNEpochs(1000)

# # Start the training process.
# Eval.CommenceTraining()