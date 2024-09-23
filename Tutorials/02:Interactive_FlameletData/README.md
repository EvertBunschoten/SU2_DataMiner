# SU2 DataMiner Tutorial 2: Flamelet data generation through the terminal

| Parameter      | Description |
| ----------- | ----------- |
| Difficulty      | Easy      |
| Requires   | ```SU2 DataMiner```,```MPI```      |
| Uses | bin/GenerateFlameletData.py|
| Author | E.C. Bunschoten |
| Version | 1.0.0 |

## Goals

The goal of this tutorial is to demonstrate the most basic method for generating data for a flamelet-based manifold. 

## Set-up
Set up ```SU2 DataMiner``` as per the [general set-up instructions](../../README.md). Other than that, this tutorial requires no scripts or reference data. 

In addition, this tutorial will be using the configuration from [tutorial 1](../01:Interactive_FGM/README.md), summarized as follows:

```
   _____ __  _____      ____        __        __  ____                
  / ___// / / /__ \    / __ \____ _/ /_____ _/  |/  (_)___  ___  _____
  \__ \/ / / /__/ /   / / / / __ `/ __/ __ `/ /|_/ / / __ \/ _ \/ ___/
 ___/ / /_/ // __/   / /_/ / /_/ / /_/ /_/ / /  / / / / / /  __/ /    
/____/\____//____/  /_____/\__,_/\__/\__,_/_/  /_/_/_/ /_/\___/_/     
                                                                      

flameletAIConfiguration: Tutorial_1

Flamelet generation settings:
Flamelet data output directory: Tutorials/01:Interactive_FGM
Reaction mechanism: h2o2.yaml
Transport model: mixture-averaged
Fuel definition: H2: 1.00e+00
Oxidizer definition: O2: 1.00e+00,N2: 3.76e+00

Reactant temperature range: 300.00 K -> 600.00 K (10 steps)
Mixture status defined as equivalence ratio
Reactant mixture status range: 3.00e-01 -> 7.00e-01  (10 steps)

Flamelet types included in manifold:
-Adiabatic free-flamelet data

Flamelet manifold data characteristics: 
Controlling variable names: ProgressVariable, EnthalpyTot, MixtureFraction
Progress variable definition: -4.96e-01 H2, -3.13e-02 O2, +5.55e-02 H2O

Average specie Lewis numbers:
H2:3.4532e-01, H:2.3124e-01, O:8.1336e-01, O2:1.2417e+00, OH:8.2708e-01, H2O:1.0149e+00, HO2:1.2245e+00, H2O2:1.2317e+00, AR:8.7322e-01, N2:7.6639e-01
```

## Tutorial

The most basic manner in which flamelet data can be generated is through the terminal, using the ```GenerateFluidData.py``` command. Running this command with no arguments displays the argument options.

```
>>> GenerateFluidData.py
usage: GenerateFluidData.py [-h] [--c CONFIG_NAME] [--np NP] [--b] [--t TYPE]

options:
  -h, --help       show this help message and exit
  --c CONFIG_NAME  Configuration file name.
  --np NP          Number of processors to use for flamelet data generation.
  --b              Generate chemical equilibrium boundary data over the full mixture range (0.0 <= Z <= 1.0).
  --t TYPE         Data type to generate: (1:FGM, 2:NICFD)
```

The main options are ```--c``` and ```--np```, which correspond to the ```SU2 DataMiner``` configuration file and number of processors respectively. The option ```--c``` accepts the configuration file path name describing the manifold. For this tutorial, the configuration from [tutorial 1](../01:Interactive_FGM/README.md) is used.

The computational cost of generating the flamelet data depends on the complexity of the chemical kinetics and the range of flamelet data. In order to reduce the computation time, the computational load can be distributed over multiple processors using the ```--np``` command. In this tutorial, 2 cores will be used for the flamelet generation process. 

Initiate the flamelet data computation process with the following command:
```
>>> GenerateFluidData --c ../01\:Interactive_FlameletData/Tutorial_1.cfg --np 2 --t 1
```

This will first print a summary of the configuration in the terminal, followed by status updates of the flamelet data generation process:
```
Successfull Freeflame simulation at phi: 0.34444444444444444 T_u: 600.0 (1/10)
Successfull Freeflame simulation at phi: 0.3 T_u: 600.0 (1/10)
Successfull Freeflame simulation at phi: 0.3 T_u: 566.6666666666666 (2/10)
Successfull Freeflame simulation at phi: 0.34444444444444444 T_u: 566.6666666666666 (2/10)
Successfull Freeflame simulation at phi: 0.3 T_u: 533.3333333333334 (3/10)
Successfull Freeflame simulation at phi: 0.34444444444444444 T_u: 533.3333333333334 (3/10)
Successfull Freeflame simulation at phi: 0.3 T_u: 500.0 (4/10)
Successfull Freeflame simulation at phi: 0.34444444444444444 T_u: 500.0 (4/10)
Successfull Freeflame simulation at phi: 0.3 T_u: 466.66666666666663 (5/10)
Successfull Freeflame simulation at phi: 0.34444444444444444 T_u: 466.66666666666663 (5/10)
Successfull Freeflame simulation at phi: 0.34444444444444444 T_u: 433.3333333333333 (6/10)
Successfull Freeflame simulation at phi: 0.34444444444444444 T_u: 400.0 (7/10)
Successfull Freeflame simulation at phi: 0.3 T_u: 433.3333333333333 (6/10)
Successfull Freeflame simulation at phi: 0.3 T_u: 400.0 (7/10)
...
```
At any time, the flamelet data generation process can be interrupted by the user by pressing ctrl + c in the terminal.

Since the output data folder was set to that of tutorial 1, the flamelet data are stored in the folder ```freeflame_data``` in the folder of tutorial 1. The flamelet data generation process for this tutorial should not take more than a couple of minutes and the flamelet data set takes up around 30 MB.


