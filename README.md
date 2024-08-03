# SU2 DataMiner
This repository describes the workflow for manifold generation for data-driven fluid modeling in SU2. The workflow allows the user to generate fluid data and convert these into tables and train multi-layer perceptrons in order to retrieve thermo-chemical quantities during simulations in SU2. The applications are currently limited to non-ideal computational fluid dynamics and flamelet-generated manifold simulations for arbitrary fluids and reactants respectively. 

## Capabilities
The SU2 DataMiner workflow allows the user to generate fluid data and convert these into look-up tables (LUT) or multi-layer perceptrons (MLP) for usage in SU2 simulations. The types of simulations for which this workflow is suitable are flamelet-generated manifold (FGM) and non-ideal computational fluid dynamics (NICFD) simulations. This tool allows the user to start from scratch and end up with a table input file or a set of MLP input files which can immediately be used within SU2. 

## Requirements and Set-Up
The SU2 DataMiner tool is python-based and was generated with python 3.11. Currently only Linux distributions are supported.
You require the following modules for this workflow to run:
- numpy
- pickle
- os
- CoolProp
- cantera
- tqdm
- csv
- matplotlib
- random 
- tensorflow
- time 
- sklearn
- pyfiglet

After cloning this repository, add the following lines to your ```~/.bashrc``` in order to update your pythonpath accordingly:

```export PYTHONPATH=$PYTHONPATH:<PATH_TO_SOURCE>```

where ```<PATH_TO_SOURCE>``` is the path to where you cloned the repository.

Tutorials can be found under ```TestCases```, proper documentation will follow soon.
