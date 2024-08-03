# SU2 DataMiner
This repository describes the workflow for manifold generation for data-driven fluid modeling in SU2. The workflow allows the user to generate fluid data and convert these into tables and train multi-layer perceptrons in order to retrieve thermo-chemical quantities during simulations in SU2. The applications are currently limited to non-ideal computational fluid dynamics and flamelet-generated manifold simulations for arbitrary fluids and reactants respectively. 

The workflow is python-based and was generated with python 3.11.
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
