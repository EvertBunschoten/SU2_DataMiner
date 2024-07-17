# PINNTraining
Manifold generation for entropy-based fluid modeling in SU2. This workflow allows the user to train physics-informed neural networks to be used for the data-driven fluid model in SU2. 

The workflow is python-based and was generated with python 3.11.
You require the following modules for this workflow to run:
- numpy==1.26.4
- pickle
- os
- CoolProp==6.6.0
- tqdm==4.65.0
- csv==1.0
- matplotlib==3.9.0
- random 
- tensorflow==2.16.2
- time 
- sklearn=1.2.2

After cloning this repository, add the following lines to your ```~/.bashrc``` in order to update your pythonpath accordingly:

```export PYTHONPATH=$PYTHONPATH:<PATH_TO_SOURCE>```

where ```<PATH_TO_SOURCE>``` is the path to where you cloned the repository.

Tutorials can be found under ```TestCases```, proper documentation will follow soon.