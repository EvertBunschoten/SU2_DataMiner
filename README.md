<p style="margin-bottom:1cm;"> </p>
<p align="center">
        <img src="Documentation/images/SU2DataMiner_logo.png" width="200"/> 
</p>
<p style="margin-bottom:1cm;"> </p
>


# SU2 DataMiner
This repository describes the workflow for manifold generation for data-driven fluid modeling in SU2. The workflow allows the user to generate fluid data and convert these into tables and train multi-layer perceptrons in order to retrieve thermo-chemical quantities during simulations in SU2. The applications are currently limited to non-ideal computational fluid dynamics and flamelet-generated manifold simulations for arbitrary fluids and reactants respectively. Documentation of the source code, the theory guide, and tutorial collection can be found on the [SU2 DataMiner web page](https://su2code.github.io/SU2_DataMiner/).

## Capabilities
The SU2 DataMiner workflow allows the user to generate fluid data and convert these into look-up tables (LUT) or multi-layer perceptrons (MLP) for usage in SU2 simulations. The types of simulations for which this workflow is suitable are flamelet-generated manifold (FGM) and non-ideal computational fluid dynamics (NICFD) simulations. This tool allows the user to start from scratch and end up with a table input file or a set of MLP input files which can immediately be used within SU2. 

## Requirements and Set-Up
The SU2 DataMiner tool is python-based and was generated with python 3.13. Currently only Linux distributions are supported.
To install the required python modules, navigate to the SU2 DataMiner source code directory and run the following command:
```
python -m pip install -r required_packages.txt
```
Alternatively, a suitable conda environment can be created using the [environment recipe](environment.yml) through the following command:
```
conda env create -f environment.yml
```

After cloning this repository, add the following lines to your ```~/.bashrc``` in order to update your pythonpath accordingly:

```
export PINNTRAINING_HOME=<PATH_TO_SOURCE>
export PYTHONPATH=$PYTHONPATH:$PINNTRAINING_HOME
export PATH=$PATH:$PINNTRAINING_HOME/bin
```

where ```<PATH_TO_SOURCE>``` is the path to where you cloned the repository.


