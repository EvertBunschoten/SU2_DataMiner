.. _tutorialconfigs:

.. sectionauthor:: Evert Bunschoten 
    
Creating SU2 DataMiner Configurations
=====================================

*SU2 DataMiner* uses a configuration class in order to store important information regarding the data generation, data mining, and 
manifold generation processes. This set of tutorials demonstrates some methods for creating *SU2 DataMiner* configurations 
for FGM and NICFD applications. *SU2 DataMiner* configurations can be generated using an interactive menu and using the python API.
The interactive method is recommended for instances where only a single configuration is required or when writing a configuration for 
the first time, while through the python interface, it is possible to create multiple configurations at the same time.  

.. important:: 

    To be able to follow the tutorials, *SU2 DataMiner* should be set up according to the instructions in
    :ref:`label_setup`.


:ref:`label_config_interactive_FGM`
:ref:`label_config_interactive_NICFD`


.. _label_config_interactive_FGM:

1. Interactive method: FGM
--------------------------

This tutorial explains how to generate a *SU2 DataMiner* configuration that contains the set-up for a **methane-air** flamelet-generated manifold (FGM) containing adiabatic 
flamelet data. 
In order to generate a *SU2 DataMiner* configuration using the interactive menu, run the following command in your terminal:

.. code-block::

    >>> GenerateConfig.py


This will initiate the interactive configuration generation menu by printing a message in the terminal.

1. Select FGM manifold type

The first step is to select the type of configuration to generate. In this tutorial, a configuration for FGM applications is defined, so option `1` is selected.

.. code-block::

       _____ __  _____      ____        __        __  ____                
      / ___// / / /__ \    / __ \____ _/ /_____ _/  |/  (_)___  ___  _____
      \__ \/ / / /__/ /   / / / / __ `/ __/ __ `/ /|_/ / / __ \/ _ \/ ___/
     ___/ / /_/ // __/   / /_/ / /_/ / /_/ /_/ / /  / / / / / /  __/ /    
    /____/\____//____/  /_____/\__,_/\__/\__,_/_/  /_/_/_/ /_/\___/_/     
                                                                        

    #====================================================================================================================================================================#
    Welcome to the SU2 DataMiner interactive configuration menu.
    Generate a configuration for a data-driven fluid manifold through terminal inputs.
    #====================================================================================================================================================================#
    Type of SU2 DataMiner configuration (1:Flamelet,2:NICFD): 1

2. Select reaction mechanism 

Currently, *SU2 DataMiner* only supports flamelet generation with `Cantera <https://cantera.org/index.html>`_. Cantera is an open-source software suite for chemical kinetics and thermodynamics which 
supports the calculation of one-dimensional detailed chemistry solutions for FGM applications. The chemical kinetics solver in Cantera obtains its information about 
the species composition, transport properties, and chemical reactions from a kinetics file. Within the interactive menu, you can specify the kinetics file used for flamelet calculations.
SU2 DataMiner can use the kinetics files included in the Cantera Python module, or any other kinetics file as long as it is accessible from the terminal.

By default, SU2 DataMiner will select the `gri30 <http://combustion.berkeley.edu/gri-mech/version30/text30.html>`_ mechanism, which is commonly used for modeling carbohydrate thermochemistry.


.. code-block::

    #===================================================================================================#
    Insert reaction mechanism file to use for flamelet computations (gri30.yaml by default): gri30.yaml
    Reaction mechanism: gri30.yaml
    #===================================================================================================#


SU2 DataMiner will give a warning if the chemical kinetics file cannot be found or if improperly formatted. 


3. Define fuel composition 

In the next steps, the user specifies the fuel and oxidizer composition by naming the species and molar weights. By default, the fuel is defined as pure methane 
.. code-block:: 

    #===================================================================#
    Insert comma-separated list of fuel species (CH4 by default): CH4
    Fuel definition: CH4:1.0
    #===================================================================#

4. Define oxidizer composition 

.. code-block:: 

    #====================================================================================================================================================================#
    Insert comma-separated list of oxidizer species (21% O2,79% N2 by default): 
    Oxidizer definition: O2:1.0,N2:3.76
    #====================================================================================================================================================================#

5. Select species transport model 


.. code-block:: 

    #====================================================================================================================================================================#
    Insert flamelet solver transport model (1:mixture-averaged,2:multicomponent,3:unity-Lewis-number) multicomponent by default: 3
    Flamelet solution transport model: unity-Lewis-number
    #====================================================================================================================================================================#


6. Define the progress variable (optional)

.. code-block:: 

    #====================================================================================================================================================================#
    Insert comma-separated list of progress variable species (CH4,O2,CO2 by default):
    Insert comma-separated list of progress variable weights (-0.5359,-0.2687,0.1953 by default):
    Progress variable definition: (-5.359e-01)*CH4+(-2.687e-01)*O2+(+1.954e-01)*CO2
    #====================================================================================================================================================================#


7. Define the mixture status 

.. code-block:: 

    #====================================================================================================================================================================#
    Define reactant mixture through mixture fracion (1:False,2:True) False by default:1
    Reactant mixture status defined as equivalence ratio.
    #====================================================================================================================================================================#

8. Set mixture range 

.. code-block:: 

    #====================================================================================================================================================================#
    Insert lower reactant mixture status value (0.200 by default): 0.5
    Insert upper reactant mixture status value (20.000 by default): 2.0
    Insert number of divisions for the mixture status range (30 by default): 6
    Lower reactant mixture status value: 0.500
    Upper reactant mixture status value: 2.000
    Number of mixture status divisions: 6
    #====================================================================================================================================================================#

9. Set reactant temperature range 

.. code-block:: 

    #====================================================================================================================================================================#
    Insert lower reactant temperature value [K] (300.000 by default): 300
    Insert upper reactant temperature value [K] (800.000 by default): 500
    Insert number of divisions for the reactant temperature range (30 by default): 6
    Lower reactant temperature value: 300.000 K
    Upper reactant temperature value: 500.000 K
    Number of reactant temperature status divisions: 6
    #====================================================================================================================================================================#

10. Select flamelet types 

.. code-block:: 

    #====================================================================================================================================================================#
    Compute adiabatic flamelet data (1:True,2:False) True by default:1
    Adiabatic flamelets are included in manifold.
    #====================================================================================================================================================================#
    Compute burner-stabilized flamelet data (1:True,2:False) True by default:2
    Burner-stabilized flamelets are ommitted in manifold.
    #====================================================================================================================================================================#
    Compute chemical equilibrium data (1:True,2:False) True by default:1
    Chemical equilibrium data are included in manifold.
    #====================================================================================================================================================================#


11. Set storage folder 

.. code-block:: 

    #====================================================================================================================================================================#
    Insert data output directory (<CURRENT_FOLDER> by default): <OUTPUT_FOLDER>
    Data output folder: <OUTPUT_FOLDER>
    #====================================================================================================================================================================#


12. Name configuration

.. code-block:: 

    #====================================================================================================================================================================#
    Set a name for the current SU2 DataMiner configuration (config_FGM by default): methane_adiabatic
    #====================================================================================================================================================================#

13. Summary

.. code-block::

    #====================================================================================================================================================================#
    Summary:
       _____ __  _____      ____        __        __  ____                
      / ___// / / /__ \    / __ \____ _/ /_____ _/  |/  (_)___  ___  _____
      \__ \/ / / /__/ /   / / / / __ `/ __/ __ `/ /|_/ / / __ \/ _ \/ ___/
     ___/ / /_/ // __/   / /_/ / /_/ / /_/ /_/ / /  / / / / / /  __/ /    
    /____/\____//____/  /_____/\__,_/\__/\__,_/_/  /_/_/_/ /_/\___/_/     
                                                                        

    Config_FGMuration: methane_adiabatic

    Flamelet generation settings:
    Flamelet data output directory: <OUTPUT_FOLDER>
    Reaction mechanism: gri30.yaml
    Transport model: unity-Lewis-number
    Fuel definition: CH4: 1.00e+00
    Oxidizer definition: O2: 1.00e+00,N2: 3.76e+00

    Reactant temperature range: 300.00 K -> 500.00 K (6 steps)
    Mixture status defined as equivalence ratio
    Reactant mixture status range: 5.00e-01 -> 2.00e+00  (6 steps)

    Flamelet types included in manifold:
    -Adiabatic free-flamelet data
    -Chemical equilibrium data

    Flamelet manifold data characteristics: 
    Controlling variable names: ProgressVariable, EnthalpyTot, MixtureFraction
    Progress variable definition: -5.36e-01 CH4, -2.69e-01 O2, +1.95e-01 CO2


14. Save configuration 

.. code-block:: 

    Save configuration and exit (1) or re-run configuration set-up (2)? 1



.. _label_config_interactive_NICFD:

2. Interactive method: NICFD
----------------------------


