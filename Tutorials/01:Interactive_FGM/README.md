# SU2 DataMiner Tutorial 1: Flamelet Manifold Set-Up through the Terminal

| Parameter      | Description |
| ----------- | ----------- |
| Difficulty      | Easy       |
| Requires   | ```SU2 DataMiner```        |
| Uses | bin/GenerateConfig.py|
| Author | E.C. Bunschoten |
| Version | 1.0.0 |

## Goals
The goal of this tutorial is to set up a flamelet-based manifold configuration through ```SU2 DataMiner``` through the interactive menu functionality. 

## Set-up
Set up ```SU2 DataMiner``` as per the [general set-up instructions](../../README.md). Other than that, this tutorial requires no scripts or reference data. 

## Tutorial

The manifold generation process starts by running the command ```GenerateConfig.py``` in the terminal. This will start the interactive menu for setting up a ```SU2 DataMiner``` configuration and display the following message in the terminal:

```
>>> GenerateConfig.py
   _____ __  _____      ____        __        __  ____                
  / ___// / / /__ \    / __ \____ _/ /_____ _/  |/  (_)___  ___  _____
  \__ \/ / / /__/ /   / / / / __ `/ __/ __ `/ /|_/ / / __ \/ _ \/ ___/
 ___/ / /_/ // __/   / /_/ / /_/ / /_/ /_/ / /  / / / / / /  __/ /    
/____/\____//____/  /_____/\__,_/\__/\__,_/_/  /_/_/_/ /_/\___/_/     
                                                                      

#====================================================================================================#
Welcome to the SU2 DataMiner interactive configuration menu.
Generate a configuration for a data-driven fluid manifold through terminal inputs.
#====================================================================================================#
Type of SU2 DataMiner configuration (Flamelet/NICFD):
```

### 1: Manifold type
The first step is to choose the type of configuration: ```Flamelet``` for an FGM-based manifold, ```NICFD``` for an NICFD-based manifold. For this tutorial, the option ```Flamelet``` is used.

```
Type of SU2 DataMiner configuration (Flamelet/NICFD): Flamelet
```

### 2: Reaction mechanism
The flamelt-based manifold consists of 1D flamelet data generated through [Cantera](https://cantera.org/). Cantera requires a chemical kinetics file in order to compute flamelet solutions. Cantera has several popular chemical kinetics systems installed by default (e.g. ```gri30.yaml```, ```h2o2.yaml```), but supports custom kinetics files as well, as long as the file is reachable on the ```$PYTHONPATH```. For this tutorial, the ```gri30``` mechanism is used.

```
#====================================================================================================#
Insert reaction mechanism file to use for flamelet computations (gri30.yaml by default): gri30.yaml
Reaction mechanism: gri30.yaml
#====================================================================================================#
```

### 3: Reactant definition
The Flamelet-based manifold supports singular and composite reactant definitions. By default, the reactants are set to methane-air. For this tutorial, the fuel is set to pure hydrogen (```H2```).
```
#====================================================================================================#
Insert comma-separated list of fuel species (CH4 by default): H2
Fuel definition: H2:1.0
#====================================================================================================#
```
When inserting multiple species, the menu asks for the respective fuel species molar fractions.

The oxidizer definition is set to air by default. This definition is used in this tutorial as well. In order to use the default settings as suggested by the menu, simply press "enter"
```
#====================================================================================================#
Insert comma-separated list of oxidizer species (21% O2,79% N2 by default): 
Oxidizer definition: O2:1.0,N2:3.76
#====================================================================================================#
```

### 4: Flamelet solver transport model
The species transport model affects the way in which species diffusion is resolved in Cantera.  See the [Cantera documentation](https://cantera.org/documentation/docs-2.5/sphinx/html/cython/importing.html#cantera.Solution) for details. In this tutorial, the transport mechanism is set to ```mixture-averaged```.
```
#====================================================================================================#
Insert flamelet solver transport model (mixture-averaged/multicomponent/unity-Lewis-number, multicomponent by default): mixture-averaged
Flamelet solution transport model: mixture-averaged
#====================================================================================================#
```

### 5: Progress variable definition
The progress variable consits of a weighted sum of species in the flamelet solution and is a measure for the "progress" of the reaction. Over the course of a flamelet solution, the progress variable value should monotonically increase. ```SU2 DataMiner``` contains methods to optimize the progress variable, but at this stage, the progress variable can be set manually. By default, the progress variable is set as the weighted sum of reactants (minus nitrogen) and the major reactant product at stochiometry. The weights are by default set to be the inverse of the species molecular weight. For example, for hydrogen-air, the default progress variable definition is

$$\mathcal{Y} = \frac{1}{W_{H_2O}}Y_{H_2O} - \frac{1}{W_{H_2}}Y_{H_2} - \frac{1}{W_{O_2}}Y_{O_2}$$

where $W_i$ is the species molecular weight in grams per mole for specie $i$. The weights of the reactants are set to be negative in order to increase monotonicity.

For this tutorial, the progress variable definition is left to be the default.
```
#====================================================================================================#
Insert comma-separated list of progress variable species (H2,O2,H2O by default):
Insert comma-separated list of progress variable weights (-0.49603174603174605,-0.03125195324707794,0.055509297807382736 by default):
Progress variable definition: (-4.960e-01)*H2+(-3.125e-02)*O2+(+5.551e-02)*H2O
#====================================================================================================#
```

### 6: Manifold bounds
The flamelet data in the manifold can be generated over a range of reactant mixtures and temperatures. This ensures the manifold has sufficient ranges along the three controlling variable axes: progress variable, total enthalpy, and mixture fraction. The range in mixture fraction can be controlled by setting defining a set of mixture ratio's for which flamelet data are generated. The mixture ratio can be defined as equivalence ratio $(\phi)$ or mixture fraction $(Z)$. For this tutorial, the mixture ratio is defined as equivalence ratio.
```
#====================================================================================================#
Insert definition for reactant mixture (1 for equivalence ratio, 2 for mixture fraction, 1 by default): 1
Reactant mixture defined as equivalence ratio.
Reactant mixture status defined as equivalence ratio.
#====================================================================================================#
```
Next, the program asks for the lower and upper bounds of the mixture ratio. The lower bound value should not exceed the upper bound value and the values should be physically sound. For example, negative mixture ratio values are not supported and neither are mixture fraction values higher than one. The range in mixture ratio depends on the application of the manifold.
For the current tutorial, the lower and upper equivalence ratio values are set to 0.3 and 0.7. 
The number of divisions in the mixture ratio range defines the number of mixture ratios for which flamelet data are generated. A high number results in a larger, more detailed flamelet data set. For this tutorial, 10 divisions are used.
```
#====================================================================================================#
Insert lower reactant mixture status value (0.200 by default): 0.3
Insert upper reactant mixture status value (20.000 by default): 0.7
Insert number of divisions for the mixture status range (30 by default): 10
Lower reactant mixture status value: 0.300
Upper reactant mixture status value: 0.700
Number of mixture status divisions: 10
#====================================================================================================#
```

The reaction mechanics are influenced significantly by total enthalpy. The range in total enthalpy of the flamelet-based manifold can be controlled by defining the lower and upper reactant temperature. Similarly to the definition of the mixture ratio range, the user sets the lower and upper bound of the reactant temperature, as well as the number of divisions. For this tutorial, a lower reactant temperature of 300 Kelvin and an upper reactant temperature of 600 Kevin are used, with 10 divisions.
```
#====================================================================================================#
Insert lower reactant temperature value [K] (300.000 by default): 300
Insert upper reactant temperature value [K] (800.000 by default): 600
Insert number of divisions for the reactant temperature range (30 by default): 10
Lower reactant temperature value: 300.000 K
Upper reactant temperature value: 600.000 K
Number of reactant temperature status divisions: 10
#====================================================================================================#
```

### 7: Flamelet types
```SU2 DataMiner``` supports multiple flamelet types which can populate the manifold. The supported options are [*adiabatic flamelets*](https://cantera.org/documentation/docs-2.5/sphinx/html/cython/onedim.html#freeflame), [*burner-stabilized flamelets*](https://cantera.org/documentation/docs-2.5/sphinx/html/cython/onedim.html#burnerflame), and *chemical equilibrium data* (pure reactants and products over the reactant temperature range). In this tutorial, only adiabatic flamelets will be used.
```
#====================================================================================================#
Compute adiabatic flamelet data (1=yes, 0=no, 1 by default): 1
Adiabatic flamelets are included in manifold.
#====================================================================================================#
Compute burner-stabilized flamelet data (1=yes, 0=no, 1 by default): 0
Burner-stabilized flamelets are ommitted in manifold.
#====================================================================================================#
Compute chemical equilibrium data (1=yes, 0=no, 1 by default): 0
Chemical equilibrium data are ommitted in manifold.
#====================================================================================================#
```

### 8: General settings
The configuration is finalized by defining the data output folder, where flamelet and manifold data are saved and naming the configuration. The output data folder should be accessible on the current hardware and is set to the current working directory by default. 
```
#====================================================================================================#
Insert data output directory (<CURRENT DIRECTORY> by default):
Data output folder: <CURRENT DIRECTORY>
#====================================================================================================#
```
The last configuration setting is the configuration name. When saving the configuration, it will be saved under the set name with a ```.cfg``` extension. The manifold settings can be accessed by loading the specific configuration. For this tutorial, the name is set as ```Tutorial_1```.
```
#====================================================================================================#
Set a name for the current SU2 DataMiner configuration (config_FGM by default): Tutorial_1
#====================================================================================================#
```
Once the configuration name is set, a summary of the configuration is displayed in the terminal, where the user can review the settings.

```
#====================================================================================================#
Summary:
   _____ __  _____      ____        __        __  ____                
  / ___// / / /__ \    / __ \____ _/ /_____ _/  |/  (_)___  ___  _____
  \__ \/ / / /__/ /   / / / / __ `/ __/ __ `/ /|_/ / / __ \/ _ \/ ___/
 ___/ / /_/ // __/   / /_/ / /_/ / /_/ /_/ / /  / / / / / /  __/ /    
/____/\____//____/  /_____/\__,_/\__/\__,_/_/  /_/_/_/ /_/\___/_/     
                                                                      

flameletAIConfiguration: Tutorial_1

Flamelet generation settings:
Flamelet data output directory: <CURRENT DIRECTORY>
Reaction mechanism: gri30.yaml
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

Save configuration and exit (1) or re-run configuration set-up (2)?
```

If the user is satisfied with the current set-up, the configuration can be saved by typing ```1``` and the menu is closed. However, it is also possible to re-run the set-up and adjust some previously set options by typing ```2```. For example, the reaction mechanism can be set to the ```h2o2.yaml``` mechanism by re-running the set-up and keeping all the previously set values, except for the reaction mechanism.
```
Save configuration and exit (1) or re-run configuration set-up (2)?2
Re-running configuration set-up
Insert reaction mechanism file to use for flamelet computations (gri30.yaml by default): h2o2.yaml
Reaction mechanism: h2o2.yaml
#====================================================================================================#
#====================================================================================================#
Insert comma-separated list of fuel species (H2 by default): 
Fuel definition: H2:1.0
#====================================================================================================#
Insert comma-separated list of oxidizer species (21% O2,79% N2 by default): 
Oxidizer definition: O2:1.0,N2:3.76
#====================================================================================================#
Insert flamelet solver transport model (mixture-averaged/multicomponent/unity-Lewis-number, mixture-averaged by default): 
Flamelet solution transport model: mixture-averaged
#====================================================================================================#
Insert comma-separated list of progress variable species (H2,O2,H2O by default):
Insert comma-separated list of progress variable weights (-0.49603174603174605,-0.03125195324707794,0.055509297807382736 by default):
Progress variable definition: (-4.960e-01)*H2+(-3.125e-02)*O2+(+5.551e-02)*H2O
#====================================================================================================#
Insert definition for reactant mixture (1 for equivalence ratio, 2 for mixture fraction, 1 by default): 
Reactant mixture status defined as equivalence ratio.
#====================================================================================================#
Insert lower reactant mixture status value (0.300 by default): 
Insert upper reactant mixture status value (0.700 by default): 
Insert number of divisions for the mixture status range (10 by default): 
Lower reactant mixture status value: 0.300
Upper reactant mixture status value: 0.700
Number of mixture status divisions: 10
#====================================================================================================#
Insert lower reactant temperature value [K] (300.000 by default): 
Insert upper reactant temperature value [K] (600.000 by default): 
Insert number of divisions for the reactant temperature range (10 by default): 
Lower reactant temperature value: 300.000 K
Upper reactant temperature value: 600.000 K
Number of reactant temperature status divisions: 10
#====================================================================================================#
Compute adiabatic flamelet data (1=yes, 0=no, 1 by default): 
#====================================================================================================#
Compute burner-stabilized flamelet data (1=yes, 0=no, 1 by default): 
#====================================================================================================#
Compute chemical equilibrium data (1=yes, 0=no, 1 by default): 
#====================================================================================================#
Insert data output directory (<CURRENT DIRECTORY> by default):
Data output folder: <CURRENT DIRECTORY>
#====================================================================================================#
Set a name for the current SU2 DataMiner configuration (Tutorial_1 by default): 
#====================================================================================================#
Summary:
   _____ __  _____      ____        __        __  ____                
  / ___// / / /__ \    / __ \____ _/ /_____ _/  |/  (_)___  ___  _____
  \__ \/ / / /__/ /   / / / / __ `/ __/ __ `/ /|_/ / / __ \/ _ \/ ___/
 ___/ / /_/ // __/   / /_/ / /_/ / /_/ /_/ / /  / / / / / /  __/ /    
/____/\____//____/  /_____/\__,_/\__/\__,_/_/  /_/_/_/ /_/\___/_/     
                                                                      

flameletAIConfiguration: Tutorial_1

Flamelet generation settings:
Flamelet data output directory: <CURRENT DIRECTORY>
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

By pressing ```1```, the configuration will be saved and can be used for subsequent flamelet data computations and manifold set-up.

