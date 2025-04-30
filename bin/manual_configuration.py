###############################################################################################
#       #      _____ __  _____      ____        __        __  ____                   #        #  
#       #     / ___// / / /__ \    / __ \____ _/ /_____ _/  |/  (_)___  ___  _____   #        #  
#       #     \__ \/ / / /__/ /   / / / / __ `/ __/ __ `/ /|_/ / / __ \/ _ \/ ___/   #        #      
#       #    ___/ / /_/ // __/   / /_/ / /_/ / /_/ /_/ / /  / / / / / /  __/ /       #        #  
#       #   /____/\____//____/  /_____/\__,_/\__/\__,_/_/  /_/_/_/ /_/\___/_/        #        #
#       #                                                                            #        #
###############################################################################################

########################### FILE NAME: manual_configuration.py ################################
#=============================================================================================#
# author: Evert Bunschoten                                                                    |
#    :PhD Candidate ,                                                                         |
#    :Flight Power and Propulsion                                                             |
#    :TU Delft,                                                                               |
#    :The Netherlands                                                                         |
#                                                                                             |
#                                                                                             |
# Description:                                                                                |
#  Interactive menus for configuration set-up.                                                |
#                                                                                             |  
# Version: 2.0.0                                                                              |
#                                                                                             |
#=============================================================================================#

# Import configuration classes
from Common.Config_base import Config
from Common.DataDrivenConfig import *

from shutil import get_terminal_size

# General print methods
def printhbar():
    print("#" + "="*(get_terminal_size()[0]-2) + "#")

def printwronginput():
    print("Wrong input, please try again.")
    return 

def InsertConfigOption(Config_in_method, default_value, input_message:str):
    """General configuration option parser function. Set a single configuration function
    value.

    :param Config_in_method: Config setter function
    :type Config_in_method: function
    :param default_value: setting default/current value.
    :type default_value: any
    :param input_message: terminal message describing the input.
    :type input_message: str
    """
    correct_input:bool = False 
    while not correct_input:
        try:
            # Check if user input is valid.
            val_input = default_value
            user_input = input(input_message)
            if not user_input == "":
                val_input = type(default_value)(user_input)
            Config_in_method(val_input)
            correct_input = True 
        except:
            # Set default value when invalid input.
            Config_in_method(default_value)
            printwronginput()
    return 

def InsertEnumerateOption(Config_in_method, values_options:list, default_value, input_message:str):
    correct_input:bool = False 
    while not correct_input:
        try:
            # Check if user input is valid.
            val_input = default_value
            options_string = " (" + ",".join(("%i:%s" % (i+1, str(values_options[i])) for i in range(len(values_options)))) + ")"
            default_string = " %s by default:" % str(default_value)
            user_input = input(input_message + options_string + default_string)
            if not user_input == "":
                val_input = values_options[int(user_input)-1]
            Config_in_method(val_input)
            correct_input = True 
        except:
            # Set default value when invalid input.
            Config_in_method(default_value)
            printwronginput()
    return

def GeneralSettings(Config_in:Config):
    """Define common configuration settings and save configuration.

    :param Config_in: SU2 DataMiner configuration class.
    :type Config_in: Config
    :return: whether to re-run set-up or save and exit.
    :rtype: bool
    """

    # Set data output directory.
    InsertConfigOption(Config_in.SetOutputDir, Config_in.GetOutputDir(), "Insert data output directory (%s by default):" % Config_in.GetOutputDir())
    print("Data output folder: %s" % Config_in.GetOutputDir())
    printhbar()

    # Set configuration name.
    InsertConfigOption(Config_in.SetConfigName, Config_in.GetConfigName(), "Set a name for the current SU2 DataMiner configuration (%s by default): " % Config_in.GetConfigName())
    printhbar()

    # Summarize current set-up.
    print("Summary:")
    Config_in.PrintBanner()

    # Re-run configuration or save and exit.
    statisfied_input:str = input("Save configuration and exit (1) or re-run configuration set-up (2)?")
    if int(statisfied_input) == 1:
        Config_in.SaveConfig()
        statisfied = True 
    else:
        print("Re-running configuration set-up")
        statisfied = False 

    return statisfied

def ManualFlameletConfiguration():
    """Define FGM-based SU2 DataMiner configuration.
    """

    # Initiate empty configuration.
    Config_in:Config_FGM = Config_FGM()
    statisfied = False 
    
    while not statisfied:
        
        # 1: Define reaction mechanism.
        InsertConfigOption(Config_in.SetReactionMechanism, Config_in.GetReactionMechanism(), "Insert reaction mechanism file to use for flamelet computations (%s by default): " % Config_in.GetReactionMechanism())
        print("Reaction mechanism: " + Config_in.GetReactionMechanism())
        printhbar()

        # 2: Define fuel species and molar fractions.
        correct_fuel_definition = False 
        fuel_definition_default = Config_in.GetFuelDefinition().copy()
        fuel_weights_default = Config_in.GetFuelWeights().copy()
        printhbar()
        while not correct_fuel_definition:
            try:
                fuel_string:str = input("Insert comma-separated list of fuel species (%s by default): " % (",".join(s for s in fuel_definition_default)))
                if not fuel_string == "":
                    fuel_species = fuel_string.split(',')
                    if len(fuel_species) > 1:
                        fuel_weights_string:str = input("Insert comma-separated list of fuel species molar fractions: ")
                        fuel_weights = [float(w) for w in fuel_weights_string.split(',')]
                    else:
                        fuel_weights = [1.0]
                    Config_in.SetFuelDefinition(fuel_species, fuel_weights)
                correct_fuel_definition = True 
            except:
                Config_in.SetFuelDefinition(fuel_definition_default, fuel_weights_default)
                print("Wrong inputs, please try again.")

        print("Fuel definition: " + Config_in.GetFuelString())
        printhbar()

        # 3: Define oxidizer species and molar fractions.
        correct_oxidizer_definition = False 
        ox_definition_default = Config_in.GetOxidizerDefinition().copy()
        ox_weights_default = Config_in.GetOxidizerWeights().copy()
        while not correct_oxidizer_definition:
            try:
                oxidizer_string:str = input("Insert comma-separated list of oxidizer species (21% O2,79% N2 by default): ")
                if not oxidizer_string == "":
                    oxidizer_species = oxidizer_string.split(',')
                    oxidizer_weights_string:str = input("Insert comma-separated list of oxidizer species molar fractions: ")
                    oxidizer_weights = [float(w) for w in oxidizer_weights_string.split(',')]

                    Config_in.SetOxidizerDefinition(oxidizer_species, oxidizer_weights)
                correct_oxidizer_definition = True 
            except:
                Config_in.SetOxidizerDefinition(ox_definition_default, ox_weights_default)
                print("Wrong inputs, please try again.")

        print("Oxidizer definition: " + Config_in.GetOxidizerString())
        printhbar()

        # 4: Define flamelet solver transport model.
        InsertEnumerateOption(Config_in.SetTransportModel, ["mixture-averaged","multicomponent","unity-Lewis-number"], Config_in.GetTransportModel(), "Insert flamelet solver transport model")
        print("Flamelet solution transport model: %s" % Config_in.GetTransportModel())
        printhbar()

        # 5: Define progress variable species and weights.
        correct_pv_definition = False 
        pv_species_default = Config_in.GetProgressVariableSpecies().copy()
        pv_weights_default = Config_in.GetProgressVariableWeights().copy()
        while not correct_pv_definition:
            try:
                pv_species_input:str = input("Insert comma-separated list of progress variable species (%s by default):" % (",".join(s for s in pv_species_default)))
                if not pv_species_input == "":
                    pv_species = pv_species_input.split(',')
                else:
                    pv_species = pv_species_default
                pv_weights_input:str = input("Insert comma-separated list of progress variable weights (%s by default):" % (",".join(str(s) for s in pv_weights_default)))
                if not pv_weights_input == "":
                    pv_weights = [float(w) for w in pv_weights_input.split(',')]
                else:
                    pv_weights = pv_weights_default  
                Config_in.SetProgressVariableDefinition(pv_species, pv_weights)
                correct_pv_definition = True 
            except:
                Config_in.SetProgressVariableDefinition(pv_species_default, pv_weights_default)
                printwronginput()
        print("Progress variable definition: " + "+".join(("(%+.3e)*%s" % (w, s)) for w, s in zip(Config_in.GetProgressVariableWeights(), Config_in.GetProgressVariableSpecies())))
        printhbar()

        # 6: Define manifold mixture status range.
        InsertEnumerateOption(Config_in.DefineMixtureStatus, [False, True], Config_in.GetMixtureStatus(), "Define reactant mixture through mixture fracion")

        if Config_in.GetMixtureStatus():
            print("Reactant mixture status defined as mixture fraction.")
        else:
            print("Reactant mixture status defined as equivalence ratio.")
        printhbar()

        correct_mixture_bounds = False 
        mix_status_lower = Config_in.GetMixtureBounds()[0]
        mix_status_upper = Config_in.GetMixtureBounds()[1]
        while not correct_mixture_bounds:
            try:
                lower_mixture_value_input = input("Insert lower reactant mixture status value (%.3f by default): " % mix_status_lower)
                upper_mixture_value_input = input("Insert upper reactant mixture status value (%.3f by default): " % mix_status_upper)
                if not lower_mixture_value_input == "":
                    mix_status_lower = float(lower_mixture_value_input)
                if not upper_mixture_value_input == "":
                    mix_status_upper = float(upper_mixture_value_input)
                Config_in.SetMixtureBounds(mix_status_lower, mix_status_upper)
                correct_mixture_bounds = True 

            except:
                try:
                    Config_in.SetMixtureBounds(mix_status_lower, mix_status_upper)
                except:
                    printwronginput()
        
        InsertConfigOption(Config_in.SetNpMix, Config_in.GetNpMix(), "Insert number of divisions for the mixture status range (%i by default): " % Config_in.GetNpMix())
        print("Lower reactant mixture status value: %.3f" % Config_in.GetMixtureBounds()[0])
        print("Upper reactant mixture status value: %.3f" % Config_in.GetMixtureBounds()[1])
        print("Number of mixture status divisions: %i" % Config_in.GetNpMix())
        printhbar()

        # 7: Define manifold reactant temperature range.
        correct_temperature_bounds = False 
        T_lower = Config_in.GetUnbTempBounds()[0]
        T_upper = Config_in.GetUnbTempBounds()[1]
        while not correct_temperature_bounds:
            try:
                lower_T_value_input = input("Insert lower reactant temperature value [K] (%.3f by default): " % T_lower)
                upper_T_value_input = input("Insert upper reactant temperature value [K] (%.3f by default): " % T_upper)
                if not lower_T_value_input == "":
                    T_lower = float(lower_T_value_input)
                if not lower_T_value_input == "":
                    T_upper = float(upper_T_value_input)
                Config_in.SetUnbTempBounds(T_lower, T_upper)
                correct_temperature_bounds = True 

            except:
                try:
                    Config_in.SetUnbTempBounds(T_lower, T_upper)
                except:
                    printwronginput()

        InsertConfigOption(Config_in.SetNpTemp, Config_in.GetNpTemp(), "Insert number of divisions for the reactant temperature range (%i by default): " % Config_in.GetNpTemp())
        print("Lower reactant temperature value: %.3f K" % Config_in.GetUnbTempBounds()[0])
        print("Upper reactant temperature value: %.3f K" % Config_in.GetUnbTempBounds()[1])
        print("Number of reactant temperature status divisions: %i" % Config_in.GetNpTemp())
        printhbar()

        # 8: Flamelet types in manifold.
        InsertEnumerateOption(Config_in.RunFreeFlames, [True, False], Config_in.GenerateFreeFlames(), "Compute adiabatic flamelet data")
        if Config_in.GenerateFreeFlames():
            print("Adiabatic flamelets are included in manifold.")
        else:
            print("Adiabatic flamelets are ommitted in manifold.")
        printhbar()

        InsertEnumerateOption(Config_in.RunBurnerFlames, [True, False], Config_in.GenerateBurnerFlames(), "Compute burner-stabilized flamelet data")
        if Config_in.GenerateBurnerFlames():
            print("Burner-stabilized flamelets are included in manifold.")
        else:
            print("Burner-stabilized flamelets are ommitted in manifold.")
        printhbar()


        InsertEnumerateOption(Config_in.RunEquilibrium, [True, False], Config_in.GenerateEquilibrium(), "Compute chemical equilibrium data")
        if Config_in.GenerateEquilibrium():
            print("Chemical equilibrium data are included in manifold.")
        else:
            print("Chemical equilibrium data are ommitted in manifold.")
        printhbar()

        # 9: General settings and finalize.
        statisfied = GeneralSettings(Config_in=Config_in)

    return 

def ManualNICFDConfiguration():
    """Define NICFD-based SU2 DataMiner configuration.
    """

    # Initiate empty configuration.
    Config_in:Config_NICFD = Config_NICFD()
    statisfied = False 
    
    while not statisfied:
        
        # 0: Define CoolProp equation of state.
        InsertConfigOption(Config_in.SetEquationOfState, DefaultSettings_NICFD.EOS_type, "Insert CoolProp equation of state (%s by default):" % DefaultSettings_NICFD.EOS_type)
        print("CoolProp equation of state: %s" % Config_in.GetEquationOfState())
        printhbar()

        # 1: Define fluid name(s).
        correct_fluid_definition = False 
        fluid_definition_default = Config_in.GetFluidNames()
        fluid_weights_default = Config_in.GetMoleFractions()
        printhbar()
        while not correct_fluid_definition:
            try:
                fluid_string:str = input("Insert comma-separated list of fluid components (%s by default): " % (",".join(s for s in fluid_definition_default)))
                if not fluid_string == "":
                    fluid_species = fluid_string.split(',')
                    if len(fluid_species) > 1:
                        fluid_weights_string:str = input("Insert comma-separated list of fluid molar fractions: ")
                        molar_weights = [float(w) for w in fluid_weights_string.split(',')]
                    else:
                        molar_weights = [1.0]
                    Config_in.SetFluid(fluid_species)
                    Config_in.SetFluidMoleFractions(molar_weights)
                else:
                    Config_in.SetFluid(fluid_definition_default)
                    Config_in.SetFluidMoleFractions(fluid_weights_default)
                correct_fluid_definition = True 
            except:
                Config_in.SetFluid(fluid_definition_default)
                Config_in.SetFluidMoleFractions(fluid_weights_default)
                printwronginput()

        print("Fluid definition: " + ",".join(("%.3e*%s" % (w, s)) for w, s in zip(Config_in.GetMoleFractions(), Config_in.GetFluidNames())))
        printhbar()

        # 2: Define fluid data controlling variable grid.
        InsertEnumerateOption(Config_in.UsePTGrid, [True, False], Config_in.GetPTGrid(), "Use pressure-temperature (True) or density-energy (False) based grid")

        if Config_in.GetPTGrid():
            grid_string = "pressure-temperature"
        else:
            grid_string = "density-energy"
        print("Fluid data grid definition: %s based." % grid_string)
        printhbar()

        # 3: Define fluid data manifold bounds.
        correct_x_bounds=False 
        correct_y_bounds=False
        if Config_in.GetPTGrid():
            x_string = "pressure"
            x_unit = "[Pa]"
            y_string = "temperature"
            y_unit = "[K]"
            x_getter_method = Config_in.GetPressureBounds
            y_getter_method = Config_in.GetTemperatureBounds
            x_setter_method = Config_in.SetPressureBounds
            y_setter_method = Config_in.SetTemperatureBounds
            Nx_setter_method = Config_in.SetNpPressure
            Nx_getter_method = Config_in.GetNpPressure
            Ny_setter_method = Config_in.SetNpTemp
            Ny_getter_method = Config_in.GetNpTemp
        else:
            x_string = "density"
            y_string = "static energy"
            x_unit = "[kg / m^3]"
            y_unit = "[J / kg]"
            x_getter_method = Config_in.GetDensityBounds
            y_getter_method = Config_in.GetEnergyBounds
            x_setter_method = Config_in.SetDensityBounds
            y_setter_method = Config_in.SetEnergyBounds
            Nx_setter_method = Config_in.SetNpDensity
            Nx_getter_method = Config_in.GetNpDensity
            Ny_setter_method = Config_in.SetNpEnergy
            Ny_getter_method = Config_in.GetNpEnergy

        x_bounds = x_getter_method()
        y_bounds = y_getter_method()
        x_lower, x_upper = x_bounds[0], x_bounds[1]
        y_lower, y_upper = y_bounds[0], y_bounds[1]
        while not correct_x_bounds:
            try:
                lower_x_input = input("Insert lower %s value (%s, %.3e by default): " % (x_string, x_unit, x_lower))
                upper_x_input = input("Insert upper %s value (%s, %.3e by default): " % (x_string, x_unit, x_upper))
                if not lower_x_input == "":
                    x_lower = float(lower_x_input)
                if not upper_x_input == "":
                    x_upper = float(upper_x_input)
                x_setter_method(x_lower, x_upper)
                correct_x_bounds = True 

            except:
                try:
                    x_setter_method(x_lower, x_upper)
                except:
                    printwronginput()
        InsertConfigOption(Config_in_method=Nx_setter_method, default_value=Nx_getter_method(), input_message="Insert number of divisions in the %s range (%i by default):" % (x_string, Nx_getter_method()))
        print("Lower fluid %s value: %.3e" % (x_string, x_getter_method()[0]))
        print("Upper fluid %s value: %.3e" % (x_string, x_getter_method()[1]))
        print("Number of fluid %s divisions: %i" % (x_string, Nx_getter_method()))
        printhbar()

        while not correct_y_bounds:
            try:
                lower_y_input = input("Insert lower %s value (%s, %.3e by default): " % (y_string, y_unit, y_lower))
                upper_y_input = input("Insert upper %s value (%s, %.3e by default): " % (y_string, y_unit, y_upper))
                if not lower_y_input == "":
                    y_lower = float(lower_y_input)
                if not upper_y_input == "":
                    y_upper = float(upper_y_input)
                y_setter_method(y_lower, y_upper)
                correct_y_bounds = True 

            except:
                try:
                    y_setter_method(y_lower, y_upper)
                except:
                    printwronginput()
        InsertConfigOption(Config_in_method=Ny_setter_method, default_value=Ny_getter_method(), input_message="Insert number of divisions in the %s range (%i by default):" % (y_string, Ny_getter_method()))
        print("Lower fluid %s value: %.3e" % (y_string, y_getter_method()[0]))
        print("Upper fluid %s value: %.3e" % (y_string, y_getter_method()[1]))
        print("Number of fluid %s divisions: %i" % (y_string, Ny_getter_method()))
        printhbar()

        # 4: General settings and finalize.
        statisfied = GeneralSettings(Config_in=Config_in)
    return