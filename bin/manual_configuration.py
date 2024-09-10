from Common.DataDrivenConfig import FlameletAIConfig, EntropicAIConfig
from shutil import get_terminal_size
h_bar = "#==================================================================================#"
def printhbar():
    print("#" + "="*(get_terminal_size()[0]-2) + "#")

def ManualFlameletConfiguration():
    Config_in:FlameletAIConfig = FlameletAIConfig()
    statisfied = False 
    def InsertFlameletConfigOption(Config_in_method, default_value, input_message:str):
        correct_input:bool = False 
        while not correct_input:
            try:
                val_input = default_value
                user_input = input(input_message)
                if not user_input == "":
                    val_input = type(default_value)(user_input)
                Config_in_method(val_input)
                correct_input = True 
            except:
                Config_in_method(default_value)
                print("Wrong input, please try again")
    
    while not statisfied:
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

        InsertFlameletConfigOption(Config_in.SetReactionMechanism, Config_in.GetReactionMechanism(), "Insert reaction mechanism file to use for flamelet computations (%s by default): " % Config_in.GetReactionMechanism())
        print("Reaction mechanism: " + Config_in.GetReactionMechanism())
        printhbar()

        InsertFlameletConfigOption(Config_in.SetTransportModel, Config_in.GetTransportModel(), "Insert flamelet solver transport model (mixture-averaged/multicomponent/unity-Lewis-number, %s by default): " % Config_in.GetTransportModel())
        print("Flamelet solution transport model: %s" % Config_in.GetTransportModel())
        printhbar()

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
                print("Wrong input, please try again.")
        print("Progress variable definition: " + "+".join(("(%+.3e)*%s" % (w, s)) for w, s in zip(Config_in.GetProgressVariableWeights(), Config_in.GetProgressVariableSpecies())))
        printhbar()

        correct_mixture_definition = False 
        while not correct_mixture_definition:
            try:
                mixture_definition:str= input("Insert definition for reactant mixture (1 for equivalence ratio, 2 for mixture fraction, 1 by default): ")
                
                if not mixture_definition == "":
                    if int(mixture_definition) == 1:
                        print("Reactant mixture defined as equivalence ratio.")
                        Config_in.DefineMixtureStatus(run_as_mixture_fraction=False)
                    elif int(mixture_definition) == 2:
                        print("Reactant mixture defined as mixture fraction.")
                        Config_in.DefineMixtureStatus(run_as_mixture_fraction=True)
                    else:
                        raise Exception()
                else:
                    Config_in.DefineMixtureStatus()
                
                correct_mixture_definition = True 
            except:
                print("Wrong input, please try again.")
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
                    print("Wrong input, please try again.")
        
        
        
        InsertFlameletConfigOption(Config_in.SetNpMix, Config_in.GetNpMix(), "Insert number of divisions for the mixture status range (%i by default): " % Config_in.GetNpMix())
        print("Lower reactant mixture status value: %.3f" % Config_in.GetMixtureBounds()[0])
        print("Upper reactant mixture status value: %.3f" % Config_in.GetMixtureBounds()[1])
        print("Number of mixture status divisions: %i" % Config_in.GetNpMix())
        printhbar()

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
                    print("Wrong input, please try again.")

            
            InsertFlameletConfigOption(Config_in.SetNpTemp, Config_in.GetNpTemp(), "Insert number of divisions for the reactant temperature range (%i by default): " % Config_in.GetNpTemp())
            print("Lower reactant temperature value: %.3f K" % Config_in.GetUnbTempBounds()[0])
            print("Upper reactant temperature value: %.3f K" % Config_in.GetUnbTempBounds()[1])
            print("Number of reactant temperature status divisions: %i" % Config_in.GetNpTemp())
            printhbar()

            InsertFlameletConfigOption(Config_in.SetOutputDir, Config_in.GetOutputDir(), "Insert flamelet data output directory (%s by default):" % Config_in.GetOutputDir())
            print("Flamelet data output folder: %s" % Config_in.GetOutputDir())
            printhbar()

            InsertFlameletConfigOption(Config_in.RunFreeFlames, Config_in.GenerateFreeFlames(), "Compute adiabatic flamelet data (1=yes, 0=no, 1 by default): ")
            if Config_in.GenerateFreeFlames():
                print("Adiabatic flamelets are included in manifold.")
            else:
                print("Adiabatic flamelets are ommitted in manifold.")
            printhbar()

            InsertFlameletConfigOption(Config_in.RunBurnerFlames, Config_in.GenerateBurnerFlames(), "Compute burner-stabilized flamelet data (1=yes, 0=no, 1 by default): ")
            if Config_in.GenerateBurnerFlames():
                print("Burner-stabilized flamelets are included in manifold.")
            else:
                print("Burner-stabilized flamelets are ommitted in manifold.")
            printhbar()

            InsertFlameletConfigOption(Config_in.RunEquilibrium, Config_in.GenerateEquilibrium(), "Compute chemical equilibrium data (1=yes, 0=no, 1 by default): ")
            if Config_in.GenerateEquilibrium():
                print("Chemical equilibrium data are included in manifold.")
            else:
                print("Chemical equilibrium data are ommitted in manifold.")
            printhbar()

            InsertFlameletConfigOption(Config_in.SetConfigName, Config_in.GetConfigName(), "Set a name for the current FlameletAI configuration (%s by default): " % Config_in.GetConfigName())
            printhbar()
            print("Summary:")
            Config_in.PrintBanner()

            statisfied_input:str = input("Save configuration and exit (1) or re-run configuration set-up (2)?")
            if int(statisfied_input) == 1:
                Config_in.SaveConfig()
                statisfied = True 
            else:
                print("Re-running configuration set-up")
    return 

def ManualNICFDConfiguration():
    print("Not implemented yet!")
    return