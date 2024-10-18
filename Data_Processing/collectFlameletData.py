###############################################################################################
#       #      _____ __  _____      ____        __        __  ____                   #        #
#       #     / ___// / / /__ \    / __ \____ _/ /_____ _/  |/  (_)___  ___  _____   #        #
#       #     \__ \/ / / /__/ /   / / / / __ `/ __/ __ `/ /|_/ / / __ \/ _ \/ ___/   #        #
#       #    ___/ / /_/ // __/   / /_/ / /_/ / /_/ /_/ / /  / / / / / /  __/ /       #        #
#       #   /____/\____//____/  /_____/\__,_/\__/\__,_/_/  /_/_/_/ /_/\___/_/        #        #
#       #                                                                            #        #
###############################################################################################

############################ FILE NAME: collectFlameletData.py ################################
#=============================================================================================#
# author: Evert Bunschoten                                                                    |
#    :PhD Candidate ,                                                                         |
#    :Flight Power and Propulsion                                                             |
#    :TU Delft,                                                                               |
#    :The Netherlands                                                                         |
#                                                                                             |
#                                                                                             |
# Description:                                                                                |
#  Class for reading flamelet data files, extracting relevant data, and generating a          |
#  homogeneous distribution of flamelet data along the progress variable, enthalpy, and       |
#  mixture fraction direction.                                                                |
#                                                                                             |
# Version: 1.0.0                                                                              |
#                                                                                             |
#=============================================================================================#

import numpy as np 
from os import path, listdir
import sys
import csv 
from tqdm import tqdm
np.random.seed(0)
from random import sample 
import os 
import matplotlib.pyplot as plt 
prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()['color']

from Common.DataDrivenConfig import FlameletAIConfig
from Common.Properties import DefaultSettings_FGM

class FlameletConcatenator:
    """Read, regularize, and concatenate flamelet data for MLP training or LUT generation.

    """
    __Config:FlameletAIConfig = None # FlameletAI configuration for current workflow.

    __Np_per_flamelet:int = 2**DefaultSettings_FGM.batch_size_exponent          # Number of data points to extract per flamelet.
    __custom_resolution:bool = False    # Overwrite average number of data points per flamelet with a specified value.

    __mfrac_skip:int = 1        # Number of mixture status folder to skip while concatenating flamelet data.

    __ignore_mixture_bounds:bool = False
    __mix_status_max:float = DefaultSettings_FGM.eq_ratio_min     # Minimum mixture status value above which to collect flamelet data.
    __mix_status_min:float = DefaultSettings_FGM.eq_ratio_max    # Maximum mixture status value below which to collect flamelet data.

    __f_train:float = DefaultSettings_FGM.train_fraction   # Fraction of the flamelet data used for training.
    __f_test:float = DefaultSettings_FGM.test_fraction    # Fraction of the flamelet data used for testing.

    __output_file_header:str = DefaultSettings_FGM.output_file_header   # File header for the concatenated flamelet data file.
    __boundary_file_header:str = DefaultSettings_FGM.boundary_file_header
    __flameletdata_dir:str = "./"               # Directory from which to read flamelet data.

    __controlling_variables:list[str] = DefaultSettings_FGM.controlling_variables
    __N_control_vars:int = len(DefaultSettings_FGM.controlling_variables)

    # Thermodynamic data to search for in flamelet data.
    __TD_train_vars = ['Temperature', 'MolarWeightMix', 'DiffusionCoefficient', 'Conductivity', 'ViscosityDyn', 'Cp']
    __TD_flamelet_data:np.ndarray = None 

    # Differential diffusion data to search for in flamelet data.
    __PD_train_vars = ['Beta_ProgVar', 'Beta_Enth_Thermal', 'Beta_Enth', 'Beta_MixFrac']
    __PD_flamelet_data:np.ndarray = None 

    __flamelet_ID_vars = ['FlameletID']
    __flamelet_ID:np.ndarray = None 

    # Passive species names for which to save production and consumption terms.
    __Species_in_FGM = []

    # Passive look-up terms to include in the manifold.
    __LookUp_vars = []
    __LookUp_flamelet_data:np.ndarray = None 

    # Progress variable source term name.
    __PPV_train_vars = ['ProdRateTot_PV']
    __Sources_vars = [__PPV_train_vars[0]]
    for s in __Species_in_FGM:
        __Sources_vars.append("Y_dot_pos-"+s)
        __Sources_vars.append("Y_dot_neg-"+s)
        __Sources_vars.append("Y_dot_net-"+s)
    __Sources_flamelet_data = None

    __CV_flamelet_data:np.ndarray = None    # Flamelet controlling variables

    __include_freeflames = True     # Read adiabatic flamelets.
    __include_burnerflames = True   # Read burner-stabilized flamelets.
    __include_equilibrium = True    # Read chemical equilibrium data.
    __include_counterflame = False 
    __include_fuzzy = False

    __write_LUT_data:bool = False   # Apply source term and equilibrium data corrections for table data preparations.

    __verbose:int=1

    def __init__(self, Config:FlameletAIConfig,verbose_level:int=1):
        """Class constructor

        :param Config: loaded FlameletAIConfig class for the current workflow.
        :type Config: FlameletAIConfig
        """
        self.__verbose = verbose_level
        if self.__verbose >0:
            print("Loading flameletAI configuration " + Config.GetConfigName())
        self.__Config = Config
        self.__SynchronizeSettings()
        
        return 
    
    def __SynchronizeSettings(self):
        # Load settins from configuration:
        self.__include_freeflames = self.__Config.GenerateFreeFlames()
        self.__include_burnerflames = self.__Config.GenerateBurnerFlames()
        self.__include_equilibrium = self.__Config.GenerateEquilibrium()
        self.__include_counterflame = self.__Config.GenerateCounterFlames()

        self.__Np_per_flamelet = self.__Config.GetNpConcatenation()
        [self.__mix_status_min, self.__mix_status_max] = self.__Config.GetMixtureBounds()
        self.__f_train = self.__Config.GetTrainFraction()
        self.__f_test = self.__Config.GetTestFraction()
        
        self.SetAuxilarySpecies(self.__Config.GetPassiveSpecies())
        self.SetLookUpVars(self.__Config.GetLookUpVariables())

        self.__flameletdata_dir = self.__Config.GetOutputDir()
        self.__output_file_header = self.__Config.GetConcatenationFileHeader()
        self.__controlling_variables = []
        for c in self.__Config.GetControllingVariables():
            self.__controlling_variables.append(c)
        self.__N_control_vars = len(self.__controlling_variables)

        return 
    
    def IgnoreMixtureBounds(self, ignore_bounds:bool=False):
        self.__ignore_mixture_bounds = ignore_bounds
        return 
    
    def WriteLUTData(self, write_LUT_data:bool=False):
        """Apply corrections to chemical equilibrium data and source terms in order to ensure boundary
        correctness for table generation.
        """
        self.__write_LUT_data = write_LUT_data
        return
    
    def SetNFlameletNodes(self, Np_per_flamelet:int):
        """Manually define the number of data points per flamelet to be included in the manifold.

        :param Np_per_flamelet: number of data points to be interpolated from each flamelet.
        :type Np_per_flamelet: int
        :raises Exception: if the number of points is lower than two.
        """
        if Np_per_flamelet < 2:
            raise Exception("Number of data points per flamelet should be higher than two.")
        self.__Np_per_flamelet = Np_per_flamelet
        self.__custom_resolution = True 
        return 
    
    def GetNFlameletNodes(self):
        return self.__Np_per_flamelet
    
    def SetMixStep(self, skip_mixtures:int):
        """Skip a number of mixture status values when reading flamelet data to reduce the concatenated file size.

        :param skip_mixtures: step size in mixture status
        :type skip_mixtures: int
        :raises Exception: if the provided step size is lower than one.
        """
        if skip_mixtures < 1:
            raise Exception("Mixture step size should be higher than one.")
        self.__mfrac_skip = skip_mixtures
        return 
    
    def SetMixStatusBounds(self, mix_status_low:float, mix_status_high:float):
        """Define the mixture status bounds between which to read flamelet data.

        :param mix_status_low: lower mixture status value.
        :type mix_status_low: float
        :param mix_status_high: upper mixture status value.
        :type mix_status_high: float
        :raises Exception: if the lower mixture status value is higher than the upper mixture status value.
        """
        if mix_status_low >= mix_status_high:
            raise Exception("Lower mixture status should be lower than upper value.")
        self.__mix_status_min = mix_status_low
        self.__mix_status_max = mix_status_high
        return 
    
    def SetAuxilarySpecies(self, species_input:list[str]):
        """Define the passive species names for which to collect source terms.

        :param input: list of species names.
        :type input: list[str]
        """
        self.__Config.SetPassiveSpecies(species_input)
        
        self.__Species_in_FGM = []
        for s in species_input:
            self.__Species_in_FGM.append(s)
        self.__Sources_vars = [self.__PPV_train_vars[0]]
        for s in species_input:
            self.__Sources_vars.append("Y_dot_pos-"+s)
            self.__Sources_vars.append("Y_dot_neg-"+s)
            self.__Sources_vars.append("Y_dot_net-"+s)
        return 

    def SetControllingVariables(self, controlling_variables:list[str]=DefaultSettings_FGM.controlling_variables):
        self.__Config.SetControllingVariables(controlling_variables)
        self.__SynchronizeSettings()
        return 
    
    def IncludeFreeFlames(self, input:bool):
        """Read adiabatic flamelet data.

        :param input: read adiabatic flamelets (True) or not (False)
        :type input: bool
        """
        self.__include_freeflames = input 
        return
    
    def IncludeBurnerFlames(self, input:bool):
        """Read burner-stabilized flamelet data.

        :param input: read burner-stabilized flamelet data (True) or not (False)
        :type input: bool
        """
        self.__include_burnerflames = input 
        return 
    
    def IncludeEquilibrium(self, input:bool):
        """Read chemical equilibrium data.

        :param input: read chemical equilibrium data (True) or not (False)
        :type input: bool
        """
        self.__include_equilibrium = input  
        return 
    
    def Include_CounterFlames(self, input:bool):
        self.__include_counterflame = input 
        return 
    
    def SetLookUpVars(self, input:list[str]):
        """Define passive look-up variables to be included in the manifold data.

        :param input: list of passive look-up variables.
        :type input: list[str]
        """
        self.__LookUp_vars = []
        for s in input:
            self.__LookUp_vars.append(s)
        return
    
    def SetFlameletDir(self, input:str):
        """Manually define the directory where the flamelet data is stored.

        :param input: path to flamelet data directory.
        :type input: str
        :raises Exception: if the provided directory doesn't exist.
        """
        if not path.isdir(input):
            raise Exception("Flamelet data directory does not exist.")
        self.__flameletdata_dir = input
        return 
    
    def SetOutputFileName(self, input:str):
        """Define the manifold output file header.

        :param input: manifold file header.
        :type input: str
        """
        self.__output_file_header = input 
        return 
    
    def SetBoundaryFileName(self, input:str):
        self.__boundary_file_header = input 
        return 
    
    def SetTrainFraction(self, input:float=DefaultSettings_FGM.train_fraction):
        """Define the fraction of concatenated flamelet data to be used for training MLP's.

        :param input: train data fraction. Should be between zero and one.
        :type input: float
        :raises Exception: if the provided fraction is lower than zero or higher than one.
        """
        if (input <= 0) or (input >= 1):
            raise Exception("Train fraction should be between zero and one.")
        self.__f_train = input 
        return
    
    def SetTestFraction(self, input:float=DefaultSettings_FGM.test_fraction):
        """Define the fraction of concatenated flamelet data to be used for accuracy testing after training MLP's.

        :param input: test data fraction. Should be between zero and one.
        :type input: float
        :raises Exception: if the provided fraction is lower than zero or higher than one.
        """
        if (input <= 0) or (input >= 1):
            raise Exception("Test fraction should be between zero and one.")
        self.__f_test = input 
        return

    def ConcatenateFlameletData(self):
        """Read flamelets and concatenate relevant flamelet data in the appropriate resolution.
        """

        # Size output data arrays according to number of flamelets and manifold resolution.
        self.__SizeDataArrays()

        i_start = 0

        # Read from the approprate file header.
        if self.__Config.GetMixtureStatus():
            folder_header = "mixfrac_"
        else:
            folder_header = "phi_"

        if self.__verbose > 0:
            print("Concatenating flamelet data...")

        # Read adiabatic flamelet data
        if self.__include_freeflames:
            if self.__verbose > 0:
                print("Reading freeflames...")
            i_freeflame_total = 0
            mixture_folders = np.sort(np.array(self.mfracs_freeflames))
            for z in mixture_folders[::self.__mfrac_skip]:
                if self.__ignore_mixture_bounds:
                    i_freeflame_total = self.__InterpolateFlameletData(self.__flameletdata_dir + "/freeflame_data/", z, 0, i_freeflame_total)
                else:
                    if z[:len(folder_header)] == folder_header:
                        mixture_status = float(z[len(folder_header):])
                        if (mixture_status <= self.__mix_status_max) and (mixture_status >= self.__mix_status_min):
                            i_freeflame_total = self.__InterpolateFlameletData(self.__flameletdata_dir + "/freeflame_data/", z, 0, i_freeflame_total)
            if self.__verbose > 0:
                print("Done!")
            i_start += i_freeflame_total

        # Read burner-stabilized flamelet data
        if self.__include_burnerflames:
            if self.__verbose > 0:
                print("Reading burnerflamelets...")
            i_burnerflame_total = 0
            mixture_folders = np.sort(np.array(self.mfracs_burnerflames))
            for z in mixture_folders[::self.__mfrac_skip]:
                if self.__ignore_mixture_bounds:
                    i_burnerflame_total = self.__InterpolateFlameletData(self.__flameletdata_dir + "/burnerflame_data/", z, i_start, i_burnerflame_total)
                else:
                    if z[:len(folder_header)] == folder_header:
                        mixture_status = float(z[len(folder_header):])
                        if (mixture_status <= self.__mix_status_max) and (mixture_status >= self.__mix_status_min):
                            i_burnerflame_total = self.__InterpolateFlameletData(self.__flameletdata_dir + "/burnerflame_data/", z, i_start, i_burnerflame_total)
            if self.__verbose > 0:
                print("Done!")
            i_start +=  i_burnerflame_total

        # Read chemical equilibrium data
        if self.__include_equilibrium:
            i_equilibrium_total = 0
            if self.__verbose>0:
                print("Reading equilibrium data...")
            mixture_folders = np.sort(np.array(self.mfracs_equilibrium))
            for z in mixture_folders[::self.__mfrac_skip]:
                if self.__ignore_mixture_bounds:
                    i_equilibrium_total = self.__InterpolateFlameletData(self.__flameletdata_dir + "/equilibrium_data/", z, i_start, i_equilibrium_total, is_equilibrium=True)
                else:
                    if z[:len(folder_header)] == folder_header:
                        mixture_status = float(z[len(folder_header):])
                        if (mixture_status <= self.__mix_status_max) and (mixture_status >= self.__mix_status_min):
                            i_equilibrium_total = self.__InterpolateFlameletData(self.__flameletdata_dir + "/equilibrium_data/", z, i_start, i_equilibrium_total, is_equilibrium=True)
            if self.__verbose > 0:
                print("Done!")
            i_start +=  i_equilibrium_total

        # Read fuzzy data
        if self.__include_fuzzy:
            i_fuzzy_total = 0
            print("Reading fuzzy data...")
            mixture_folders = np.sort(np.array(self.mfracs_fuzzy))
            for z in mixture_folders[::self.__mfrac_skip]:
                if z[:len(folder_header)] == folder_header:
                    mixture_status = float(z[len(folder_header):])
                    if (mixture_status <= self.__mix_status_max) and (mixture_status >= self.__mix_status_min):
                        i_fuzzy_total = self.__InterpolateFlameletData(self.__flameletdata_dir + "/fuzzy_data/", z, i_start, i_fuzzy_total, is_fuzzy=True)
            print("Done!")
            i_start +=  i_fuzzy_total

        if self.__include_counterflame:
            i_counterflame_total = 0
            print("Reading counter-flow diffusion flame data...")
            i_counterflame_total = self.__InterpolateFlameletData(self.__flameletdata_dir + "/counterflame_data/", "", i_start, i_counterflame_total)
            print("Done!")
            i_start += i_counterflame_total

        # Once all flamelet data has been read and interpolated, write output files.
        if self.__verbose > 0:
            print("Writing output data...")
        self.__WriteOutputFiles()
        if self.__verbose > 0:
            print("Done!")

    def CollectBoundaryData(self):
        self.IgnoreMixtureBounds(True)
        self.Include_CounterFlames(False)
        self.IncludeBurnerFlames(False)
        self.IncludeFreeFlames(False)
        self.IncludeEquilibrium(True)
        self.__output_file_header = self.__boundary_file_header
        self.ConcatenateFlameletData()
        return 
    
    def __WriteOutputFiles(self):
        """Collect all flamelet data arrays, split into train, test, and validation portions, and write to appropriately named files.
        """

        # Collect all variable names in the manifold.
        total_variables = "ProgressVariable,EnthalpyTot,MixtureFraction,"
        total_variables += ",".join(self.__TD_train_vars)+","
        if self.__Config.PreferentialDiffusion():
            total_variables += ",".join(self.__PD_train_vars)+","
        total_variables += ",".join(self.__Sources_vars)+","
        total_variables += ",".join(self.__LookUp_vars)+","
        total_variables += ",".join(self.__flamelet_ID_vars)

        # Concatenate all flamelet data arrays into a single 2D array.
        total_data = np.append(self.__CV_flamelet_data, self.__TD_flamelet_data,axis=1)
        if self.__Config.PreferentialDiffusion():
            total_data = np.append(total_data, self.__PD_flamelet_data, axis=1)
        total_data = np.append(total_data, self.__Sources_flamelet_data, axis=1)
        total_data = np.append(total_data, self.__LookUp_flamelet_data, axis=1)
        total_data = np.append(total_data, self.__flamelet_ID,axis=1)

        # Filter unique data points.
        _, idx_unique = np.unique(self.__CV_flamelet_data, axis=0, return_index=True)
        total_data = total_data[idx_unique, :]

        # Remove any empty rows.
        total_data = total_data[~np.all(total_data == 0, axis=1)]

        # Shuffle flamelet data to remove bias.
        np.random.shuffle(total_data)

        # Number of data points for training and testing.
        np_train = int(self.__f_train*len(total_data))
        np_val = int(self.__f_test*len(total_data))

        train_data = total_data[:np_train, :]
        val_data = total_data[np_train:np_train+np_val, :]
        test_data = total_data[np_train+np_val:, :]

        # Write full, train, validation, and test data files.
        if self.__verbose > 0:
            print("Writing output files with header " + self.__output_file_header)
        fid = open(self.__flameletdata_dir +"/"+ self.__output_file_header + "_full.csv", "w+")
        fid.write(total_variables + "\n")
        csvwriter = csv.writer(fid)
        csvwriter.writerows(total_data)
        fid.close()

        fid = open(self.__flameletdata_dir +"/"+ self.__output_file_header + "_train.csv", "w+")
        fid.write(total_variables + "\n")
        csvwriter = csv.writer(fid)
        csvwriter.writerows(train_data)
        fid.close()

        fid = open(self.__flameletdata_dir +"/"+ self.__output_file_header + "_val.csv", "w+")
        fid.write(total_variables + "\n")
        csvwriter = csv.writer(fid)
        csvwriter.writerows(val_data)
        fid.close()

        fid = open(self.__flameletdata_dir +"/"+ self.__output_file_header + "_test.csv", "w+")
        fid.write(total_variables + "\n")
        csvwriter = csv.writer(fid)
        csvwriter.writerows(test_data)
        fid.close()
        if self.__verbose > 0:
            print("Done!")
        return 
    
    def __SizeDataArrays(self):
        """Size the output data arrays according to the number of flamelets and manifold resolution.
        """

        # Total number of converged flamelet solutions.
        n_freeflames = 0
        n_burnerflames = 0
        n_eq = 0
        n_counterflames = 0
        n_fuz = 0

        # Total number of flamelet data points.
        Np_tot = 0

        # Get the appropriate folder header according to the mixture status definition.
        if self.__Config.GetMixtureStatus():
            folder_header = "mixfrac_"
        else:
            folder_header = "phi_"

        # Count the number of adiabatic flamelets.
        if self.__include_freeflames:
            if self.__verbose > 0:
                print("Counting adabatic free-flame data...")
            self.mfracs_freeflames = listdir(self.__flameletdata_dir + "/freeflame_data")
            mixture_folders = np.sort(np.array(self.mfracs_freeflames))
            for z in mixture_folders[::self.__mfrac_skip]:
                if self.__ignore_mixture_bounds:
                    n_freeflames += len(listdir(self.__flameletdata_dir + "/freeflame_data/" + z))
                    for f in listdir(self.__flameletdata_dir + "/freeflame_data/" + z):
                        with open(self.__flameletdata_dir + "/freeflame_data/" + z + "/" + f, "r") as fid:
                            Np_tot += len(fid.readlines())-1
                else:
                    if z[:len(folder_header)] == folder_header:
                        mixture_status = float(z[len(folder_header):])
                        if (mixture_status <= self.__mix_status_max) and (mixture_status >= self.__mix_status_min):
                            n_freeflames += len(listdir(self.__flameletdata_dir + "/freeflame_data/" + z))
                            for f in listdir(self.__flameletdata_dir + "/freeflame_data/" + z):
                                with open(self.__flameletdata_dir + "/freeflame_data/" + z + "/" + f, "r") as fid:
                                    Np_tot += len(fid.readlines())-1

        # Count the number of burner-stabilized flamelets.
        if self.__include_burnerflames:
            if self.__verbose > 0:
                print("Counting burner-stabilized flame data...")
            self.mfracs_burnerflames = listdir(self.__flameletdata_dir + "/burnerflame_data")
            mixture_folders = np.sort(np.array(self.mfracs_burnerflames))
            for z in mixture_folders[::self.__mfrac_skip]:
                if self.__ignore_mixture_bounds:
                    n_burnerflames += len(listdir(self.__flameletdata_dir + "/burnerflame_data/" + z))
                    for f in listdir(self.__flameletdata_dir + "/burnerflame_data/" + z):
                        with open(self.__flameletdata_dir + "/burnerflame_data/" + z + "/" + f, "r") as fid:
                            Np_tot += len(fid.readlines())-1
                else:
                    if z[:len(folder_header)] == folder_header:
                        mixture_status = float(z[len(folder_header):])
                        if (mixture_status <= self.__mix_status_max) and (mixture_status >= self.__mix_status_min):
                            n_burnerflames += len(listdir(self.__flameletdata_dir + "/burnerflame_data/" + z))
                            for f in listdir(self.__flameletdata_dir + "/burnerflame_data/" + z):
                                with open(self.__flameletdata_dir + "/burnerflame_data/" + z + "/" + f, "r") as fid:
                                    Np_tot += len(fid.readlines())-1
        
        # Count the number of chemical equilibrium data files.
        if self.__include_equilibrium:
            if self.__verbose > 0:
                print("Counting chemical equilibrium data...")
            self.mfracs_equilibrium = listdir(self.__flameletdata_dir + "/equilibrium_data")
            mixture_folders = np.sort(np.array(self.mfracs_equilibrium))
            for z in mixture_folders[::self.__mfrac_skip]:
                if self.__ignore_mixture_bounds:
                    n_eq += len(listdir(self.__flameletdata_dir + "/equilibrium_data/" + z))
                    for f in listdir(self.__flameletdata_dir + "/equilibrium_data/" + z):
                        with open(self.__flameletdata_dir + "/equilibrium_data/" + z + "/" + f, "r") as fid:
                            Np_tot += len(fid.readlines())-1
                else:
                    if z[:len(folder_header)] == folder_header:
                        mixture_status = float(z[len(folder_header):])
                        if (mixture_status <= self.__mix_status_max) and (mixture_status >= self.__mix_status_min):
                            n_eq += len(listdir(self.__flameletdata_dir + "/equilibrium_data/" + z))
                            for f in listdir(self.__flameletdata_dir + "/equilibrium_data/" + z):
                                with open(self.__flameletdata_dir + "/equilibrium_data/" + z + "/" + f, "r") as fid:
                                    Np_tot += len(fid.readlines())-1

        # Count the number of chemical equilibrium data files.
        if self.__include_fuzzy:
            print("Counting fuzzy flamelet data...")
            self.mfracs_fuzzy = listdir(self.__flameletdata_dir + "/fuzzy_data")
            mixture_folders = np.sort(np.array(self.mfracs_fuzzy))
            for z in mixture_folders[::self.__mfrac_skip]:
                if z[:len(folder_header)] == folder_header:
                    mixture_status = float(z[len(folder_header):])
                    if (mixture_status <= self.__mix_status_max) and (mixture_status >= self.__mix_status_min):
                        n_fuz += len(listdir(self.__flameletdata_dir + "/fuzzy_data/" + z))
                        for f in listdir(self.__flameletdata_dir + "/fuzzy_data/" + z):
                            with open(self.__flameletdata_dir + "/fuzzy_data/" + z + "/" + f, "r") as fid:
                                Np_tot += len(fid.readlines())-1

        # Count the number of counter-flow diffusion flamelets.
        if self.__include_counterflame:
            print("Counting counter-flow diffusion flame data...")
            counterflame_files = listdir(self.__flameletdata_dir + "/counterflame_data")
            n_counterflames += len(counterflame_files)
            for f in counterflame_files:
                with open(self.__flameletdata_dir + "/counterflame_data/" + f, 'r') as fid:
                    Np_tot += len(fid.readlines())-1
        if self.__verbose > 0:
            print("Done!")

        # Compute the average number of data points per flamelet.
        n_flamelets_total = n_freeflames + n_burnerflames + n_eq + n_counterflames + n_fuz
        if not self.__custom_resolution:
            #self.__Np_per_flamelet = 2**self.__Config.GetBatchExpo()
            self.__Np_per_flamelet = int(Np_tot / n_flamelets_total)
        if self.__verbose > 0:
            print("Number of data-points per flamelet: %i " % self.__Np_per_flamelet)

        # Size output data arrays according to manifold resolution.
        self.__CV_flamelet_data = np.zeros([n_flamelets_total * self.__Np_per_flamelet, self.__N_control_vars])
        self.__TD_flamelet_data = np.zeros([n_flamelets_total * self.__Np_per_flamelet, len(self.__TD_train_vars)])
        if self.__Config.PreferentialDiffusion():
            self.__PD_flamelet_data = np.zeros([n_flamelets_total * self.__Np_per_flamelet, len(self.__PD_train_vars)])
        self.__Sources_flamelet_data = np.zeros([n_flamelets_total * self.__Np_per_flamelet, 1 + 3 * len(self.__Species_in_FGM)])
        self.__LookUp_flamelet_data = np.zeros([n_flamelets_total * self.__Np_per_flamelet, len(self.__LookUp_vars)])
        self.__flamelet_ID = np.zeros([n_flamelets_total * self.__Np_per_flamelet, len(self.__flamelet_ID_vars)],dtype=int)

        self.__N_p_total = n_flamelets_total * self.__Np_per_flamelet
        return 
    
    
    def __InterpolateFlameletData(self, flamelet_dir:str, eq_file:str, i_start:int, i_flamelet_total:int, is_fuzzy:bool=False, is_equilibrium:bool=False):

        flamelets = listdir(flamelet_dir + "/" + eq_file)
        for i_flamelet, f in enumerate(flamelets):
            BurningFlamelet:bool = True 

            # Get flamelet variables
            fid = open(flamelet_dir + "/" + eq_file + "/" + f, 'r')
            variables = fid.readline().strip().split(',')
            fid.close()

            # Load flamelet data
            if is_equilibrium and self.__write_LUT_data:
                D = np.loadtxt(flamelet_dir + "/" + eq_file + "/" + f, delimiter=',',skiprows=1,max_rows=1)[np.newaxis, :]
            else:
                D = np.loadtxt(flamelet_dir + "/" + eq_file + "/" + f, delimiter=',',skiprows=1)

            # Set flamelet controlling variables
            CV_flamelet = np.zeros([len(D), self.__N_control_vars])
            for iCV in range(self.__N_control_vars):
                if self.__controlling_variables[iCV] == DefaultSettings_FGM.name_pv:
                    pv_flamelet = self.__Config.ComputeProgressVariable(variables, D)
                    
                    CV_flamelet[:, iCV] = pv_flamelet
                else:
                    CV_flamelet[:, iCV] = D[:, variables.index(self.__controlling_variables[iCV])]

            CV_min, CV_max = np.min(CV_flamelet, axis=0), np.max(CV_flamelet, axis=0)
            CV_norm = (CV_flamelet - CV_min)/(CV_max - CV_min + 1e-10)
            delta_CV = CV_norm[1:] - CV_norm[:-1]
            dS = np.sqrt(np.sum(np.power(delta_CV, 2), axis=1))
            S_flamelet = np.append(np.array([0]), np.cumsum(dS))
            is_valid_flamelet = True 

            if (max(S_flamelet) == 0) and not is_equilibrium:
                print("Dodgy flamelet data file: " + flamelet_dir + "/" + eq_file + "/" + f)
                is_valid_flamelet = False

            if is_valid_flamelet:
                S_flamelet_norm = S_flamelet / (max(S_flamelet)+1e-32)

                T_flamelet = D[:, variables.index("Temperature")]
                if np.max(T_flamelet) < DefaultSettings_FGM.T_threshold:
                    BurningFlamelet = False 

                sourceterm_zero_line_numbers = [0, -1]

                if self.__write_LUT_data:
                    # Set source terms to zero near the start and end of the flamelet.
                    temp_margin = 2e-2
                    T_max, T_min = np.max(T_flamelet), np.min(T_flamelet)
                    deltaT = temp_margin*(T_max - T_min)
                    sourceterm_zero_line_numbers = np.logical_or((T_flamelet - T_min) < deltaT,\
                                                                 ((T_max - T_flamelet) < deltaT))

                # Load flamelet thermophysical property data
                TD_data = np.zeros([len(D), len(self.__TD_train_vars)])
                if BurningFlamelet:

                    for iVar_TD, TD_var in enumerate(self.__TD_train_vars):
                        if TD_var == "DiffusionCoefficient":
                            idx_cp = variables.index("Cp")
                            idx_cond = variables.index("Conductivity")
                            idx_density = variables.index("Density")
                            TD_data[:, iVar_TD] = D[:, idx_cond] / (D[:, idx_cp] * D[:, idx_density])
                        else:
                            idx_var_flamelet = variables.index(TD_var)
                            TD_data[:, iVar_TD] = D[:, idx_var_flamelet]

                # Load flamelet look-up variable data
                LookUp_data = np.zeros([len(D), len(self.__LookUp_vars)])
                if BurningFlamelet:
                    for iVar_LookUp, LookUp_var in enumerate(self.__LookUp_vars):
                        idx_var_flamelet = variables.index(LookUp_var)
                        LookUp_data[:, iVar_LookUp] = D[:, idx_var_flamelet]
                        if LookUp_var == "Heat_Release":
                            LookUp_data[sourceterm_zero_line_numbers, iVar_LookUp] = 0.0
                
                # Load species sources data
                species_production_rate = np.zeros([len(D), len(self.__Species_in_FGM)])
                species_destruction_rate = np.zeros([len(D), len(self.__Species_in_FGM)])
                species_net_rate = np.zeros([len(D), len(self.__Species_in_FGM)])
                if BurningFlamelet:
                    for iSp, Sp in enumerate(self.__Species_in_FGM):
                        if Sp == "NOx":
                            species_production_rate[:, iSp] = np.zeros(len(D))
                            species_destruction_rate[:, iSp] = np.zeros(len(D))
                            species_net_rate[:, iSp] = np.zeros(len(D))
                            for NOsp in ["NO2","NO","N2O"]:
                                species_production_rate[:, iSp] += D[:, variables.index("Y_dot_pos-"+NOsp)]
                                species_destruction_rate[:, iSp] += D[:, variables.index("Y_dot_neg-"+NOsp)]
                                species_net_rate[:, iSp] += D[:, variables.index("Y_dot_net-"+NOsp)]
                        else:
                            species_production_rate[:, iSp] = D[:, variables.index("Y_dot_pos-"+Sp)]
                            species_destruction_rate[:, iSp] = D[:, variables.index("Y_dot_neg-"+Sp)]
                            species_net_rate[:, iSp] = D[:, variables.index("Y_dot_net-"+Sp)]

                Sources_data = np.zeros([len(D), 1 + 3 * len(self.__Species_in_FGM)])
                if BurningFlamelet:
                    ppv_flamelet = self.__Config.ComputeProgressVariable_Source(variables, D)
                    Sources_data[:, 0] = ppv_flamelet

                    for iSp in range(len(self.__Species_in_FGM)):
                        Sources_data[:, 1 + 3*iSp] = species_production_rate[:, iSp]
                        Sources_data[:, 1 + 3*iSp + 1] = species_destruction_rate[:, iSp]
                        Sources_data[:, 1 + 3*iSp + 2] = species_net_rate[:, iSp]

                    Sources_data[sourceterm_zero_line_numbers, :] = 0.0

                # Compute preferential diffusion scalars
                if self.__Config.PreferentialDiffusion() and BurningFlamelet:
                    beta_pv_flamelet, beta_h1_flamelet, beta_h2_flamelet, beta_z_flamelet = self.__Config.ComputeBetaTerms(variables, D)
                    PD_data = np.zeros([len(D), len(self.__PD_train_vars)])
                    PD_data[:, 0] = beta_pv_flamelet
                    PD_data[:, 1] = beta_h1_flamelet
                    PD_data[:, 2] = beta_h2_flamelet
                    PD_data[:, 3] = beta_z_flamelet

                if BurningFlamelet:
                    if is_fuzzy:
                        if self.__Np_per_flamelet > len(D):
                            samples = [k for k in range(len(D))] + [0]*(self.__Np_per_flamelet - len(D))
                        else:
                            samples = sample(range(len(D)), self.__Np_per_flamelet)
                        CV_sampled = CV_flamelet[samples, :]
                        TD_sampled = TD_data[samples, :]
                        if self.__Config.PreferentialDiffusion():
                            PD_sampled = PD_data[samples, :]
                        lookup_sampled = LookUp_data[samples, :]
                        sources_sampled = Sources_data[samples, :]
                    else:
                        # Define query controlling variable range
                        if is_equilibrium and self.__write_LUT_data:
                            CV_sampled = CV_flamelet*np.ones([self.__Np_per_flamelet, np.shape(CV_flamelet)[1]])
                            TD_sampled = TD_data*np.ones([self.__Np_per_flamelet, np.shape(CV_flamelet)[1]])
                            if self.__Config.PreferentialDiffusion():
                                PD_sampled = PD_data*np.ones([self.__Np_per_flamelet, np.shape(CV_flamelet)[1]])
                            lookup_sampled = LookUp_data*np.ones([self.__Np_per_flamelet, np.shape(CV_flamelet)[1]])
                            sources_sampled = np.zeros([self.__Np_per_flamelet, 1 + 3*len(self.__Species_in_FGM)])
                        else:
                            if is_equilibrium:
                                S_q = np.linspace(0, 1.0, self.__Np_per_flamelet)
                            else:
                                S_q = 0.5 - 0.5*np.cos(np.linspace(0, np.pi, self.__Np_per_flamelet))
                            CV_sampled = np.zeros([self.__Np_per_flamelet, np.shape(CV_flamelet)[1]])
                            TD_sampled = np.zeros([self.__Np_per_flamelet, np.shape(TD_data)[1]])
                            if self.__Config.PreferentialDiffusion():
                                PD_sampled = np.zeros([self.__Np_per_flamelet, np.shape(PD_data)[1]])
                            lookup_sampled = np.zeros([self.__Np_per_flamelet, np.shape(LookUp_data)[1]])
                            sources_sampled = np.zeros([self.__Np_per_flamelet, 1 + 3*len(self.__Species_in_FGM)])
                            for i_CV in range(self.__N_control_vars):
                                CV_sampled[:, i_CV] = np.interp(S_q, S_flamelet_norm, CV_flamelet[:, i_CV])
                            for iVar_TD in range(len(self.__TD_train_vars)):
                                TD_sampled[:, iVar_TD] = np.interp(S_q, S_flamelet_norm, TD_data[:, iVar_TD])
                            if self.__Config.PreferentialDiffusion():
                                for iVar_PD in range(len(self.__PD_train_vars)):
                                    PD_sampled[:, iVar_PD] = np.interp(S_q, S_flamelet_norm, PD_data[:, iVar_PD])

                            for iVar_LU in range(len(self.__LookUp_vars)):
                                lookup_sampled[:, iVar_LU] = np.interp(S_q, S_flamelet_norm, LookUp_data[:, iVar_LU])
                            for iVar_Source in range(1 + 3*len(self.__Species_in_FGM)):
                                sources_sampled[:, iVar_Source] = np.interp(S_q, S_flamelet_norm, Sources_data[:, iVar_Source])

                    start = (i_start + i_flamelet + i_flamelet_total) * self.__Np_per_flamelet
                    end = (i_start + i_flamelet+1+ i_flamelet_total)*self.__Np_per_flamelet
                    self.__CV_flamelet_data[start:end, :] = CV_sampled
                    self.__TD_flamelet_data[start:end, :] = TD_sampled
                    if self.__Config.PreferentialDiffusion():
                        self.__PD_flamelet_data[start:end, :] = PD_sampled 
                    self.__LookUp_flamelet_data[start:end, :] = lookup_sampled
                    self.__Sources_flamelet_data[start:end, :] = sources_sampled
                    self.__flamelet_ID[start:end, :] = i_start + i_flamelet + i_flamelet_total

        return len(flamelets) + i_flamelet_total

class GroupOutputs:
    """Class which groups flamelet data variables into MLP outputs based on their affinity.
    """

    __Config:FlameletAIConfig = None    # FlameletAI configuration for the current problem.
    __controlling_variables:list[str] = DefaultSettings_FGM.controlling_variables
    __vars_to_exclude:list[str] = DefaultSettings_FGM.controlling_variables + ["FlameletID"]   # Variables to exclude from grouping; controlling variables by default.
    __flamelet_variables:list[str]  # Flamelet data variable names.
    
    __free_variables:list[str]      # Flamelet variables considered for grouping.
    __flamelet_data_filepath:str    # File path where flamelet data collection file is located.
    __flamelet_data:np.ndarray      # Concatenated flamelet data.
    __correlation_matrix:np.ndarray # Cross-correlation values between flamelet data variables.
    __iVar_remove:list[int]         # Variable indices to exclude from data set.

    __theta_threshold:float = 0.7   # Affinity threshold above which groups are accepted.

    __group_leaders_orig:list[str] = [] # Lead variables forced to represent separate groups.
    __n_groups:list[int]                # Number of groups in each combination.
    __group_variables:list[list[str]]   # Variables in each group.
    __group_affinity:list[list[float]]  # Minimum affinity for each combination of groups.

    # Combinations of variables for FGM evaluation. 
    __evaluations_TD:list[str] = ["Temperature", "ViscosityDyn","MolarWeightMix","Cp","Conductivity","DiffusionCoefficient"]
    __evaluations_PD:list[str] = ["Beta_ProgVar","Beta_Enth_Thermal","Beta_Enth","Beta_MixFrac"]
    __evaluations_Sources:list[str] = ["ProdRateTot_PV"]

    __most_interesting_groups:list[list[list[str]]] = []    # Combinations of groups with highest affinity for a certain group count.

    __best_group:int = 0
    def __init__(self, Config_in:FlameletAIConfig):
        """Class constructor, load flamelet data.

        :param Config_in: FlameletAI configuration for the current problem.
        :type Config_in: FlameletAIConfig
        """
        self.__Config = Config_in 
        self.__flamelet_data_filepath = self.__Config.GetOutputDir()+"/"+self.__Config.GetConcatenationFileHeader()+"_full.csv"
        
        self.__controlling_variables = self.__Config.GetControllingVariables()
        self.__vars_to_exclude = []
        for var in self.__controlling_variables:
            self.__vars_to_exclude.append(var)
        self.__vars_to_exclude.append("FlameletID")
        self.__FilterVariables(self.__Config.GetControllingVariables() + ["FlameletID"])
        return 
    
    def SetFlameletDataFile(self, filepath_in:str):
        """Define a custom flamelet data file for which to compute output groups.

        :param filepath_in: file path name of flamelet data file.
        :type filepath_in: str
        :raises Exception: if specified path does not exist on current hardware.
        """
        if not os.path.isdir(filepath_in):
            raise Exception("Supplied flamelet data file does not exist.")
        self.__flamelet_data_filepath = filepath_in 
        self.__FilterVariables(self.__vars_to_exclude)

    def SetControllingVariables(self, control_vars:list[str]):
        """Define controlling variable names to always be excluded from flamelet data grouping.

        :param control_vars: list with controlling variable names
        :type control_vars: list[str]
        :raises Exception: if any of the specified variable names is not present in flamelet data set.
        """
        for var in control_vars:
            if var not in self.__flamelet_variables:
                raise Exception("Controlling variable " + var + " not present in flamelet data set.")
        
        vars_originally_excluded = self.__vars_to_exclude[len(self.__controlling_variables):]
        self.__controlling_variables = []
        self.__vars_to_exclude = []
        for var in control_vars:
            self.__controlling_variables.append(var)
            self.__vars_to_exclude.append(var)
        for var in vars_originally_excluded:
            self.__vars_to_exclude.append(var)

    def ExcludeVariables(self, vars_to_exclude:list[str]):
        """Add variables to be excluded from the output grouping.

        :param vars_to_exclude: list with variable names to be omitted from grouping.
        :type vars_to_exclude: list[str]
        :raises Exception: if any of the specified variables is not present in flamelet data set.
        """
        self.__vars_to_exclude = [c for c in self.__controlling_variables]
        self.__vars_to_exclude.append("FlameletID")
        for var in vars_to_exclude:
            if var not in self.__flamelet_variables:
                raise Exception("Variable "+var+" not present in flamelet data set.")
            self.__vars_to_exclude.append(var)
        return 
    
    def SetAffinityThreshold(self, val_threshold:float=0.7):
        """Specify the threshold value for affinity below which groups are not considered.

        :param val_threshold: affinity threshold value. Should be between zero and one.
        :type val_threshold: float
        :raises Exception: if threshold value is not within range.
        """
        if val_threshold <= 0 or val_threshold >= 1:
            raise Exception("Threshold value should be between zero and one.")
        self.__theta_threshold = val_threshold 

    def SetGroupLeaders(self, group_leaders_in:list[str]):
        """Specify a set of variables which are forced into separate groups.

        :param group_leaders_in: list of group leading variables.
        :type group_leaders_in: list[str]
        :raises Exception: if any of the variables is not present in the flamelet data set.
        """
        for g in group_leaders_in:
            if g not in self.__flamelet_variables:
                raise Exception("Variable " + g + " not present in flamelet data set.")
        self.__group_leaders_orig = []
        for g in group_leaders_in:
            self.__group_leaders_orig.append(g) 

    def __FilterVariables(self, vars_to_remove:list[str]):
        with open(self.__flamelet_data_filepath, 'r') as fid:
            flamelet_variables = fid.readline().strip().split(',')
            self.__flamelet_variables = flamelet_variables
        self.__free_variables = []
        for var in self.__flamelet_variables:
            self.__free_variables.append(var)

        self.__iVar_remove = []
        for var in vars_to_remove:
            if var not in flamelet_variables:
                raise Exception("Variable " + var + " not present in flamelet data.")
            self.__iVar_remove.append(flamelet_variables.index(var))
            self.__free_variables.remove(var)

        self.__LoadFlameletData()
        self.__GenerateCorrelationMatrix()

        self.__correlation_matrix = np.delete(self.__correlation_matrix, self.__iVar_remove,0)
        self.__correlation_matrix = np.delete(self.__correlation_matrix, self.__iVar_remove,1)
        
    def __LoadFlameletData(self):
        self.__flamelet_data = np.loadtxt(self.__flamelet_data_filepath, delimiter=',',skiprows=1)

    def __GenerateCorrelationMatrix(self):
        self.__correlation_matrix = np.corrcoef(self.__flamelet_data.T)

    def __UpdateGroupLeaders(self, group_leaders_in:list[str]):
        group_variables = []
        group_indices = []
        group_affinity = []
        free_var_indices = [i for i in range(len(self.__correlation_matrix))]
        free_vars = [var for var in self.__free_variables]
        for g in group_leaders_in:
            group_indices.append([self.__free_variables.index(g)])
            group_affinity.append([1])
            group_variables.append([g])
            free_var_indices.remove(self.__free_variables.index(g))
            free_vars.remove(g)
        return group_variables, group_indices, group_affinity, free_var_indices, free_vars 

    def __AffinityFunction(self, group_indices:list[int], iVar:int):
        theta = 1
        for k in group_indices:
                theta *= np.abs(self.__correlation_matrix[iVar,iVar])*np.abs(self.__correlation_matrix[iVar, k])
        return theta
    
    def EvaluateGroups(self):
        """Perform affinity evaluation and generate combinations of groups with a minimum affinity beyond the threshold value.
        """
        self.__FilterVariables(self.__vars_to_exclude)

        self.__group_affinity = []
        self.__group_variables = []
        self.__n_groups = []

        # Specify initial groups according to group leaders.
        group_variables, group_indices, group_affinity, _, free_vars_orig = self.__UpdateGroupLeaders(self.__group_leaders_orig)

        # Repeat 1000 times to come up with plenty of potential groups.
        for _ in tqdm(range(1000)):
            repeat = True 

            while repeat:
                # Randomly select species from list of remaining species to act as additional group leaders.
                new_group_vars = sample(free_vars_orig, np.random.randint(1, len(free_vars_orig)))

                group_leaders = [g for g in self.__group_leaders_orig] + [g for g in new_group_vars]
                
                # Randomly select a species and add to an appropriate group by computing maximum affinity with that group.
                group_variables, group_indices, group_affinity, _, free_vars = self.__UpdateGroupLeaders(group_leaders)

                n_free_vars = len(free_vars)
                while n_free_vars > 0:
                    var_sample = sample(free_vars, 1)[0]
                    iVar = self.__free_variables.index(var_sample)
                    affinity_groups = []
                    for iGroup in range(len(group_leaders)):
                        theta = self.__AffinityFunction(group_indices=group_indices[iGroup], iVar=iVar)
                        affinity_groups.append(theta)
                    best_group_index = np.argmax(affinity_groups)
                    group_variables[best_group_index].append(var_sample)
                    group_indices[best_group_index].append(iVar)
                    group_affinity[best_group_index].append(max(affinity_groups))
                    free_vars.remove(var_sample)
                    n_free_vars -= 1
                repeat = False
                for g in group_affinity:
                    if min(g) < self.__theta_threshold:
                        repeat = True 
            min_affinity = 1
            for g in group_affinity:
                min_affinity = min(min_affinity, min(g))
            n_groups = len(group_leaders)
            self.__n_groups.append(n_groups)
            self.__group_variables.append(group_variables)
            self.__group_affinity.append(min_affinity)

        self.PostProcessGroups()
        return 
    
    def __ComputeNumberofEvaluations(self, group_variables:list[list[str]]):
        n_networks_eval = 0
   
        for g in group_variables:
            this_group_TD = False
            this_group_PD = False
            this_group_sources = False 
            for var in self.__evaluations_TD:
                if var in g:
                    this_group_TD = True 
            if self.__Config.PreferentialDiffusion():
                for var in self.__evaluations_PD:
                    if var in g:
                        this_group_PD = True 
            for var in self.__evaluations_Sources:
                if var in g:
                    this_group_sources = True 

            if this_group_TD:
                n_networks_eval += 1
            if this_group_PD:
                n_networks_eval += 1
            if this_group_sources:
                n_networks_eval += 1 
        return n_networks_eval
    
    def PostProcessGroups(self):
        """Extract the combinations of variables with the highest affinity and fewest number of network evaluations. 
        Groups with most potential are visualized in a figure.
        """
        min_group = min(self.__n_groups)
        max_group = max(self.__n_groups)

        unique_groups = range(min_group, max_group + 1)

        n_network_evals = []
        interesting_groups = []
        self.__most_interesting_groups = []
        for j,i in enumerate(unique_groups):
            same_number_of_groups = np.argwhere(np.array(self.__n_groups) == i)[:,0]
            affinities_combinations = np.array(self.__group_affinity)[same_number_of_groups]
            iMax_affinity = np.argmax(affinities_combinations)
            interesting_group = self.__group_variables[same_number_of_groups[iMax_affinity]]
            n_network_evals.append(self.__ComputeNumberofEvaluations(interesting_group))
            interesting_groups.append(interesting_group)
            self.__most_interesting_groups.append(interesting_group)

        group_fewest_evaluations = np.argmin(np.array(n_network_evals))
        print("Output combinations with fewest number of network evaluations:")
        for iGroup, g in enumerate(interesting_groups[group_fewest_evaluations]):
            print("Output group " + str(iGroup)+ ": [" + ",".join("\"" + s + "\"" for s in g) + "]")
        self.__best_group = group_fewest_evaluations 

    def GetInterestingGroup(self, iGroup:int=-1):
        """Get the group or groups with the highest efficiency.

        :param iGroup: combination index for which to display the output groups. If none provided, all combinations are returned.
        :type iGroup: int, optional
        :raises Exception: if specified index exceeds number of combinations.
        :return: list of output groups or list of combinations.
        :rtype: list[str]
        """
        if iGroup == -1:
            return self.__most_interesting_groups[self.__best_group]
        else:
            if iGroup >= len(self.__most_interesting_groups):
                raise Exception("Index exceeds number of best combinations")
            return self.__most_interesting_groups[iGroup]
    
    def PlotCorrelationMatrix(self, combination_index:int=-1):
        """Plots cross-correlation matrix between filtered flamelet data.

        :param combination_index: variable combination for which to plot output groups, defaults to 0
        :type combination_index: int, optional
        :raises Exception: if index exceeds number of combinations.
        """
        if combination_index == -1:
            combination_index = self.__best_group
        if combination_index >= len(self.__most_interesting_groups):
            raise Exception("Index exceeds number of best combinations")
        fig = plt.figure(figsize=[10,10])
        ax = plt.axes()
        ax.matshow(np.abs(self.__correlation_matrix))
        for i in range(len(self.__free_variables)):
            for j in range(len(self.__free_variables)):
                ax.text(i, j, "%.2f" % (np.abs(self.__correlation_matrix[i,j])),\
                        fontsize=12,\
                        horizontalalignment='center',\
                        verticalalignment='center')

        for iGroup, g in enumerate(self.__most_interesting_groups[combination_index]):
            color = colors[iGroup]
            for iVar, v in enumerate(g):
                if iVar == 0:
                    ax.plot(self.__free_variables.index(v), self.__free_variables.index(v), 'o',markerfacecolor='none',color=color,markersize=24, markeredgewidth=3,label="Group "+str(iGroup+1))
                else:
                    ax.plot(self.__free_variables.index(v), self.__free_variables.index(g[0]), 'o',markerfacecolor='none',color=color,markersize=24, markeredgewidth=3)
        
        ax.set_xticks(range(len(self.__free_variables)))
        ax.set_yticks(range(len(self.__free_variables)))
        ax.set_xticklabels(self.__free_variables)
        ax.set_yticklabels(self.__free_variables)
        ax.tick_params(axis='x',labelrotation=90)
        ax.tick_params(which='both',labelsize=18)
        ax.legend(fontsize=20, bbox_to_anchor=(1.0, 0.5))
        fig.savefig(self.__Config.GetOutputDir()+"/Group_correlation_matrix.pdf",format='pdf',bbox_inches='tight')
        plt.tight_layout()
        plt.show()

        return 
    
    def UpdateConfig(self, combination_index:int=-1):
        """Update the output groups in the FlameletAI configuration

        :param combination_index: group combination index to store in config, defaults to -1
        :type combination_index: int, optional
        """

        # By default, select the combination with the fewest number of function 
        # evaluations.
        if combination_index == -1:
            combination_index = self.__best_group

        # Clear output groups present in the configuration.
        self.__Config.ClearOutputGroups()

        # Add flamelet variable groups to configuration.
        for group in self.__most_interesting_groups[combination_index]:
            self.__Config.AddOutputGroup(group)

        # Save configuration.
        self.__Config.SaveConfig()
        return