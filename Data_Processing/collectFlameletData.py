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

from Common.DataDrivenConfig import FlameletAIConfig

class FlameletConcatenator:
    """Read, regularize, and concatenate flamelet data for MLP training or LUT generation.

    """
    __Config:FlameletAIConfig = None # FlameletAI configuration for current workflow.

    __Np_per_flamelet:int = 60          # Number of data points to extract per flamelet.
    __custom_resolution:bool = False    # Overwrite average number of data points per flamelet with a specified value.

    __mfrac_skip:int = 1        # Number of mixture status folder to skip while concatenating flamelet data.

    __mix_status_max:float = 20     # Minimum mixture status value above which to collect flamelet data.
    __mix_status_min:float = 0.1    # Maximum mixture status value below which to collect flamelet data.

    __f_train:float = 0.8   # Fraction of the flamelet data used for training.
    __f_test:float = 0.1    # Fraction of the flamelet data used for testing.

    __output_file_header:str = "flameletdata"   # File header for the concatenated flamelet data file.
    __flameletdata_dir:str = "./"               # Directory from which to read flamelet data.

    # Thermodynamic data to search for in flamelet data.
    __TD_train_vars = ['Temperature', 'MolarWeightMix', 'DiffusionCoefficient', 'Conductivity', 'ViscosityDyn', 'Cp']
    __TD_flamelet_data:np.ndarray = None 

    # Differential diffusion data to search for in flamelet data.
    __PD_train_vars = ['Beta_ProgVar', 'Beta_Enth_Thermal', 'Beta_Enth', 'Beta_MixFrac']
    __PD_flamelet_data:np.ndarray = None 

    __flamelet_ID_vars = ['FlameletID']
    __flamelet_ID:np.ndarray = None 

    # Passive species names for which to save production and consumption terms.
    __Species_in_FGM = ['H2', 'H2O']

    # Passive look-up terms to include in the manifold.
    __LookUp_vars = ['Heat_Release']
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

    def __init__(self, Config:FlameletAIConfig):
        """Class constructor

        :param Config: loaded FlameletAIConfig class for the current workflow.
        :type Config: FlameletAIConfig
        """
        print("Loading flameletAI configuration " + Config.GetConfigName())
        self.__Config = Config

        # Load settins from configuration:
        self.__include_freeflames = self.__Config.GenerateFreeFlames()
        self.__include_burnerflames = self.__Config.GenerateBurnerFlames()
        self.__include_equilibrium = self.__Config.GenerateEquilibrium()
        self.__include_counterflame = False#self.__Config.GenerateCounterFlames()

        self.__Np_per_flamelet = self.__Config.GetNpConcatenation()
        [self.__mix_status_min, self.__mix_status_max] = self.__Config.GetMixtureBounds()
        self.__f_train = self.__Config.GetTrainFraction()
        self.__f_test = self.__Config.GetTestFraction()
        
        self.SetAuxilarySpecies(self.__Config.GetPassiveSpecies())
        self.SetLookUpVars(self.__Config.GetLookUpVariables())

        self.__flameletdata_dir = self.__Config.GetOutputDir()
        self.__output_file_header = self.__Config.GetConcatenationFileHeader()

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

    def SetMixStep(self, skip_mixtures:int):
        """Skip a number of mixture status values when reading flamelet data to reduce the concatenated file size.

        :param skip_mixtures: step size in mixture status
        :type skip_mixtures: int
        :raises Exception: if the provided step size is lower than one.
        """
        if skip_mixtures < 1:
            raise Exception("Mixture step size should be higher than one.")
        self.__mfrac_skip = skip_mixtures

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

    def SetAuxilarySpecies(self, species_input:list[str]):
        """Define the passive species names for which to collect source terms.

        :param input: list of species names.
        :type input: list[str]
        """
        self.__Species_in_FGM = []
        for s in species_input:
            self.__Species_in_FGM.append(s)
        self.__Sources_vars = [self.__PPV_train_vars[0]]
        for s in species_input:
            self.__Sources_vars.append("Y_dot_pos-"+s)
            self.__Sources_vars.append("Y_dot_neg-"+s)
            self.__Sources_vars.append("Y_dot_net-"+s)

    def IncludeFreeFlames(self, input:bool):
        """Read adiabatic flamelet data.

        :param input: read adiabatic flamelets (True) or not (False)
        :type input: bool
        """
        self.__include_freeflames = input 
    def IncludeBurnerFlames(self, input:bool):
        """Read burner-stabilized flamelet data.

        :param input: read burner-stabilized flamelet data (True) or not (False)
        :type input: bool
        """
        self.__include_burnerflames = input 
    def IncludeEquilibrium(self, input:bool):
        """Read chemical equilibrium data.

        :param input: read chemical equilibrium data (True) or not (False)
        :type input: bool
        """
        self.__include_equilibrium = input  
    def Include_CounterFlames(self, input:bool):
        self.__include_counterflame = input 

    def SetLookUpVars(self, input:list[str]):
        """Define passive look-up variables to be included in the manifold data.

        :param input: list of passive look-up variables.
        :type input: list[str]
        """
        self.__LookUp_vars = []
        for s in input:
            self.__LookUp_vars.append(s)
        
    def SetFlameletDir(self, input:str):
        """Manually define the directory where the flamelet data is stored.

        :param input: path to flamelet data directory.
        :type input: str
        :raises Exception: if the provided directory doesn't exist.
        """
        if not path.isdir(input):
            raise Exception("Flamelet data directory does not exist.")
        self.__flameletdata_dir = input

    def SetOutputFileName(self, input:str):
        """Define the manifold output file header.

        :param input: manifold file header.
        :type input: str
        """
        self.__output_file_header = input 

    def SetTrainFraction(self, input:float):
        """Define the fraction of concatenated flamelet data to be used for training MLP's.

        :param input: train data fraction. Should be between zero and one.
        :type input: float
        :raises Exception: if the provided fraction is lower than zero or higher than one.
        """
        if (input <= 0) or (input >= 1):
            raise Exception("Train fraction should be between zero and one.")
        self.__f_train = input 

    def SetTestFraction(self, input:float):
        """Define the fraction of concatenated flamelet data to be used for accuracy testing after training MLP's.

        :param input: test data fraction. Should be between zero and one.
        :type input: float
        :raises Exception: if the provided fraction is lower than zero or higher than one.
        """
        if (input <= 0) or (input >= 1):
            raise Exception("Test fraction should be between zero and one.")
        self.__f_test = input 
        

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

        print("Concatenating flamelet data...")

        # Read adiabatic flamelet data
        if self.__include_freeflames:
            print("Reading freeflames...")
            i_freeflame_total = 0
            mixture_folders = np.sort(np.array(self.mfracs_freeflames))
            for z in tqdm(mixture_folders[::self.__mfrac_skip]):
                if z[:len(folder_header)] == folder_header:
                    mixture_status = float(z[len(folder_header):])
                    if (mixture_status <= self.__mix_status_max) and (mixture_status >= self.__mix_status_min):
                        i_freeflame_total = self.__InterpolateFlameletData(self.__flameletdata_dir + "/freeflame_data/", z, 0, i_freeflame_total)
            print("Done!")
            i_start += i_freeflame_total

        # Read burner-stabilized flamelet data
        if self.__include_burnerflames:
            print("Reading burnerflamelets...")
            i_burnerflame_total = 0
            mixture_folders = np.sort(np.array(self.mfracs_burnerflames))
            for z in tqdm(mixture_folders[::self.__mfrac_skip]):
                if z[:len(folder_header)] == folder_header:
                    mixture_status = float(z[len(folder_header):])
                    if (mixture_status <= self.__mix_status_max) and (mixture_status >= self.__mix_status_min):
                        i_burnerflame_total = self.__InterpolateFlameletData(self.__flameletdata_dir + "/burnerflame_data/", z, i_start, i_burnerflame_total)
            print("Done!")
            i_start +=  i_burnerflame_total

        # Read chemical equilibrium data
        if self.__include_equilibrium:
            i_equilibrium_total = 0
            print("Reading equilibrium data...")
            mixture_folders = np.sort(np.array(self.mfracs_equilibrium))
            for z in tqdm(mixture_folders[::self.__mfrac_skip]):
                if z[:len(folder_header)] == folder_header:
                    mixture_status = float(z[len(folder_header):])
                    if (mixture_status <= self.__mix_status_max) and (mixture_status >= self.__mix_status_min):
                        i_equilibrium_total = self.__InterpolateFlameletData(self.__flameletdata_dir + "/equilibrium_data/", z, i_start, i_equilibrium_total)
            print("Done!")
            i_start +=  i_equilibrium_total

        # Read fuzzy data
        if self.__include_fuzzy:
            i_fuzzy_total = 0
            print("Reading fuzzy data...")
            mixture_folders = np.sort(np.array(self.mfracs_fuzzy))
            for z in tqdm(mixture_folders[::self.__mfrac_skip]):
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
        print("Writing output data...")
        self.__WriteOutputFiles()
        print("Done!")

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

        _, idx_unique = np.unique(self.__CV_flamelet_data, axis=0, return_index=True)
        total_data = total_data[idx_unique, :]

        # Shuffle flamelet data to remove bias.
        np.random.shuffle(total_data)

        # Number of data points for training and testing.
        np_train = int(self.__f_train*len(total_data))
        np_val = int(self.__f_test*len(total_data))

        train_data = total_data[:np_train, :]
        val_data = total_data[np_train:np_train+np_val, :]
        test_data = total_data[np_train+np_val:, :]

        # Write full, train, validation, and test data files.
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
        print("Done!")
        
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
            print("Counting adabatic free-flame data...")
            self.mfracs_freeflames = listdir(self.__flameletdata_dir + "/freeflame_data")
            mixture_folders = np.sort(np.array(self.mfracs_freeflames))
            for z in tqdm(mixture_folders[::self.__mfrac_skip]):
                if z[:len(folder_header)] == folder_header:
                    mixture_status = float(z[len(folder_header):])
                    if (mixture_status <= self.__mix_status_max) and (mixture_status >= self.__mix_status_min):
                        n_freeflames += len(listdir(self.__flameletdata_dir + "/freeflame_data/" + z))
                        for f in listdir(self.__flameletdata_dir + "/freeflame_data/" + z):
                            with open(self.__flameletdata_dir + "/freeflame_data/" + z + "/" + f, "r") as fid:
                                Np_tot += len(fid.readlines())-1

        # Count the number of burner-stabilized flamelets.
        if self.__include_burnerflames:
            print("Counting burner-stabilized flame data...")
            self.mfracs_burnerflames = listdir(self.__flameletdata_dir + "/burnerflame_data")
            mixture_folders = np.sort(np.array(self.mfracs_burnerflames))
            for z in tqdm(mixture_folders[::self.__mfrac_skip]):
                if z[:len(folder_header)] == folder_header:
                    mixture_status = float(z[len(folder_header):])
                    if (mixture_status <= self.__mix_status_max) and (mixture_status >= self.__mix_status_min):
                        n_burnerflames += len(listdir(self.__flameletdata_dir + "/burnerflame_data/" + z))
                        for f in listdir(self.__flameletdata_dir + "/burnerflame_data/" + z):
                            with open(self.__flameletdata_dir + "/burnerflame_data/" + z + "/" + f, "r") as fid:
                                Np_tot += len(fid.readlines())-1
        
        # Count the number of chemical equilibrium data files.
        if self.__include_equilibrium:
            print("Counting chemical equilibrium data...")
            self.mfracs_equilibrium = listdir(self.__flameletdata_dir + "/equilibrium_data")
            mixture_folders = np.sort(np.array(self.mfracs_equilibrium))
            for z in tqdm(mixture_folders[::self.__mfrac_skip]):
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
            for z in tqdm(mixture_folders[::self.__mfrac_skip]):
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
            for f in tqdm(counterflame_files):
                with open(self.__flameletdata_dir + "/counterflame_data/" + f, 'r') as fid:
                    Np_tot += len(fid.readlines())-1

        print("Done!")

        # Compute the average number of data points per flamelet.
        n_flamelets_total = n_freeflames + n_burnerflames + n_eq + n_counterflames + n_fuz
        if not self.__custom_resolution:
            self.__Np_per_flamelet = int(Np_tot / n_flamelets_total)
        print("Number of data-points per flamelet: %i " % self.__Np_per_flamelet)

        # Size output data arrays according to manifold resolution.
        self.__CV_flamelet_data = np.zeros([n_flamelets_total * self.__Np_per_flamelet, 3])
        self.__TD_flamelet_data = np.zeros([n_flamelets_total * self.__Np_per_flamelet, len(self.__TD_train_vars)])
        if self.__Config.PreferentialDiffusion():
            self.__PD_flamelet_data = np.zeros([n_flamelets_total * self.__Np_per_flamelet, len(self.__PD_train_vars)])
        self.__Sources_flamelet_data = np.zeros([n_flamelets_total * self.__Np_per_flamelet, 1 + 3 * len(self.__Species_in_FGM)])
        self.__LookUp_flamelet_data = np.zeros([n_flamelets_total * self.__Np_per_flamelet, len(self.__LookUp_vars)])
        self.__flamelet_ID = np.zeros([n_flamelets_total * self.__Np_per_flamelet, len(self.__flamelet_ID_vars)],dtype=int)

    
    def __InterpolateFlameletData(self, flamelet_dir:str, eq_file:str, i_start:int, i_flamelet_total:int, is_fuzzy:bool=False):

        flamelets = listdir(flamelet_dir + "/" + eq_file)
        for i_flamelet, f in enumerate(flamelets):

            # Get flamelet variables
            fid = open(flamelet_dir + "/" + eq_file + "/" + f, 'r')
            variables = fid.readline().strip().split(',')
            fid.close()

            # Load flamelet data
            D = np.loadtxt(flamelet_dir + "/" + eq_file + "/" + f, delimiter=',',skiprows=1)
            
            # Compute the progress variable and reaction rates

            pv_flamelet = self.__Config.ComputeProgressVariable(variables, D)
            ppv_flamelet = self.__Config.ComputeProgressVariable_Source(variables, D)

            # Load the flamelet enthalpy and mixture fraction
            try:
                enth_flamelet = D[:, variables.index('Total_Enthalpy')]
                mfrac_flamelet = D[:, variables.index('Mixture_Fraction')]
            except:
                enth_flamelet = D[:, variables.index('EnthalpyTot')]
                mfrac_flamelet = D[:, variables.index('MixtureFraction')]

            # Set flamelet controlling variables
            CV_flamelet = np.zeros([len(D), 3])
            CV_flamelet[:, 0] = pv_flamelet
            CV_flamelet[:, 1] = enth_flamelet
            CV_flamelet[:, 2] = mfrac_flamelet
            CV_min, CV_max = np.min(CV_flamelet, axis=0), np.max(CV_flamelet, axis=0)
            CV_norm = (CV_flamelet - CV_min)/(CV_max - CV_min + 1e-10)

            delta_CV = CV_norm[1:] - CV_norm[:-1]
            dS = np.sqrt(np.sum(np.power(delta_CV, 2), axis=1))
            S_flamelet = np.append(np.array([0]), np.cumsum(dS))
            is_valid_flamelet = True 

            if max(S_flamelet) == 0:
                print("Dodgy flamelet data file: " + flamelet_dir + "/" + eq_file + "/" + f)
                is_valid_flamelet = False

            if is_valid_flamelet:
                S_flamelet_norm = S_flamelet / (max(S_flamelet))

                # Load flamelet thermophysical property data
                TD_data = np.zeros([len(D), len(self.__TD_train_vars)])
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
                for iVar_LookUp, LookUp_var in enumerate(self.__LookUp_vars):
                    idx_var_flamelet = variables.index(LookUp_var)
                    LookUp_data[:, iVar_LookUp] = D[:, idx_var_flamelet]
        
                
                # Load species sources data
                species_production_rate = np.zeros([len(D), len(self.__Species_in_FGM)])
                species_destruction_rate = np.zeros([len(D), len(self.__Species_in_FGM)])
                species_net_rate = np.zeros([len(D), len(self.__Species_in_FGM)])
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
                Sources_data[:, 0] = ppv_flamelet

                for iSp in range(len(self.__Species_in_FGM)):
                    Sources_data[:, 1 + 3*iSp] = species_production_rate[:, iSp]
                    Sources_data[:, 1 + 3*iSp + 1] = species_destruction_rate[:, iSp]
                    Sources_data[:, 1 + 3*iSp + 2] = species_net_rate[:, iSp]

                # Compute preferential diffusion scalars
                if self.__Config.PreferentialDiffusion():
                    beta_pv_flamelet, beta_h1_flamelet, beta_h2_flamelet, beta_z_flamelet = self.__Config.ComputeBetaTerms(variables, D)
                    PD_data = np.zeros([len(D), len(self.__PD_train_vars)])
                    PD_data[:, 0] = beta_pv_flamelet
                    PD_data[:, 1] = beta_h1_flamelet
                    PD_data[:, 2] = beta_h2_flamelet
                    PD_data[:, 3] = beta_z_flamelet

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
                    S_q = 0.5 - 0.5*np.cos(np.linspace(0, np.pi, self.__Np_per_flamelet))
                    CV_sampled = np.zeros([self.__Np_per_flamelet, np.shape(CV_flamelet)[1]])
                    TD_sampled = np.zeros([self.__Np_per_flamelet, np.shape(TD_data)[1]])
                    if self.__Config.PreferentialDiffusion():
                        PD_sampled = np.zeros([self.__Np_per_flamelet, np.shape(PD_data)[1]])
                    lookup_sampled = np.zeros([self.__Np_per_flamelet, np.shape(LookUp_data)[1]])
                    sources_sampled = np.zeros([self.__Np_per_flamelet, 1 + 3*len(self.__Species_in_FGM)])
                    for i_CV in range(3):
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
                    # for i_CV in range(3):
                    #     self.__CV_flamelet_data[(i_start + i_flamelet + i_flamelet_total) * self.__Np_per_flamelet: (i_start + i_flamelet+1+ i_flamelet_total)*self.__Np_per_flamelet, i_CV] = np.interp(S_q, S_flamelet_norm, CV_flamelet[:, i_CV])
                    # for iVar_TD in range(len(self.__TD_train_vars)):
                    #     self.__TD_flamelet_data[(i_start + i_flamelet+ i_flamelet_total) * self.__Np_per_flamelet: (i_start + i_flamelet+1+ i_flamelet_total)*self.__Np_per_flamelet, iVar_TD] = np.interp(S_q, S_flamelet_norm, TD_data[:, iVar_TD])
                    # if self.__Config.PreferentialDiffusion():
                    #     for iVar_PD in range(len(self.__PD_train_vars)):
                    #         self.__PD_flamelet_data[(i_start + i_flamelet+ i_flamelet_total) * self.__Np_per_flamelet: (i_start + i_flamelet+1+ i_flamelet_total)*self.__Np_per_flamelet, iVar_PD] = np.interp(S_q, S_flamelet_norm, PD_data[:, iVar_PD])
                    # for iVar_LU in range(len(self.__LookUp_vars)):
                    #     self.__LookUp_flamelet_data[(i_start + i_flamelet+ i_flamelet_total) * self.__Np_per_flamelet: (i_start + i_flamelet+1+ i_flamelet_total)*self.__Np_per_flamelet, iVar_LU] = np.interp(S_q, S_flamelet_norm, LookUp_data[:, iVar_LU])
                    # for iVar_Source in range(1 + 3*len(self.__Species_in_FGM)):
                    #     self.__Sources_flamelet_data[(i_start + i_flamelet+ i_flamelet_total) * self.__Np_per_flamelet: (i_start + i_flamelet+1+ i_flamelet_total)*self.__Np_per_flamelet, iVar_Source] = np.interp(S_q, S_flamelet_norm, Sources_data[:, iVar_Source])
        return len(flamelets) + i_flamelet_total

if __name__ == "__main__":
    config_input_file = sys.argv[-1]
    Config = FlameletAIConfig(config_input_file)
    F = FlameletConcatenator(Config)
    F.SetMixStatusBounds(0, 600)
    F.SetNFlameletNodes(21)
    #F.SetOutputFileName("LUT_data")
    F.ConcatenateFlameletData()
