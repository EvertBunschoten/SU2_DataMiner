import CoolProp.CoolProp as CP
import numpy as np 
from tqdm import tqdm
import csv 
import matplotlib.pyplot as plt 
from Common.EntropicAIConfig import EntropicAIConfig 

class DataGenerator_CoolProp:

    __Config:EntropicAIConfig = None 
    __fluid = None 

    __use_PT:bool = True 
    __T_min:float 
    __T_max:float 
    __Np_T:int 

    __P_min:float 
    __P_max:float 
    __Np_P:int 

    __T_grid:np.ndarray[float]
    __P_grid:np.ndarray[float]
    
    __rho_min:float 
    __rho_max:float 
    __e_min:float 
    __e_max:float 
    __e_grid:np.ndarray[float]
    __rho_grid:np.ndarray[float]

    __s_fluid:np.ndarray[float]
    __dsdrho_e_fluid:np.ndarray[float]
    __dsde_rho_fluid:np.ndarray[float]
    __d2sdrho2_fluid:np.ndarray[float]
    __d2sde2_fluid:np.ndarray[float]
    __d2sdedrho_fluid:np.ndarray[float]
    __T_fluid:np.ndarray[float]
    __P_fluid:np.ndarray[float]
    __e_fluid:np.ndarray[float]
    __rho_fluid:np.ndarray[float]
    __c2_fluid:np.ndarray[float]
    __success_locations:np.ndarray[bool]

    def __init__(self, Config_in:EntropicAIConfig):
        self.__Config = Config_in 
        self.__use_PT = self.__Config.GetPTGrid()
        if len(self.__Config.GetFluidNames()) > 1:
            fluid_names = self.__Config.GetFluidNames()
            CAS_1 = CP.get_fluid_param_string(fluid_names[0], "CAS")
            CAS_2 = CP.get_fluid_param_string(fluid_names[1], "CAS")
            CP.apply_simple_mixing_rule(CAS_1, CAS_2,'linear')
        self.__fluid = CP.AbstractState("HEOS", self.__Config.GetFluidName())
        if len(self.__Config.GetFluidNames()) > 1:
            mole_fractions = self.__Config.GetMoleFractions()
            self.__fluid.set_mole_fractions(mole_fractions)
        self.__use_PT = self.__Config.GetPTGrid()
        P_bounds = self.__Config.GetPressureBounds()
        T_bounds = self.__Config.GetTemperatureBounds()
        self.__P_min, self.__P_max = P_bounds[0], P_bounds[1]
        self.__Np_P = self.__Config.GetNpPressure()

        self.__T_min, self.__T_max = T_bounds[0], T_bounds[1]
        self.__Np_T = self.__Config.GetNpTemp()

    def __GenerateDataGrid(self):
        P_range = np.linspace(self.__P_min, self.__P_max, self.__Np_P)
        T_range = np.linspace(self.__T_min, self.__T_max, self.__Np_T)
        self.__P_grid, self.__T_grid = np.meshgrid(P_range, T_range)

        if not self.__use_PT:
            P_dataset = self.__P_grid.flatten()
            T_dataset = self.__T_grid.flatten()
            self.__rho_min = 1e32
            self.__rho_max = -1e32
            self.__e_min = 1e32
            self.__e_max = -1e32 
            for p,T in zip(P_dataset, T_dataset):
                try:
                    self.__fluid.update(CP.PT_INPUTS, p, T)
                    rho = self.__fluid.rhomass()
                    e = self.__fluid.umass()
                    self.__rho_max = max(rho, self.__rho_max)
                    self.__rho_min = min(rho, self.__rho_min)
                    self.__e_max = max(e, self.__e_max)
                    self.__e_min = min(e, self.__e_min)
                except:
                    pass 
            rho_range = (self.__rho_min - self.__rho_max)* (np.cos(np.linspace(0, 0.5*np.pi, self.__Np_P))) + self.__rho_max
            e_range = np.linspace(self.__e_min, self.__e_max, self.__Np_T)
            self.__rho_grid, self.__e_grid = np.meshgrid(rho_range, e_range)

    def PreprocessData(self):
        self.__GenerateDataGrid()
    
    def VisualizeDataGrid(self):
        if self.__use_PT:
            fig = plt.figure(figsize=[10,10])
            ax = plt.axes()
            ax.plot(self.__P_grid.flatten(), self.__T_grid.flatten(), 'k.')
            ax.set_xlabel(r"Pressure $(p)[Pa]",fontsize=20)
            ax.set_ylabel(r"Temperature $(T)[K]",fontsize=20)
            ax.tick_params(which='both',labelsize=18)
            ax.grid()
        else:
            fig = plt.figure(figsize=[10,10])
            ax = plt.axes()
            ax.plot(self.__rho_grid.flatten(), self.__e_grid.flatten(), 'k.')
            ax.set_xlabel(r"Density $(\rho)[kg m^{-3}]",fontsize=20)
            ax.set_ylabel(r"Internal energy $(e)[J kg^{-1}]",fontsize=20)
            ax.tick_params(which='both',labelsize=18)
            ax.grid()
        plt.show()

    def ComputeFluidData(self):
        

        self.__s_fluid = np.zeros([self.__Np_P, self.__Np_T])
        self.__dsde_rho_fluid = np.zeros([self.__Np_P, self.__Np_T])
        self.__dsdrho_e_fluid = np.zeros([self.__Np_P, self.__Np_T])
        self.__d2sde2_fluid = np.zeros([self.__Np_P, self.__Np_T])
        self.__d2sdrho2_fluid = np.zeros([self.__Np_P, self.__Np_T])
        self.__d2sdedrho_fluid = np.zeros([self.__Np_P, self.__Np_T])
        self.__P_fluid = np.zeros([self.__Np_P, self.__Np_T])
        self.__T_fluid = np.zeros([self.__Np_P, self.__Np_T])
        self.__c2_fluid = np.zeros([self.__Np_P, self.__Np_T])
        self.__rho_fluid = np.zeros([self.__Np_P, self.__Np_T])
        self.__e_fluid = np.zeros([self.__Np_P, self.__Np_T])

        self.__success_locations = np.ones([self.__Np_P, self.__Np_T],dtype=bool)
        
        for i in tqdm(range(self.__Np_P)):
            for j in range(self.__Np_T):
                try:
                    if self.__use_PT:
                        self.__fluid.update(CP.PT_INPUTS, self.__P_grid[i,j], self.__T_grid[i,j])
                    else:
                        self.__fluid.update(CP.DmassUmass_INPUTS, self.__rho_grid[j,i], self.__e_grid[j,i])

                    if (self.__fluid.phase() != 0) and (self.__fluid.phase() != 3) and (self.__fluid.phase() != 6):
                        self.__s_fluid[i,j] = self.__fluid.smass()
                        self.__dsde_rho_fluid[i,j] = self.__fluid.first_partial_deriv(CP.iSmass, CP.iUmass, CP.iDmass)
                        self.__dsdrho_e_fluid[i,j] = self.__fluid.first_partial_deriv(CP.iSmass, CP.iDmass, CP.iUmass)
                        self.__d2sde2_fluid[i,j] = self.__fluid.second_partial_deriv(CP.iSmass, CP.iUmass, CP.iDmass, CP.iUmass, CP.iDmass)
                        self.__d2sdedrho_fluid[i,j] = self.__fluid.second_partial_deriv(CP.iSmass, CP.iUmass, CP.iDmass, CP.iDmass, CP.iUmass)
                        self.__d2sdrho2_fluid[i,j] = self.__fluid.second_partial_deriv(CP.iSmass, CP.iDmass, CP.iUmass, CP.iDmass, CP.iUmass)
                        self.__P_fluid[i,j] = self.__fluid.p()
                        self.__T_fluid[i,j] = self.__fluid.T()
                        self.__c2_fluid[i,j] = self.__fluid.speed_sound()**2
                        self.__rho_fluid[i,j] = self.__fluid.rhomass()
                        self.__e_fluid[i,j] = self.__fluid.umass()
                    else:
                        self.__success_locations[i,j] = False
                        self.__s_fluid[i, j] = None
                        self.__dsde_rho_fluid[i,j] = None 
                        self.__dsdrho_e_fluid[i,j] = None 
                        self.__d2sde2_fluid[i,j] = None 
                        self.__d2sdrho2_fluid[i,j] = None 
                        self.__d2sdedrho_fluid[i,j] = None 
                        self.__c2_fluid[i,j] = None 
                        self.__P_fluid[i,j] = None 
                        self.__T_fluid[i,j] = None 
                        self.__rho_fluid[i,j] = None 
                        self.__e_fluid[i,j] = None 
                except:
                    self.__success_locations[i,j] = False 
                    self.__s_fluid[i, j] = None
                    self.__dsde_rho_fluid[i,j] = None 
                    self.__dsdrho_e_fluid[i,j] = None 
                    self.__d2sde2_fluid[i,j] = None 
                    self.__d2sdrho2_fluid[i,j] = None 
                    self.__d2sdedrho_fluid[i,j] = None 
                    self.__c2_fluid[i,j] = None 
                    self.__P_fluid[i,j] = None 
                    self.__T_fluid[i,j] = None 
                    self.__rho_fluid[i,j] = None 
                    self.__e_fluid[i,j] = None 

    def VisualizeFluidData(self):

        fig = plt.figure(figsize=[10,10])
        ax = plt.axes(projection='3d')
        if self.__use_PT:
            ax.plot_surface(self.__P_fluid, self.__T_fluid, self.__s_fluid)
        else:
            ax.plot_surface(self.__rho_fluid, self.__e_fluid, self.__s_fluid)
        plt.show()

    def SaveFluidData(self):
        output_dir = self.__Config.GetOutputDir()

        full_file = output_dir + "/" + self.__Config.GetConcatenationFileHeader() + "_full.csv"
        train_file = output_dir + "/" + self.__Config.GetConcatenationFileHeader() + "_train.csv"
        test_file = output_dir + "/" + self.__Config.GetConcatenationFileHeader() + "_test.csv"
        val_file = output_dir + "/" + self.__Config.GetConcatenationFileHeader() + "_val.csv"

        controlling_vars = ["Density", "Energy"]
        entropic_vars = ["s","dsdrho_e","dsde_rho","d2sdrho2","d2sde2","d2sdedrho"]
        TD_vars = ["T","p","c2"]

        all_vars = controlling_vars + entropic_vars + TD_vars

        CV_data = np.vstack((self.__rho_fluid.flatten(), \
                             self.__e_fluid.flatten())).T 
        entropic_data = np.vstack((self.__s_fluid.flatten(),\
                                   self.__dsdrho_e_fluid.flatten(),\
                                   self.__dsde_rho_fluid.flatten(),\
                                   self.__d2sdrho2_fluid.flatten(),\
                                   self.__d2sde2_fluid.flatten(),\
                                   self.__d2sdedrho_fluid.flatten())).T 
        TD_data = np.vstack((self.__T_fluid.flatten(),\
                             self.__P_fluid.flatten(),\
                             self.__c2_fluid.flatten())).T
        
        full_data = np.hstack((CV_data, entropic_data, TD_data))
        full_data = full_data[self.__success_locations.flatten(), :]

        np.random.shuffle(full_data)

        Np_full = np.shape(full_data)[0]
        Np_train = int(self.__Config.GetTrainFraction()*Np_full)
        Np_test = int(self.__Config.GetTestFraction()*Np_full)

        train_data = full_data[:Np_train, :]
        test_data = full_data[Np_train:Np_train+Np_test, :]
        val_data = full_data[Np_train+Np_test:, :]

        with open(full_file,"w+") as fid:
            fid.write(",".join(v for v in all_vars) + "\n")
            csvWriter = csv.writer(fid)
            csvWriter.writerows(full_data)
        
        with open(train_file,"w+") as fid:
            fid.write(",".join(v for v in all_vars) + "\n")
            csvWriter = csv.writer(fid)
            csvWriter.writerows(train_data)

        with open(test_file,"w+") as fid:
            fid.write(",".join(v for v in all_vars) + "\n")
            csvWriter = csv.writer(fid)
            csvWriter.writerows(test_data)

        with open(val_file,"w+") as fid:
            fid.write(",".join(v for v in all_vars) + "\n")
            csvWriter = csv.writer(fid)
            csvWriter.writerows(val_data)
# fluidName = 'MM'
# P_min = 2e4
# P_max = 2.0e6
# T_min = 300.0
# T_max = 600.0

# density_min = 0.5
# density_max = +300.0

# #energy_min = 3.50604240e+05
# energy_min = 3e5
# energy_max = 5.5e+05

# Np_grid = 150

# f_train = 0.8
# f_test = 0.1

# use_PT = True

# P_range = np.linspace(P_min, P_max, Np_grid)
# T_range = np.linspace(T_min, T_max, Np_grid)
# P_grid, T_grid = np.meshgrid(P_range, T_range)

# P_dataset = P_grid.flatten()
# T_dataset = T_grid.flatten()

# fluid = CP.AbstractState("HEOS", fluidName)
# if use_PT:
#     density_min=1e5
#     density_max=0
#     energy_min=5.5e5
#     energy_max=0
#     for p,T in zip(P_dataset, T_dataset):
#         fluid.update(CP.PT_INPUTS, p, T)
#         rho = fluid.rhomass()
#         e = fluid.umass()
#         density_max = max(rho, density_max)
#         density_min = min(rho, density_min)
#         energy_max = max(e, energy_max)
#         energy_min = min(e, energy_min)

# with open("../PR_ORCHID_stator_data.csv",'r') as fid:
#     line = fid.readline().strip().split(",")
#     vars_PRData = [v.strip("\"") for v in line]

# ORCHID_stator_data = np.loadtxt("../PR_ORCHID_stator_data.csv",delimiter=',',skiprows=1)
# T_ORCHID = ORCHID_stator_data[:, vars_PRData.index("Temperature")]
# p_ORCHID = ORCHID_stator_data[:, vars_PRData.index("Pressure")]

# rho_range = np.linspace(density_min, density_max, Np_grid)
# rho_range = (density_max - density_min) * (1 - np.cos(np.linspace(0, 0.5*np.pi, Np_grid))) + density_min

# e_range = np.linspace(energy_min, energy_max, Np_grid)
# #e_range = (energy_max - energy_min) * (1 - np.cos(np.linspace(0, 0.5*np.pi, Np_grid))) + energy_min
# rho_grid, e_grid = np.meshgrid(rho_range, e_range)

# P_dataset = P_grid.flatten()
# T_dataset = T_grid.flatten()


# rho_dataset = rho_grid.flatten()
# e_dataset = e_grid.flatten()
# # if use_PT:
# #     rho_dataset = np.zeros(P_dataset.shape)
# #     e_dataset = np.zeros(P_dataset.shape)
# # else:
# #     rho_dataset = rho_grid.flatten()
# #     e_dataset = e_grid.flatten()
# s_dataset = np.zeros(P_dataset.shape)
# dsde_dataset = np.zeros(P_dataset.shape)
# dsdrho_dataset = np.zeros(P_dataset.shape)
# d2sde2_dataset = np.zeros(P_dataset.shape)
# d2sdedrho_dataset = np.zeros(P_dataset.shape)
# d2sdrho2_dataset = np.zeros(P_dataset.shape)
# C2_dataset = np.zeros(P_dataset.shape)

# s_ORCHID = np.zeros(T_ORCHID.shape)
# dsde_ORCHID = np.zeros(T_ORCHID.shape)
# dsdrho_ORCHID = np.zeros(T_ORCHID.shape)
# d2sdrho2_ORCHID = np.zeros(T_ORCHID.shape)
# d2sdedrho_ORCHID = np.zeros(T_ORCHID.shape)
# d2sde2_ORCHID = np.zeros(T_ORCHID.shape)
# rho_ORCHID = np.zeros(T_ORCHID.shape)
# e_ORCHID = np.zeros(T_ORCHID.shape) 
# C2_ORCHID = np.zeros(T_ORCHID.shape)

# fluid = CP.AbstractState("HEOS", fluidName)
# idx_failed_below = []
# idx_failed_above = []
# idx_successful = []
# for i in tqdm(range(len(P_dataset))):
#     try:
#         fluid.update(CP.DmassUmass_INPUTS, rho_dataset[i], e_dataset[i])
#         if (fluid.phase() != 0) and (fluid.phase() != 3) and (fluid.phase() != 6):
#             s_dataset[i] = fluid.smass()
#             dsde_dataset[i] = fluid.first_partial_deriv(CP.iSmass, CP.iUmass, CP.iDmass)
#             dsdrho_dataset[i] = fluid.first_partial_deriv(CP.iSmass, CP.iDmass, CP.iUmass)
#             d2sde2_dataset[i] = fluid.second_partial_deriv(CP.iSmass, CP.iUmass, CP.iDmass, CP.iUmass, CP.iDmass)
#             d2sdedrho_dataset[i] = fluid.second_partial_deriv(CP.iSmass, CP.iUmass, CP.iDmass, CP.iDmass, CP.iUmass)
#             d2sdrho2_dataset[i] = fluid.second_partial_deriv(CP.iSmass, CP.iDmass, CP.iUmass, CP.iDmass, CP.iUmass)
#             P_dataset[i] = fluid.p()
#             T_dataset[i] = fluid.T()
#             C2_dataset[i] = fluid.speed_sound()**2
#             idx_successful.append(i)
#         else:
#             idx_failed_above.append(i)
#     except:
#         idx_failed_above.append(i)
#         print("CP failed at pressure "+str(rho_dataset[i]) + ", temperature "+str(e_dataset[i]))

# for i in tqdm(range(len(p_ORCHID))):
#     fluid.update(CP.PT_INPUTS, p_ORCHID[i], T_ORCHID[i])
#     s_ORCHID[i] = fluid.smass()
#     dsde_ORCHID[i] = fluid.first_partial_deriv(CP.iSmass, CP.iUmass, CP.iDmass)
#     dsdrho_ORCHID[i] = fluid.first_partial_deriv(CP.iSmass, CP.iDmass, CP.iUmass)
#     d2sde2_ORCHID[i] = fluid.second_partial_deriv(CP.iSmass, CP.iUmass, CP.iDmass, CP.iUmass, CP.iDmass)
#     d2sdedrho_ORCHID[i] = fluid.second_partial_deriv(CP.iSmass, CP.iUmass, CP.iDmass, CP.iDmass, CP.iUmass)
#     d2sdrho2_ORCHID[i] = fluid.second_partial_deriv(CP.iSmass, CP.iDmass, CP.iUmass, CP.iDmass, CP.iUmass)
#     rho_ORCHID[i] = fluid.rhomass()
#     e_ORCHID[i] = fluid.umass()
#     C2_ORCHID[i] = fluid.speed_sound()**2

# ORCHID_data = np.vstack([rho_ORCHID, 
#                             e_ORCHID, 
#                             s_ORCHID, 
#                             dsde_ORCHID, 
#                             dsdrho_ORCHID, 
#                             d2sde2_ORCHID, 
#                             d2sdedrho_ORCHID, 
#                             d2sdrho2_ORCHID,
#                             p_ORCHID,
#                             T_ORCHID,
#                             C2_ORCHID]).T

# fig = plt.figure(figsize=[10,10])
# ax = plt.axes(projection='3d')
# ax.plot_surface(rho_dataset.reshape(P_grid.shape), e_dataset.reshape(P_grid.shape), s_dataset.reshape(P_grid.shape))
# plt.show()

# rho_dataset = rho_dataset[idx_successful]
# e_dataset = e_dataset[idx_successful]
# s_dataset = s_dataset[idx_successful]
# dsde_dataset = dsde_dataset[idx_successful]
# dsdrho_dataset = dsdrho_dataset[idx_successful]
# d2sde2_dataset = d2sde2_dataset[idx_successful]
# d2sdedrho_dataset = d2sdedrho_dataset[idx_successful]
# d2sdrho2_dataset = d2sdrho2_dataset[idx_successful]
# P_dataset = P_dataset[idx_successful]
# T_dataset= T_dataset[idx_successful]
# C2_dataset = C2_dataset[idx_successful]

# collected_data = np.vstack([rho_dataset, 
#                                   e_dataset, 
#                                   s_dataset, 
#                                   dsde_dataset, 
#                                   dsdrho_dataset, 
#                                   d2sde2_dataset, 
#                                   d2sdedrho_dataset, 
#                                   d2sdrho2_dataset,
#                                   P_dataset,
#                                   T_dataset,
#                                   C2_dataset]).T


# np.random.shuffle(collected_data)

# np_train = int(f_train*len(rho_dataset))
# np_val = int(f_test*len(rho_dataset))
# np_test = len(rho_dataset) - np_train - np_val



# train_data = collected_data[:np_train, :]
# dev_data = collected_data[np_train:(np_train+np_val), :]
# test_data = collected_data[(np_train+np_val):, :]
# with open("single_dataset_full.csv", "w+") as fid:
#     fid.write("Density,Energy,s,dsde_rho,dsdrho_e,d2sde2,d2sdedrho,d2sdrho2,p,T,c2\n")
#     csvWriter = csv.writer(fid,delimiter=',')
#     csvWriter.writerows(collected_data)

# with open("single_dataset_train.csv", "w+") as fid:
#     fid.write("Density,Energy,s,dsde_rho,dsdrho_e,d2sde2,d2sdedrho,d2sdrho2,p,T,c2\n")
#     csvWriter = csv.writer(fid,delimiter=',')
#     csvWriter.writerows(train_data)

# with open("single_dataset_dev.csv", "w+") as fid:
#     fid.write("Density,Energy,s,dsde_rho,dsdrho_e,d2sde2,d2sdedrho,d2sdrho2,p,T,c2\n")
#     csvWriter = csv.writer(fid,delimiter=',')
#     csvWriter.writerows(dev_data)

# with open("single_dataset_test.csv", "w+") as fid:
#     fid.write("Density,Energy,s,dsde_rho,dsdrho_e,d2sde2,d2sdedrho,d2sdrho2,p,T,c2\n")
#     csvWriter = csv.writer(fid,delimiter=',')
#     csvWriter.writerows(test_data)

# with open("ORCHID_dataset.csv", "w+") as fid:
#     fid.write("Density,Energy,s,dsde_rho,dsdrho_e,d2sde2,d2sdedrho,d2sdrho2,p,T,c2\n")
#     csvWriter = csv.writer(fid,delimiter=',')
#     csvWriter.writerows(ORCHID_data)
