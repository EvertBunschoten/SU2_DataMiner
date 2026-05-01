###############################################################################################
#       #      _____ __  _____      ____        __        __  ____                   #        #  
#       #     / ___// / / /__ \    / __ \____ _/ /_____ _/  |/  (_)___  ___  _____   #        #  
#       #     \__ \/ / / /__/ /   / / / / __ `/ __/ __ `/ /|_/ / / __ \/ _ \/ ___/   #        #      
#       #    ___/ / /_/ // __/   / /_/ / /_/ / /_/ /_/ / /  / / / / / /  __/ /       #        #  
#       #   /____/\____//____/  /_____/\__,_/\__/\__,_/_/  /_/_/_/ /_/\___/_/        #        #
#       #                                                                            #        #
###############################################################################################

############################ FILE NAME: DataGenerator_NICFD.py ################################
#=============================================================================================#
# author: Evert Bunschoten                                                                    |
#    :PhD Candidate ,                                                                         |
#    :Flight Power and Propulsion                                                             |
#    :TU Delft,                                                                               |
#    :The Netherlands                                                                         |
#                                                                                             |
#                                                                                             |
# Description:                                                                                |
#  Class for generating fluid data for NI-CFD data mining operations.                         |                                                               
#                                                                                             |  
# Version: 3.1.0                                                                              |
#                                                                                             |
#=============================================================================================#

#---------------------------------------------------------------------------------------------#
# Importing general packages
#---------------------------------------------------------------------------------------------#
import CoolProp.CoolProp as CP
import CoolProp as CoolP
import numpy as np 
from tqdm import tqdm
import csv 
import matplotlib.pyplot as plt 
np.random.seed(2)

#---------------------------------------------------------------------------------------------#
# Importing DataMiner classes and functions
#---------------------------------------------------------------------------------------------#
from Common.DataDrivenConfig import Config_NICFD
from Common.Properties import DefaultSettings_NICFD, EntropicVars
from Data_Generation.DataGenerator_Base import DataGenerator_Base


class DataGenerator_CoolProp(DataGenerator_Base):
    """Class for generating fluid data using CoolProp
    """
    _Config:Config_NICFD
    fluid = None 
    __accepted_phases:list[int] =  []
    # Pressure and temperature limits
    __use_PT:bool = DefaultSettings_NICFD.use_PT_grid
    __T_min:float = DefaultSettings_NICFD.T_min
    __T_max:float = DefaultSettings_NICFD.T_max
    __Np_Y:int = DefaultSettings_NICFD.Np_temp

    __P_min:float = DefaultSettings_NICFD.P_min
    __P_max:float = DefaultSettings_NICFD.P_max
    __Np_X:int = DefaultSettings_NICFD.Np_p

    __auto_range:bool=False
    # Density and static energy limits
    __rho_min:float = DefaultSettings_NICFD.Rho_min
    __rho_max:float = DefaultSettings_NICFD.Rho_max
    __e_min:float = DefaultSettings_NICFD.Energy_min 
    __e_max:float = DefaultSettings_NICFD.Energy_max 
    __X_grid:np.ndarray[float] = None
    __Y_grid:np.ndarray[float] = None 

    # Entropy derivatives.
    __StateVars_fluid:np.ndarray[float] = None 
    __success_locations:np.ndarray[bool] = None 
    __mixture:bool = False 

    __fd_step_size_rho:float = 7e-3
    __fd_step_size_e:float = 7e-3

    def __init__(self, Config_in:Config_NICFD=None):
        DataGenerator_Base.__init__(self, Config_in=Config_in)

        if Config_in is None:
            print("Initializing NICFD data generator with default settings.")
            self._Config = Config_NICFD()
        else:
            # Load configuration and set default properties.
            self.__use_PT = self._Config.GetPTGrid()
            if len(self._Config.GetFluidNames()) > 1:
                self.__mixture = True 

            # Define phases for which to generate fluid data.
            self.EnableTwophase(self._Config.TwoPhase())
            self.EnableLiquidPhase(self._Config.LiquidPhase())
            self.EnableGasPhase(self._Config.GasPhase())
            self.EnableSuperCritical(self._Config.SuperCritical())

            self.fluid = CP.AbstractState(self._Config.GetEquationOfState(), self._Config.GetFluid())
            self.__auto_range = self._Config.GetAutoRange()

            if len(self._Config.GetFluidNames()) > 1:
                mole_fractions = self._Config.GetMoleFractions()
                self.fluid.set_mole_fractions(mole_fractions)
            
            self.__use_PT = self._Config.GetPTGrid()
            P_bounds = self._Config.GetPressureBounds()
            T_bounds = self._Config.GetTemperatureBounds()
            rho_bounds = self._Config.GetDensityBounds()
            e_bounds = self._Config.GetEnergyBounds()

            self.__P_min, self.__P_max = P_bounds[0], P_bounds[1]
            self.__rho_min, self.__rho_max = rho_bounds[0], rho_bounds[1]
            self.__Np_X = self._Config.GetNpPressure()

            self.__T_min, self.__T_max = T_bounds[0], T_bounds[1]
            self.__e_min, self.__e_max = e_bounds[0], e_bounds[1]
            self.__Np_Y = self._Config.GetNpTemp()

        return 
    
    def UseAutoRange(self, use_auto_range:bool=True):
        """Automatically set controlling variable ranges depending on the fluid triple point and critical point."""
        self.__auto_range = use_auto_range
        return 
    
    def PreprocessData(self):
        """Generate density and static energy grid at which to evaluate fluid properties.
        """
        if self.__auto_range:
            pmin = CP.PropsSI("PTRIPLE", self._Config.GetFluid())
            pmax = self.fluid.pmax()
            Tmin = self.fluid.Tmin()
            Tmax = self.fluid.Tmax()
            p_range = np.linspace(pmin, pmax, self.__Np_X)
            T_range = np.linspace(Tmin, Tmax, self.__Np_Y)
            pp, TT = np.meshgrid(p_range, T_range)
            dd = np.zeros(np.shape(pp))
            uu = np.zeros(np.shape(TT))
            for i in range(len(T_range)):
                for j in range(len(p_range)):
                    try:
                        self.fluid.update(CP.PT_INPUTS, pp[i,j], TT[i,j])
                        if self.fluid.phase() in self.__accepted_phases:
                            dd[i,j] = self.fluid.rhomass()
                            uu[i,j] = self.fluid.umass()
                        else:
                            dd[i,j] = float("nan")
                            uu[i,j] = float("nan")
                    except:
                        dd[i,j] = float("nan")
                        uu[i,j] = float("nan")
            idx_valid = np.invert(np.isnan(dd))
            if self.__use_PT:
                X_min, X_max = np.min(pp[idx_valid]), np.max(pp[idx_valid])
                Y_min, Y_max = np.min(TT[idx_valid]), np.max(TT[idx_valid])
                self.__P_min, self.__P_max = X_min, X_max
                self.__T_min, self.__T_max = Y_min, Y_max
            else:
                X_min, X_max = np.min(dd[idx_valid]), np.max(dd[idx_valid])
                Y_min, Y_max = np.min(uu[idx_valid]), np.max(uu[idx_valid])
                self.__rho_min, self.__rho_max = X_min, X_max
                self.__e_min, self.__e_max = Y_min, Y_max
            self.UpdateConfig()
        else:
            if self.__use_PT:
                X_min = self.__P_min
                X_max = self.__P_max
                Y_min = self.__T_min
                Y_max = self.__T_max
            else:
                X_min = self.__rho_min
                X_max = self.__rho_max
                Y_min = self.__e_min
                Y_max = self.__e_max 
        X_range = (X_min - X_max) * np.cos(np.linspace(0, 0.5*np.pi, self.__Np_X)) + X_max
        Y_range = np.linspace(Y_min, Y_max, self.__Np_Y)
        self.__X_grid, self.__Y_grid = np.meshgrid(X_range, Y_range)
        return 
    
    def UpdateConfig(self):
        if self.__use_PT:
            self._Config.SetPressureBounds(self.__P_min, self.__P_max)
            self._Config.SetTemperatureBounds(self.__T_min, self.__T_max)
        else:
            self._Config.SetDensityBounds(self.__rho_min, self.__rho_max)
            self._Config.SetEnergyBounds(self.__e_min, self.__e_max)
        self._Config.SaveConfig()
        return 
    
    def VisualizeDataGrid(self):
        """Visualize query points at which fluid data are evaluated.
        """
        fig = plt.figure(figsize=[10,10])
        ax = plt.axes()
        ax.plot(self.__X_grid.flatten(), self.__Y_grid.flatten(), 'k.')
        if self. __use_PT:
            ax.set_xlabel(r"Pressure $(p)[Pa]",fontsize=20)
            ax.set_ylabel(r"Temperature $(T)[K]",fontsize=20)
        else:
            ax.set_xlabel(r"Density $(\rho)[kg m^{-3}]$",fontsize=20)
            ax.set_ylabel(r"Static energy $(e)[J kg^{-1}]$",fontsize=20)
        ax.tick_params(which='both',labelsize=18)
        ax.grid()
        
        plt.show()
        return 
    
    def SetTemperatureBounds(self, T_lower:float=DefaultSettings_NICFD.T_min, T_upper:float=DefaultSettings_NICFD.T_max):
        """Set the upper and lower temperature limits for the fluid data grid.

        :param T_lower: lower temperature limit in Kelvin.
        :type T_lower: float
        :param T_upper: upper temperature limit in Kelvin.
        :type T_upper: float
        :raises Exception: if lower temperature limit exceeds upper temperature limit.
        """
        if (T_lower >= T_upper):
            raise Exception("Lower temperature should be lower than upper temperature.")
        else:
            self.__T_min = T_lower
            self.__T_max = T_upper
        return
    
    def GetTemperatureBounds(self):
        """Get fluid temperature limits.

        :return: list with minimum and maximum temperature.
        :rtype: list[float]
        """
        return [self.__T_min, self.__T_max]
    
    def SetNpDensity(self, Np_density:int=DefaultSettings_NICFD.Np_p):
        self.SetNpPressure(Np_P=Np_density)
        return 

    def GetNpDensity(self):
        return self.GetNpPressure()
    
    def SetDensityBounds(self, Density_lower:float=DefaultSettings_NICFD.Rho_min, Density_upper:float=DefaultSettings_NICFD.Rho_max):
        self.__rho_min = Density_lower
        self.__rho_max = Density_upper
        return 
    
    def SetNpEnergy(self, Np_energy:int=DefaultSettings_NICFD.Np_temp):
        self.SetNpTemp(Np_Temp=Np_energy)
        return 
    
    def GetNpEnergy(self):
        return self.GetNpTemp()
    
    def SetEnergyBounds(self, Energy_lower:float=DefaultSettings_NICFD.Energy_min, Energy_upper:float=DefaultSettings_NICFD.Energy_max):
        self.__e_min = Energy_lower
        self.__e_max = Energy_upper
        return 
    
    
    def SetNpTemp(self, Np_Temp:int=DefaultSettings_NICFD.Np_temp):
        """
        Set number of divisions for the temperature grid.

        :param Np_Temp: Number of divisions for the temperature range.
        :type Np_Temp: int
        :rase: Exception: If the number of divisions is lower than one.

        """
        if (Np_Temp <= 0):
            raise Exception("Number of unburnt temperature samples should be higher than one.")
        else:
            self.__Np_Y = Np_Temp
        return 
    
    def GetNpTemp(self):
        """
        Get the number of divisions for the fluid temperature range.

        :return: Number of divisions for the fluid temperature range.
        :rtype: int

        """
        return self.__Np_Y
    

    
    def SetNpPressure(self, Np_P:int=DefaultSettings_NICFD.Np_p):
        """
        Set number of divisions for the fluid pressure grid.

        :param Np_Temp: Number of divisions for the fluid pressure.
        :type Np_Temp: int
        :rase: Exception: If the number of divisions is lower than one.

        """
        if (Np_P <= 0):
            raise Exception("Number of unburnt temperature samples should be higher than one.")
        else:
            self.__Np_X = Np_P 
        return 
    
    def GetNpPressure(self):
        return self.__Np_X
    
    def SetPressureBounds(self, P_lower:float=DefaultSettings_NICFD.P_min, P_upper:float=DefaultSettings_NICFD.P_max):
        """Set the upper and lower limits for the fluid pressure.

        :param P_lower: lower pressure limit in Pa.
        :type P_lower: float
        :param P_upper: upper pressure limit in Pa.
        :type P_upper: float
        :raises Exception: if lower pressure limit exceeds upper pressure limit.
        """
        if (P_lower >= P_upper):
            raise Exception("Lower temperature should be lower than upper temperature.")
        else:
            self.__P_min = P_lower
            self.__P_max = P_upper
        return 
    
    def GetPressureBounds(self):
        """Get minimum and maximum pressure.

        :return: list with minimum and maximum fluid pressure.
        :rtype: list[float]
        """
        return [self.__P_min, self.__P_max]
    
    def GetDensityBounds(self):
        """Get fluid density bounds values.

        :return: minimum and maximum fluid density in data set.
        :rtype: tuple
        """
        return self.__rho_min, self.__rho_max 
    
    def GetEnergyBounds(self):
        """Get fluid static energy bounds values.

        :return: minimum and maximum fluid static energy in data set.
        :rtype: tuple
        """
        return self.__e_min, self.__e_max 
    
    def IncludeTransportProperties(self, calc_transport_properties:bool=False):
        """Include transport properties in fluid data calculation

        :param calc_transport_properties: evaluate transport properties, defaults to False
        :type calc_transport_properties: bool, optional
        """
        self._Config.IncludeTransportProperties(calc_transport_properties)
        return 
    
    def CalcTransportProperties(self):
        return self._Config.CalcTransportProperties()
    
    def EnableGasPhase(self, gas_phase:bool=True):
        """Include thermophysical data of the fluid in gas phase.

        :param gas_phase: include gas phase data, defaults to True
        :type gas_phase: bool, optional
        """
        self._Config.EnableGasPhase(gas_phase)
        if self._Config.GasPhase() and CoolP.iphase_gas not in self.__accepted_phases:
            self.__accepted_phases.append(CoolP.iphase_gas)
            if self._Config.SuperCritical():
                self.__accepted_phases.append(CoolP.iphase_supercritical_gas)
        if not self._Config.GasPhase() and CoolP.iphase_gas in self.__accepted_phases:
            self.__accepted_phases.remove(CoolP.iphase_gas)
            if CoolP.iphase_supercritical_gas in self.__accepted_phases:
                self.__accepted_phases.remove(CoolP.iphase_supercritical_gas)
        return 

    def GasPhase(self):
        """Whether gas phase data are included in the fluid data set.

        :return: gas phase data included in fluid data set.
        :rtype: bool
        """
        return self._Config.GasPhase()
           
    def EnableTwophase(self, two_phase:bool=False):
        """Include two-phase region in fluid data.

        :param two_phase: include two-phase data in fluid data, defaults to False
        :type two_phase: bool, optional
        """
        self._Config.EnableTwophase(two_phase)
        if self._Config.TwoPhase() and CoolP.iphase_twophase not in self.__accepted_phases:
            self.__accepted_phases.append(CoolP.iphase_twophase)
        if not self._Config.TwoPhase() and CoolP.iphase_twophase in self.__accepted_phases:
            self.__accepted_phases.remove(CoolP.iphase_twophase)
        return 
    
    def TwoPhase(self):
        return self._Config.TwoPhase()
    
    def EnableLiquidPhase(self, liquid_phase:bool=False):
        """Include thermophysical data of the fluid in liquid phase.

        :param liquid_phase: include liquid phase fluid data, defaults to False
        :type liquid_phase: bool, optional
        """
        self._Config.EnableLiquidPhase(liquid_phase)
        if self._Config.LiquidPhase() and CoolP.iphase_liquid not in self.__accepted_phases:
            self.__accepted_phases.append(CoolP.iphase_liquid)
            if self._Config.SuperCritical():
                self.__accepted_phases.append(CoolP.iphase_supercritical_liquid)
        if not self._Config.LiquidPhase() and CoolP.iphase_liquid in self.__accepted_phases:
            self.__accepted_phases.remove(CoolP.iphase_liquid)
            if CoolP.iphase_supercritical_liquid in self.__accepted_phases:
                self.__accepted_phases.remove(CoolP.iphase_supercritical_liquid)
        return 
    
    def LiquidPhase(self):
        """Whether to include fluid data in liquid phase.

        :return: liquid phase data are included in the data set.
        :rtype: bool
        """
        return self._Config.LiquidPhase()
    
    def EnableSuperCritical(self, supercritical:bool=True):
        """Whether to include fluid data in the supercritical phase in the data set.

        :param supercritical: include data in the supercritical phase, defaults to True
        :type supercritical: bool, optional
        """
        self._Config.EnableSuperCritical(supercritical)
        if self._Config.SuperCritical():
            if CoolP.iphase_supercritical not in self.__accepted_phases:
                self.__accepted_phases.append(CoolP.iphase_supercritical)
            if CoolP.iphase_gas in self.__accepted_phases and CoolP.iphase_supercritical_gas not in self.__accepted_phases:
                self.__accepted_phases.append(CoolP.iphase_supercritical_gas)
            if (CoolP.iphase_liquid in self.__accepted_phases) and (CoolP.iphase_supercritical_liquid not in self.__accepted_phases):
                self.__accepted_phases.append(CoolP.iphase_supercritical_liquid)

        else:
            if CoolP.iphase_supercritical in self.__accepted_phases:
                self.__accepted_phases.remove(CoolP.iphase_supercritical)
            if CoolP.iphase_supercritical_gas in self.__accepted_phases:
                self.__accepted_phases.remove(CoolP.iphase_supercritical_gas)
            if CoolP.iphase_supercritical_liquid in self.__accepted_phases:
                self.__accepted_phases.remove(CoolP.iphase_supercritical_liquid)
        return 
    
    def SetConductivityModel(self, conductivity_model:str=DefaultSettings_NICFD.conductivity_model):
        """Specify the two-phase conductivity model.

        :param conductivity_model: two-phase conductivity model option, defaults to "volume"
        :type conductivity_model: str, optional
        :raises Exception: if the specified option is not supported.
        """
        self._Config.SetConductivityModel(conductivity_model)
        if not self.TwoPhase():
            self.EnableTwophase(True)
            print("Two-phase conductivity model specified, including two-phase fluid data.")
        return 
    
    def GetConductivityModel(self):
        """Get two-phase conductivity model.

        :return: two-phase conductivity model option
        :rtype: str
        """
        return self._Config.GetConductivityModel()
    
    def SetViscosityModel(self, viscosity_model:str=DefaultSettings_NICFD.viscosity_model):
        """Specify the two-phase viscosity model.

        :param viscosity_model: two-phase viscosity model option, defaults to "mcadams"
        :type viscosity_model: str, optional
        :raises Exception: if the specified option is not supported.
        """
        self._Config.SetViscosityModel(viscosity_model)
        if not self.TwoPhase():
            self.EnableTwophase(True)
            print("Two-phase viscosity model specified, including two-phase fluid data.")
        return 
    
    def GetViscosityModel(self):
        """Get two-phase viscosity model.

        :return: two-phase viscosity model option
        :rtype: str
        """
        return self._Config.GetViscosityModel()
    

    def ComputeData(self):
        super().ComputeData()
        """Evaluate fluid properties on the density-energy grid or pressure-temperature grid.
        """

        # Initiate empty fluid property arrays.
        
        self.__StateVars_fluid = np.zeros([self.__Np_X, self.__Np_Y, EntropicVars.N_STATE_VARS.value])
        self.__success_locations = np.ones([self.__Np_X, self.__Np_Y],dtype=bool)
        # Loop over density-based or pressure-based grid.
        for i in tqdm(range(self.__Np_X)):
            for j in range(self.__Np_Y):
                try:
                    if self.__use_PT:
                        self.fluid.update(CP.PT_INPUTS, self.__X_grid[i,j], self.__Y_grid[i,j])
                    else:
                        self.fluid.update(CP.DmassUmass_INPUTS, self.__X_grid[j,i], self.__Y_grid[j,i])
                        p = self.fluid.p()
                        T = self.fluid.T()
                        self.fluid.update(CP.PT_INPUTS, p, T)
                    # Check if fluid phase is not vapor or liquid
                    self.__StateVars_fluid[i,j,:], self.__success_locations[i,j] = self.GetStateVector()
                except:
                    self.__success_locations[i,j] = False 
                    self.__StateVars_fluid[i, j, :] = None
        return 
    
    def __TransportProperties(self, q:float):
        """Evaluate fluid transport properties

        :param q: vapor quality
        :type q: float
        :return: dynamic viscosity, thermal conductivity, vapor quality
        :rtype: float, float, float
        """
        p = self.fluid.p()
        rho = self.fluid.rhomass()
        e = self.fluid.umass()
        phase = self.fluid.phase()
        is_twophase = (phase == CoolP.iphase_twophase)
        if is_twophase:
                vapor_q = q
                sat_props = self.__get_saturated_transport_properties(p)
        
                if sat_props is not None:
                    # Compute void fraction for volume-weighted models
                    alpha = self.__compute_void_fraction(
                        q, sat_props['rho_l'], sat_props['rho_g'])
                    # Mixture viscosity
                    viscosity = self.__compute_twophase_viscosity(
                        q, sat_props['mu_l'], sat_props['mu_g'],
                        sat_props['rho_l'], sat_props['rho_g'], alpha)
                    
                    # Mixture thermal conductivity
                    conductivity = self.__compute_twophase_conductivity(
                        q, sat_props['k_l'], sat_props['k_g'],
                        sat_props['rho_l'], sat_props['rho_g'], alpha)
                else:
                    # Fallback: try to get properties at current state (might fail)
                    try:
                        self.fluid.update(CP.DmassUmass_INPUTS, rho, e)
                        viscosity = self.fluid.viscosity()
                        conductivity = self.fluid.conductivity()
                    except:
                        # Last resort: interpolate based on quality
                        viscosity = 1e-5  # Reasonable default
                        conductivity = 0.1  # Reasonable default
        else:
            vapor_q = -1.0
            viscosity = self.fluid.viscosity()
            conductivity = self.fluid.conductivity()
        return viscosity, conductivity, vapor_q
    
    def __EntropicEoS(self, rho, e, s, derivs, state_vector_struct:dict):
        state_vector_struct[EntropicVars.Density.name] = rho 
        state_vector_struct[EntropicVars.Energy.name] = e 
        state_vector_struct[EntropicVars.s.name] = s
        dsdrho_e = derivs["dsdrho_e"]
        state_vector_struct[EntropicVars.dsdrho_e.name] = dsdrho_e
        dsde_rho = derivs["dsde_rho"]
        state_vector_struct[EntropicVars.dsde_rho.name] = dsde_rho
        d2sdrho2 = derivs["d2sdrho2"]
        state_vector_struct[EntropicVars.d2sdrho2.name] = d2sdrho2
        d2sde2 = derivs["d2sde2"]
        state_vector_struct[EntropicVars.d2sde2.name] = d2sde2
        d2sdedrho = derivs["d2sdedrho"]
        state_vector_struct[EntropicVars.d2sdedrho.name] = d2sdedrho
        Temperature = pow(dsde_rho, -1)
        state_vector_struct[EntropicVars.T.name] = Temperature
        Pressure = -rho * rho * Temperature * dsdrho_e
        state_vector_struct[EntropicVars.p.name] = Pressure
        dPde_rho = -rho*rho * Temperature * (-Temperature * (d2sde2 * dsdrho_e) + d2sdedrho)
        state_vector_struct[EntropicVars.dpde_rho.name] = dPde_rho
        dPdrho_e = - rho * Temperature * (dsdrho_e * (2 - rho * Temperature * d2sdedrho) + rho * d2sdrho2)
        state_vector_struct[EntropicVars.dpdrho_e.name] = dPdrho_e
        SoundSpeed2 = dPdrho_e - (dsdrho_e / dsde_rho) * dPde_rho
        state_vector_struct[EntropicVars.c2.name] = SoundSpeed2
        dTde_rho = -Temperature * Temperature * d2sde2
        state_vector_struct[EntropicVars.dTde_rho.name] = dTde_rho
        dTdrho_e = -Temperature * Temperature * d2sdedrho
        state_vector_struct[EntropicVars.dTdrho_e.name] = dTdrho_e
        
        dhde_rho = 1 + dPde_rho / rho 
        dhdrho_e = -Pressure * np.power(rho, -2) + dPdrho_e / rho 
        state_vector_struct[EntropicVars.dhde_rho.name] = dhdrho_e
        state_vector_struct[EntropicVars.dhdrho_e.name] = dhde_rho

        # drhode_p = -dPde_rho / dPdrho_e 
        # dTde_p = dTde_rho + dTdrho_e * drhode_p
        # dhde_p = dhde_rho + drhode_p*dhdrho_e 
        # Cp = dhde_p / (dTde_p+np.sign(dTde_p)*1e-8)
        # Cv = 1 / (dTde_rho+np.sign(dTde_rho)*1e-8)

        X = self.fluid.Q()
        self.fluid.update(CoolP.PQ_INPUTS, Pressure, 1)
        cp_vap, cv_vap,rho_vap = self.fluid.cpmass(), self.fluid.cvmass(), self.fluid.rhomass()
        self.fluid.update(CoolP.PQ_INPUTS, Pressure, 0)
        cp_liq, cv_liq = self.fluid.cpmass(), self.fluid.cvmass()
        alpha=X*rho/rho_vap
        Cp =alpha*cp_vap+(1-alpha)*cp_liq
        Cv =alpha*cv_vap+(1-alpha)*cv_liq
        self.fluid.update(CP.DmassUmass_INPUTS, rho, e)

        state_vector_struct[EntropicVars.cp.name] = Cp
        state_vector_struct[EntropicVars.cv.name] = Cv
        dhdrho_P = dhdrho_e - dhde_rho * (1 / dPde_rho) * dPdrho_e
        dhdP_rho = dhde_rho * (1 / dPde_rho)
        dsdrho_P = dsdrho_e - dPdrho_e * (1 / dPde_rho) * dsde_rho
        dsdP_rho = dsde_rho / dPde_rho
        state_vector_struct[EntropicVars.dhdrho_p.name] = dhdrho_P
        state_vector_struct[EntropicVars.dhdp_rho.name] = dhdP_rho
        state_vector_struct[EntropicVars.dsdrho_p.name] = dsdrho_P
        state_vector_struct[EntropicVars.dsdp_rho.name] = dsdP_rho
        return 
        

        
    def __EquationofState(self, s:float, rho:float, e:float, derivs:dict, state_vector_struct:dict):
        """Calculate thermodynamic state variables with entropic equation of state

        :param s: fluid entropy
        :type s: float
        :param rho: fluid density
        :type rho: float
        :param e: fluid static energy
        :type e: float
        :param derivs: entropy Jacobian and Hessian
        :type derivs: dict
        :param state_vector_struct: state struct
        :type state_vector_struct: dict
        """

        # Retrieve entropy Jacobian and Hessian components
        dsdrho_e = derivs["dsdrho_e"]
        dsde_rho = derivs["dsde_rho"]
        d2sdrho2 = derivs["d2sdrho2"]
        d2sde2 = derivs["d2sde2"]
        d2sdedrho = derivs["d2sdedrho"]

        # Store entropy, Jacobian, and Hessian components
        state_vector_struct[EntropicVars.s.name] = s 
        state_vector_struct[EntropicVars.Density.name] = rho 
        state_vector_struct[EntropicVars.Energy.name] = e 
        state_vector_struct[EntropicVars.dsdrho_e.name] = dsdrho_e
        state_vector_struct[EntropicVars.dsde_rho.name] = dsde_rho
        state_vector_struct[EntropicVars.d2sdrho2.name] = d2sdrho2
        state_vector_struct[EntropicVars.d2sde2.name] = d2sde2
        state_vector_struct[EntropicVars.d2sdedrho.name] = d2sdedrho
        
        # Compute and store thermodynamic state variables
        Temperature = self.fluid.T()#pow(dsde_rho, -1)
        state_vector_struct[EntropicVars.T.name] = Temperature
        Pressure = self.fluid.p()#-rho * rho * Temperature * dsdrho_e
        state_vector_struct[EntropicVars.p.name] = Pressure 
        Enthalpy = self.fluid.hmass()
        state_vector_struct[EntropicVars.Enthalpy.name] = Enthalpy

        X = self.fluid.Q()
        if X<=0 or X>=1:
            state_vector_struct[EntropicVars.dpde_rho.name] = self.fluid.first_partial_deriv(CP.iP, CP.iUmass, CP.iDmass)
            state_vector_struct[EntropicVars.dpdrho_e.name] = self.fluid.first_partial_deriv(CP.iP, CP.iDmass, CP.iUmass)
            state_vector_struct[EntropicVars.c2.name] = self.fluid.speed_sound()**2
            state_vector_struct[EntropicVars.dTde_rho.name] = self.fluid.first_partial_deriv(CP.iT, CP.iUmass, CP.iDmass)
            state_vector_struct[EntropicVars.dTdrho_e.name] = self.fluid.first_partial_deriv(CP.iT, CP.iDmass, CP.iUmass)
            state_vector_struct[EntropicVars.cp.name] = self.fluid.cpmass()
            state_vector_struct[EntropicVars.cv.name] = self.fluid.cvmass()
            state_vector_struct[EntropicVars.dhdrho_e.name] = self.fluid.first_partial_deriv(CP.iHmass, CP.iDmass, CP.iUmass)
            state_vector_struct[EntropicVars.dhde_rho.name] = self.fluid.first_partial_deriv(CP.iHmass, CP.iUmass, CP.iDmass)
            state_vector_struct[EntropicVars.dhdrho_p.name] = self.fluid.first_partial_deriv(CP.iHmass, CP.iDmass, CP.iP)
            state_vector_struct[EntropicVars.dhdp_rho.name] = self.fluid.first_partial_deriv(CP.iHmass, CP.iP, CP.iDmass)
            state_vector_struct[EntropicVars.dsdrho_p.name] = self.fluid.first_partial_deriv(CP.iSmass, CP.iDmass, CP.iP)
            state_vector_struct[EntropicVars.dsdp_rho.name] = self.fluid.first_partial_deriv(CP.iSmass, CP.iP, CP.iDmass)
        else:
            self.__EntropicEoS(rho, e, s, derivs, state_vector_struct)
        return 
        
    def __ThermodynamicState(self):
        """Evaluate the thermodynamic state variables

        :return: thermodynamic state variables.
        :rtype: struct
        """
        state_vector_vals = {v.name : None for v in EntropicVars}

        phase = self.fluid.phase()
        rho = self.fluid.rhomass()
        e = self.fluid.umass()
        s = self.fluid.smass()
        is_twophase = (phase == CoolP.iphase_twophase)
        # Evaluate entropy Jacobian and Hessian components depending on phase
        derivs = {"dsdrho_e":0,"dsde_rho":0,"d2sdrho2":0,"d2sde2":0,"d2sdedrho":0}
        if is_twophase:
            derivs_twophase = self.__compute_derivatives_fd(rho, e, s)
            derivs["dsdrho_e"] = derivs_twophase["dsdrho_e"]
            derivs["dsde_rho"] = derivs_twophase["dsde_rho"]
            derivs["d2sdrho2"] = derivs_twophase["d2sdrho2"]
            derivs["d2sde2"] = derivs_twophase["d2sde2"]
            derivs["d2sdedrho"] = derivs_twophase["d2sdedrho"]
            self.UpdateFluid(rho, e)
        else:
            derivs["dsdrho_e"] = self.fluid.first_partial_deriv(CP.iSmass, CP.iDmass, CP.iUmass)
            derivs["dsde_rho"] = self.fluid.first_partial_deriv(CP.iSmass, CP.iUmass, CP.iDmass)
            derivs["d2sdrho2"] = self.fluid.second_partial_deriv(CP.iSmass, CP.iDmass, CP.iUmass, CP.iDmass, CP.iUmass)
            derivs["d2sde2"] = self.fluid.second_partial_deriv(CP.iSmass, CP.iUmass, CP.iDmass, CP.iUmass, CP.iDmass)
            derivs["d2sdedrho"] = self.fluid.second_partial_deriv(CP.iSmass, CP.iUmass, CP.iDmass, CP.iDmass, CP.iUmass)

        # Calculate thermodynamic state variables with entropy-based equation of state
        self.__EquationofState(s, rho, e, derivs, state_vector_vals)

        return state_vector_vals
        

    def GetStateVector(self):
        """Retrieve thermodynamic or thermophysical state information

        :return: array with fluid properties, whether phase is supported
        :rtype: np.ndarray[float], bool
        """
        state_vector_vals = np.ones(EntropicVars.N_STATE_VARS.value)

        phase = self.fluid.phase()
        q = self.fluid.Q()
        correct_phase = True
        if phase in self.__accepted_phases:
            state_vals = self.__ThermodynamicState()
            if self._Config.CalcTransportProperties():
                viscosity, conductivity,vapor_quality = self.__TransportProperties(q)
                state_vals[EntropicVars.Conductivity.name] = conductivity
                state_vals[EntropicVars.ViscosityDyn.name] = viscosity
                state_vals[EntropicVars.VaporQuality.name] = vapor_quality
            for s in EntropicVars:
                if s.value < EntropicVars.N_STATE_VARS.value:
                    state_vector_vals[s.value] = state_vals[s.name]

        else:
            correct_phase = False
            state_vector_vals[:] = None 
        return state_vector_vals, correct_phase
    

    def __get_entropy_safe(self, rho:float, e:float):
        """Retrieve fluid entropy and check if CoolProp converges

        :param rho: fluid density
        :type rho: float
        :param e: fluid static energy
        :type e: float
        :return: fluid entropy
        :rtype: float
        """
        try:
            self.fluid.update(CP.DmassUmass_INPUTS, rho, e)
            phase = self.fluid.phase()
            if phase not in self.__accepted_phases:
                return None
            return self.fluid.smass()
        except:
            return None
    
    def SetFDStepSizes(self, fd_step_rho:float=1e-5, fd_step_e:float=1e-5):
        """Set relative step sizes for density and static energy used for finite-difference analysis.

        :param fd_step_rho: relative density FD step, defaults to 1e-5
        :type fd_step_rho: float, optional
        :param fd_step_e: relative static energy FD step, defaults to 1e-5
        :type fd_step_e: float, optional
        :raises Exception: if negative step sizes are provided
        """
        if fd_step_rho <= 0 or fd_step_e <= 0:
            raise Exception("Relative step sizes for finite-differences should be positive")
        self.__fd_step_size_rho = fd_step_rho
        self.__fd_step_size_e = fd_step_e
        return 
    
    def ComputeSaturationCurve(self):
        """Compute the density and static energy along the fluid saturation curve

        :return: density and static energy array of saturation curve
        :rtype: np.ndarray[float]
        """
        eos_fluid = "%s::%s" % (self._Config.GetEquationOfState(), self._Config.GetFluid())
        ptriple = CP.PropsSI("PTRIPLE", eos_fluid)
        pcrit = CP.PropsSI("PCRIT", eos_fluid)
        Psat = np.linspace(ptriple, pcrit, 4000)

        rhoLiq=CP.PropsSI("D","P",Psat,"Q",0,eos_fluid)
        rhoVap=CP.PropsSI("D","P",Psat,"Q",1,eos_fluid)

        eLiq=CP.PropsSI("U","P",Psat,"Q",0,eos_fluid)
        eVap=CP.PropsSI("U","P",Psat,"Q",1,eos_fluid)

        rho_sat=np.concatenate((rhoLiq[:-1],np.flip(rhoVap)))
        e_sat=np.concatenate((eLiq[:-1],np.flip(eVap)))
        sat_curve=np.column_stack((rho_sat,e_sat))
        return sat_curve 
    
    def GetFDStepSizes(self):
        return self.__fd_step_size_rho, self.__fd_step_size_e
    
    def __get_saturated_transport_properties(self, p:float):
        """Get transport properties at saturation conditions for both liquid and vapor phases.

        :param p: pressure [Pa]
        :type p: float
        :return: dict with keys: 'mu_l', 'mu_g', 'k_l', 'k_g', 'rho_l', 'rho_g'
        :rtype: dict
        """
        try:
            # Create temporary fluid states for saturated liquid and vapor
            # Use pressure-quality inputs to get saturation properties
            
            # Saturated liquid (Q=0)
            self.fluid.update(CP.PQ_INPUTS, p, 0.0)
            mu_l = self.fluid.viscosity()
            k_l = self.fluid.conductivity()
            rho_l = self.fluid.rhomass()
            
            # Saturated vapor (Q=1)
            self.fluid.update(CP.PQ_INPUTS, p, 1.0)
            mu_g = self.fluid.viscosity()
            k_g = self.fluid.conductivity()
            rho_g = self.fluid.rhomass()
            
            return {
                'mu_l': mu_l, 'mu_g': mu_g,
                'k_l': k_l, 'k_g': k_g,
                'rho_l': rho_l, 'rho_g': rho_g
            }
        except Exception as ex:
            return None
    
    def __compute_void_fraction(self, quality:float, rho_l:float, rho_g:float):
        """Compute void fractio from quality using the homogeneous model.

        :param quality: vapor quality (mass fraction of vapor)
        :type quality: float
        :param rho_l: Saturated liquid density [kg/m3]
        :type rho_l: float
        :param rho_g: Saturated vapor density [kg/m3]
        :type rho_g: float
        :return: Void fraction (volume fraction of vapor)
        :rtype: float
        """

        if quality <= 0:
            return 0.0
        elif quality >= 1:
            return 1.0
        else:
            # Homogeneous model void fraction
            return 1.0 / (1.0 + (1.0 - quality) / quality * rho_g / rho_l)


    def __compute_twophase_viscosity(self, quality:float, mu_l:float, mu_g:float, rho_l:float=None, rho_g:float=None, alpha:float=None):
        """Compute two-phase mixture viscosity using selected mixing model.

        :param quality: Vapor quality (mass fraction)
        :type quality: float
        :param mu_l: Saturated liquid viscosity [Pas]
        :type mu_l: float
        :param mu_g: Saturated vapor viscosity [Pas]
        :type mu_g: float
        :param rho_l:  Saturated liquid density [kg/m3] (needed for some models), defaults to None
        :type rho_l: float, optional
        :param rho_g: Saturated vapor density [kg/m3] (needed for some models), defaults to None
        :type rho_g: float, optional
        :param alpha: Void fraction, defaults to None
        :type alpha: float, optional
        :return: Two-phase mixture viscosity [Pas]
        :rtype: float
        """
        
        x = quality
        if self._Config.GetViscosityModel() == "mcadams":
            # McAdams et al. (1942) - Reciprocal average (recommended for most cases)
            if x <= 0:
                return mu_l
            elif x >= 1:
                return mu_g
            else:
                return 1.0 / (x / mu_g + (1.0 - x) / mu_l)
                
        elif self._Config.GetViscosityModel() == "cicchitti":
            # Cicchitti et al. (1960) - Mass-weighted average
            return x * mu_g + (1.0 - x) * mu_l
            
        elif self._Config.GetViscosityModel() == "dukler":
            # Dukler et al. (1964) - Volume-weighted (needs void fraction)
            if alpha is None:
                alpha = self.__compute_void_fraction(x, rho_l, rho_g)
            return alpha * mu_g + (1.0 - alpha) * mu_l
            
        else:
            # Default to McAdams
            if x <= 0:
                return mu_l
            elif x >= 1:
                return mu_g
            else:
                return 1.0 / (x / mu_g + (1.0 - x) / mu_l)

    def __compute_twophase_conductivity(self, quality:float, k_l:float, k_g:float, rho_l:float=None, rho_g:float=None, alpha:float=None):
        """Compute two-phase mixture thermal conductivity using selected mixing model.

        :param quality: Vapor quality (mass fraction)
        :type quality: float
        :param k_l: Saturated liquid thermal conductivity [W/(mK)]
        :type k_l: float
        :param k_g: Saturated vapor thermal conductivity [W/(mK)]
        :type k_g: float
        :param rho_l: Saturated liquid density [kg/m3], defaults to None
        :type rho_l: float, optional
        :param rho_g: Saturated vapor density [kg/m3], defaults to None
        :type rho_g: float, optional
        :param alpha: Void fraction, defaults to None
        :type alpha: float, optional
        :return: Two-phase mixture thermal conductivity [W/(mK)]
        :rtype: float
        """
     
        x = quality
        
        if alpha is None and rho_l is not None and rho_g is not None:
            alpha = self.__compute_void_fraction(x, rho_l, rho_g)
        
        if self._Config.GetConductivityModel() == "volume":
            # Volume-weighted average (recommended)
            if alpha is None:
                # Fallback to mass-weighted if void fraction unavailable
                return x * k_g + (1.0 - x) * k_l
            return alpha * k_g + (1.0 - alpha) * k_l
            
        elif self._Config.GetConductivityModel() == "mass":
            # Mass-weighted average
            return x * k_g + (1.0 - x) * k_l
            
        else:
            # Default to volume-weighted
            if alpha is None:
                return x * k_g + (1.0 - x) * k_l
            return alpha * k_g + (1.0 - alpha) * k_l
    

    def DiscretizationError(self, rho, e):
        self.UpdateFluid(rho, e) 
        state_vector,_ = self.GetStateVector()
        self.UpdateFluid(rho, e) 
        dPde_rho_ref = self.fluid.first_partial_deriv(CP.iP, CP.iUmass, CP.iDmass)
        dPdrho_e_ref = self.fluid.first_partial_deriv(CP.iP, CP.iDmass, CP.iUmass)
        dTde_rho_ref = self.fluid.first_partial_deriv(CP.iT, CP.iUmass, CP.iDmass)
        dTdrho_e_ref = self.fluid.first_partial_deriv(CP.iT, CP.iDmass, CP.iUmass)
        P_ref = self.fluid.p()
        T_ref = self.fluid.T()
        ref_data = np.array([P_ref, T_ref])

        dPde_rho_test = state_vector[EntropicVars.dpde_rho.value]
        dPdrho_e_test = state_vector[EntropicVars.dpdrho_e.value]
        dTde_rho_test = state_vector[EntropicVars.dTde_rho.value]
        dTdrho_e_test = state_vector[EntropicVars.dTdrho_e.value]
        P_test = state_vector[EntropicVars.p.value]
        T_test = state_vector[EntropicVars.T.value]
        test_data = np.array([P_test, T_test])
        
        e = np.sum(np.power((test_data - ref_data)/test_data,2))
        return e
    
    def __compute_derivatives_fd(self, rho:float, e:float, s_center:float):
        """Compute all entropy derivatives using central finite differences. Works for both single-phase and two-phase regions.

        :param rho: fluid density [kg/m3]
        :type rho: float
        :param e: fluid static energy [J/kg]
        :type e: float
        :param s_center: fluid entropy [J/kg]
        :type s_center: float
        :return: dict with keys: 'dsdrho_e', 'dsde_rho', 'd2sdrho2', 'd2sde2', 'd2sdedrho'
        :rtype: dict
        """

        # Compute step sizes
        drho = rho * self.__fd_step_size_rho#max(rho * self.__fd_step_size_rho, 1e-6)  # Minimum absolute step
        de = abs(e) * self.__fd_step_size_e#max(abs(e) * self.__fd_step_size_e, 1.0)     # Minimum 1 J/kg step
        
        # Get entropy at stencil points for first derivatives (central difference)
        s_rho_plus = self.__get_entropy_safe(rho + drho, e)
        s_rho_minus = self.__get_entropy_safe(rho - drho, e)
        s_e_plus = self.__get_entropy_safe(rho, e + de)
        s_e_minus = self.__get_entropy_safe(rho, e - de)
        
        # Check if first derivative stencil is valid
        if any(s is None for s in [s_rho_plus, s_rho_minus, s_e_plus, s_e_minus]):
            # Try one-sided differences as fallback
            return self.__compute_derivatives_fd_onesided(rho, e, s_center, drho, de)
        
        # First derivatives (central difference: O(h^2) accuracy)
        dsdrho_e = (s_rho_plus - s_rho_minus) / (2 * drho)
        dsde_rho = (s_e_plus - s_e_minus) / (2 * de)
        
        # Second derivatives
        d2sdrho2 = (s_rho_plus - 2*s_center + s_rho_minus) / (drho**2)
        d2sde2 = (s_e_plus - 2*s_center + s_e_minus) / (de**2)
        
        # Mixed derivative d2s/drhode
        # Use central difference: [s(rho+,e+) - s(rho+,e-) - s(rho-,e+) + s(rho-,e-)] / (4 * drho * de)
        s_pp = self.__get_entropy_safe(rho + drho, e + de)
        s_pm = self.__get_entropy_safe(rho + drho, e - de)
        s_mp = self.__get_entropy_safe(rho - drho, e + de)
        s_mm = self.__get_entropy_safe(rho - drho, e - de)
        
        if any(s is None for s in [s_pp, s_pm, s_mp, s_mm]):
            # Mixed derivative failed - use simpler approximation
            d2sdedrho = 0.0  # This is less critical than first derivatives
        else:
            d2sdedrho = (s_pp - s_pm - s_mp + s_mm) / (4 * drho * de)
        
        return {
            'dsdrho_e': dsdrho_e,
            'dsde_rho': dsde_rho,
            'd2sdrho2': d2sdrho2,
            'd2sde2': d2sde2,
            'd2sdedrho': d2sdedrho
        }
    
    def __compute_derivatives_fd_onesided(self, rho:float, e:float, s_center:float, drho:float, de:float):
        """Fallback: compute derivatives using one-sided finite differences. Less accurate but more robust near boundaries.

        :param rho: fluid density [kg/m3]
        :type rho: float
        :param e: fluid static energy [J/kg]
        :type e: float
        :param s_center: fluid entropy [J/kg]
        :type s_center: float
        :param drho: density increment [kg/m3]
        :type drho: float
        :param de: static energy increment [J/kg]
        :type de: float
        :return: dict with keys: 'dsdrho_e', 'dsde_rho', 'd2sdrho2', 'd2sde2', 'd2sdedrho'
        :rtype: dict
        """
        
        # Fallback: compute derivatives using one-sided finite differences.
        # Less accurate but more robust near boundaries.
        
        derivs = {}
        
        # Try forward difference for ds/drho
        s_rho_plus = self.__get_entropy_safe(rho + drho, e)
        s_rho_minus = self.__get_entropy_safe(rho - drho, e)
        
        if s_rho_plus is not None and s_rho_minus is not None:
            derivs['dsdrho_e'] = (s_rho_plus - s_rho_minus) / (2 * drho)
            derivs['d2sdrho2'] = (s_rho_plus - 2*s_center + s_rho_minus) / (drho**2)
        elif s_rho_plus is not None:
            derivs['dsdrho_e'] = (s_rho_plus - s_center) / drho
            derivs['d2sdrho2'] = 0.0
        elif s_rho_minus is not None:
            derivs['dsdrho_e'] = (s_center - s_rho_minus) / drho
            derivs['d2sdrho2'] = 0.0
        else:
            return None  # Cannot compute
        
        # Try for ds/de
        s_e_plus = self.__get_entropy_safe(rho, e + de)
        s_e_minus = self.__get_entropy_safe(rho, e - de)
        
        if s_e_plus is not None and s_e_minus is not None:
            derivs['dsde_rho'] = (s_e_plus - s_e_minus) / (2 * de)
            derivs['d2sde2'] = (s_e_plus - 2*s_center + s_e_minus) / (de**2)
        elif s_e_plus is not None:
            derivs['dsde_rho'] = (s_e_plus - s_center) / de
            derivs['d2sde2'] = 0.0
        elif s_e_minus is not None:
            derivs['dsde_rho'] = (s_center - s_e_minus) / de
            derivs['d2sde2'] = 0.0
        else:
            return None
        
        # Mixed derivative - just set to zero if we're already using fallback
        derivs['d2sdedrho'] = 0.0
        
        return derivs
    
    def UpdateFluid(self, val_rho:float, val_e:float):
        """Update the CoolProp fluid object with a given density and static energy.

        :param val_rho: fluid density [kg/m3]
        :type val_rho: float
        :param val_e: fluid static energy [J/kg]
        :type val_e: float
        """

        self.fluid.update(CP.DmassUmass_INPUTS, val_rho, val_e)
        return 
    
    def VisualizeFluidData(self):
        """Visualize computed fluid data.
        """

        fig = plt.figure(figsize=[27, 9])
        ax0 = fig.add_subplot(1, 3, 1, projection='3d')
        ax0.plot_surface(self.__StateVars_fluid[:, :, EntropicVars.Density.value],\
                         self.__StateVars_fluid[:, :, EntropicVars.Energy.value],\
                         self.__StateVars_fluid[:, :, EntropicVars.p.value])
        ax0.set_xlabel("Density [kg/m3]",fontsize=20)
        ax0.set_ylabel("Static Energy [J/kg]",fontsize=20)
        ax0.set_zlabel("Pressure [Pa]",fontsize=20)
        ax0.tick_params(which='both',labelsize=18)
        ax0.grid()
        ax0.set_title("Fluid pressure data",fontsize=22)

        ax1 = fig.add_subplot(1, 3, 2, projection='3d')
        ax1.plot_surface(self.__StateVars_fluid[:, :, EntropicVars.Density.value],\
                         self.__StateVars_fluid[:, :, EntropicVars.Energy.value],\
                         self.__StateVars_fluid[:, :, EntropicVars.T.value])
        ax1.set_xlabel("Density [kg/m3]",fontsize=20)
        ax1.set_ylabel("Static Energy [J/kg]",fontsize=20)
        ax1.set_zlabel("Temperature [K]",fontsize=20)
        ax1.tick_params(which='both',labelsize=18)
        ax1.grid()
        ax1.set_title("Fluid temperature data",fontsize=22)

        ax2 = fig.add_subplot(1, 3, 3, projection='3d')
        ax2.plot_surface(self.__StateVars_fluid[:, :, EntropicVars.Density.value],\
                         self.__StateVars_fluid[:, :, EntropicVars.Energy.value],\
                         np.sqrt(self.__StateVars_fluid[:, :, EntropicVars.c2.value]))
        ax2.set_xlabel("Density [kg/m3]",fontsize=20)
        ax2.set_ylabel("Static Energy [J/kg]",fontsize=20)
        ax2.set_zlabel("Speed of sound [m/s]",fontsize=20)
        ax2.tick_params(which='both',labelsize=18)
        ax2.grid()
        ax2.set_title("Fluid speed of sound data",fontsize=22)

        plt.show()
        return 
    
    def SaveData(self):
        """Save fluid data in separate files for train, test and validation.
        """

        # Define output files for all, train, test, and validation data.
        full_file = self.GetOutputDir() + "/" + self.GetConcatenationFileHeader() + "_full.csv"
        train_file = self.GetOutputDir() + "/" + self.GetConcatenationFileHeader() + "_train.csv"
        test_file = self.GetOutputDir() + "/" + self.GetConcatenationFileHeader() + "_test.csv"
        val_file = self.GetOutputDir() + "/" + self.GetConcatenationFileHeader() + "_val.csv"

        # Append controlling and training variable data.
        controlling_vars = [EntropicVars.Density, EntropicVars.Energy]
        if self.__mixture:
            entropic_vars=[EntropicVars.s]
        else:
            entropic_vars = [EntropicVars.s, \
                             EntropicVars.dsdrho_e, \
                             EntropicVars.dsde_rho, \
                             EntropicVars.d2sdrho2, \
                             EntropicVars.d2sdedrho, \
                             EntropicVars.d2sde2]
        TD_vars = [EntropicVars.T, EntropicVars.p, EntropicVars.c2]
        secondary_vars = [EntropicVars.dTdrho_e, EntropicVars.dTde_rho, EntropicVars.dpdrho_e, EntropicVars.dpde_rho,\
                          EntropicVars.dhdrho_e, EntropicVars.dhde_rho, EntropicVars.dhdrho_p, EntropicVars.dhdp_rho,\
                          EntropicVars.dsdp_rho, EntropicVars.dsdrho_p,EntropicVars.cp]
        all_vars = controlling_vars + entropic_vars + TD_vars + secondary_vars

        
        CV_data = np.vstack(self.__StateVars_fluid[:, :, [v.value for v in controlling_vars]])
        entropic_data = np.vstack(self.__StateVars_fluid[:, :, [v.value for v in entropic_vars]])
        secondary_data = np.vstack(self.__StateVars_fluid[:, :, [v.value for v in secondary_vars]])
        TD_data = np.vstack(self.__StateVars_fluid[:, :, [v.value for v in TD_vars]])

        full_data = np.hstack((CV_data, entropic_data, TD_data, secondary_data))
        full_data = full_data[self.__success_locations.flatten(), :]
  
        # remove inf values
        full_data = full_data[~np.isinf(full_data).any(axis=1), :]
        full_data = full_data[~np.isnan(full_data).any(axis=1), :]
        # Shuffle data array.
        #np.random.shuffle(full_data)
        full_data_n = full_data.copy()
        np.random.shuffle(full_data_n)
        # print(",".join("%+.16e" % f for f in full_data[0,:]))
        # print(np.shape(full_data))
        # Define number of training and test data points.
        Np_full = np.shape(full_data_n)[0]
        Np_train = int(self.GetTrainFraction()*Np_full)
        Np_test = int(self.GetTestFraction()*Np_full)

        train_data = full_data_n[:Np_train, :]
        test_data = full_data_n[Np_train:Np_train+Np_test, :]
        val_data = full_data_n[Np_train+Np_test:, :]
        # Write output data files.
        with open(full_file,"w+") as fid:
            fid.write(",".join(v.name for v in all_vars) + "\n")
            csvWriter = csv.writer(fid)
            csvWriter.writerows(full_data)
        
        with open(train_file,"w+") as fid:
            fid.write(",".join(v.name for v in all_vars) + "\n")
            csvWriter = csv.writer(fid)
            csvWriter.writerows(train_data)

        with open(test_file,"w+") as fid:
            fid.write(",".join(v.name for v in all_vars) + "\n")
            csvWriter = csv.writer(fid)
            csvWriter.writerows(test_data)

        with open(val_file,"w+") as fid:
            fid.write(",".join(v.name for v in all_vars) + "\n")
            csvWriter = csv.writer(fid)
            csvWriter.writerows(val_data)
            
        return 
    
    def GetStateData(self):
        return self.__StateVars_fluid, self.__success_locations
    
    def GetFluidDataGrid(self):
        return self.__X_grid, self.__Y_grid