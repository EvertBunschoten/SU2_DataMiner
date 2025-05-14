#!/usr/bin/env python3
###############################################################################################
#       #      _____ __  _____      ____        __        __  ____                   #        #  
#       #     / ___// / / /__ \    / __ \____ _/ /_____ _/  |/  (_)___  ___  _____   #        #  
#       #     \__ \/ / / /__/ /   / / / / __ `/ __/ __ `/ /|_/ / / __ \/ _ \/ ___/   #        #      
#       #    ___/ / /_/ // __/   / /_/ / /_/ / /_/ /_/ / /  / / / / / /  __/ /       #        #  
#       #   /____/\____//____/  /_____/\__,_/\__/\__,_/_/  /_/_/_/ /_/\___/_/        #        #
#       #                                                                            #        #
###############################################################################################

############################# FILE NAME: consistency_check.py #################################
#=============================================================================================#
# author: Evert Bunschoten                                                                    |
#    :PhD Candidate ,                                                                         |
#    :Flight Power and Propulsion                                                             |
#    :TU Delft,                                                                               |
#    :The Netherlands                                                                         |
#                                                                                             |
#                                                                                             |
# Description:                                                                                |
#  Unit test for verifying the consistency of the equation of state based on entropy potential|
#                                                                                             |  
# Version: 2.0.0                                                                              |
#                                                                                             |
#=============================================================================================#


import numpy as np 
import CoolProp.CoolProp as CP
import CoolProp as CoolP
import matplotlib.pyplot as plt
from CoolProp.CoolProp import get_global_param_string

# Entropic equation of stat is only valid for the gas and supercritical phases.
__accepted_phases:list[int] = [CoolP.iphase_gas, CoolP.iphase_supercritical_gas, CoolP.iphase_supercritical]

def EntropicEOS(rho:float,e:float, s:float, dsdrhoe:list[float], d2sdrho2e2:list[list[float]]):
    """Entropic equation of state used in the SU2 data-driven fluid class and in the PINN training method in
    Manifold_Generation/MLP/Trainers_NICFD/Trainers.py.

    :param rho: fluid density [kg m^-3]
    :type rho: float
    :param e: fluid internal energy [J kg^-1]
    :type e: float
    :param s: fluid entropy [J kg^-1]
    :type s: float
    :param dsdrhoe: Jacobian of fluid entropy w.r.t. density and energy.
    :type dsdrhoe: list[float]
    :param d2sdrho2e2: Hessian of fluid entropy w.r.t. density and energy.
    :type d2sdrho2e2: list[list[float]]
    :return: Primary and derived caloric fluid properties.
    :rtype: dict
    """

    # Retrieve entropy Jacobian and Hessian components
    dsdrho_e = dsdrhoe[0]
    dsde_rho = dsdrhoe[1]
    d2sdrho2 = d2sdrho2e2[0][0]
    d2sdedrho = d2sdrho2e2[0][1]
    d2sde2 = d2sdrho2e2[1][1]


    rho2 = rho*rho
    T = 1 / dsde_rho
    P = -rho2 * T * dsdrho_e
    dTde_rho = -T*T * d2sde2 
    dTdrho_e = -T*T * d2sdedrho 

    dPde_rho = -rho2 * (dTde_rho * dsdrho_e + T * d2sdedrho)
    dPdrho_e = -2 * rho * T * dsdrho_e - rho2 * (dTdrho_e * dsdrho_e + T * d2sdrho2)
    dhdrho_e = -P * (1.0/rho2) + dPdrho_e / rho
    dhde_rho = 1 + dPde_rho / rho

    dhdrho_P = dhdrho_e - dhde_rho * (1 / dPde_rho) * dPdrho_e
    dhdP_rho = dhde_rho * (1 / dPde_rho)
    dsdrho_P = dsdrho_e - dPdrho_e * (1 / dPde_rho) * dsde_rho
    dsdP_rho = dsde_rho / dPde_rho

    drhode_p = -dPde_rho/dPdrho_e
    dTde_p = dTde_rho + dTdrho_e*drhode_p
    dhde_p = dhde_rho + drhode_p*dhdrho_e
    Cp = dhde_p / dTde_p
    
    c2 = dPdrho_e - dsdrho_e * dPde_rho / dsde_rho
    
    state = {"rho" : rho,\
             "e" : e,\
             "T" : T,\
             "p" : P,\
             "c2" : c2,\
             "s" : s,\
             "dsdrho_e" : dsdrho_e,\
             "dsde_rho" : dsde_rho,\
             "d2sdrho2" : d2sdrho2,\
             "d2sdedrho" : d2sdedrho,\
             "d2sde2" :d2sde2,\
             "dTdrho_e" : dTdrho_e, \
             "dTde_rho" : dTde_rho, \
             "dPdrho_e" : dPdrho_e, \
             "dPde_rho" : dPde_rho, \
             "dhdrho_e" : dhdrho_e, \
             "dhde_rho" : dhde_rho, \
             "dhdP_rho" : dhdP_rho, \
             "dhdrho_P" : dhdrho_P, \
             "dsdP_rho" : dsdP_rho, \
             "dsdrho_P" : dsdrho_P, \
             "Cp" : Cp
            }
    return state

def GetFluidState(fluid:CP.AbstractState, rho:float, e:float):
    """Calculate the fluid caloric properties using the entropic equation of state.

    :param fluid: CoolProp abstract state object
    :type fluid: CP.AbstractState
    :param rho: fluid density [kg m^-3]
    :type rho: float
    :param e: fluid static energy [J kg^-1]
    :type e: float
    :return: fluid caloric state
    :rtype: dict
    """
    fluid.update(CP.DmassUmass_INPUTS, rho, e)
    if fluid.phase() in __accepted_phases:
            s = fluid.smass()
            dsde_rho = fluid.first_partial_deriv(CP.iSmass, CP.iUmass, CP.iDmass)
            dsdrho_e = fluid.first_partial_deriv(CP.iSmass, CP.iDmass, CP.iUmass)
            d2sde2 = fluid.second_partial_deriv(CP.iSmass, CP.iUmass, CP.iDmass, CP.iUmass, CP.iDmass)
            d2sdedrho = fluid.second_partial_deriv(CP.iSmass, CP.iUmass, CP.iDmass, CP.iDmass, CP.iUmass)
            d2sdrho2 = fluid.second_partial_deriv(CP.iSmass, CP.iDmass, CP.iUmass, CP.iDmass, CP.iUmass)
            state = EntropicEOS(rho, e, s, [dsdrho_e, dsde_rho], [[d2sdrho2, d2sdedrho],[d2sdedrho, d2sde2]])
            return state 
    else: 
        return 0 
    
# Retrieve all currently supported fluids
fluids = get_global_param_string("FluidsList").split(',')

all_fluids_are_consistent = True
# Perform consistency check for all fluids
for fluid_name in fluids:

    # Use Helmholtz Equation of State
    EoS = "HEOS"

    # Define density and energy ranges within which consistency checks are performed
    Np_X = 100
    Np_Y = 100
    fluid = CP.AbstractState(EoS, fluid_name)
    pmin = CP.PropsSI("PTRIPLE", fluid_name)
    pmax = fluid.pmax()
    Tmin = fluid.Tmin()
    Tmax = fluid.Tmax()
    p_range = np.linspace(pmin, pmax, Np_X)
    T_range = np.linspace(Tmin, Tmax, Np_Y)
    pp, TT = np.meshgrid(p_range, T_range)
    dd = np.zeros(np.shape(pp))
    uu = np.zeros(np.shape(TT))
    for i in range(len(T_range)):
        for j in range(len(p_range)):
            try:
                fluid.update(CP.PT_INPUTS, pp[i,j], TT[i,j])
                if fluid.phase() in __accepted_phases:
                    dd[i,j] = fluid.rhomass()
                    uu[i,j] = fluid.umass()
                else:
                    dd[i,j] = float("nan")
                    uu[i,j] = float("nan")
            except:
                dd[i,j] = float("nan")
                uu[i,j] = float("nan")
    idx_valid = np.invert(np.isnan(dd))

    X_min, X_max = np.min(dd[idx_valid]), np.max(dd[idx_valid])
    Y_min, Y_max = np.min(uu[idx_valid]), np.max(uu[idx_valid])
    __rho_min, __rho_max = X_min, X_max
    __e_min, __e_max = Y_min, Y_max

    # Define step sizes for density and energy used to calculate the derived fluid properties
    # using finite-differences.
    delta_rho_FD = 1e-6*(__rho_max - __rho_min)
    delta_e_FD = 1e-6*(__e_max - __e_min)

    consistent_EoS = True 
    # Check consistency for 500 random density-energy combinations within the established ranges.
    for i in range(500):
        val_rho_test = np.random.rand()*(__rho_max - __rho_min) + __rho_min
        val_e_test = np.random.rand()*(__e_max - __e_min) + __e_min

        # Check if CoolProp converges and check whether fluid state is supported by EEoS.
        try:
            state_base = GetFluidState(fluid, val_rho_test, val_e_test)
            state_e_p = GetFluidState(fluid, val_rho_test, val_e_test + delta_e_FD)
            state_e_m = GetFluidState(fluid, val_rho_test, val_e_test - delta_e_FD)
            state_rho_p = GetFluidState(fluid, val_rho_test + delta_rho_FD, val_e_test)
            state_rho_m = GetFluidState(fluid, val_rho_test - delta_rho_FD, val_e_test)
            if any([state_base == 0, state_e_p ==0, state_e_m == 0, state_rho_m==0, state_rho_p==0]):
                valid_point = False 
            else:
                valid_point = True 
        except:
            valid_point = False


        if valid_point:
            # Retrieve temperature-energy and pressure-energy derivatives from CoolProp, finite-differences,
            # and the entropic equation of state.
            dTde_rho_CoolProp = fluid.first_partial_deriv(CP.iT, CP.iUmass, CP.iDmass)
            dTde_rho_FD = (state_e_p["T"] - state_e_m["T"])/(2*delta_e_FD)
            dTde_rho_EoS = state_base["dTde_rho"]
            dpde_rho_CoolProp = fluid.first_partial_deriv(CP.iP, CP.iUmass, CP.iDmass)
            dpde_rho_FD = (state_e_p["p"] - state_e_m["p"])/(2*delta_e_FD)
            dpde_rho_EoS = state_base["dPde_rho"]

            # Retrieve temperature-density and pressure-density derivatives from CoolProp, finite-differences,
            # and the entropic equation of state.
            dTdrho_e_CoolProp = fluid.first_partial_deriv(CP.iT, CP.iDmass, CP.iUmass)
            dTdrho_e_FD = (state_rho_p["T"] - state_rho_m["T"])/(2*delta_rho_FD)
            dTdrho_e_EoS = state_base["dTdrho_e"]
            dpdrho_e_CoolProp = fluid.first_partial_deriv(CP.iP, CP.iDmass, CP.iUmass)
            dpdrho_e_FD = (state_rho_p["p"] - state_rho_m["p"])/(2*delta_rho_FD)
            dpdrho_e_EoS = state_base["dPdrho_e"]
            
            # Calculate discretization errors between finite-differences and reference data.
            disc_error_dTde_rho = 100*abs(dTde_rho_FD - dTde_rho_CoolProp)/(abs(dTde_rho_CoolProp) + 1e-6)
            disc_error_dTdrho_e = 100*abs(dTdrho_e_FD - dTdrho_e_CoolProp)/(abs(dTdrho_e_CoolProp) + 1e-6)
            disc_error_dpde_rho = 100*abs(dpde_rho_FD - dpde_rho_CoolProp)/(abs(dpde_rho_CoolProp) + 1e-6)
            disc_error_dpdrho_e = 100*abs(dpdrho_e_FD - dpdrho_e_CoolProp)/(abs(dpdrho_e_CoolProp) + 1e-6)
            disc_errors = [disc_error_dTdrho_e, disc_error_dTde_rho, disc_error_dpdrho_e, disc_error_dpde_rho]
            
            # Calculate consistency errors between EEoS and finite-differneces.
            const_error_dTde_rho = 100*abs(dTde_rho_FD - dTde_rho_EoS)/(abs(dTde_rho_FD) + 1e-6)
            const_error_dTdrho_e = 100*abs(dTdrho_e_FD - dTdrho_e_EoS)/(abs(dTdrho_e_FD) + 1e-6)
            const_error_dpde_rho = 100*abs(dpde_rho_FD - dpde_rho_EoS)/(abs(dpde_rho_FD) + 1e-6)
            const_error_dpdrho_e = 100*abs(dpdrho_e_FD - dpdrho_e_EoS)/(abs(dpdrho_e_FD) + 1e-6)
            const_errors = [const_error_dTde_rho, const_error_dTdrho_e, const_error_dpde_rho, const_error_dpde_rho]
            
            # Model is consistent if the consistency error values are lower than the discretization error values.
            consistent = all([c <= d] for c, d in zip(const_errors, disc_errors))
            if not consistent:
                consistent_EoS = False
    
    if not consistent_EoS:
        with open("consistency_check.txt","a+") as fid:
            fid.write("Entropic equation of state is inconsistent for %s\n" % fluid_name)
        print("Entropic equation of state is inconsistent for %s" % fluid_name)      
        all_fluids_are_consistent = False     
if all_fluids_are_consistent:
    with open("consistency_check.txt","a+") as fid:
            fid.write("Consistent!")
    print("Consistent!")
