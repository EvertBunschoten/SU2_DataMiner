.. sectionauthor:: Evert Bunschoten


.. _NICFD:

SU2 DataMiner Configuration for NICFD
=====================================

.. contents:: :depth: 2

    
The following section describes the most important SU2 DataMiner configuration class functions for NICFD applications. 
*SU2 DataMiner* uses *CoolProp* abstract state to calculate thermodynamic states. The equation of state used by *CoolProp* can be set to 
the Helmholtz equation of state ("HEOS") or RefProp ("REFPROP") if a local installation of RefProp is detected. A warning is raised if the equation of state model 
is not recognized by *CoolProp*.

Equation of State and Fluid 
---------------------------


.. autofunction:: Common.DataDrivenConfig.Config_NICFD.SetEquationOfState 


The fluid for which thermodynamic states are generated is specified with the following function. The available list of fluids supported by *CoolProp* can be 
retrieved by running the following code snippet:

::

    from CoolProp.CoolProp import get_global_param_string
    fluids = get_global_param_string("FluidsList").split(',')
    print(fluids)


.. note::

    Support for mixtures of fluids has not yet been validated. 


.. autofunction:: Common.DataDrivenConfig.Config_NICFD.SetFluid

.. _nicfd_data_limits: 

Range of the Thermodynamic State Data 
-------------------------------------


The following functions are used for setting the limits within which thermodynamic states are generated. 
By default, the limits of the fluid data are automatically determined based on the temperature and pressure limits supported by *CoolProp*.

.. autofunction:: Common.DataDrivenConfig.Config_NICFD.UseAutoRange 


The fluid data can be generated on a pressure-termperature grid or a density-static energy grid.


.. autofunction:: Common.DataDrivenConfig.Config_NICFD.UsePTGrid


.. autofunction:: Common.DataDrivenConfig.Config_NICFD.GetPTGrid


The limits of the fluid data grid are set with the following functions 


.. autofunction:: Common.DataDrivenConfig.Config_NICFD.SetTemperatureBounds 


.. autofunction:: Common.DataDrivenConfig.Config_NICFD.SetPressureBounds 


.. autofunction:: Common.DataDrivenConfig.Config_NICFD.SetEnergyBounds 


.. autofunction:: Common.DataDrivenConfig.Config_NICFD.SetDensityBounds


Phases and transport properties 
-------------------------------

Thermophysical state data can be generated at various phases. The following functions can be used to specify in which phases thermophysical state data are included in the data generation process.
*SU2 DataMiner* currently only supports the inclusion of fluid data in the gaseous, liquid, two-phase, supercritical, supercritical gas, and supercritical liquid phases. 
Information about the phase data can be accessed with the following functions.

.. autofunction:: Common.DataDrivenConfig.Config_NICFD.EnableGasPhase 

.. autofunction:: Common.DataDrivenConfig.Config_NICFD.EnableLiquidPhase 

.. autofunction:: Common.DataDrivenConfig.Config_NICFD.EnableSuperCritical 

.. autofunction:: Common.DataDrivenConfig.Config_NICFD.GasPhase 

.. autofunction:: Common.DataDrivenConfig.Config_NICFD.LiquidPhase 

.. autofunction:: Common.DataDrivenConfig.Config_NICFD.SuperCritical 

Transport properties can also be included in the fluid data set. Currently, these are limited to the fluid *thermal conductivity*, *dynamic viscosity*, and *vapor quality*. 
Information on whether transport properties are included in the fluid data set can be accessed with the following functions: 

.. autofunction:: Common.DataDrivenConfig.Config_NICFD.IncludeTransportProperties

.. autofunction:: Common.DataDrivenConfig.Config_NICFD.CalcTransportProperties


.. _resolutionsettings:

Resolution 
----------


Fluid data are generated on a 2D grid (density-energy or pressure-temperature), for which the number of samples along each direction is set with the following functions 
A linear distribution is used for the temperature and static energy dimension of the data grid, while a cosine distribution is used for the density or pressure dimension to increase the density of the data sets in dilute conditions.
The following distribution was used 


.. math::

    x_i = (x_{\mathrm{min}} - x_{\mathrm{max}}) * \cos(\theta_i) + x_{\mathrm{max}}\quad 0\leq \theta_i \leq \frac{1}{2}\pi,


where :math:`x_{\mathrm{min}}` and :math:`x_{\mathrm{max}}` are the lower and upper bound values.


.. autofunction:: Common.DataDrivenConfig.Config_NICFD.SetNpTemp 


.. autofunction:: Common.DataDrivenConfig.Config_NICFD.SetNpPressure 


.. autofunction:: Common.DataDrivenConfig.Config_NICFD.SetNpEnergy 


.. autofunction:: Common.DataDrivenConfig.Config_NICFD.SetNpDensity



Machine Learning 
----------------

*SU2 DataMiner* can be used to train networks according to the EEoS-PINN method documented in :ref:`Bunschoten et al. (2026) <eeos_article>`. The training loss function is the fitting loss of a set of thermodynamic variables.
By default, these are entropy, pressure, temperature, and the speed of sound, but a custom set can be accessed with the following function.


.. autofunction:: Common.DataDrivenConfig.Config_NICFD.SetStateVars 


.. autofunction:: Common.DataDrivenConfig.Config_NICFD.GetStateVars 




Tabulation 
----------

In addition to the EEoS-PINN method, thermophysical state information can also be stored in look-up tables. 
The thermodynamic state space can be discretized with a *Cartesian* method or with an *adaptive* method. 
With the Cartesian method, the thermodynamic state space is divided into equally sized, rectangular cells in the density-static energy space.
The number of cells along the density and energy axes are specified with the :ref:`setters <resolutionsettings>` for thermodynamic state space resolution.
With the adaptive method, the thermodynamic state space is divided into triangular elements and can be locally refined in areas of interest such as around the saturation curve.
For both methods, the limits of the table can be specified with the :ref:`setters <nicfd_data_limits>` for the density and static energy ranges. 

Information on the table discretization method can be accessed with the following functions:

.. autofunction:: Common.DataDrivenConfig.Config_NICFD.SetTableDiscretization

.. autofunction:: Common.DataDrivenConfig.Config_NICFD.GetTableDiscretization


References
----------

.. _MLPCpp : https://github.com/EvertBunschoten/MLPCpp.git 

.. _eeos_article:

E.Bunschoten, A.Cappiello, M.Pini "Data-driven regression of thermodynamic models in entropic form using physics-informed machine learning". In: Computers and Fluids 306 (2026), DOI:https://doi.org/10.1016/j.compfluid.2025.106932