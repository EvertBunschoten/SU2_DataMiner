.. _nicfddata:


Generation of Thermodynamic Data
================================

.. contents:: :depth: 2


.. important:: 

    DOCUMENTATION IN PROGRESS


This page provides the documentation on the methods used for generating thermodynamic data with *SU2 DataMiner*. 

.. autofunction:: Data_Generation.DataGenerator_NICFD.DataGenerator_CoolProp.__init__


.. _thermodynamicstates:

Thermodynamic State Calculations
--------------------------------

.. autofunction:: Data_Generation.DataGenerator_NICFD.DataGenerator_CoolProp.PreprocessData

.. autofunction:: Data_Generation.DataGenerator_NICFD.DataGenerator_CoolProp.ComputeData

.. autofunction:: Data_Generation.DataGenerator_NICFD.DataGenerator_CoolProp.SaveData



.. _configoverwrite_nicfd:

Overwiting Configuration Settings
---------------------------------

.. autofunction:: Data_Generation.DataGenerator_NICFD.DataGenerator_CoolProp.UseAutoRange

.. autofunction:: Data_Generation.DataGenerator_NICFD.DataGenerator_CoolProp.SetTemperatureBounds

.. autofunction:: Data_Generation.DataGenerator_NICFD.DataGenerator_CoolProp.SetPressureBounds

.. autofunction:: Data_Generation.DataGenerator_NICFD.DataGenerator_CoolProp.SetDensityBounds

.. autofunction:: Data_Generation.DataGenerator_NICFD.DataGenerator_CoolProp.SetEnergyBounds 

.. autofunction:: Data_Generation.DataGenerator_NICFD.DataGenerator_CoolProp.UpdateConfig