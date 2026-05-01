.. _flameletdata:

Generation of Flamelet Data
===========================

.. contents:: :depth: 2

.. important:: 

    DOCUMENTATION IN PROGRESS

This page provides the documentation on the methods used for generating flamelet data with *SU2 DataMiner*. 


.. _flameletgen:

Flamelet Calculations
---------------------

.. autofunction:: Data_Generation.DataGenerator_FGM.DataGenerator_Cantera.ComputeFreeFlames


.. autofunction:: Data_Generation.DataGenerator_FGM.DataGenerator_Cantera.ComputeBurnerFlames


.. autofunction:: Data_Generation.DataGenerator_FGM.DataGenerator_Cantera.ComputeEquilibrium


.. autofunction:: Data_Generation.DataGenerator_FGM.DataGenerator_Cantera.ComputeFlameletsOnMixStatus


.. autofunction:: Data_Generation.DataGenerator_FGM.DataGenerator_Cantera.ComputeFlamelets


.. autofunction:: Data_Generation.DataGenerator_FGM.ComputeFlameletData


.. autofunction:: Data_Generation.DataGenerator_FGM.ComputeBoundaryData


.. _configoverwrite:

Overwiting Configuration Settings
---------------------------------


.. autofunction:: Data_Generation.DataGenerator_FGM.DataGenerator_Cantera.SetOutputDir

.. autofunction:: Data_Generation.DataGenerator_FGM.DataGenerator_Cantera.SetTransportModel 

.. autofunction:: Data_Generation.DataGenerator_FGM.DataGenerator_Cantera.SetReactionMechanism

.. autofunction:: Data_Generation.DataGenerator_FGM.DataGenerator_Cantera.RunFreeFlames

.. autofunction:: Data_Generation.DataGenerator_FGM.DataGenerator_Cantera.RunBurnerFlames 

.. autofunction:: Data_Generation.DataGenerator_FGM.DataGenerator_Cantera.RunEquilibrium

.. autofunction:: Data_Generation.DataGenerator_FGM.DataGenerator_Cantera.RunMixtureFraction

.. autofunction:: Data_Generation.DataGenerator_FGM.DataGenerator_Cantera.RunEquivalenceRatio

.. autofunction:: Data_Generation.DataGenerator_FGM.DataGenerator_Cantera.SetFuelDefinition

.. autofunction:: Data_Generation.DataGenerator_FGM.DataGenerator_Cantera.SetOxidizerDefinition