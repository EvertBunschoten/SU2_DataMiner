.. _doc_nicfd_tabulation:

.. sectionauthor:: Evert Bunschoten 

Tabulation methods for NICFD applications 
=========================================


This page documents the tabulation methods for NICFD applications in *SU2 DataMiner* 

.. contents:: :depth: 2


.. autofunction:: Manifold_Generation.LUT.LUTGenerators.SU2TableGenerator_NICFD.__init__ 


.. _doc_nicfd_tabulation_refinement:

Refinement settings 
-------------------

.. autofunction:: Manifold_Generation.LUT.LUTGenerators.SU2TableGenerator_NICFD.SetTableDiscretization 

Adaptive table settings:

The following methods can be used to specify the table resolution for adaptive tables.
The input value corresponds to the **length scale** of the coarse and refined cells, relative to the scaled thermodynamic state space.
Therefore, a length scale of 0.1 would result in the look-up table to be populated by approximately 100 elements. 

.. autofunction:: Manifold_Generation.LUT.LUTGenerators.SU2TableGenerator_NICFD.SetCellSize_Coarse
.. autofunction:: Manifold_Generation.LUT.LUTGenerators.SU2TableGenerator_NICFD.SetCellSize_Refined 
.. autofunction:: Manifold_Generation.LUT.LUTGenerators.SU2TableGenerator_NICFD.SetRefinement_Radius 
.. autofunction:: Manifold_Generation.LUT.LUTGenerators.SU2TableGenerator_NICFD.AddRefinementCriterion

.. autofunction:: Manifold_Generation.LUT.LUTGenerators.SU2TableGenerator_NICFD.SetNpDensity
.. autofunction:: Manifold_Generation.LUT.LUTGenerators.SU2TableGenerator_NICFD.SetNpEnergy
.. autofunction:: Manifold_Generation.LUT.LUTGenerators.SU2TableGenerator_NICFD.SetDensityBounds
.. autofunction:: Manifold_Generation.LUT.LUTGenerators.SU2TableGenerator_NICFD.SetEnergyBounds


.. _doc_nicfd_tabulation_table_generation:

Table Generation 
----------------

.. autofunction:: Manifold_Generation.LUT.LUTGenerators.SU2TableGenerator_NICFD.SetTableVars 
.. autofunction:: Manifold_Generation.LUT.LUTGenerators.SU2TableGenerator_NICFD.GenerateTable 

Output of tabulation files 
--------------------------

.. autofunction:: Manifold_Generation.LUT.LUTGenerators.SU2TableGenerator_NICFD.WriteTableFile

.. autofunction:: Manifold_Generation.LUT.LUTGenerators.SU2TableGenerator_NICFD.WriteOutParaview

