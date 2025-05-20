DataDrivenConfig Base Class
===========================

*SU2 DataMiner* uses a configuration class in order to store important information regarding the data generation, data mining, and 
manifold generation processes. This page lists some of the important functions of the *Config* class which acts as the base 
class for configurations specific to the application such as NICFD and FGM.

Storage Location and Configuration Information
----------------------------------------------

During the various processes in *SU2 DataMiner*, data are generated, processed, and analyzed. All information regarding these
processes is stored on the current hardware in a user-defined location. *SU2 DataMiner* configurations can be saved locally under 
different names in order to keep track of various data sets and manifolds at once. 
The following functions can be used to manipulate and access the storage location for fluid data and manifolds of the *SU2 DataMiner* configuration 
and save and load configurations.

.. autofunction:: Common.Config_base.Config.SetConfigName 

.. autofunction:: Common.Config_base.Config.GetConfigName 


.. autofunction:: Common.Config_base.Config.SaveConfig 


.. autofunction:: Common.Config_base.Config.SetOutputDir 


.. autofunction:: Common.Config_base.Config.GetOutputDir 


.. autofunction:: Common.Config_base.Config.PrintBanner
    



