.. _base:

SU2 DataMiner Configuration Base Class
======================================
*SU2 DataMiner* uses a configuration class in order to store important information regarding the data generation, data mining, and 
manifold generation processes. This page lists some of the important functions of the *Config* class which acts as the :ref:`base<base>` 
class for configurations specific to the application such as :ref:`NICFD<NICFD>` and FGM :ref:`FGM<FGM>`, for which additional settings can be specified.

.. contents:: :depth: 2


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

The code snippet below shows how to create multiple *SU2 DataMiner* configurations with different settings. Two *SU2 DataMiner* configurations are created titled *test.cfg*
and *test_2.cfg* which are locally stored in binary format. 


.. code-block::

       from su2dataminer.config import Config 

       c = Config()
       c.SetConfigName("test")
       # define settings #
       c.SaveConfig()
    
       c.SetConfigName("test_2")
       # adjust settings #
       c.SaveConfig()


The following functions regard the location where fluid data and other information are stored during the various steps in the *SU2 DataMiner* workflow. The current working directory is used by default and a warning will
show when the specified directory is inaccessible. 

.. important:: 

    *SU2 DataMiner* has been configured for Linux operating systems, so the path separator is currently hard-coded as `/`. Windows compatibility is a work in progress.


.. autofunction:: Common.Config_base.Config.SetOutputDir 

.. autofunction:: Common.Config_base.Config.GetOutputDir 


For training artificial neural networks and generating tables, *SU2 DataMiner* accumulates all generated fluid data into comma-separated ASCII files. The header of these files can be specified with the following functions.


.. autofunction:: Common.Config_base.Config.SetConcatenationFileHeader 

.. autofunction:: Common.Config_base.Config.GetConcatenationFileHeader 


Within the Python API, important settings in the configuration can be displayed in the terminal with the following function, which can also be called from 
the terminal itself by running 

.. code-block::

    >>> DisplayConfig.py --c <CONFIG FILE NAME>


.. autofunction:: Common.Config_base.Config.PrintBanner
    

.. _MACHINELEARNING:


Training Data Sets and Learning Rate
------------------------------------

The data-driven fluid modeling applications of *SU2 DataMiner* involve the use of multi-layer perceptrons to calculate the thermodynamic state of the fluid during flow calculations in SU2. 
The values of the weights and biases, the hidden layer architecture(s) and other parameters needed to train the network can be stored in and retrieved from the *SU2 DataMiner* configuration. 

.. Information about the controlling variables can be accessed through the following functions. For FGM applications, the controlling variables are by default the progress variable and total enthalpy and the mixture GetTestFraction
.. is automatically included when the *multicomponent* transport model is selected for flamelet calculations. 
.. For NICFD applications, the controlling variables are fluid density and static energy by default. 


.. .. autofunction:: Common.Config_base.Config.SetControllingVariables

.. .. autofunction:: Common.Config_base.Config.GetControllingVariables


The fraction of data samples used for training, testing, and validation during machine learning processes can be accessed with the following functions. The training and testing samples are 
selected by shuffling the unique samples in the complete fluid data set and dividing it into the training, testing, and validation data sets, which are locally saved in separate files. 
The fraction of samples reserved for the validation data set is calculated as


.. math::

    f_{\mathrm{validation}} = 1.0 - f_{\mathrm{train}} - f_{\mathrm{test}}


.. autofunction:: Common.Config_base.Config.SetTrainFraction

.. autofunction:: Common.Config_base.Config.GetTrainFraction

.. autofunction:: Common.Config_base.Config.SetTestFraction

.. autofunction:: Common.Config_base.Config.GetTestFraction   


The following functions are used to specify the parameters used for training the networks on the fluid data. Currently, *SU2 DataMiner* uses supervised, gradient-based methods for training where the learning 
rate follows the exponential decay method

.. math:: 

    r_{\mathrm{l}} = r_{\mathrm{l,0}} d^{\frac{i}{N_\mathrm{d}}}


where :math:`r_{\mathrm{l,0}}` is the value of the initial learning rate, :math:`d` the learning rate decay parameter, :math:`i` the iteration, and :math:`N_\mathrm{d}` the number of decay steps.
The number of decay steps is automatically calculated as 

.. math::

    N_\mathrm{d} = 1e-3 N_\mathrm{e}\frac{N_\mathrm{train}}{N_\mathrm{b}}


where :math:`N_\mathrm{e}` is the number of epochs, :math:`N_\mathrm{train}` the number of samples in the training data set, and :math:`N_\mathrm{b}` the number of samples in each training batch.


The value of the initial learning rate is calculated as an exponent with base 10 using

.. math:: 

    r_{\mathrm{l,0}} = 10^{\alpha} 


and the value of :math:`\alpha` can be accessed and specified through the following functions


.. autofunction:: Common.Config_base.Config.SetAlphaExpo

.. autofunction:: Common.Config_base.Config.GetAlphaExpo


The value of the learning rate decay parameter :math:`d` is accessed through 


.. autofunction:: Common.Config_base.Config.SetLRDecay

.. autofunction:: Common.Config_base.Config.GetLRDecay
   

The networks are trained using batches of training data. The number of samples in each training batch are calculated with

.. math::

    N_\mathrm{b} = 2^b 


and the exponent :math:`b` can be specified through 

.. autofunction:: Common.Config_base.Config.SetBatchExpo

.. autofunction:: Common.Config_base.Config.GetBatchExpo


Multi-Layer Perceptrons 
-----------------------

The following functions regard the specification of multi-layer perceptrons (MLP) within *SU2 DataMiner*. The number of nodes in the *hidden layers* of the network can be specified with

.. autofunction:: Common.Config_base.Config.SetHiddenLayerArchitecture

.. autofunction:: Common.Config_base.Config.GetHiddenLayerArchitecture

Currently, *SU2 DataMiner* supports the use of a single activation function for the nodes in all hidden layers of the network, while a linear function is automatically applied to the input and output layer. 
The activation functions currently supported by *SU2 DataMiner* are: 

1. "linear" (linear function) :math:`y = x`
2. "elu" (:ref:`exponential linear unit <eluarticle>`) :math:`y=\begin{cases}x,\quad x>0\\e^x\quad x\leq 0\end{cases}`
3. "relu" (:ref:`rectified linear unit <reluarticle>`) :math:`y=\begin{cases}x,\quad x>0\\0\quad x\leq 0\end{cases}`
4. "tanh" (hyperbolic tangent) :math:`y=\tan^{-1}(x)`
5. "exponential" :math:`y=e^x`
6. "gelu" (:ref:`Gaussian error linear unit <geluarticle>`) :math:`y=\frac{x}{2}\left(1 + \mathrm{erf}\left(\frac{x}{\sqrt{2}}\right)\right)`
7. "sigmoid" :math:`y=\frac{e^{x}}{1 + e^{x}}`
8. "swish" (:ref:`sigmoid linear unit <geluarticle>`) :math:`y=x\frac{e^{x}}{1 + e^{x}}`


.. autofunction:: Common.Config_base.Config.SetActivationFunction

.. autofunction:: Common.Config_base.Config.GetActivationFunction


The weights and bias values are stored in the *SU2 DataMiner* configuration after training and they can also be manually accessed through the following functions

.. autofunction:: Common.Config_base.Config.SetWeights 

.. autofunction:: Common.Config_base.Config.SetBiases

.. autofunction:: Common.Config_base.Config.GetWeightsBiases


All the settings regarding the training parameters, architecture, weights, and biases of the trained network can be stored automatically after training a network from the :ref:`Trainer` class 


.. autofunction:: Common.Config_base.Config.UpdateMLPHyperParams


To use the network stored in the configuration in SU2 simulations, the network information needs to be written to an ASCII file such that it can be loaded in SU2 through the `MLPCpp`_ module. 
All relevant information about the network is automatically written to a properly formatted ASCII file using 


.. autofunction:: Common.Config_base.Config.WriteSU2MLP



.. _MLPCpp : https://github.com/EvertBunschoten/MLPCpp.git 