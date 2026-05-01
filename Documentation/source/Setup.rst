.. _label_setup:

Set-up and Requirements
=======================

*SU2 DataMiner* is an open-source, python-based software suite that is available on `GitHub <https://github.com/su2code/SU2_DataMiner.git>`_. 

.. important::

    *SU2 DataMiner* was developed for Linux OS. Enabling compatibility with Windows and Mac OS is an ongoing development.


Requirements
------------

*SU2 DataMiner* is currently only compatible with Linux-based operating systems. The following programs are required to use *SU2 DataMiner*:

* python 3.11 or higher
* git 
* pip 
* build-essential

Recommended:

* python3-virtualenv

Downloading the source code 
---------------------------

The *SU2 DataMiner* source code can be downloaded from GitHub by cloning the repository


.. code-block::

    >>> git clone https://github.com/su2code/SU2_DataMiner.git <PATH_TO_SOURCE>


where `<PATH_TO_SOURCE>` refers to the target location where the *SU2 DataMiner* source code will be stored. 

For python to have access to the various modules in *SU2 DataMiner*, it is recommended to update the `PATH` and `PYTHONPATH` environment
variables by writing the following lines in your `.bashrc` file:

.. code-block::

    export SU2_DM_HOME=<PATH_TO_SOURCE>
    export PYTHONPATH=$PYTHONPATH:$SU2_DM_HOME
    export PATH=$PATH:$SU2_DM_HOME/bin

After updating the environment variables by opening a new terminal or running 

.. code-block::
    
    >>> source .bashrc 

your system will be able to access the *SU2 DataMiner* python modules.


Creating the python environment
-------------------------------

*SU2 DataMiner* needs a number of python modules to function. These modules are listed in `required_packages.txt`. You can 
install these manually, but it is recommended to create a separate python environment containing all the necessary packages. 
To create the python environment named `su2dm_venv` with all the required python modules, run the following commands in your terminal:

.. code-block:: 

    >>> cd ~
    >>> python3 -m venv su2dm_venv
    >>> virtualenv -p python3 su2dm_venv
    >>> . su2dm_venv/bin/activate 
    >>> python3 -m pip install -r $SU2DATAMINER_HOME/required_packages.txt > pip_install_log.log


By adding the following line to your `.bashrc` file, the *SU2 DataMiner* python environment is automatically activated when opening a new terminal.

.. code-block::

    . ~/su2dm_env/bin/activate
