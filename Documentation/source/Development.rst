.. _developerguide:

Developers Guide
================

*SU2 DataMiner* is an open-source software library, meaning that anyone can download, use, and contribute for free. 
This page contains instructions on how to contribute to the development of the code. 

.. important::

    Documentation in development 


Gitting Started, Opening a Pull Request
---------------------------------------


Clone and fork repository

Checkout develop branch 

Create new branch for your feature 

Write your code 

Publish branch, push changes 

Open a pull request on the GitHub page, review PR checklist 


Regression Tests 
----------------

When applicable, add a regression test 


Documentation
-------------

When adding new functionalities and/or tutorials, it is important to include these in the source code documentation. 
The *SU2 DataMiner* documentation is powered by `sphinx <https://www.sphinx-doc.org/en/master/index.html>`_. 

The official documentation is updated only when the ```main``` branch is updated. To update the documentation and inspect your changes, you can compile the documentation locally with the following command:


.. code-block::

    >>> cd Documentation 
    >>> sphinx-build -b html source ../_site 


This will compile the *SU2 DataMiner* and build the file ``_site/index.html``. By opening this file in your browser, you can inspect the documentation as it will appear after deployment. 


