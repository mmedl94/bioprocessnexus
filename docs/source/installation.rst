Installation
============

There are two main methods to launch the BioProcessNexus GUI:

Installation via .exe (Windows)
-------------------------------

To launch the BioProcessNexus GUI download and run
`BioProcessNexus.exe <https://drive.boku.ac.at/library/92f9f47a-3e8b-4711-9009-0594d99e2d76/BioProcessNexus/Software>`_.

Installation via PyPI (Linux/macOS)
-----------------------------------

If you want to launch the BioProcessNexus GUI directly via Python you can do so by installing 
from `PyPI <https://pypi.org/project/bioprocessnexus/>`_.

.. code-block:: console

 $ pip install bioprocessnexus

To install all required dependencies you can download the
`requirements.txt <https://github.com/mmedl94/bioprocessnexus/blob/main/docs/requirements.txt>`_ file and run

.. code-block:: console

 $ pip install -r requirements.txt

The GUI can then be launched in Python via

.. code-block:: python

   import bioprocessnexus
   bioprocessnexus.launch_nexus()

