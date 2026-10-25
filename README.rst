.. image:: https://github.com/samadel98/CAT/actions/workflows/pythonpackage.yml/badge.svg?branch=master
   :target: https://github.com/samadel98/CAT/actions/workflows/pythonpackage.yml?query=branch%3Amaster
.. image:: https://github.com/samadel98/CAT/actions/workflows/docs.yml/badge.svg?branch=master
   :target: https://samadel98.github.io/CAT/
.. image:: https://zenodo.org/badge/169708827.svg
   :target: https://zenodo.org/badge/latestdoi/169708827
.. image:: https://badge.fury.io/py/nlesc-CAT.svg
   :target: https://badge.fury.io/py/nlesc-CAT

|

.. image:: https://img.shields.io/badge/python-3.10-blue.svg
   :target: https://docs.python.org/3.10/
.. image:: https://img.shields.io/badge/python-3.11-blue.svg
   :target: https://docs.python.org/3.11/
.. image:: https://img.shields.io/badge/python-3.12-blue.svg
   :target: https://docs.python.org/3.12/
.. image:: https://img.shields.io/badge/python-3.13-blue.svg
   :target: https://docs.python.org/3.13/
.. image:: https://img.shields.io/badge/python-3.14-blue.svg
   :target: https://docs.python.org/3.14/

########################
Compound Attachment Tool
########################

**CAT** is a collection of tools designed for the construction of various chemical compounds.
Further information is provided in the documentation_.

Package installation
--------------------
**CAT** requires Python 3.10 or later and can be installed from this repository with pip:

- **CAT**: ``pip install git+https://github.com/samadel98/CAT``

  - ``pip install nlesc-CAT`` installs the last release on PyPI (1.1.0, from 2022),
    which predates the fixes in this repository (*e.g.* NumPy 2, pandas 2 and Python 3.13 support).

Note that, while not strictly necessary, it is recommended to first create a conda environment:

- Download and install miniconda for python3: miniconda_ (also you can install the complete anaconda_ version).

- Create a new virtual environment:  ``conda create --name CAT python``

- Activate the environment:: ``conda activate CAT``

Input files
============

Running **CAT** and can be done with the following command:
``init_cat my_settings.yaml``. The user merely has to provide a yaml_ file
with the job settings, settings which can be tweaked and altered to suit ones
purposes (see example1_). Alternatively, **CAT** can be run like a regular
python script, bypassing the command-line interface
(*i.e.* ``python input.py``, see example2_).

An extensive description of the various available settings is available in
the documentation_.

References
==========
* Belić, J.; van Beek, B.; Menzel, J. P.; Buda, F.; Visscher, L. `Systematic Computational Design and Optimization of Light Absorbing Dyes <https://doi.org/10.1021/acs.jpca.0c04506>`__. *J. Phys. Chem. A* **2020**, *124* (31), 6380–6388.
* van Beek, B.; Zito, J.; Visscher, L.; Infante, I. CAT: A Compound Attachment Tool for the construction of composite chemical compounds. *J. Chem. Inf. Model.* (submitted).

.. _yaml: https://yaml.org/
.. _documentation: https://samadel98.github.io/CAT/
.. _miniconda: http://conda.pydata.org/miniconda.html
.. _anaconda: https://www.continuum.io/downloads
.. _installConda: https://docs.anaconda.com/anaconda/install/
.. _HDF5: http://www.h5py.org/
.. _here: https://www.python.org/downloads/
.. _rdkit: http://www.rdkit.org
.. _PLAMS: https://github.com/SCM-NV/PLAMS
.. _QMFlows: https://github.com/SCM-NV/qmflows
.. _example1: https://github.com/BvB93/CAT/blob/master/examples/input_settings.yaml
.. _example2: https://github.com/BvB93/CAT/blob/master/examples/input.py
