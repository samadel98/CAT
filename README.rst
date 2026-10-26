.. image:: https://github.com/samadel98/nano-CAT/actions/workflows/pythonpackage.yml/badge.svg?branch=master
   :target: https://github.com/samadel98/nano-CAT/actions/workflows/pythonpackage.yml?query=branch%3Amaster
.. image:: https://github.com/samadel98/CAT/actions/workflows/docs.yml/badge.svg?branch=master
   :target: https://samadel98.github.io/CAT/
.. image:: https://badge.fury.io/py/nano-CAT.svg
   :target: https://badge.fury.io/py/nano-CAT

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

########
Nano-CAT
########

**Nano-CAT** is a collection of tools for the analysis of nanocrystals,
building on the framework of the Compound Attachment Tools package (CAT_).


Installation
============

- Download miniconda for python3: miniconda_ (also you can install the complete anaconda_ version).

- Install according to: installConda_.

- Create a new virtual environment, for Python 3.10 or later, using the following commands:

  - ``conda create --name CAT -c conda-forge python pip``

- The virtual environment can be enabled and disabled by, respectively, typing:

  - Enable: ``conda activate CAT``

  - Disable: ``conda deactivate``


Package installation
--------------------
Finally, install **Nano-CAT**, together with its CAT_ and Auto-FOX_ dependencies, using pip:

- ``pip install git+https://github.com/samadel98/CAT git+https://github.com/samadel98/auto-FOX git+https://github.com/samadel98/nano-CAT``

  - ``pip install nano-CAT`` installs the last release on PyPI, which (just like the CAT and
    Auto-FOX releases on PyPI) predates the fixes in these repositories (*e.g.* NumPy 2 and pandas 2 support).

  - Like CAT, Nano-CAT works with PLAMS 1.5.1 as well as PLAMS 2024 and later, but installs PLAMS 1.5.1
    by default, as no QMFlows release supports the newer PLAMS versions yet.
    To use a newer PLAMS, install it with uv and CAT's
    `latest-plams-overrides.txt <https://github.com/samadel98/CAT/blob/master/.github/latest-plams-overrides.txt>`_
    (``uv pip install ... --override latest-plams-overrides.txt``), which swaps in the QMFlows fork.
    The conceptual DFT workflow needs PLAMS 1.5.1.

Now you are ready to use **Nano-CAT**.


.. _miniconda: http://conda.pydata.org/miniconda.html
.. _anaconda: https://www.continuum.io/downloads
.. _installConda: https://docs.anaconda.com/anaconda/install/
.. _CAT: https://github.com/samadel98/CAT
.. _Auto-FOX: https://github.com/samadel98/auto-FOX
