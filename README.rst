
.. image:: https://github.com/nlesc-nano/data-CAT/workflows/Build%20with%20Conda/badge.svg
    :target: https://github.com/nlesc-nano/data-CAT/actions?query=workflow%3A%22Build+with+Conda%22
.. image:: https://codecov.io/gh/nlesc-nano/data-CAT/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/nlesc-nano/data-CAT
.. image:: https://badge.fury.io/py/Data-CAT.svg
    :target: https://badge.fury.io/py/Data-CAT

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
Data-CAT
########

Data-CAT is a databasing framework for the Compound Attachment Tools package (CAT_).


Installation
============

**Data-CAT** requires Python 3.10 or later and can be installed with pip:

- **Data-CAT**: ``pip install data-CAT --upgrade``

All dependencies, including rdkit_ and h5py_, are available on PyPI and will be installed automatically.
Alternatively, they can be installed beforehand in a conda_ environment:

- ``conda create --name CAT --channel conda-forge python rdkit h5py``

- The virtual environment can be enabled and disabled by, respectively, typing:

    - Enable: ``conda activate CAT``

    - Disable: ``conda deactivate``

The optional MongoDB_ component of ``dataCAT.Database`` requires a running MongoDB server;
see `mongoconfig.md <mongoconfig.md>`_ for how to run one in a Docker container.

Now you are ready to use **Data-CAT**.


.. _conda: https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html
.. _MongoDB: https://www.mongodb.com/
.. _CAT: https://github.com/nlesc-nano/CAT
.. _rdkit: http://www.rdkit.org
.. _h5py: http://www.h5py.org/
