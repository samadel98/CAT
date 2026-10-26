
.. image:: https://github.com/samadel98/data-CAT/actions/workflows/pythonpackage.yml/badge.svg?branch=master
    :target: https://github.com/samadel98/data-CAT/actions/workflows/pythonpackage.yml?query=branch%3Amaster
.. image:: https://github.com/samadel98/CAT/actions/workflows/docs.yml/badge.svg?branch=master
    :target: https://samadel98.github.io/CAT/7_database.html
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
Results are exported to MongoDB with ``Database.update_mongodb()`` and read back as a DataFrame,
filtered by any index or property, with ``Database.query_mongodb()``:

.. code:: python

    from dataCAT import Database

    db = Database('path/to/database', host='localhost', port=27017)
    db.update_mongodb('ligand')

    # All ligands with a cone angle below 120 degrees
    df = db.query_mongodb('ligand', {('cone_angle', 'dist=0.0'): {'$lt': 120}})

Now you are ready to use **Data-CAT**.


.. _conda: https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html
.. _MongoDB: https://www.mongodb.com/
.. _CAT: https://github.com/samadel98/CAT
.. _rdkit: http://www.rdkit.org
.. _h5py: http://www.h5py.org/
