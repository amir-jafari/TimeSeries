Installation
============

Requirements
------------

* Python 3.8 or later
* NumPy ≥ 1.19
* SciPy ≥ 1.5
* Matplotlib ≥ 3.3

Installing from PyPI
--------------------

.. code-block:: bash

   pip install timeseries-toolbox

This installs the ``TimeSeriesSRC`` package and all runtime dependencies.

Installing from source
----------------------

.. code-block:: bash

   git clone https://github.com/amir-jafari/TimeSeries.git
   cd TimeSeries
   pip install -e .

The ``-e`` flag installs in *editable* mode so any changes to the source are
immediately reflected without reinstalling.

Optional dependencies
---------------------

For running the example Jupyter notebooks:

.. code-block:: bash

   pip install timeseries-toolbox pandas notebook

For building the documentation locally:

.. code-block:: bash

   pip install timeseries-toolbox[docs]
   cd docs
   make html

For running the test suite:

.. code-block:: bash

   pip install timeseries-toolbox[dev]
   pytest

Verifying the installation
--------------------------

.. code-block:: python

   import TimeSeriesSRC
   print(TimeSeriesSRC.__version__)   # → 0.1.0