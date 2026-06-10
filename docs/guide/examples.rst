Examples
========

Four end-to-end worked examples ship with the toolbox.  Each is available as
both a Python script (VS Code / Spyder ``# %%`` cell format) and a Jupyter
notebook.

Running the examples
--------------------

.. code-block:: bash

   cd TimeSeriesSRC/Examples/PyFiles
   python 01_ARMA_Model.py

Or open the notebooks in ``TimeSeriesSRC/Examples/NoteBooks/``.

Example 1 — ARMA model (Series A)
----------------------------------

**Dataset**: Chemical-plant concentration, 197 observations.

**Goal**: Identify the best ARMA(:math:`n_c, n_d`) order and fit the model.

**Script**: ``Examples/PyFiles/01_ARMA_Model.py``

**Notebook**: ``Examples/NoteBooks/01_ARMA_Model.ipynb``

Key steps::

   y = Series_A_Chemical_Concentration   (197 obs)

   uniAnal(y)          # ACF, PACF, GPAC → order hints
   selpmod(...)        # grid search → best AIC/BIC order
   estimate(pm, y)     # LM estimation
   uniChi(e, ...)      # residual whiteness test
   pmoddisp(pm, stat)  # confidence intervals

Example 2 — ARIMA model (Series C)
------------------------------------

**Dataset**: Chemical-plant temperature, 226 observations (non-stationary trend).

**Goal**: Apply regular differencing to achieve stationarity, then fit ARMA.

**Script**: ``Examples/PyFiles/02_ARIMA_Model.py``

**Notebook**: ``Examples/NoteBooks/02_ARIMA_Model.ipynb``

Key steps::

   y = Series_C_Chemical_Temperature   (226 obs)

   uniAnal(y, diff=[1])          # differenced series analysis
   pmodel(..., diff=[1])         # ARIMA structure
   estimate(pm, y)

Example 3 — Seasonal ARIMA (Series G)
---------------------------------------

**Dataset**: International airline passengers, 144 monthly observations.

**Goal**: Identify and fit a seasonal ARIMA model with period :math:`s = 12`.

**Script**: ``Examples/PyFiles/03_Seasonal_ARIMA_Model.py``

**Notebook**: ``Examples/NoteBooks/03_Seasonal_ARIMA_Model.ipynb``

Key steps::

   y = Series_G_Airline_Passengers   (144 monthly obs, s=12)

   uniAnal(y, diff=[1,1], per=[0,12])   # seasonal differencing
   pmodel(..., diff=[1,1], per=[0,12])  # seasonal ARIMA structure
   estimate(pm, y)

Example 4 — BJTF / ARMAX / ARX (Series J)
-------------------------------------------

**Dataset**: Box-Jenkins gas-furnace, 296 observations (input: CH4 flow, output: CO2).

**Goal**: Compare ARX, ARMAX, and BJTF model classes on the same dataset.

**Script**: ``Examples/PyFiles/04_BJTF_Model.py``

**Notebook**: ``Examples/NoteBooks/04_BJTF_Model.ipynb``

Key steps::

   y, u = Series_J_Gas_Furnace   (296 obs, MISO)

   multiAnal(y, u)                     # impulse response, cross-correlation
   pmodel("arx",   na=[2],nb=[2], delay=[3])
   pmodel("armax", na=[2],nb=[2],nc=[1], delay=[3])
   pmodel("bjtf",  nb=[2],nc=[1],nd=[1],nf=[2], delay=[3])
   estimate(pm, y, u=u)
   multiChi(e, u)                      # residual/input independence test

Benchmark datasets
------------------

All datasets are in ``TimeSeriesSRC/TestData/`` and can be loaded with:

.. code-block:: python

   import pathlib, pandas as pd
   import TimeSeriesSRC as ts

   data_dir = pathlib.Path(ts.__file__).parent / "TestData"
   y = pd.read_csv(data_dir / "Series_A_Chemical_Concentration.csv").values.flatten()

.. list-table::
   :header-rows: 1
   :widths: 20 40 15 10

   * - File
     - Description
     - Length
     - Type
   * - Series_A_Chemical_Concentration.csv
     - Chemical process concentration
     - 197
     - Univariate
   * - Series_B_IBM_Stock.csv
     - IBM weekly stock price
     - 369
     - Univariate
   * - Series_C_Chemical_Temperature.csv
     - Chemical process temperature
     - 226
     - Univariate
   * - Series_D_Chemical_Viscosity.csv
     - Chemical process viscosity
     - —
     - Univariate
   * - Series_E_Sunspot_Numbers.csv
     - Annual sunspot numbers
     - —
     - Univariate
   * - Series_F_Chemical_Yields.csv
     - Chemical process yields
     - —
     - Univariate
   * - Series_G_Airline_Passengers.csv
     - Monthly airline passengers
     - 144
     - Univariate (seasonal)
   * - Series_J_Gas_Furnace.csv
     - Gas furnace (CH4 in, CO2 out)
     - 296
     - Bivariate