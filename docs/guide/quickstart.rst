Quick Start
===========

This page walks through the **four-step Box-Jenkins workflow** using the
classic Series A (chemical-plant concentration) dataset that ships with the
toolbox.

Step 1 — Choose a model class
------------------------------

Examine the raw series and decide on a model family.  For a univariate series
with no external input, start with *ARMA*.

.. code-block:: python

   import pandas as pd
   import pathlib

   import TimeSeriesSRC as ts

   # Load one of the bundled benchmark datasets
   data_dir = pathlib.Path(ts.__file__).parent / "TestData"
   y = pd.read_csv(data_dir / "Series_A_Chemical_Concentration.csv").values.flatten()

Step 2 — Select model order
----------------------------

Use :func:`~TimeSeriesSRC.uniAnal` to compute and plot the ACF, PACF, and
GPAC.  The patterns guide the choice of :math:`n_c` and :math:`n_d`.

.. code-block:: python

   yacf, ypacf, ygpac = ts.uniAnal(y, na=20, nump=10, nrg=5)

A decaying ACF with a cut-off PACF at lag 1 suggests AR(1), i.e., ARMA with
:math:`n_c = 0,\ n_d = 1`.  A decaying PACF with a cut-off ACF at lag 2
suggests MA(2), i.e., ARMA with :math:`n_c = 2,\ n_d = 0`.

Alternatively, let the toolbox search automatically:

.. code-block:: python

   pm_best = ts.selpmod(
       y,
       model_type="arma",
       nc_range=range(0, 4),
       nd_range=range(0, 4),
       criterion="aic",
   )

Step 3 — Estimate parameters
-----------------------------

Build the model structure and call :func:`~TimeSeriesSRC.estimate`:

.. code-block:: python

   pm = ts.pmodel("arma", nc=[2], nd=[1], diff=[0], per=[])
   pm_est, trec, stat = ts.estimate(pm, y)

``stat`` contains:

* ``stat['sigma']``  — residual variance :math:`\hat{\sigma}^2`
* ``stat['stdx']``   — standard deviation of each parameter
* ``stat['perf']``   — MSE at each epoch

Step 4 — Validate the model
----------------------------

Check that residuals are white noise and display confidence intervals:

.. code-block:: python

   # Retrieve one-step-ahead residuals
   e = trec["e"]

   # Chi-square whiteness test
   ts.uniChi(e, na=20, npar=3)

   # Parameter table + error-bar plot
   ts.pmoddisp(pm_est, stat)

   # Pole-zero map (stability check)
   from TimeSeriesSRC.Model.pmoddisp import func_pmodpzplot
   func_pmodpzplot(pm_est)

If any test fails, return to Step 2 and adjust the order.

Transfer function model (BJTF)
-------------------------------

For an input-output dataset (Series J — gas furnace):

.. code-block:: python

   df = pd.read_csv(data_dir / "Series_J_Gas_Furnace.csv")
   u  = df.iloc[:, 0].values.flatten()   # CH4 flow rate
   y  = df.iloc[:, 1].values.flatten()   # CO2 concentration

   # Examine cross-correlation and impulse response
   yacf, uacf, ygpac, ugpac, imp = ts.multiAnal(y, u, na=20, nump=10, nrg=3)

   # Fit a BJTF model: G(q)=B/F, H(q)=C/D, delay=3
   pm = ts.pmodel("bjtf", nb=[2], nc=[1], nd=[1], nf=[2], delay=[3])
   pm_est, trec, stat = ts.estimate(pm, y, u=u)

   # Validate residuals and cross-correlation
   ts.multiChi(trec["e"], u, na=20, npar=6)