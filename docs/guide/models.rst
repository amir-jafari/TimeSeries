Model Reference
===============

All models share the general structure

.. math::

   y(t) = G(q)\,u(t) + H(q)\,e(t)

where :math:`G(q)` is the transfer function from external input to output,
:math:`H(q)` is the noise model, :math:`q^{-1}` is the backward-shift
operator (:math:`q^{-k}y(t)=y(t-k)`), and :math:`e(t)` is white noise with
variance :math:`\sigma^2`.

A polynomial of order :math:`n` in :math:`q^{-1}` is written

.. math::

   P(q) = 1 + p_1 q^{-1} + p_2 q^{-2} + \cdots + p_n q^{-n}

---

Univariate models (no external input)
--------------------------------------

AR — Autoregressive
~~~~~~~~~~~~~~~~~~~

.. math::

   D(q)\,y(t) = e(t)

The AR(:math:`n_d`) model.  Stationary when all roots of :math:`D(z)=0` lie
outside the unit circle.

.. code-block:: python

   pm = pmodel("arma", nc=[0], nd=[nd])

MA — Moving Average
~~~~~~~~~~~~~~~~~~~~

.. math::

   y(t) = C(q)\,e(t)

The MA(:math:`n_c`) model.  Invertible when all roots of :math:`C(z)=0` lie
outside the unit circle.

.. code-block:: python

   pm = pmodel("arma", nc=[nc], nd=[0])

ARMA — Autoregressive Moving Average
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   D(q)\,y(t) = C(q)\,e(t)

.. code-block:: python

   pm = pmodel("arma", nc=[nc], nd=[nd], diff=[0], per=[])

ARIMA — ARMA with Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apply :math:`d` regular differences :math:`\nabla^d = (1-q^{-1})^d` before
fitting ARMA:

.. math::

   D(q)\,\nabla^d y(t) = C(q)\,e(t)

.. code-block:: python

   pm = pmodel("arma", nc=[nc], nd=[nd], diff=[d], per=[])

Seasonal ARIMA
~~~~~~~~~~~~~~

Combines regular and seasonal differences, each with their own ARMA factors:

.. math::

   D(q)\,D_s(q^{-s})\,\nabla^d\nabla_s^{d_s}\,y(t)
   = C(q)\,C_s(q^{-s})\,e(t)

.. code-block:: python

   # e.g. SARIMA(1,1,1)(1,1,1)_12
   pm = pmodel("arma",
               nc=[1, 1], nd=[1, 1],
               diff=[1, 1], per=[0, 12])

---

Transfer function models (with external input)
-----------------------------------------------

ARX — Autoregressive with eXogenous input
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   A(q)\,y(t) = B(q)\,q^{-k}\,u(t) + e(t)

The noise enters through the same :math:`A(q)` polynomial as the output.

.. code-block:: python

   pm = pmodel("arx", na=[na], nb=[nb], delay=[k])

ARMAX — ARX with Moving-Average noise
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   A(q)\,y(t) = B(q)\,q^{-k}\,u(t) + C(q)\,e(t)

.. code-block:: python

   pm = pmodel("armax", na=[na], nb=[nb], nc=[nc], delay=[k])

BJTF — Box-Jenkins Transfer Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The most general model.  :math:`G` and :math:`H` are identified independently:

.. math::

   y(t) = \frac{B(q)}{F(q)}\,q^{-k}\,u(t)
         + \frac{C(q)}{D(q)}\,e(t)

.. code-block:: python

   pm = pmodel("bjtf", nb=[nb], nc=[nc], nd=[nd], nf=[nf], delay=[k])

Regression
~~~~~~~~~~

Static linear model:

.. math::

   y(t) = b_0 + b_1 u_1(t) + \cdots + b_m u_m(t)

.. code-block:: python

   pm = pmodel("regr", nb=[1]*m)

---

Model order arguments
---------------------

All order arguments are **lists** to support seasonal models with multiple
polynomial blocks.

.. list-table::
   :header-rows: 1
   :widths: 10 45 45

   * - Arg
     - Meaning
     - Example
   * - ``nc``
     - Orders of :math:`C(q)` polynomial(s)
     - ``[2]`` or ``[1, 1]`` (seasonal)
   * - ``nd``
     - Orders of :math:`D(q)` polynomial(s)
     - ``[1]``
   * - ``na``
     - Order of :math:`A(q)` (ARX / ARMAX)
     - ``[2]``
   * - ``nb``
     - Orders of :math:`B(q)` polynomial(s)
     - ``[3]``
   * - ``nf``
     - Orders of :math:`F(q)` (BJTF only)
     - ``[2]``
   * - ``delay``
     - Input delay :math:`k` (samples)
     - ``[3]``
   * - ``diff``
     - Differencing orders
     - ``[1]`` or ``[1, 1]``
   * - ``per``
     - Seasonal periods
     - ``[]`` or ``[0, 12]``

---

Estimation
----------

All models are estimated by minimizing the **sum of squared one-step-ahead
prediction errors** using the **Levenberg-Marquardt** algorithm:

.. math::

   \hat{\theta} = \arg\min_\theta \sum_{t} \hat{e}(t|\theta)^2

See :func:`~TimeSeriesSRC.estimate` for full API details.