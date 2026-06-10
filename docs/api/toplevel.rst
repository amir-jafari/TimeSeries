Top-level API
=============

All public symbols are importable directly from ``TimeSeriesSRC``:

.. code-block:: python

   import TimeSeriesSRC as ts
   ts.pmodel(...)
   ts.estimate(...)

.. currentmodule:: TimeSeriesSRC

Model building
--------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   pmodel
   estimate
   selpmod

Model assessment
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   pmoddisp
   pmodsim
   pmodaic
   pmodbic
   pmodmse

Time series analysis
--------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   uniAnal
   multiAnal
   uniChi
   multiChi

Utilities
---------

.. autosummary::
   :toctree: generated
   :nosignatures:

   sdiff
   xcorr
   parcor
   partoacf
   gpac
   impest