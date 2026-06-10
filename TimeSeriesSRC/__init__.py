"""
TimeSeries Toolbox
==================
A Python toolbox for time series analysis and prediction modeling
based on the classical Box-Jenkins framework.

Supports ARMA, ARIMA, Seasonal ARIMA, ARX, ARMAX, and BJTF
(Box-Jenkins Transfer Function) models with Levenberg-Marquardt
parameter estimation and full model diagnostics.

Main entry points
-----------------
pmodel        : Build a prediction model structure.
estimate      : Fit a model to data via Levenberg-Marquardt.
selpmod       : Automatic model-order selection (AIC / BIC grid search).
uniAnal       : Univariate analysis — ACF, PACF, GPAC.
multiAnal     : Multivariate analysis — impulse response, residual ACF.
uniChi        : Box-Pierce chi-square test on residuals.
multiChi      : Chi-square test on residual / input cross-correlation.
pmoddisp      : Display parameter table and confidence intervals.
pmodsim       : Simulate model output.
sdiff         : Seasonal differencing.

References
----------
Box, G. E. P., Jenkins, G. M., & Reinsel, G. C. (1970).
    *Time Series Analysis: Forecasting and Control*. Holden-Day.
Ljung, L. (1987).
    *System Identification: Theory for the User*. Prentice Hall.
"""

__author__  = "Amir Jafari, Martin Hagan, Lilian S. De Rivera"
__email__   = "ajafari@gwu.edu"
__version__ = "0.1.2"
__license__ = "MIT"

# ── Model classes and estimation ─────────────────────────────────────────────
from .Model.model    import pmodel
from .Model.estimate import estimate
from .Model.selpmod  import func_selpmod  as selpmod
from .Model.pmoddisp import func_pmoddisp as pmoddisp
from .Model.pmodsim  import func_pmodsim  as pmodsim
from .Model.pmodaic  import func_pmodaic  as pmodaic
from .Model.pmodbic  import func_pmodbic  as pmodbic
from .Model.pmodmse  import func_pmodmse  as pmodmse

# ── Analysis and diagnostics ──────────────────────────────────────────────────
from .basefunctions.uniAnal  import func_uniAnal  as uniAnal
from .basefunctions.multiAnal import func_multiAnal as multiAnal
from .basefunctions.uniChi   import func_uniChi   as uniChi
from .basefunctions.multiChi import func_multiChi  as multiChi
from .basefunctions.sdiff    import func_sdiff     as sdiff
from .basefunctions.xcorr    import func_xcorr     as xcorr
from .basefunctions.parcor   import func_parcor    as parcor
from .basefunctions.partoacf import func_partoacf  as partoacf
from .basefunctions.gpac     import func_gpac      as gpac
from .basefunctions.impest   import func_impest    as impest

__all__ = [
    # Model
    "pmodel", "estimate", "selpmod",
    "pmoddisp", "pmodsim",
    "pmodaic", "pmodbic", "pmodmse",
    # Analysis
    "uniAnal", "multiAnal",
    "uniChi",  "multiChi",
    "sdiff", "xcorr", "parcor", "partoacf", "gpac", "impest",
]