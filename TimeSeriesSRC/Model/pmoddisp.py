import numpy as np
import matplotlib.pyplot as plt


def func_pmoddisp(pmod, stat):
    """Print a parameter table and produce an error-bar plot.

    Displays each estimated parameter with its ±2σ confidence interval in a
    formatted table, then plots the same information as a horizontal error-bar
    chart.

    Parameter ordering follows ``pmod.getmX()``:

    - ``bjtf``  — b, c, d, f  (one block per input / seasonal period)
    - ``armax`` — a, b, c
    - ``arx``   — a, b
    - ``arma``  — c, d
    - ``regr``  — b

    Parameters
    ----------
    pmod : pmodel
        Estimated prediction model as returned by :func:`estimate`.
    stat : dict
        Statistics dictionary returned by :func:`estimate`:

        - ``stat['sigma']`` — residual variance :math:`\\hat{\\sigma}^2`.
        - ``stat['stdx']``  — standard deviation of each parameter (same
          order as ``pmod.getmX()``).

    Examples
    --------
    >>> import pathlib, pandas as pd
    >>> import TimeSeriesSRC as ts
    >>> data_dir = pathlib.Path(ts.__file__).parent / 'TestData'
    >>> y = pd.read_csv(data_dir / 'Series_A_Chemical_Concentration.csv').values.flatten()
    >>> pm = ts.pmodel('arma', nc=[2], nd=[1], diff=[0], per=[])
    >>> pm_est, trec, stat = ts.estimate(pm, y, show_plot=False, show_output=False)
    >>> ts.pmoddisp(pm_est, stat)

    See Also
    --------
    func_pmodpzplot : Pole-zero map for stability/invertibility check.
    estimate        : Returns the ``stat`` dict consumed here.
    """

    X    = np.asarray(pmod.getmX()).ravel()
    stdx = np.asarray(stat['stdx']).ravel()
    sigma = float(stat['sigma'])

    labels = _make_labels(pmod)

    if len(labels) != len(X):
        raise ValueError(
            f'Label count ({len(labels)}) does not match getmX length ({len(X)}). '
            'Check that pmod parameters match the model type.')

    # --- printed table ---
    title = f'Parameter estimates — {pmod.type.upper()} model'
    print(title)
    print('-' * len(title))
    col_w = max(len(lb) for lb in labels)
    hdr = f'  {"Param":<{col_w}}   {"Value":>10}   {"±2σ":>10}   {"95% CI"}'
    print(hdr)
    print('  ' + '-' * (len(hdr) - 2))
    for lb, val, sd in zip(labels, X, stdx):
        lo = val - 2 * sd
        hi = val + 2 * sd
        print(f'  {lb:<{col_w}}   {val:>10.4f}   {2*sd:>10.4f}   ({lo:>10.4f}, {hi:>10.4f})')
    print()
    print(f'  Residual std  σ = {np.sqrt(sigma):.6f}')
    print(f'  Residual var  σ² = {sigma:.6f}')

    # --- error-bar plot ---
    n = len(labels)
    y_pos = np.arange(n)

    fig, ax = plt.subplots(figsize=(7, max(3, 0.45 * n + 1.5)))
    ax.errorbar(X, y_pos, xerr=2 * stdx,
                fmt='o', color='steelblue', ecolor='steelblue',
                capsize=4, capthick=1.5, linewidth=1.5, markersize=5)
    ax.axvline(0, color='k', linewidth=0.8, linestyle='--')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel('Parameter value')
    ax.set_title(f'Parameter estimates ± 2σ  —  {pmod.type.upper()} model')
    ax.grid(axis='x', linestyle=':', linewidth=0.7, alpha=0.7)
    plt.tight_layout()
    plt.show()


def func_pmodpzplot(pmod):
    """Plot poles and zeros of the G and H transfer functions.

    Displays an annotated pole-zero map on the complex plane with the unit
    circle as a stability/invertibility reference.

    Polynomial conventions (all in the forward-shift operator :math:`q`):

    - B polynomial — ``[b0, b1, ..., b_nb]`` (zeros of G)
    - F polynomial — ``[1, f1, ..., f_nf]``, combined with A → poles of G
    - A polynomial — ``[1, a1, ..., a_na]``
    - C polynomial — ``[1, c1, ..., c_nc]`` (zeros of H, regular part)
    - D polynomial — ``[1, d1, ..., d_nd]``, combined with A → poles of H

    For seasonal models (``pmod.period`` non-empty) the H transfer function is
    decomposed into one panel per seasonal component.  For ``arma`` models G is
    omitted.  For ``regr`` (static) models the function prints a notice and
    returns immediately.

    Parameters
    ----------
    pmod : pmodel
        Estimated prediction model (result of :func:`estimate` or manually
        constructed).

    Examples
    --------
    >>> import pathlib, pandas as pd
    >>> import TimeSeriesSRC as ts
    >>> data_dir = pathlib.Path(ts.__file__).parent / 'TestData'
    >>> y = pd.read_csv(data_dir / 'Series_A_Chemical_Concentration.csv').values.flatten()
    >>> pm = ts.pmodel('arma', nc=[2], nd=[1], diff=[0], per=[])
    >>> pm_est, trec, stat = ts.estimate(pm, y, show_plot=False, show_output=False)
    >>> ts.pmodpzplot(pm_est)

    See Also
    --------
    func_pmoddisp : Parameter table and error-bar plot.
    estimate      : Fit model parameters from data.
    """

    def _poly(lst, idx=0, prepend_one=True):
        '''Extract polynomial coefficients from a pmod attribute list.'''
        try:
            arr = np.asarray(lst[idx]).ravel()
        except (IndexError, TypeError, ValueError):
            arr = np.array([])
        if arr.size == 0:
            return np.array([1.0])
        return np.concatenate([[1.0], arr]) if prepend_one else arr

    xtype = pmod.type
    if xtype == 'regr':
        print('pmodpzplot: regr is a static model — no poles or zeros.')
        return

    a_poly = _poly(pmod.a)                    # [1, a1, ...]
    c_poly = _poly(pmod.c)                    # [1, c1, ...] — C[0] regular part
    d_poly = _poly(pmod.d)                    # [1, d1, ...] — D[0] regular part
    f_poly = _poly(pmod.f)                    # [1, f1, ...]
    b_poly = _poly(pmod.b, prepend_one=False) # [b0, b1, ...]

    # G: num = B,  den = F·A
    g_zeros = np.roots(b_poly) if b_poly.size > 1 else np.array([])
    g_den   = np.convolve(f_poly, a_poly)
    g_poles = np.roots(g_den)  if g_den.size  > 1 else np.array([])

    # H regular: num = C[0],  den = D[0]·A
    h_zeros = np.roots(c_poly) if c_poly.size > 1 else np.array([])
    h_den   = np.convolve(d_poly, a_poly)
    h_poles = np.roots(h_den)  if h_den.size  > 1 else np.array([])

    # Seasonal H components — roots in compressed variable w = B^s.
    # Cs(B^s) = 1 + c1*w + c2*w^2 + ... treated as a plain polynomial in w.
    # The unit circle in w-space still correctly separates invertible/non-invertible.
    seasonal_panels = []
    for i, per in enumerate(pmod.period):
        per = int(per)
        if i + 1 < len(pmod.c):
            cs_poly = np.concatenate([[1.0], np.asarray(pmod.c[i + 1]).ravel()])
        else:
            cs_poly = np.array([1.0])
        if i + 1 < len(pmod.d):
            ds_poly = np.concatenate([[1.0], np.asarray(pmod.d[i + 1]).ravel()])
        else:
            ds_poly = np.array([1.0])
        hs_zeros = np.roots(cs_poly) if cs_poly.size > 1 else np.array([])
        hs_poles = np.roots(ds_poly) if ds_poly.size > 1 else np.array([])
        seasonal_panels.append((hs_poles, hs_zeros, f'H — Cs/Ds  (period s = {per})', per))

    has_g = xtype != 'arma'
    n_plots = (1 if has_g else 0) + 1 + len(seasonal_panels)

    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    ax_idx = 0
    if has_g:
        _draw_pz(axes[ax_idx], g_poles, g_zeros, 'G  —  poles: F·A,  zeros: B')
        ax_idx += 1

    h_title = 'H — C/D (regular)' if seasonal_panels else 'H  —  poles: D·A,  zeros: C'
    _draw_pz(axes[ax_idx], h_poles, h_zeros, h_title)
    ax_idx += 1

    for sp, sz, stitle, per in seasonal_panels:
        _draw_pz(axes[ax_idx], sp, sz, stitle)
        axes[ax_idx].set_xlabel(f'Re(w),  w = B^{per}')
        axes[ax_idx].set_ylabel(f'Im(w),  w = B^{per}')
        ax_idx += 1

    fig.suptitle(f'Pole-Zero Map  —  {xtype.upper()} model')
    plt.tight_layout()
    plt.show()


def _draw_pz(ax, poles, zeros, title):
    '''Draw poles (×), zeros (○) and unit circle on ax.'''
    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', linewidth=0.8, alpha=0.55)
    ax.axhline(0, color='k', linewidth=0.4)
    ax.axvline(0, color='k', linewidth=0.4)

    if poles.size > 0:
        ax.plot(poles.real, poles.imag, 'bx', ms=10, mew=2, label='Poles')
    if zeros.size > 0:
        ax.plot(zeros.real, zeros.imag, 'ro', ms=8, mfc='none', mew=2, label='Zeros')

    # axis limits: at least ±1.2, expand to fit all roots
    r_max = 1.2
    if poles.size > 0:
        r_max = max(r_max, np.abs(poles).max() * 1.25)
    if zeros.size > 0:
        r_max = max(r_max, np.abs(zeros).max() * 1.25)
    ax.set_xlim(-r_max, r_max)
    ax.set_ylim(-r_max, r_max)

    ax.set_xlabel('Real')
    ax.set_ylabel('Imaginary')
    ax.set_title(title)
    ax.set_aspect('equal')
    if poles.size > 0 or zeros.size > 0:
        ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, linestyle=':', alpha=0.5)


# ---------------------------------------------------------------------------
# Internal helper — builds parameter name labels in the same order as getmX
# ---------------------------------------------------------------------------

def _make_labels(pmod):

    def _expand(letter, arrays, start=1):
        # start=1 for a/c/d/f (leading 1 is implicit, not stored)
        # start=0 for b (b0 is the first stored coefficient)
        n = len(arrays)
        out = []
        for i, arr in enumerate(arrays):
            arr = np.asarray(arr).ravel()
            if len(arr) == 0:
                continue
            prefix = letter if n == 1 else f'{letter}({i + 1})'
            for k in range(len(arr)):
                out.append(f'{prefix}{k + start}')
        return out

    xtype = pmod.type
    if xtype == 'regr':
        return _expand('b', pmod.b, start=0)
    elif xtype == 'arma':
        return _expand('c', pmod.c) + _expand('d', pmod.d)
    elif xtype == 'arx':
        return _expand('a', [pmod.a[0]]) + _expand('b', pmod.b, start=0)
    elif xtype == 'armax':
        return _expand('a', [pmod.a[0]]) + _expand('b', pmod.b, start=0) + _expand('c', [pmod.c[0]])
    elif xtype == 'bjtf':
        return _expand('b', pmod.b, start=0) + _expand('c', pmod.c) + _expand('d', pmod.d) + _expand('f', pmod.f)
    return []
