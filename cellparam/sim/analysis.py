"""Per-cell time-series statistics used by the sanity checks. Inputs are (frames, cells) arrays."""

import numpy as np


def _dev(x):
    return x - x.mean(axis=0)


def amplitude(areas):
    """Typical size of each cell's area fluctuation (std over time, averaged over cells)."""
    a = areas / areas.mean(axis=1, keepdims=True)
    return float(a.std(axis=0).mean())


def cross_correlation(x, y, max_lag):
    """Cell-averaged normalized correlation of x(t + lag) with y(t); positive peak lag = x lags y."""
    x, y = _dev(x), _dev(y)
    norm = np.sqrt((x ** 2).sum(0) * (y ** 2).sum(0))
    norm[norm == 0] = np.inf
    lags = np.arange(-max_lag, max_lag + 1)
    out = np.empty(lags.size)
    for k, lag in enumerate(lags):
        if lag >= 0:
            c = (x[lag:] * y[:len(y) - lag]).sum(0)
        else:
            c = (x[:len(x) + lag] * y[-lag:]).sum(0)
        out[k] = np.mean(c / norm)
    return lags, out


def autocorrelation(x, max_lag):
    lags, c = cross_correlation(x, x, max_lag)
    return lags[max_lag:], c[max_lag:]


def oscillation_score(x, max_lag):
    """(depth, lag) of the first dip in the autocorrelation; depth < 0 means the signal swings back."""
    lags, c = autocorrelation(x, max_lag)
    dips = np.flatnonzero((c[1:-1] < c[:-2]) & (c[1:-1] <= c[2:])) + 1
    if dips.size == 0:
        return float(c[-1]), int(lags[-1])
    i = dips[0]
    return float(c[i]), int(lags[i])
