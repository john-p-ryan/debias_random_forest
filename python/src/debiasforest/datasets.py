"""Simulated benchmark functions from the paper and the UCI Yacht data set."""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import numpy as np
from sklearn.utils import check_random_state

__all__ = ["simulate", "load_yacht", "DatasetUnavailableError", "FUNCTION_NAMES"]

# Coefficients of Function 3 as printed in the paper (drawn once from U(0.5, 1)).
_F3_COEF = np.array([0.878, 0.642, 0.766, 0.666, 0.933])


def _fig1(X):
    return 3 * X[:, 0] + X[:, 1] ** 2 + np.sin(X[:, 2]) - np.sqrt(X[:, 3])


def _f1(X):  # Friedman 1
    return (10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5) ** 2
            + 10 * X[:, 3] + 5 * X[:, 4])


def _f2(X):
    # As printed in the paper (p. 123): ... + 3 x2 + 2 x4 + x5 (x6..x10 are noise
    # variables).  Note Friedman's (1991) original has 3 x3 instead of 3 x2.
    return (0.1 * np.exp(4 * X[:, 0]) + 4.0 / (1.0 + np.exp(-20 * X[:, 1] + 10))
            + 3 * X[:, 1] + 2 * X[:, 3] + X[:, 4])


def _f3(X):  # linear, x6..x10 are noise variables
    return 10 * (X[:, :5] @ _F3_COEF)


def _f4(X):  # polynomial / non-linear, x5 is a noise variable
    return (10 * X[:, 0] + 0.5 * X[:, 1] ** 2 + 15 * X[:, 1]
            - 25 * np.sqrt(X[:, 2]) + 0.05 * X[:, 3])


def _f5(X):  # non-linear
    return (10 * np.cos(np.pi * X[:, 0] * X[:, 2]) + 10 * X[:, 2] ** 2
            + 10 * (X[:, 0] + X[:, 3]))


# name -> (n_features, function, default sigma)
_FUNCS = {
    "fig1": (4, _fig1, 0.5),
    "f1": (5, _f1, 1.0),
    "f2": (10, _f2, 1.0),
    "f3": (10, _f3, 1.0),
    "f4": (5, _f4, 1.0),
    "f5": (5, _f5, 1.0),
}
_ALIASES = {"friedman1": "f1", "function1": "f1", "function2": "f2", "function3": "f3",
            "function4": "f4", "function5": "f5", "figure1": "fig1"}
FUNCTION_NAMES = tuple(_FUNCS)


def _resolve(name):
    key = str(name).lower().replace(" ", "")
    key = _ALIASES.get(key, key)
    if key not in _FUNCS:
        raise ValueError(f"unknown function {name!r}; choose from {sorted(_FUNCS)} "
                         f"(or 'friedman1')")
    return key


def default_sigma(name):
    """Default noise standard deviation of a design (0.5 for ``fig1``, else 1)."""
    return _FUNCS[_resolve(name)][2]


def simulate(name, n, sigma=None, rng=None):
    """Draw ``n`` rows from one of the paper's simulation designs.

    ``name`` is ``"fig1"`` (4 variables, sigma 0.5), ``"friedman1"``/``"f1"`` (5),
    ``"f2"`` (10), ``"f3"`` (10), ``"f4"`` (5) or ``"f5"`` (5); sigma defaults to 1 except
    for ``fig1``.  All predictors are i.i.d. U(0, 1).  ``rng`` is a seed, ``RandomState``
    or ``None``.  Returns ``(X, y, f)`` with ``y = f + N(0, sigma^2)`` noise.
    """
    key = _resolve(name)
    p, fun, default_sigma = _FUNCS[key]
    sigma = default_sigma if sigma is None else float(sigma)
    rng = check_random_state(rng)
    X = rng.uniform(size=(int(n), p))
    f = fun(X)
    y = f + sigma * rng.standard_normal(int(n))
    return X, y, f


class DatasetUnavailableError(RuntimeError):
    """Raised when a data set can be neither read from the cache nor downloaded."""


_YACHT_URLS = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data",
    "https://raw.githubusercontent.com/yaringal/DropoutUncertaintyExps/master/UCI_Datasets/"
    "yacht/data/data.txt",
)


def _default_cache_dir():
    env = os.environ.get("DEBIASFOREST_DATA")
    return Path(env) if env else Path.home() / ".cache" / "debiasforest"


def _parse_yacht(text):
    rows = [ln.split() for ln in text.strip().splitlines() if ln.strip()]
    arr = np.array([[float(v) for v in r] for r in rows])
    if arr.shape != (308, 7):
        raise ValueError(f"unexpected yacht data shape {arr.shape}")
    return arr


def load_yacht(cache_dir=None, timeout=20):
    """UCI Yacht Hydrodynamics data: ``X`` (308 x 6) and ``y`` (residuary resistance).

    The file is downloaded from the UCI repository (falling back to a GitHub mirror) and
    cached under ``cache_dir`` (default ``$DEBIASFOREST_DATA`` or
    ``~/.cache/debiasforest``).  Raises :class:`DatasetUnavailableError` when it is not
    cached and cannot be downloaded (e.g. offline), so callers can skip gracefully.
    """
    cache = Path(cache_dir) if cache_dir is not None else _default_cache_dir()
    path = cache / "yacht_hydrodynamics.data"
    arr = None
    if path.exists():
        try:
            arr = _parse_yacht(path.read_text())
        except (ValueError, OSError):
            arr = None
    errors = []
    if arr is None:
        for url in _YACHT_URLS:
            try:
                with urllib.request.urlopen(url, timeout=timeout) as resp:
                    text = resp.read().decode("utf-8", "replace")
                arr = _parse_yacht(text)
                cache.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
                break
            except Exception as exc:  # network, HTTP or parse errors
                errors.append(f"{url}: {exc}")
    if arr is None:
        raise DatasetUnavailableError("could not obtain the yacht data set: "
                                      + "; ".join(errors))
    return arr[:, :-1], arr[:, -1]
