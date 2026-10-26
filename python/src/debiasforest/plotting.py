"""Diagnostic plot: predicted versus observed (or true) values with a 45 degree line."""
from __future__ import annotations

import numpy as np

__all__ = ["bias_plot"]


def bias_plot(reference, predictions, ax=None, xlabel="observed", ylabel="predicted",
              title=None, alpha=0.6, s=10):
    """Scatter of predictions against ``reference`` for one or several methods.

    Parameters
    ----------
    reference : observed responses (or true function values) on the x axis.
    predictions : array-like, or dict ``label -> array`` (several methods share the axes,
        each with its own colour and legend entry).
    ax : optional matplotlib axes; a new figure is created otherwise.

    Points above (below) the red 45 degree line are over- (under-) predicted; systematic
    curvature around the line indicates bias that a correction can remove.  matplotlib is
    imported lazily, so the rest of the package does not require it.  Returns the axes.
    """
    import matplotlib.pyplot as plt

    ref = np.asarray(reference, dtype=float)
    if not isinstance(predictions, dict):
        predictions = {None: predictions}
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    allv = [ref]
    for lab, pred in predictions.items():
        pred = np.asarray(pred, dtype=float)
        ax.scatter(ref, pred, s=s, alpha=alpha, label=lab)
        allv.append(pred)
    lo = min(float(np.min(v)) for v in allv)
    hi = max(float(np.max(v)) for v in allv)
    ax.plot([lo, hi], [lo, hi], color="red", lw=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if any(k is not None for k in predictions):
        ax.legend()
    return ax
