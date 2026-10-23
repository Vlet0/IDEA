"""Compatibility shim: the plotting code lives in the ``figures`` package.

``train.py`` imports the two per-run figures from here.  The earlier module also held an
unused gate-sweep plot, orientation heuristics that the canonical Z-up pipeline makes
unnecessary, and a global ``matplotlib.use('Agg')``; all were removed.
"""

from figures.plots import plot_latent_distribution, plot_skeleton_demo

__all__ = ["plot_latent_distribution", "plot_skeleton_demo"]
