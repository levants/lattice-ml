"""Narrow import shim for Overcomplete 0.3.0 and current Matplotlib."""

import matplotlib
import matplotlib.cm


if not hasattr(matplotlib.cm, 'get_cmap'):
    matplotlib.cm.get_cmap = matplotlib.colormaps.get_cmap
