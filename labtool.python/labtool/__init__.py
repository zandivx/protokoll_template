"""A python library for LU Experimentalphysik 2"""

# dunders
__author__ = "Andreas Zach"
__version__ = 0.1

# 3rd party library imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import uncertainties as u
import uncertainties.unumpy as unp

# own library imports
from .src.classes import *
from .src.functions import *
from .src import monkeypatch_uncertainties

# __all__
from .src.classes import __all__ as cls_all
from .src.functions import __all__ as func_all
__all__ = sorted(cls_all + func_all
                 + ["np", "pd", "plt", "u", "unp"])  # type: ignore
del cls_all
del func_all


# apply monkey patches
monkeypatch_uncertainties.Rounding.display()
monkeypatch_uncertainties.Rounding.init()
