"""
A python library for LU Experimentalphysik 2
"""

# standard library imports
# currently none

# 3rd party library imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import sem
import uncertainties as u
import uncertainties.unumpy as unp

# own library imports
from .src.__init__ import *


# monkey patch
monkeypatch.uncertainties.Rounding.display()  # type: ignore[name-defined]
monkeypatch.uncertainties.Rounding.init()     # type: ignore[name-defined]
