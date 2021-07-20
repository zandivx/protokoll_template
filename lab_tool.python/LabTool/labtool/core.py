"""
core.py of lab_tool
"""

# 3rd party
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
# from scipy.stats import sem
# from uncertainties import ufloat

# own
from functions import *
from classes import *


# patches
MonkeyPatch.uncertainties().rounding()
