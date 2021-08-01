"""
./src/core.py of package 'labtool'
"""

# standard library
# None

# 3rd party
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import numpy
import pandas
from matplotlib import pyplot
from scipy.stats import sem
import uncertainties
from uncertainties import unumpy

# own libraries
from classes import *  # should be relative import: from .classes import *
from functions import *  # to be relative import


# synonyms for libraries
np = numpy
pd = pandas
plt = pyplot
u = uncertainties
unp = unumpy


# monkey patches
MonkeyPatch.uncertainties.rounding()  # type:ignore
