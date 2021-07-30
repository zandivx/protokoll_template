"core.py of lab_tool"


# 3rd party libraries
import numpy
import pandas
import matplotlib.pyplot as pyplot
import uncertainties
import uncertainties.unumpy as unumpy
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.stats import sem


# own libraries
from classes import *
from functions import *


# synonyms for libraries
np = numpy
pd = pandas
plt = pyplot
u = uncertainties
unp = unumpy


# monkey patches
MonkeyPatch.uncertainties.rounding()
