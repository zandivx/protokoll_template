"""labtool/src/classes.py"""

# dunders
__author__ = "Andreas Zach"
__all__ = ["CDContxt", "SysPathContxt", "CurveFit", "Interpolate", "Student"]

# std library
import os
import sys
from typing import Callable, Union

# 3rd party
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.stats import sem
from uncertainties import ufloat
from uncertainties.unumpy import uarray

# own
from .functions import cd


class CDContxt:
    """A context manager that changes the working directory temporarily to the directory
    of the calling script or to an optional path.
    """

    def __init__(self, path: str = ""):
        self.path = path

    def __enter__(self):
        self.olddir = os.getcwd()

        if self.path != "":
            os.chdir(self.path)
        else:
            cd()

    def __exit__(self, type, value, traceback):
        os.chdir(self.olddir)


class SysPathContxt:
    """A context manager that appends the sys.path-variable with a given init-path.
    Useful for relative imports in a package, where the importing module should be run
    as a script.

    """

    def __init__(self, path: str):
        self.path = path

    def __enter__(self):
        sys.path.append(self.path)

    def __exit__(self, *args):
        sys.path.pop()


class AbstractFit:
    """An abstract class as superclass for other fit-like classes.
    Currently for:
    -> CurveFit
    -> Interpolate
    """

    def __init__(self,
                 x_in: ArrayLike,
                 y_in: ArrayLike,
                 f: Callable,
                 divisions: int = 0):

        self.f = f
        self.x_in = x_in
        self.y_in = y_in

        a = np.min(x_in)
        b = np.max(x_in)
        c = divisions if divisions > 0 else int(20 * (b - a))

        self.x_out = np.linspace(a, b, c)
        self.y_out = f(self.x_out)

    def plot(self,
             style_in: str = "-",
             style_out: str = "-",
             label_in: str = "Data in",
             label_out: str = "Data out",
             title: str = "",
             xlabel: str = "",
             ylabel: str = "",
             xlim: Union[bool, tuple] = False,
             ylim: Union[bool, tuple] = False,
             grid: bool = False,
             plot: bool = True
             ) -> None:

        # make sure to start with an empty figure
        plt.clf()

        plt.plot(self.x_in, self.y_in, style_in, label=label_in)
        plt.plot(self.x_out, self.y_out, style_out, label=label_out)
        plt.title(title)
        plt.legend()

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        if xlim:
            plt.xlim(xlim)
        if ylim:
            plt.ylim(ylim)
        if grid:
            plt.grid()

        # pass plot=None if other plt.functions() are needed,
        # plt.show() afterwards
        if plot:
            plt.show()

        return None


class CurveFit(AbstractFit):
    """A class for fits with scipy.optimize.curve_fit"""

    def __init__(self,
                 function: Callable,
                 x: ArrayLike,
                 y: ArrayLike,
                 p0: Union[ArrayLike, None] = None,
                 sigma: Union[ArrayLike, None] = None,
                 bounds: tuple[ArrayLike, ArrayLike] = ((-np.inf,), (np.inf,)),
                 divisions: int = 0):

        self.p, pcov, *_ = curve_fit(function, x, y, p0=p0, sigma=sigma,
                                     absolute_sigma=True, bounds=bounds)
        self.u = np.sqrt(np.diag(pcov))
        super().__init__(x, y, lambda x: function(x, *self.p), divisions=divisions)

    def __str__(self):
        uarr = uarray(self.p, self.u)
        ufloat_df = pd.DataFrame({"n": [x.n for x in uarr],
                                  "s": [x.s for x in uarr]})
        precise_df = pd.DataFrame({"n": self.p,
                                   "s": self.u})

        return f"Fit parameters:\n\nufloats:\n{ufloat_df}\n\nprecisely:\n{precise_df}"

    def __repr__(self):
        return self.__str__()


class Interpolate(AbstractFit):
    """A class for interpolation with scipy.interpolate.interp1d"""

    def __init__(self,
                 x: ArrayLike,
                 y: ArrayLike,
                 kind: str = "linear",
                 divisions: int = 0):

        func = interp1d(x, y, kind=kind)
        super().__init__(x, y, lambda x: func(x), divisions=divisions)
        self.data = pd.DataFrame({"x": self.x_out, "y": self.y_out})

    def __str__(self):
        return f"Interpolation: table of values\n\n{self.data}"

    def __repr__(self):
        return self.__str__()


class Student:
    """A class for Student-t distributions.

    Calculate the mean of a given series and the uncertainty of the mean
    with a given sigma-niveau.
    """

    # class attributes
    _t_df_old = pd.DataFrame({"N": [2, 3, 4, 5, 6, 8, 10, 20, 30, 50, 100, 200],
                              "1": [1.84, 1.32, 1.20, 1.15, 1.11, 1.08, 1.06, 1.03, 1.02, 1.01, 1.00, 1.00],
                              "2": [13.97, 4.53, 3.31, 2.87, 2.65, 2.43, 2.32, 2.14, 2.09, 2.05, 2.03, 2.01],
                              "3": [235.8, 19.21, 9.22, 6.62, 5.51, 4.53, 4.09, 3.45, 3.28, 3.16, 3.08, 3.04]})

    t_df = pd.DataFrame({"1": [1.84, 1.32, 1.2, 1.15, 1.11, 1.09, 1.08, 1.07, 1.06, 1.051, 1.045, 1.04, 1.036,
                               1.033, 1.032, 1.031, 1.03, 1.03, 1.03, 1.03, 1.029, 1.028, 1.027, 1.026, 1.025,
                               1.024, 1.022, 1.021, 1.02, 1.019, 1.018, 1.017, 1.016, 1.016, 1.015, 1.014, 1.014,
                               1.013, 1.013, 1.012, 1.012, 1.012, 1.011, 1.011, 1.011, 1.011, 1.01, 1.01, 1.01],
                         "2": [13.97, 4.53, 3.31, 2.87, 2.65, 2.53, 2.43, 2.364, 2.32, 2.285, 2.255, 2.23, 2.209,
                               2.191, 2.177, 2.165, 2.155, 2.147, 2.14, 2.133, 2.127, 2.121, 2.116, 2.111, 2.106,
                               2.102, 2.097, 2.094, 2.09, 2.087, 2.083, 2.08, 2.078, 2.075, 2.072, 2.07, 2.068,
                               2.066, 2.064, 2.062, 2.06, 2.059, 2.057, 2.056, 2.055, 2.053, 2.052, 2.051, 2.05],
                         "3": [235.8, 19.21, 9.22, 6.62, 5.51, 5.02, 4.53, 4.31, 4.09, 4.026, 3.962, 3.898, 3.834,
                               3.77, 3.706, 3.642, 3.578, 3.514, 3.45, 3.433, 3.416, 3.399, 3.382, 3.365, 3.348,
                               3.331, 3.314, 3.297, 3.28, 3.274, 3.268, 3.262, 3.256, 3.25, 3.244, 3.238, 3.232,
                               3.226, 3.22, 3.214, 3.208, 3.202, 3.196, 3.19, 3.184, 3.178, 3.172, 3.166, 3.16]},
                        index=list(range(2, 51)))

    def __init__(self, series: ArrayLike, sigma: str = "1"):

        # test if sigma is reasonable
        assert sigma in {"1", "2", "3"}, \
            "Sigma must be amongst the following: ['1', '2', '3']"

        # maximum length of series is 50
        try:
            self.t = self.t_df.loc[len(series), sigma]  # type: ignore
        except KeyError:
            raise KeyError("Series is too big, maximum length is 50")

        self.series = np.array(series)

        self._n = np.mean(self.series)
        self._s = sem(self.series)
        self.mean = ufloat(self._n, self.t*self._s)

        if self.mean.s == ufloat(0, self._s).s:
            print("Student: t-factor is negligible!")

    def __str__(self):
        df = pd.DataFrame({"n": [self.mean.n, self._n],
                           "s": [self.mean.s, self._s]},
                          index=["ufloat:", "precisely:"])

        return f"Student-t distribution of series:\n{self.series}\n{df}"

    def __repr__(self):
        return self.__str__()
