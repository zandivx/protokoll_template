"""labtool/src/classes.py"""

# dunders
__author__ = "Andreas Zach"
__all__ = ["CDContxt", "CurveFit", "Interpolation", "Student"]

# std library
from os import chdir, getcwd
# from sys import path as sys_path
from typing import Callable, Union

# 3rd party
from matplotlib import pyplot as plt
from numpy import array, diag, linspace, max, min, mean, ndarray, sqrt
from numpy.typing import ArrayLike
from pandas import DataFrame
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
        self.olddir = getcwd()

        if self.path != "":
            chdir(self.path)
        else:
            cd()

    def __exit__(self, type, value, traceback):
        chdir(self.olddir)


# class SysPathContxt:
#     """A context manager that appends the sys.path-variable with a given init-path.
#     Useful for relative imports in a package, where the importing module should be run
#     as a script.
#     """

#     def __init__(self, path: str):
#         self.path = path

#     def __enter__(self):
#         sys_path.append(self.path)

#     def __exit__(self, *args):
#         sys_path.pop()


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
                 divisions: int,
                 type_: str = "AbstractFit",
                 ):
        """Initiate an AbstractFit instance"""

        self.f = f
        self.x_in = x_in
        self.y_in = y_in

        a = min(x_in)
        b = max(x_in)
        c = int(divisions) if divisions > 0 else int(20 * (b - a))

        self.x_out = linspace(a, b, c)
        self.y_out = f(self.x_out)

        # to differentiate subclasses of ABC
        self._type = type_

    def __call__(self, *args):
        """Call the instance like a function"""
        return self.f(*args)

    def __str__(self):
        return "AbstractFit string representation"

    def plot(self,
             style_in: str = "-",
             style_out: str = "-",
             label_in: str = "Data in",
             label_out: str = "Data out",
             title: Union[str, None] = None,
             **kwargs,
             ) -> None:
        """Plot AbstractFit instance. Transmit kwargs to pyplot in
        the style of method: value
        """

        # plot data
        plt.clf()
        plt.plot(self.x_in, self.y_in, style_in, label=label_in)
        plt.plot(self.x_out, self.y_out, style_out, label=label_out)

        # check title
        title = "" if title is not None else self._type

        # additional calls to pyplot
        _plot(title=title, **kwargs)

        return None

    def save(self, path: str) -> None:
        """Save fit to a file"""

        df_in = DataFrame({"x_in": self.x_in,
                           "y_in": self.y_in},
                          ).to_string()

        df_out = DataFrame({"x_out": self.x_out,
                            "y_out": self.y_out},
                           ).to_string()

        with open(path, "w", encoding="utf-8") as f:
            f.write(f"{self}\n\n{df_in}\n\n{df_out}")

        return None


class CurveFit(AbstractFit):
    """A class for fits with scipy.optimize.curve_fit"""

    def __init__(self,
                 f: Callable,
                 x: ArrayLike,
                 y: ArrayLike,
                 divisions: int = 0,
                 **kwargs,
                 ):
        """
        kwargs are transmitted to curve_fit

        Attributes:
        -> f
        -> x_in
        -> y_in
        -> x_out
        -> y_out
        -> p
        -> u
        -> _p_names
        """

        curve_fit_return = curve_fit(f, x, y, **kwargs)

        self.p: ndarray = curve_fit_return[0]
        pcov: ndarray = curve_fit_return[1]
        self.u: ndarray = sqrt(diag(pcov))

        super().__init__(x, y, lambda x: f(x, *self.p),  # array-unpacking (same as tuple unpacking)
                         divisions, type_="CurveFit")

        # store names of parameters in a tuple
        self._p_names = f.__code__.co_varnames[1:]

    def __str__(self):
        uarr = uarray(self.p, self.u)
        ufloat_df = DataFrame({"n": [x.n for x in uarr],
                               "s": [x.s for x in uarr]},
                              index=self._p_names)

        precise_df = DataFrame({"n": self.p,
                                "s": self.u},
                               index=self._p_names)

        return f"Fit parameters:\n\nufloats:\n{ufloat_df}\n\nprecisely:\n{precise_df}"

    def __repr__(self):
        return self.__str__()


class Interpolation(AbstractFit):
    """A class for interpolations with scipy.interpolate.interp1d"""

    def __init__(self,
                 x: ArrayLike,
                 y: ArrayLike,
                 divisions: int = 0,
                 **kwargs,
                 ):
        """
        kwargs are transmitted to interp1d

        Attributes:
        -> f
        -> x_in
        -> y_in
        -> x_out
        -> y_out
        -> data
        """
        f = interp1d(x, y, **kwargs)
        super().__init__(x, y, lambda x: f(x), divisions, type_="Interpolation")
        self.data = DataFrame({"x": self.x_out, "y": self.y_out})

    def __str__(self):
        return "Interpolation"

    def __repr__(self):
        return self.__str__()


class Student:
    """A class for Student-t distributions.

    Calculate the mean of a given series and the uncertainty of the mean
    with a given sigma-niveau.
    """

    # class attributes
    # _t_df_old = DataFrame({"N": [2, 3, 4, 5, 6, 8, 10, 20, 30, 50, 100, 200],
    #                        "1": [1.84, 1.32, 1.20, 1.15, 1.11, 1.08, 1.06, 1.03, 1.02, 1.01, 1.00, 1.00],
    #                        "2": [13.97, 4.53, 3.31, 2.87, 2.65, 2.43, 2.32, 2.14, 2.09, 2.05, 2.03, 2.01],
    #                        "3": [235.8, 19.21, 9.22, 6.62, 5.51, 4.53, 4.09, 3.45, 3.28, 3.16, 3.08, 3.04]})

    t_df = DataFrame({"1": [1.84, 1.32, 1.2, 1.15, 1.11, 1.09, 1.08, 1.07, 1.06, 1.051, 1.045, 1.04, 1.036,
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

    def __init__(self,
                 series: ArrayLike,
                 sigma: int = 1,
                 ):
        """Attributes:
        -> series
        -> t
        -> mean
        -> _n
        -> _s
        -> _factor_used
        """

        # test if sigma is reasonable
        if sigma not in {1, 2, 3}:
            raise ValueError(
                "Sigma must be amongst the following integers: [1, 2, 3]")

        # maximum length of series is 50
        # try-except is faster than if
        try:
            self.t = self.t_df.loc[len(series), str(sigma)]  # type: ignore
        except KeyError:
            raise IndexError("Series is too big, maximum length is 50")

        self.series = array(series)

        # precise n
        self._n = mean(self.series)

        # precise s
        self._s = sem(self.series)

        # ufloat mean
        self.mean = ufloat(self._n, self.t*self._s)

        # check if t is neglibible
        self._factor_used = True if self.mean.s != self._s else False

    def __str__(self):
        ret = f"Student-t distribution of series:\n{self.series}\n\n"
        ret += "" if self._factor_used else "!!! t-factor is negligible!\n\n"
        ret += f"Mean:\nufloat:\n\t{self.mean.n} +/- {self.mean.s}\nprecisely:\n\t{self._n} +/- {self._s}"
        return ret

    def __repr__(self):
        return self.__str__()

    def plot(self,
             style: str = "-",
             label: str = "Data",
             label_u: str = "Mean with uncertainty band",
             title: str = "Student",
             **kwargs,
             ) -> None:
        """Plot Student-t distribution"""

        # plot data
        plt.clf()
        plt.plot(self.series, style, label=label)

        length = len(self.series)
        plt.plot([self.mean.n]*length, "k--", label=label_u)

        n = self.mean.n
        s = self.mean.s
        plt.fill_between(range(length), n-s, n+s, color="gray", alpha=0.2)

        # additional calls to pyplot
        _plot(title=title, **kwargs)

        return None

    def save(self, path: str) -> None:
        """Save Student-t data to a file"""

        with open(path, "w", encoding="utf-8") as f:
            f.write(str(self))

        return None


def _plot(**kwargs) -> None:
    """Wrapper for pyplot calls"""

    # standard value = True
    for key in ("legend", "grid", "show"):
        kwargs[key] = kwargs.get(key, True)
        # print(f"Updated {key} to {kwargs[key]}")

    # check all other methods
    for key, value in kwargs.items():
        if key == "show":
            continue
        elif type(value) is bool:  # catch all booleans
            if value:  # only if True
                plt.__dict__[key]()
        else:
            plt.__dict__[key](value)

    # show must be last one to be called
    if kwargs["show"]:
        plt.show()

    return None
