"""labtool/src/classes.py"""

# dunders
__author__ = "Andreas Zach"
__all__ = ["CDContxt", "CurveFit", "Interpolation", "Student", "StudentArray"]

# std library
from os import chdir, getcwd
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
from .functions import cd, separate_uarray


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
        """Initiate an AbstractFit instance

        Attributes:
        -> x_in\tx data input
        -> y_in\ty data input
        -> x_out\toutgoing x data (numpy.linspace)
        -> y_out\toutgoing y data (fit_function(x_out))
        -> _f\tfit_function
        -> _type\tinternal class recognition

        Methods:
        -> plot
        -> save
        """

        self.x_in = array(x_in)
        self.y_in = array(y_in)

        a = min(self.x_in)
        b = max(self.x_in)
        c = int(divisions) if divisions > 0 else 3000

        self.x_out = linspace(a, b, c)
        self.y_out = f(self.x_out)

        self._f = f
        self._type = type_

    def __call__(self, *args):
        """Call the instance like a function"""
        return self._f(*args)

    def __len__(self):
        return len(self.x_in)

    def plot(self,
             style_in: str = "-",
             style_out: str = "-",
             label_in: str = "Data in",
             label_out: str = "Data out",
             title: Union[str, None] = None,
             **kwargs,
             ):  # TODO: return type
        """Plot AbstractFit instance. Transmit kwargs to pyplot in
        the style of method: value
        """

        # plot data
        fig, ax = plt.subplots()
        ax.plot(self.x_in, self.y_in, style_in, label=label_in)
        ax.plot(self.x_out, self.y_out, style_out, label=label_out)

        # check title
        title = title if title is not None else self._type

        # additional calls to pyplot
        _plot(fig, ax, title=title, **kwargs)

        return fig, ax

    def save(self, path: str) -> None:
        """Save fit to a file"""

        df_in_str = DataFrame({"x_in": self.x_in,
                               "y_in": self.y_in},
                              ).to_string()

        df_out_str = DataFrame({"x_out": self.x_out,
                                "y_out": self.y_out},
                               ).to_string()

        with open(path, "w", encoding="utf-8") as f:
            f.write(f"{self}\n\n{df_in_str}\n\n{df_out_str}")

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
        kwargs are provided to scipy.optimize.curve_fit

        Attributes:
        -> f\tfunction where x_in and y_in should be fitted on (see scipy.optimize.curve_fit documentation)
        -> x_in\tx data input
        -> y_in\ty data input
        -> x_out\tx data output (numpy.linspace)
        -> y_out\ty data output (fit_function(x_out))
        -> p\tcalculated parameters of f
        -> u\tuncertainties of the parameter
        -> pu\tuarray with ufloats
        -> df\tpandas.DataFrame with parameter names, nominal values and uncertainites
        -> _p_names\tnames of the parameters p in definition of function f

        Methods:
        -> plot
        -> save
        """

        self.p, pcov, *_ = curve_fit(f, x, y, **kwargs)
        self.u = sqrt(diag(pcov))
        self.pu = uarray(self.p, self.u)

        super().__init__(x, y, lambda x: f(x, *self.p),  # array-unpacking (same as tuple unpacking)
                         divisions, type_="CurveFit")

        # store names of parameters in a tuple
        self._p_names = f.__code__.co_varnames[1:]

        # create a dataframe of parameters with uncertainties
        n, s = separate_uarray(self.pu)
        self.df = DataFrame({"n": n,
                             "s": s},
                            index=self._p_names)

    def __str__(self):
        precise_df = DataFrame({"n": self.p,
                                "s": self.u},
                               index=self._p_names)

        return f"Fit parameters:\n\nufloats:\n{self.df}\n\nprecisely:\n{precise_df}"


class Interpolation(AbstractFit):
    """A class for interpolations with scipy.interpolate.interp1d"""

    def __init__(self,
                 x: ArrayLike,
                 y: ArrayLike,
                 divisions: int = 0,
                 **kwargs,
                 ):
        """
        kwargs are provided to scipy.interpolate.interp1d

        Attributes:
        -> x_in
        -> y_in
        -> x_out
        -> y_out
        -> data

        Methods:
        -> plot
        -> save
        """
        f = interp1d(x, y, **kwargs)
        super().__init__(x, y, lambda x: f(x), divisions, type_="Interpolation")
        self.data = DataFrame({"x": self.x_out, "y": self.y_out})


class Student:
    """A class for Student-t distributions.

    Calculate the mean of a given series and the uncertainty of the mean
    with a given sigma-niveau.
    """

    # class attributes

    # from "Einführung in die physikalischen Messmethoden": S.7, Tabelle 2"
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

        Methods:
        -> save
        """

        # test if sigma is reasonable
        if sigma not in {1, 2, 3}:
            raise ValueError("Sigma must be amongst the following integers: {1, 2, 3}")

        # maximum length of series in order to get a distinct t-factor is 50
        # t = 1, 2, 3 otherwise
        # try-except is faster than if
        try:
            self.t = self.t_df.loc[len(series), str(sigma)]  # type: ignore
        except KeyError:
            self.t = sigma

        self.series = array(series)

        # precise n (no rounding by uncertainties)
        self._n = mean(self.series)

        # precise s (no rounding by uncertainties)
        self._s = sem(self.series)

        # ufloat mean (rounded by uncertainties)
        self.mean = ufloat(self._n, self.t*self._s)

        # check if t is neglibible
        self._factor_used = True if self.mean.s != self._s else False

    def __str__(self):
        ret = (f"Student-t distribution\n\n"
               f"t-factor:\n\t{self.t if self._factor_used else 'unused'}\n"
               f"ufloat:\n\t{self.mean}\n"
               f"precisely:\n\t{self._n}+/-{self._s}")
        return ret

    def __repr__(self):
        return str(self.mean)

    def __len__(self):
        return len(self.series)

    def __getitem__(self, key):
        return self.series[key]

    def save(self, path: str) -> None:
        """Save Student-t data to a file"""

        with open(path, "w", encoding="utf-8") as f:
            f.write(str(self))

        return None


class StudentArray:
    """A class for Student-t-distributions in every point over multiple arrays"""

    def __init__(self,
                 *series: ArrayLike,
                 sigma: int = 1,
                 ):
        """Attributes:
        -> raw_data
        -> array
        -> n
        -> s
        """

        # check if provided tuple is already ArrayLike
        if len(series) == 1:
            data = array(series[0])
        else:
            data = array(*series).T  # type: ignore

        # initialize DataFrame
        self.raw_data = DataFrame(data, index=range(len(series)))

        # create uarray
        student = self.raw_data.apply(lambda x: Student(x, sigma), axis=1)
        self.array = array([x.mean for x in student])

        # create n and s arrays
        self.n, self.s = separate_uarray(self.array)

    # container methods
    # https://rszalski.github.io/magicmethods/  # for Python 2
    # https://stackoverflow.com/questions/56468788/list-of-custom-class-converted-to-numpy-array

    def __len__(self):
        return len(self.array)

    def __getitem__(self, key):
        return self.array[key]


def _plot(fig, ax, **kwargs) -> None:
    """Wrapper for pyplot calls"""

    # standard value = True
    for key in {"legend", "grid", "show"}:
        kwargs[key] = kwargs.get(key, True)
        # print(f"Updated {key} to {kwargs[key]}")

    # check all other methods
    for key, value in kwargs.items():
        if key == "show":
            continue
        elif type(value) is bool:  # catch all booleans
            if value:  # only if True
                ax.__dict__[key]()
        else:
            ax.__dict__[key](value)

    # show must be last one to be called
    if kwargs["show"]:
        plt.show()

    return None
