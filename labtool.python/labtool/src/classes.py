"""
labtool/src/classes.py
"""

# std lib
from typing import Callable, Union, Any

# 3rd party
from numpy.typing import ArrayLike


class cdContextManager:
    """
    A context manager that changes the working directory temporarily to the directory
    of the calling script or to an optional path.
    """

    import os
    from .functions import cd

    def __init__(self, path: str = ""):
        self.path = path

    def __enter__(self):
        self.olddir = self.os.getcwd()

        if self.path != "":
            self.os.chdir(self.path)
        else:
            self.cd()

    def __exit__(self, type, value, traceback):
        self.os.chdir(self.olddir)


class SysPathContextManager:
    """
    A context manager that appends the sys.path-variable with a given init-path.
    Useful for relative imports in a package, where the importing module should be run
    as a script.

    """
    import sys

    def __init__(self, path: str):
        self.path = path

    def __enter__(self):
        exec("#type:ignore")
        self.sys.path.append(self.path)

    def __exit__(self, *args):
        self.sys.path.pop()


class Fit:
    import numpy as np

    def __init__(self,
                 function: Callable,
                 x: ArrayLike,
                 y: ArrayLike,
                 p0: Union[ArrayLike, None] = None,
                 sigma: Union[ArrayLike, None] = None,
                 bounds: tuple[ArrayLike, ArrayLike] = ((-np.inf,), (np.inf,))):

        from scipy.optimize import curve_fit

        self.f = function
        self.x_in = x
        self.y_in = y

        self.p, pcov, *_ = curve_fit(self.f, self.x_in, self.y_in,
                                     p0=p0, sigma=sigma, absolute_sigma=True, bounds=bounds)
        self.u = self.np.sqrt(self.np.diag(pcov))

    def plot(self,
             label_in: str = "Data in",
             label_out: str = "Data out",
             title: str = "",
             xlabel: str = "",
             ylabel: str = "",
             xlim: Union[bool, tuple] = False,
             ylim: Union[bool, tuple] = False,
             grid: bool = False,
             divisions: int = 0) -> None:

        import matplotlib.pyplot as plt

        a = self.np.min(self.x_in)
        b = self.np.max(self.x_in)
        c = divisions if isinstance(divisions, int) \
            and divisions > 0 else 10 * (b - a)

        self.x_out = self.np.linspace(a, b, c)
        self.y_out = self.f(self.x_out, *self.p)

        fig, ax = plt.subplots()  # type: ignore

        ax.plot(self.x_in, self.y_in, label=label_in)
        ax.plot(self.x_out, self.y_out, label=label_out)
        ax.set_title(title)
        ax.legend()

        ax.set_xlabel = xlabel
        ax.set_ylabel = ylabel

        if xlim:
            ax.set_xlim = xlim
        if ylim:
            ax.set_ylim = ylim
        if grid:
            ax.grid()

        plt.show()

        return None

    def __str__(self):
        from uncertainties.unumpy import uarray
        return "Fit:\n" + str(uarray(self.p, self.u))

    def precise(self) -> str:
        return f"Fit:\n{self.p}\n{self.u}"


class Interpolate:
    def __init__(self):
        pass


class Student:
    "A class for student t distributions."

    from pandas import DataFrame

    __t_values = DataFrame({"N": [2, 3, 4, 5, 6, 8, 10, 20, 30, 50, 100, 200],
                            "1": [1.84, 1.32, 1.20, 1.15, 1.11, 1.08, 1.06, 1.03, 1.02, 1.01, 1.00, 1.00],
                            "2": [13.97, 4.53, 3.31, 2.87, 2.65, 2.43, 2.32, 2.14, 2.09, 2.05, 2.03, 2.01],
                            "3": [235.8, 19.21, 9.22, 6.62, 5.51, 4.53, 4.09, 3.45, 3.28, 3.16, 3.08, 3.04]})

    t_df = DataFrame({"N": list(range(2, 51)),
                      "1": [1.84, 1.32, 1.2, 1.15, 1.11, 1.09, 1.08, 1.07, 1.06, 1.051, 1.045, 1.04, 1.036, 1.033, 1.032, 1.031, 1.03, 1.03, 1.03, 1.03, 1.029, 1.028, 1.027, 1.026, 1.025, 1.024, 1.022, 1.021, 1.02, 1.019, 1.018, 1.017, 1.016, 1.016, 1.015, 1.014, 1.014, 1.013, 1.013, 1.012, 1.012, 1.012, 1.011, 1.011, 1.011, 1.011, 1.01, 1.01, 1.01],
                      "2": [13.97, 4.53, 3.31, 2.87, 2.65, 2.53, 2.43, 2.364, 2.32, 2.285, 2.255, 2.23, 2.209, 2.191, 2.177, 2.165, 2.155, 2.147, 2.14, 2.133, 2.127, 2.121, 2.116, 2.111, 2.106, 2.102, 2.097, 2.094, 2.09, 2.087, 2.083, 2.08, 2.078, 2.075, 2.072, 2.07, 2.068, 2.066, 2.064, 2.062, 2.06, 2.059, 2.057, 2.056, 2.055, 2.053, 2.052, 2.051, 2.05],
                      "3": [235.8, 19.21, 9.22, 6.62, 5.51, 5.02, 4.53, 4.31, 4.09, 4.026, 3.962, 3.898, 3.834, 3.77, 3.706, 3.642, 3.578, 3.514, 3.45, 3.433, 3.416, 3.399, 3.382, 3.365, 3.348, 3.331, 3.314, 3.297, 3.28, 3.274, 3.268, 3.262, 3.256, 3.25, 3.244, 3.238, 3.232, 3.226, 3.22, 3.214, 3.208, 3.202, 3.196, 3.19, 3.184, 3.178, 3.172, 3.166, 3.16]})

    def __init__(self, series: Any = [], sigma: str = "1"):
        assert sigma in ("1", "2", "3")
        self.series = series


# TODO
# print(Student([1, 2, 3, 4, 5]).t_factor())
