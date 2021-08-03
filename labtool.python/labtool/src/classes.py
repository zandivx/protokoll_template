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
             x_label: str = "",
             y_label: str = "",
             x_lim: Union[bool, tuple] = False,
             y_lim: Union[bool, tuple] = False,
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

        ax.set_xlabel = x_label
        ax.set_ylabel = y_label

        if x_lim:
            ax.set_xlim = x_lim
        if y_lim:
            ax.set_ylim = y_lim
        if grid:
            ax.grid()

        print()
        plt.show()

        return None

    def __str__(self):
        from uncertainties.unumpy import uarray
        return "Fit:\n" + str(uarray(self.p, self.u))


class Student:
    "A class for student t distributions."

    from pandas import DataFrame

    t_values = DataFrame({"N": [2, 3, 4, 5, 6, 8, 10, 20, 30, 50, 100, 200],
                          "1": [1.84, 1.32, 1.20, 1.15, 1.11, 1.08, 1.06, 1.03, 1.02, 1.01, 1.00, 1.00],
                          "2": [13.97, 4.53, 3.31, 2.87, 2.65, 2.43, 2.32, 2.14, 2.09, 2.05, 2.03, 2.01],
                          "3": [235.8, 19.21, 9.22, 6.62, 5.51, 4.53, 4.09, 3.45, 3.28, 3.16, 3.08, 3.04]})

    def __init__(self, series: Any = [], sigma: str = "1"):
        assert sigma in ("1", "2", "3")
        self.series = series


# TODO
# print(Student([1, 2, 3, 4, 5]).t_factor())
