"""
./src/classes.py of package 'labtool'
"""

# std lib
from typing import Union, Any


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


class Student:
    "A class for student t distributions."

    import pandas as pd

    t_df = pd.DataFrame({"N": [2, 3, 4, 5, 6, 8, 10, 20, 30, 50, 100, 200],
                         "68.3%": [1.84, 1.32, 1.20, 1.15, 1.11, 1.08, 1.06, 1.03, 1.02, 1.01, 1.00, 1.00],
                         "95.5%": [13.97, 4.53, 3.31, 2.87, 2.65, 2.43, 2.32, 2.14, 2.09, 2.05, 2.03, 2.01],
                         "99.7%": [235.8, 19.21, 9.22, 6.62, 5.51, 4.53, 4.09, 3.45, 3.28, 3.16, 3.08, 3.04]})
    niveau_dict = {1: "68.3%", 2: "95.5%", 3: "99.7%"}

    def __init__(self, series: Any = [], sigma: int = 1) -> None:
        assert sigma in (1, 2, 3)
        self.niveau = self.niveau_dict[sigma]
        self.series = series

    def t_factor(self) -> float:
        if x := len(self.series) in self.t_df["N"]:
            t = self.t_df.loc[x, self.niveau]
        else:
            t = self.t_df.loc[20, self.niveau]
        return t  # type:ignore

# TODO
# print(Student([1, 2, 3, 4, 5]).t_factor())


del Union
del Any
