"""
./src/classes.py of package 'labtool'
"""

# typing imports
from typing import Union, Any

from numpy.core.fromnumeric import std


class MonkeyPatch:
    "Monkey patches for certain 3rd party libraries"

    class uncertainties:
        """
        Modifications of the module 'uncertainties'
        Uncertainties: a Python package for calculations with uncertainties,
        Eric O. LEBIGOT, http://pythonhosted.org/uncertainties/
        """

        @staticmethod
        def rounding(msg: bool = False) -> None:
            """
            Update uncertainties' rounding function to a convention used in "Einführung in
            die physikalischen Messmethoden" (EPM), scriptum version 7.
            """

            import uncertainties.core as uc
            from math import ceil

            def EPM_precision(std_dev):
                """
                Return the number of significant digits to be used for the given
                standard deviation, according to the rounding rules of EPM.
                Also returns the effective standard deviation to be used for display.
                """

                exponent = uc.first_digit(std_dev)
                normalized_float = std_dev * 10**(-exponent)

                # return (significant digits, uncertainty)
                if normalized_float <= 1.9:
                    return 2, ceil(normalized_float * 10) * 10**(exponent-1)
                else:
                    return 1, ceil(normalized_float) * 10**exponent

            def round_correct(nominal_value, std_dev):
                exponent = uc.first_digit(std_dev)
                sig_dig, s = EPM_precision(std_dev)
                exponent += sig_dig - 1
                n = round(nominal_value, -exponent)
                return n, s

            def new_Variable__repr__(self):
                "A modified version of uncertainties.core.Variable.__repr__"

                if self.tag is None:
                    return uc.AffineScalarFunc.__str__(self)
                else:
                    return f"< {self.tag} = {uc.AffineScalarFunc.__repr__(self)} >"

            def new_Variable__init__(self, value, std_dev, tag=None):
                value, self.std_dev = round_correct(value, std_dev)
                uc.AffineScalarFunc.__init__(
                    self, value, uc.LinearCombination({self: 1.0}))
                self.tag = tag

            # ufloat is a factory function with return type uncertainties.core.Variable
            # which inherits from uncertainties.core.AffineScalarFunc
            # uncertainties.core.PDG_precision is used for uncertainties.core.AffineScalarFunc.__format__
            # which is used for uncertainties.core.AffineScalarFunc.__str__ (printing)
            # therefore changing the behavior of that function changes the way ufloats are diplayed
            uc.PDG_precision = EPM_precision

            # uncertainties.unumpy.core.uarray is a factory function which vectorizes uncertainties.core.Variable (__init__)
            # however class Variable does not have __str__ defined, but __repr__ instead, which just plain prints the input
            # -> change __repr__ to more sophisticated behavior of uncertainties.core.AffineScalarFunc.__str__
            uc.Variable.__repr__ = new_Variable__repr__
            # easier way if tag functionality can be omitted completely:
            # uc.Variable.__repr__ = uc.AffineScalarFunc.__str__

            # TODO
            uc.Variable.__init__ = new_Variable__init__

            if msg:
                print("Monkey patch successful: MonkeyPatch.uncertainties.rounding")

            return None


class cdContextManager:
    """A context manager that changes the working directory temporarily to the directory
    of the calling script or to an optional path."""

    import os
    from functions import cd

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
