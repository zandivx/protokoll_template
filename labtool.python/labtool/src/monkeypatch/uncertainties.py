"""
Monkey patches for package 'uncertainties':

Uncertainties: a Python package for calculations with uncertainties,
Eric O. LEBIGOT, http://pythonhosted.org/uncertainties/
"""

# std lib
from typing import Union

# 3rd party
import uncertainties.core as uc


class Rounding:
    """
    A class which only contains two staticmethods:
    -> display
    -> init

    Function 'display' changes the style ufloats are formatted and printed.

    Function 'init' changes how uncertainties.core.Variable-instances are initialized,
    nominal value and standard deviation for instantiation are rounded by the same
    principles they are displayed.
    """

    @staticmethod
    def display() -> None:
        """
        Update uncertainties' formatting function to a convention used in "Einführung
        in die physikalischen Messmethoden" (EPM), scriptum version 7.
        """

        def EPM_precision(std_dev: float) -> tuple[int, float]:
            """
            Return the number of significant digits to be used for the given
            standard deviation, according to the rounding rules of EPM.
            Also returns the effective standard deviation to be used for display.
            """

            dig, _, s = _digits_exponent_std_dev(std_dev)

            return dig, s

        def new__repr__(self):
            "A modified version of uncertainties.core.Variable.__repr__"

            if self.tag is None:
                return uc.AffineScalarFunc.__str__(self)
            else:
                return f"< {self.tag} = {uc.AffineScalarFunc.__repr__(self)} >"

        # ufloat is a factory function with return type uncertainties.core.Variable
        # which inherits from uncertainties.core.AffineScalarFunc
        # uncertainties.core.PDG_precision is used for uncertainties.core.AffineScalarFunc.__format__
        # which is used for uncertainties.core.AffineScalarFunc.__str__ (printing)
        # therefore changing the behavior of that function changes the way ufloats are diplayed
        uc.PDG_precision = EPM_precision

        # uncertainties.unumpy.core.uarray is a factory function which vectorizes uncertainties.core.Variable (__init__)
        # however class Variable does not have __str__ defined, but __repr__ instead, which just plain prints the input
        # -> change __repr__ to more sophisticated behavior of uncertainties.core.AffineScalarFunc.__str__
        uc.Variable.__repr__ = new__repr__
        # easier way if tag functionality can be omitted completely:
        # uc.Variable.__repr__ = uc.AffineScalarFunc.__str__

        return None

    @staticmethod
    def init() -> None:
        """
        Round nominal value and standard deviation according to the convention of
        EPM at the instantiation of an uncertainties.core.Variable.
        """

        def round_n_s(nominal_value: Union[int, float], std_dev: Union[int, float]) -> tuple[float, float]:
            "Round nominal value and standard deviation according to EPM."

            _, exponent, s = _digits_exponent_std_dev(std_dev)
            n = round(nominal_value, -exponent)

            return n, s

        def new__init__(self, value, std_dev, tag=None):
            "A modified version of uncertainties.core.Variable.__init__"

            value, self.std_dev = round_n_s(value, std_dev)
            uc.AffineScalarFunc.__init__(
                self, value, uc.LinearCombination({self: 1.0}))
            self.tag = tag

        # changes uncertainties.core.Variable.__init__
        uc.Variable.__init__ = new__init__

        return None


def _digits_exponent_std_dev(std_dev: float) -> tuple[int, int, float]:
    """
    Find the amount of significant digits and in reference to that the exponent
    of base 10. Also returns the effective standard deviation.

    This function provides data needed by Rounding.display (EPM_precision) and
    Rounding.init (round_conventional)
    """

    from math import ceil

    # exponent of base 10
    exponent = uc.first_digit(std_dev)

    # normalized floating point number
    # round to 4 decimals to minimize machine epsilon
    normalized_float = round(std_dev * 10**(-exponent), 4)

    # significant digits to consider for rounding
    sig_digits = 1

    # two significant digits if first digit is 1, one digit otherwise
    if normalized_float <= 1.9:
        sig_digits += 1
        exponent -= 1
        normalized_float *= 10

    # round up according to significant digits
    s = ceil(normalized_float) * 10**exponent

    return sig_digits, exponent, s
