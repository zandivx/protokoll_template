"""Monkey patches for package 'uncertainties':

Uncertainties: a Python package for calculations with uncertainties,
Eric O. LEBIGOT, http://pythonhosted.org/uncertainties/
"""

# dunders
__author__ = "Andreas Zach"
__all__ = ["display", "init", "undo"]

# std lib
from importlib import reload
from math import ceil

# 3rd party
import uncertainties.core as uc


def display() -> None:
    """Update uncertainties' formatting function to a convention used in "Einführung
    in die physikalischen Messmethoden" (EPM), scriptum version 7.
    """

    def EPM_precision(std_dev: float) -> tuple[int, float]:
        """Return the number of significant digits to be used for the given
        standard deviation, according to the rounding rules of EPM instead
        of PDG (Particle Data Group).
        Also return the effective standard deviation to be used for display
        """
        dig, _, s = _digits_exponent_std_dev(std_dev)
        return dig, s

    def new__repr__(self) -> str:
        """A modified version of uncertainties.core.Variable.__repr__"""
        if self.tag is None:
            return uc.AffineScalarFunc.__str__(self)
        else:
            return f"< {self.tag} = {uc.AffineScalarFunc.__repr__(self)} >"

    #old__format__ = _copy_func(uc.AffineScalarFunc.__format__)

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

    # possible patch in future
    #uc.AffineScalarFunc.__format__ = new__format__

    return None


def init() -> None:
    """Round nominal value and standard deviation according to the convention
    of EPM at the instantiation of an uncertainties.core.Variable
    """

    def round_n_s(nominal_value: float, std_dev: float) -> tuple[float, float]:
        """Round nominal value and standard deviation according to EPM"""
        _, exponent, s = _digits_exponent_std_dev(std_dev)
        # don't round if std_dev == exponent == 0
        n = round(nominal_value, -exponent) if s else nominal_value
        return n, s

    def new__init__(self, value, std_dev, tag=None):
        """A modified version of uncertainties.core.Variable.__init__"""
        value, self.std_dev = round_n_s(value, std_dev)
        uc.AffineScalarFunc.__init__(
            self, value, uc.LinearCombination({self: 1.0}))
        self.tag = tag

    # changes uncertainties.core.Variable.__init__
    uc.Variable.__init__ = new__init__

    return None


def undo() -> None:
    """Reload uncertainties.core and remove applied monkey patches"""
    reload(uc)
    return None


def _digits_exponent_std_dev(std_dev: float) -> tuple[int, int, float]:
    """Find the amount of significant digits the exponent of those digits to the base 10.
    Also return the effective standard deviation.

    Provide data needed by function 'display' (EPM_precision) and function 'init' (round_conventional)
    """

    if std_dev:  # std_dev != 0

        # exponent of base 10
        exponent = uc.first_digit(std_dev)

        # calculate mantissa of std_dev
        # round to 3 digits to minimize machine epsilon
        mantissa = round(std_dev * 10**(-exponent), 3)

        # significant digits to consider for rounding
        sig_digits = 1

        # two significant digits if first digit is 1, one digit otherwise
        if mantissa <= 1.9:
            sig_digits += 1
            exponent -= 1
            mantissa *= 10

        # round up according to significant digits
        s: float = ceil(mantissa) * 10**exponent

        return sig_digits, exponent, s

    else:  # std_dev == 0
        return 0, 0, 0.0
