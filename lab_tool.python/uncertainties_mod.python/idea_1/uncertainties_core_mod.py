import uncertainties.core as uc
from uncertainties.core import *
from math import ceil


def _PDG_precision(std_dev):
    """
    Return the number of significant digits to be used for the given
    standard deviation, according to the rounding rules of "Einführung in
    die physikalischen Messmethoden" (v7)

    Modified function of uncertainties original one
    """

    exponent = uc.first_digit(std_dev)

    normalized_float = std_dev * 10**-exponent

    # return (signigicant digits, uncertainty)
    if normalized_float <= 1.9:
        return 2, ceil(normalized_float * 10) * 10**(exponent-1)
    else:
        return 1, ceil(normalized_float) * 10**exponent


uc.PDG_precision = _PDG_precision
