# imports
from typing import Tuple, Union
import numpy as np
from uncertainties.core import Variable
import uncertainties as u
#from timeit import default_timer


# class
class UncFloat(Variable):
    def __init__(self, nominal_value, uncertainty):
        n, s = round_correct(nominal_value, uncertainty)
        super().__init__(n, s)


# type aliases
Int_Float = Union[int, float]
Nums_Arraylikes = Union[Int_Float,
                        list[Int_Float], tuple[Int_Float], np.ndarray]
Int_Array = Union[int, np.ndarray]


# functions
def dimension_conventional(unc: Nums_Arraylikes) -> Int_Array:

    def dimension_of_first_digit(unc: Nums_Arraylikes) -> Int_Array:
        with np.errstate(divide='ignore'):  # ignore dividing by zero warning
            array = np.floor(np.log10(np.abs(unc)))
        try:
            array[np.isinf(array)] = 0
            array = array.astype("int64")
        except TypeError:  # scalar instead of array
            if np.isinf(array):
                array = 0
            array = int(array)
        return array

    exponent = dimension_of_first_digit(unc)
    normalized_float = unc * 10**-exponent
    mask = np.floor(normalized_float) == 1

    try:
        exponent[mask] -= 1  # type:ignore  # pylance ignore
    except TypeError:
        if mask:
            exponent -= 1

    return exponent


def round_correct(n: Nums_Arraylikes, s: Nums_Arraylikes = 0) -> Tuple[Nums_Arraylikes, Nums_Arraylikes]:
    exponent = dimension_conventional(s)
    n = np.round(n, -exponent)
    s = np.ceil(s * 10**-exponent) * 10**exponent
    return n, s


test = (1.31890, 0.123)
a = UncFloat(*test)
print(
    f"rounding: {round_correct(*test)}\nown float: {a}\noriginal float: {u.ufloat(*test)}")
print(type(2*a))  # type:ignore


def first_digit(value):
    '''
    Return the first digit position of the given value, as an integer.

    0 is the digit just before the decimal point. Digits to the right
    of the decimal point have a negative position.

    Return 0 for a null value.
    '''
    if isinstance(value, np.ndarray):
        # nice try, but slower
        # masked = np.ma.masked_equal(value, 0)  # mask array where value == 0
        # masked_and_computed = np.floor(np.log10(np.abs(masked)))
        # ret = np.ma.filled(masked_and_computed, fill_value=0)
        with np.errstate(divide='ignore'):  # ignore dividing by zero warning
            array = np.floor(np.log10(np.abs(value)))
            # array[array == -np.inf] = 0
            array[np.isinf(array)] = 0
        return array

    else:
        try:  # try except faster if exceptions are rare, if slows down everytime, catching only if exception, dann aber more
            return np.log10(abs(value)) // 1
            # return int(np.floor(np.log10(abs(value))))
        except ValueError:  # Case of value == 0
            return 0


def round_right(std_dev):
    exponent = first_digit(std_dev)

    with_exponent_1 = std_dev * 10**-exponent

    if isinstance(std_dev, np.ndarray):
        mask_1 = np.floor(with_exponent_1) == 1
        mask_not_1 = np.invert(mask_1)

        with_exponent_1[mask_1] = np.ceil(
            with_exponent_1[mask_1] * 10) * 10**(exponent[mask_1] - 1)  # type:ignore

        with_exponent_1[mask_not_1] = np.ceil(
            with_exponent_1[mask_not_1]) * 10**exponent[mask_not_1]  # type:ignore

        return with_exponent_1  # wrong name

    else:
        if int(with_exponent_1) == 1:
            with_exponent_2 = np.ceil(with_exponent_1 * 10)
            return (with_exponent_2 * 10**(exponent - 1), exponent)
        else:
            return (np.ceil(with_exponent_1) * 10**exponent, exponent)


not_yet = """
def PDG_precision(std_dev):
    '''
    Return the number of significant digits to be used for the given
    standard deviation, according to the rounding rules of the
    Particle Data Group (2010)
    (http://pdg.lbl.gov/2010/reviews/rpp2010-rev-rpp-intro.pdf).

    Also returns the effective standard deviation to be used for
    display.
    '''

    exponent = first_digit(std_dev)

    # The first three digits are what matters: we get them as an
    # integer number in [100; 999).
    #
    # In order to prevent underflow or overflow when calculating
    # 10**exponent, the exponent is slightly modified first and a
    # factor to be applied after "removing" the new exponent is
    # defined.
    #
    # Furthermore, 10**(-exponent) is not used because the exponent
    # range for very small and very big floats is generally different.
    if exponent >= 0:
        # The -2 here means "take two additional digits":
        exponent, factor = exponent-2, 1
    else:
        exponent, factor = exponent+1, 1000
    digits = int(std_dev * 10**-exponent * factor)  # int rounds towards zero

    # Rules:
    if digits <= 354:
        return (2, std_dev)
    elif digits <= 949:
        return (1, std_dev)
    else:
        # The parentheses matter, for very small or very large
        # std_dev:
        return (2, 10.**exponent*(1000/factor))


def signif_dgt_to_limit(value, num_signif_d):
    '''
    Return the precision limit necessary to display value with
    num_signif_d significant digits.

    The precision limit is given as -1 for 1 digit after the decimal
    point, 0 for integer rounding, etc. It can be positive.
    '''

    fst_digit = first_digit(value)

    limit_no_rounding = fst_digit - num_signif_d + 1

    # The number of significant digits of the uncertainty, when
    # rounded at this limit_no_rounding level, can be too large by 1
    # (e.g., with num_signif_d = 1, 0.99 gives limit_no_rounding = -1, but
    # the rounded value at that limit is 1.0, i.e. has 2
    # significant digits instead of num_signif_d = 1). We correct for
    # this effect by adjusting limit if necessary:
    rounded = round(value, -limit_no_rounding)
    fst_digit_rounded = first_digit(rounded)

    if fst_digit_rounded > fst_digit:
        # The rounded limit is fst_digit_rounded-num_signif_d+1;
        # but this can only be 1 above the non-rounded limit:
        limit_no_rounding += 1

    return limit_no_rounding"""
