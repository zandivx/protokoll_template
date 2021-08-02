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
                array = 0  # type:ignore
            array = int(array)  # type:ignore
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
