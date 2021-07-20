from typing import Union


class MonkeyPatch:
    "Monkey patches for certain 3rd party libraries"

    class uncertainties:
        """
        Modifications of the module 'uncertainties'
        Uncertainties: a Python package for calculations with uncertainties,
        Eric O. LEBIGOT, http://pythonhosted.org/uncertainties/
        """

        def rounding(self, msg: bool = False) -> None:
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

            uc.PDG_precision = EPM_precision

            if msg:
                print("MonkeyPatch.uncertainties.rounding")


class ChdirCls:
    "A context manager, that changes the working directory temporarily to the one of the script calling this class."

    def __enter__(self):
        from os import getcwd
        from functions import chdir
        self.olddir = getcwd()
        chdir()

    def __exit__(self, type, value, traceback):
        from os import chdir
        chdir(self.olddir)


class Student():
    "A class for student t distributions."

    from pandas import read_csv

    with ChdirCls():
        sigma_df = read_csv("data/student_factors.txt")
    sigma_dict = {1: "68.3%", 2: "95.5%", 3: "99.7%"}

    def __init__(self, series=[], sigma: int = 1) -> None:
        assert sigma in (1, 2, 3)
        self.niveau = self.sigma_dict[sigma]
        self.series = series

    def t_factor(self):
        if x := len(self.series) in self.sigma_df["N"]:
            t = self.sigma_df.loc[x, self.niveau]
        else:
            t = self.sigma_df.loc[20, self.niveau]
        return t


#print(Student([1, 2, 3, 4, 5]).t_factor())
