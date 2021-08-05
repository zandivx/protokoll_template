"""labtool/src/functions.py"""

# dunders
__author__ = "Andreas Zach"
__all__ = ["cd", "plt_latex", "pd_format", "write_table", "profile"]

# typing imports
from typing import Callable, Union, Any


def cd() -> None:
    """Change current directory (working directory) to directory of
    the script that executes this function
    """

    import os

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # future:
    # additional information through module inspect

    return None


def plt_latex() -> None:
    """Use LaTeX as backend for matplotlib.pyplot"""

    import matplotlib.pyplot as plt

    plt.rcParams.update({  # type: ignore
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{lmodern}",
        "font.family": "Latin Modern Roman"
    })

    return None


def pd_format(format_spec: str) -> None:
    """Update float formatting of pandas.DataFrame"""

    import pandas as pd

    pd.options.display.float_format = f"{{:{format_spec}}}".format

    return None


def write_table(df: Any,
                path: str,
                environ: str = "tblr",
                colspec: str = "",
                inner_settings: list[str] = [],
                columns: Union[bool, list[str]] = False,
                index: bool = False,
                format_spec: Union[None, str] = None,
                uarray: bool = False,
                sisetup: list[str] = [],
                msg: bool = False
                ) -> None:
    """Creates a tex-file with a correctly formatted table for LaTeX-package 'tabularray' from the given input-array.

    Mandatory parameters:
    -> df\t\t\tarray-like, must be convertible to pandas.DataFrame
    -> path\t\t\tname (or relative path) to tex-file for writing the table to

    Optional parameters:
    -> environ='tblr'\t\ttblr environment specified in file 'tabularray-environments.tex'
    -> colspec=''\t\tcolumn specifier known from standard LaTeX tables
    -> inner_settings=[]\tadditional settings for the tabularray environment (see documentation), input as list of strings
    -> columns=False\t\twrite table header (first row), input as list of strings or boolean
    -> index=False\t\tboolean if indices of rows (first column) should be written to table
    -> format_spec=None\t\tfloat formatter (e.g. .3f) or None if floats should not be formatted specifically
    -> uarray=False\t\tboolean if input is uncertainties.unumpy.uarray
    -> sisetup=[]\t\tlist with options for \\sisetup before tblr gets typeset
    -> msg=False\t\tboolean if a success-message and a reduced exception-message should be printed to the console
    """

    import pandas as pd
    import re

    # input must be convertible to pandas.DataFrame
    df = pd.DataFrame(df)

    # format_specifier
    formatter = f"{{:{format_spec}}}".format if format_spec is not None else None

    # append column specifier to inner settings
    if colspec:
        inner_settings.append(f"colspec={{{colspec}}}")
        # double curly braces produce literal curly brace in f string
        # three braces: evaluation surrounded by single braces

    # make strings safe for tabularry's siunitx S columns
    if columns == True:  # boolean check with == because columns could be a non-empty container
        columns = df.columns.tolist()
    # if columns was True, it's now a list
    if isinstance(columns, list):
        columns = [f"{{{{{{{col}}}}}}}" for col in columns]

    # strings
    sisetup_str = ", ".join(sisetup)
    inner_settings_str = ", ".join(inner_settings)
    df_str = df.to_csv(sep="&", line_terminator="\\\\\n",  # to_csv without path returns string
                       float_format=formatter, header=columns, index=index)

    if uarray:
        # delete string quotes
        df_str = df_str.replace('"', '')

        # replace +/- with +-
        df_str = re.sub(r"(\d)\+/-(\d)", r"\1 +- \2", df_str)

        # delete parantheses and make extra spaces if exponents
        df_str = re.sub(r"\((\d+\.?\d*) \+- (\d+\.?\d*)\)e",
                        r"\1 +- \2 e", df_str)

    # write to file
    with open(path, "w", encoding="utf-8") as f:  # open() does not encode in utf-8 by default

        if sisetup_str:
            f.write(f"\\sisetup{{{sisetup_str}}}\n\n")

        f.write(f"\\begin{{{environ}}}{{{inner_settings_str}}}\n"
                f"{df_str}"
                f"\\end{{{environ}}}")

    # message printing
    if msg:
        pd.options.display.float_format = formatter

        with open(path, "r", encoding="utf-8") as f:
            string = f.read()

        print(f"\nSuccessfully written pandas.DataFrame:\n\n{df}\n\n"
              f"as tabularray environment '{environ}' to file: '{path}'\n\n"
              f"output:\n\n{string}")

    return None


def profile(func: Callable) -> Callable:
    """A decorator for profiling a certain function call"""

    import cProfile
    import pstats

    def decorator(*args, **kwargs):
        with cProfile.Profile() as pr:
            func(*args, **kwargs)
        stats = pstats.Stats(pr)
        stats.sort_stats(pstats.SortKey.TIME)
        stats.dump_stats(f"_profiling_{func.__name__}.snakeviz")
        stats.print_stats()
    return decorator
