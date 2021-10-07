"""labtool/src/functions.py"""

# dunders
__author__ = "Andreas Zach"
__all__ = ["cd", "plt_latex", "pd_format", "write_table", "profile", "tracer"]

# std library
from cProfile import Profile
from os import chdir, path
from pstats import Stats, SortKey
from re import sub
from typing import Callable, Union, Any as DataFrameLike

# 3rd party
from pandas import DataFrame, options
from matplotlib import rcParams


def cd() -> None:
    """Change the current working directory to the directory of the calling script."""
    chdir(path.dirname(path.abspath(__file__)))
    return None


def plt_latex() -> None:
    """Use LaTeX as backend for matplotlib"""
    rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{lmodern}",
        "font.family": "Latin Modern Roman",
    })
    return None


def pd_format(format_spec: str) -> None:
    """Update float-formatting of pandas.DataFrame."""
    options.display.float_format = f"{{:{format_spec}}}".format
    return None


def write_table(content: DataFrameLike,
                path: str,
                environ: str = "tblr",
                colspec: Union[str, list[str]] = "",
                inner_settings: list[str] = [],
                columns: Union[bool, list[str]] = False,
                index: bool = False,
                format_spec: Union[None, str] = None,
                uarray: bool = False,
                sisetup: list[str] = [],
                hlines_old: bool = False,
                msg: bool = False
                ) -> str:
    """Create a tex-file with a correctly formatted table for LaTeX-package 'tabularray' from the given
    input content. Return the created string.

    Mandatory parameters:
    -> content\t\t\tmust be convertible to pandas.DataFrame
    -> path\t\t\tname (or relative path) to tex-file for writing the table to, can be left empty
    \t\t\t\tin order to just return the string

    Optional parameters:
    -> environ='tblr'\t\ttblr environment specified in file 'tabularray-environments.tex',
    \t\t\t\toptional 'tabular' to use a standard LaTeX-table
    -> colspec=''\t\tcolumn specifier known from standard LaTeX tables (only suited for tabularray!),
    \t\t\t\tone string or list of strings
    -> inner_settings=[]\tadditional settings for the tabularray environment (see documentation), input as list of strings
    \t\t\t\tif standard 'tabular' environment is used, the column specifiers can be put here as one entry of the list
    -> columns=False\t\twrite table header (first row), input as list of strings or boolean
    -> index=False\t\tboolean if indices of rows (first column) should be written to table
    -> format_spec=None\t\tfloat formatter (e.g. .3f) or None if floats should not be formatted specifically
    -> uarray=False\t\tboolean if input was created with uncertainties.unumpy.uarray
    -> sisetup=[]\t\tlist with options for \\sisetup before tblr gets typeset
    -> hlines_old=False\t\tif standard tabular environment is used, this can be set to True to draw all hlines
    -> msg=False\t\tboolean if the reformatted DataFrame and the created string should be printed to the console
    """
    # input must be convertible to pandas.DataFrame
    df = DataFrame(content)

    # format_specifier
    formatter = f"{{:{format_spec}}}".format if format_spec is not None else None

    # append column specifier to inner settings
    if colspec:
        if isinstance(colspec, list):
            colspec = "".join(colspec)

        inner_settings.append(f"colspec={{{colspec}}}")
        # double curly braces produce literal curly brace in f string
        # three braces: evaluation surrounded by single braces

    # prepare columns for siunitx S columns
    # columns could be bool or Iterable[str]

    # check if columns has any truthy value
    if columns:

        # identity check with 'is' because columns could be a non-empty container
        # alternative: isinstance(columns, bool)
        if columns is True:
            columns = df.columns.tolist()

        # non-empty container
        else:
            # check if right amount of column labels was provided
            if len(columns) != len(df.columns):
                raise IndexError(
                    "'content' had a different amount of columns than provided 'columns'")
            else:
                # update columns of DataFrame
                df.columns = columns  # type: ignore

        # if columns was True, it's now a list
        # else it's still the provided Iterable with correct length
        # make strings safe for tabularry's siunitx S columns
        columns = [f"{{{{{{{col}}}}}}}" for col in columns]  # type: ignore

    # if falsy value, it should be False altogether
    else:
        columns = False

    # strings
    sisetup_str = ", ".join(sisetup)
    inner_settings_str = ",\n".join(inner_settings)
    hlines_str = "\\hline" if hlines_old else ""
    df_str: str = df.to_csv(sep="&", line_terminator=f"\\\\{hlines_str}\n",  # to_csv without path returns string
                            float_format=formatter, header=columns, index=index)  # type: ignore

    if uarray:
        # delete string quotes
        df_str = df_str.replace('"', '')

        # replace +/- with +-
        df_str = sub(r"(\d)\+/-(\d)", r"\1 +- \2", df_str)

        # delete parantheses and make extra spaces if exponents
        df_str = sub(r"\((\d+\.?\d*) \+- (\d+\.?\d*)\)e",
                     r"\1 +- \2 e", df_str)

    # create complete string
    complete_str = f"\\sisetup{{{sisetup_str}}}\n\n" if sisetup_str else ""
    complete_str += (f"\\begin{{{environ}}}{{{inner_settings_str}}}{hlines_str}\n"
                     f"{df_str}"
                     f"\\end{{{environ}}}")

    # write to file if path provided
    if path:
        # open() does not encode in utf-8 by default
        with open(path, "w", encoding="utf-8") as f:
            f.write(complete_str)

    # message printing
    if msg:
        # pd.options
        options.display.float_format = formatter

        print(f"Wrote pandas.DataFrame\n\n{df}\n\n"
              f"as tabularray environment '{environ}' to file '{path}'\n\n\n"
              f"output:\n\n{complete_str}")

    return complete_str


def profile(func: Callable) -> Callable:
    """A decorator for profiling a certain function call"""

    def decorator(*args, **kwargs):
        with Profile() as pr:
            func(*args, **kwargs)
        stats = Stats(pr)
        stats.sort_stats(SortKey.TIME)
        stats.dump_stats(f"_profiling_{func.__name__}.snakeviz")
        stats.print_stats()

    return decorator


def tracer(frame, event, arg):
    """Copy from StackOverflow"""
    indent = [0]

    def list_arguments():
        try:
            for i in range(frame.f_code.co_argcount):
                name = frame.f_code.co_varnames[i]
                print(f"\tArgument {name} = {frame.f_locals[name]}")
        except Exception as e:
            string = f"EXCEPTION: {e}"
            line = "\n" + "-" * len(string) + "\n"
            print(line + string + line)

    if event == "call":
        indent[0] += 2
        print("-" * indent[0] + "> call function", frame.f_code.co_name)
        list_arguments()
    elif event == "return":
        print("<" + "-" * indent[0], "exit function", frame.f_code.co_name)
        indent[0] -= 2
        list_arguments()
    else:
        pass
