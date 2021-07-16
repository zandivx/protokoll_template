# imports
import pandas as pd
from typing import Union, Any


def write_table(
    df: Any,
    path: str,
    environ: str = "tblr",
    colspec: str = "",
    inner_settings: list[str] = [],
    columns: Union[bool, list[str]] = False,
    index: bool = False,
    format_spec: Union[None, str] = None,
    msg: bool = False
) -> None:
    """
    Creates a tex-file with a correctly formatted table for LaTeX-package 'tabularray' from the given input-array.

    Mandatory parameters:
    -> df\t\t\tarray-like, must be convertible to pandas.DataFrame
    -> path\t\t\tname (opt. relative path) to tex-file for writing the table to

    Optional parameters:
    -> environ='tblr'\t\ttblr environment specified in file 'tabularray-environments.tex'
    -> colspec=''\t\tcolumn specifier known from standard LaTeX tables
    -> inner_settings=[]\tadditional settings for the tabularray environment (see documentation), input as list of strings
    -> columns=False\t\twrite table header (first row), input as list of strings or boolean
    -> index=False\t\tboolean if indices of rows (first column) should be written to table
    -> format_spec=None\t\tfloat formatter (e.g. .3f) or None if floats should not be formatted specifically
    -> msg=False\t\tboolean if a success-message should be printed to the console
    """
    try:
        # input must be convertible to pandas.DataFrame
        df = pd.DataFrame(df)

        # append column specifier to inner settings
        if colspec:
            # double curly braces produce literal curly brace in f string
            # three braces: evaluation surrounded by single braces
            inner_settings.append(f"colspec={{{colspec}}}")
        inner_settings_str = ", ".join(inner_settings)

        # format_specifier
        formatter = f"{{:{format_spec}}}".format if format_spec is not None else None

        # write mode
        with open(path, "w") as f:
            f.write(f"\\begin{{{environ}}}{{{inner_settings_str}}}\n")

        # append mode
        df.to_csv(path, header=columns, index=index, mode="a",
                  sep="&", line_terminator="\\\\\n", float_format=formatter)

        # append mode
        with open(path, "a") as f:
            f.write(f"\\end{{{environ}}}")

        # message printing
        if msg:
            pd.options.display.float_format = formatter
            print(
                f"Successfully written pandas.DataFrame:\n{df}\nas tabularray environment '{environ}' to file: '{path}'.")

    except Exception as exc:
        print(f"ERROR!\n{exc}")
        # import traceback
        # traceback.print_exc(limit=1)
