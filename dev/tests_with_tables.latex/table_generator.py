from typing import Union, Any
import numpy as np
#from numpy.typing import ArrayLike
import pandas as pd

# change working directory to path of the script
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def write_table(
    df: Any,
    file: str,
    environ: str = "tblr",
    colspec: str = "",
    inner_settings: list[str] = [],
    columns: Union[bool, list[str]] = False,
    index: bool = False,
    digits: int = 0,
    msg: bool = False
) -> None:

    try:
        df = pd.DataFrame(df)
    except ValueError:
        raise ValueError(
            "Table content must be convertible to pandas.DataFrame!")
    else:
        if colspec:
            inner_settings.append("colspec={" + colspec + "}")
        inner_settings_str = ", ".join(inner_settings)

        with open(file, "w") as f:  # write mode
            f.write(r"\begin{" + environ + "}{" + inner_settings_str + "}\n")

        df.to_csv(file, header=columns, index=index, mode="a",  # append mode
                  sep="&", line_terminator=r"\\"+"\n", float_format=f"%.{digits}f")

        with open(file, "a") as f:  # append mode
            f.write(r"\end{" + environ + "}")

        if msg:
            print(
                f"Successfully written\n{df}\nas tabularray table to file '{file}'.")


columns = ["Gerät", "Hersteller", "Modell",
           r"{{{Unsicherheit}}}", r"{{{Anmerkung}}}"]
df = pd.DataFrame(np.round(np.linspace(0, 11, 20),
                  3).reshape((-1, 5)), columns=columns)

# write_table(df, "table2.tex", columns=columns,
#             colspec="Q|Q|Q|S|S", digits=3)

write_table(df, "table3.tex",
            # inner_settings=[r"columns={l,h}", r"hlines={blue}"],
            colspec="SSSSS",
            environ="tblrx",
            digits=3,
            msg=True)
