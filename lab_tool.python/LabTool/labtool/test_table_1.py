# change working directory to path of the script
from uncertainties.core import ufloat
from write_table import write_table
import numpy as np
import pandas as pd
import uncertainties as u
import uncertainties.unumpy as unp
import os
import metrolopy as m
os.chdir(os.path.dirname(os.path.abspath(__file__)))

if True:
    columns = ["Gerät", "Hersteller", "Modell",
               "Unsicherheit", "Anmerkung"]
    df = pd.DataFrame(np.round(np.linspace(0.1, 1e8, 35),
                      3).reshape((-1, 5)), columns=columns)

    # write_table(df, "additionals/tests_with_tables/table2.tex", columns=columns,
    #             colspec="Q|Q|Q|S|S", digits=3)

    write_table(df, "additionals/tests_with_tables.latex/table3.tex",
                inner_settings=["hlines={blue}"],
                colspec="S"*5,
                environ="tblr-x",
                columns=True,
                format_spec="#.3e",
                )

uarr = unp.uarray(np.linspace(1, 1e9, 36).reshape(-1, 4),
                  np.array([i**2 for i in range(36)]).reshape(-1, 4))

df = pd.DataFrame(uarr)
df = df.astype("string")


write_table(df, path="additionals/tests_with_tables.latex/table4.tex",
            colspec="S[table-number-alignment = center]"*4,
            columns=["Test +/- 4e", "1", "2", "3"],
            uarray=True,
            sisetup=["uncertainty-mode=compact"])

# print(write_table.__doc__)
