# change working directory to path of the script
from write_table import write_table
import numpy as np
import pandas as pd
import uncertainties as u
import uncertainties.unumpy as unp
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))


columns = [r"{{{Gerät}}}", r"{{{Hersteller}}}", r"{{{Modell}}}",
           r"{{{Unsicherheit}}}", r"{{{Anmerkung}}}"]
df = pd.DataFrame(np.round(np.linspace(0.1, 1e8, 35),
                  3).reshape((-1, 5)), columns=columns)

# write_table(df, "table2.tex", columns=columns,
#             colspec="Q|Q|Q|S|S", digits=3)

write_table(df, "tests_with_tables.latex/table3.tex",
            # inner_settings=[r"columns={l,h}", r"hlines={blue}"],
            colspec="SSSSS",
            environ="tblr-x",
            columns=True,
            format_spec="#.3e",
            # msg=True
            )

uarr = unp.uarray(np.linspace(1, 1e9, 35).reshape(-1, 5),
                  np.array([i**2 for i in range(35)]).reshape(-1, 5))

write_table(uarr, "tests_with_tables.latex/table4.tex",
            colspec="SSSSS",
            msg=True)

# print(write_table.__doc__)
