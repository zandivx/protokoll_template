from labtool_ import labtool as lt


def table1():
    columns = ["Gerät", "Hersteller", "Modell",
               "Unsicherheit", "Anmerkung"]
    df = lt.pd.DataFrame(lt.np.round(lt.np.linspace(0.1, 1e8, 35),
                                     3).reshape((-1, 5)), columns=columns)

    lt.write_table(df, "tables.latex/table3.tex",
                   inner_settings=["hlines={blue}"],
                   colspec="S"*5,
                   environ="tblr-x",
                   columns=True,
                   format_spec="#.3e",
                   )


def table2():
    uarr = lt.unp.uarray(lt.np.linspace(1, 1e9, 36).reshape(-1, 4),
                         lt.np.array([i**2 for i in range(36)]).reshape(-1, 4))

    df = lt.pd.DataFrame(uarr, dtype="string")

    lt.write_table(df, "tables.latex/table4.tex",
                   colspec="S[table-number-alignment = center]"*4,
                   columns=["Test +/- 4e", "1", "2", "3"],
                   uarray=True,
                   sisetup=["uncertainty-mode=compact"],
                   msg=True)

# print(write_table.__doc__)


table2()
