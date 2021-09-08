import labtool as lt


def table3():
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


def table4():
    uarr = lt.unp.uarray(lt.np.linspace(1, 1e8, 36).reshape(-1, 4),
                         (lt.np.arange(36)**1.5).reshape(-1, 4))

    lt.write_table(uarr, "tables.latex/table4.tex",
                   colspec="S[table-number-alignment = center]" * 4,
                   columns=False,  # ["Test +/- 4e", "1", "2", "3"],
                   uarray=True,
                   sisetup=["uncertainty-mode=compact"],
                   msg=True)


def table5():
    arr = lt.np.arange(24).reshape(-1, 4)

    lt.write_table(arr, "tables.latex/table5.tex",
                   environ="tabular",
                   inner_settings=["|c|c|c|c|"],
                   columns=["T1", "T2", "T3", "T4"],
                   hlines_old=True,
                   msg=True)


print(lt.write_table.__doc__)
print(lt.cd.__doc__)
# table4()
