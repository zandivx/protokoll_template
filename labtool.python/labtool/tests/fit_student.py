import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import core

a = """
if False:
    df = core.Student.t_df

    popt, *_ = core.curve_fit(lambda x, a: a * x **
                              (-2.0) + 1, df["N"], df["68.3%"])
    print(popt)
    x = np.linspace(0, 200, 300)
    y = x**popt
    plt.plot(x, y, label="fit")

    plt.plot(df["N"], df["68.3%"], label="data")
    #plt.plot(df["N"], df["95.5%"])
    #plt.plot(df["N"], df["99.7%"])
    plt.ylim(0.9, 2)
    plt.legend()
    plt.show()
"""
