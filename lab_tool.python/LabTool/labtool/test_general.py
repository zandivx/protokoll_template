import core
import matplotlib.pyplot as plt
import pandas as pd
import os
import inspect

# core.plt_latex()
# core.chdir()

#plt.plot([i for i in range(10)], [i**2 for i in range(10)])
#plt.title("LaTeX test ß €")
# plt.show()

#req = pd.read_csv("requirements.txt")
# print(req)

df = core.Student().sigma_df
print(df)
