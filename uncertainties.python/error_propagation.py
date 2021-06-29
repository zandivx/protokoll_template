import uncertainties as u
import uncertainties.umath as um
import uncertainties.unumpy as unp

tuple_ = (u, um, unp)

for i in tuple_:
    print(f"{i}: {dir(i)}\n")


# 1 comment 4 test
test = 123
