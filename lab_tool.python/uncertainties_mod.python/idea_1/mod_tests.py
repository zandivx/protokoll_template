# type:ignore
import uncertainties_mod as u
import unumpy_mod as unp

#print(um.ufloat(3.4, 0.182).format("L"))

uarr = unp.uarray([3.4, 3.5], [0.182, 0.25])
print(uarr)
