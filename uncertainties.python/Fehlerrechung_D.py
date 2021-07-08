from uncertainties import ufloat
from uncertainties.umath import *

alpha = ufloat(radians(13.5), radians(0.5))
s = ufloat(0.1485, 0.0005)
t = ufloat(0.76, 0.12)
g = 9.81

names = ["alpha", "s", "t", "g"]
variables = [alpha, s, t]

function = (sin(alpha) - (2*s)/(g * t**2)) / cos(alpha)

derivatives = [function.derivatives[variable] for variable in variables]
terms = [abs(derivative * variable.s) for variable, derivative in zip(variables, derivatives)]

for name, derivative, term in zip(names, derivatives, terms):
    print(f"Derivative {name}:")
    print(f"  Factor:\t{derivative}")
    print(f"  Term:\t\t{term}\n")

print(f"Total uncertainty is {sum(terms)}")

# comment
