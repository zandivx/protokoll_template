from uncertainties.umath import *

def evaluate(function, variables, constants):
    return eval(function, None, variables | constants)
