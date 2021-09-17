from timeit import default_timer as ts


def exponentiation(values: list):
    mapped = map(lambda x: x * 10**7, values)
    return list(mapped)


def e(values: list):
    mapped = map(lambda x: float(f"{x}e7"), values)
    return list(mapped)


list_ = [i/11 for i in range(20)]

t_0 = ts()
exponentiation(list_)
t_1 = ts()
e(list_)
t_2 = ts()

print(f"Exponentiation: {t_1-t_0} s\nFloat conversion: {t_2-t_1} s")
