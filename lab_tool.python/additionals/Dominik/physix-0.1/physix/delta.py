class Student:
    def __init__(self, extent, niveau="low"):
        names = {"low": "68.3%", "medium": "95.5%", "high": "99.7%"}
        self.niveau = names[niveau]
        
        import os as system
        
        location = system.path.dirname(system.path.realpath(__file__))
        path = system.path.join(location, "data", "student")
        
        import pandas as backend
        
        data = backend.read_csv(path, delimiter="\s+")
        
        pairs = [(row["N"], row[self.niveau]) for index, row in data.iterrows()]
        
        distances = [abs(extent - key) for key, value in pairs]
        index = distances.index(min(distances))
        
        if index > 0 and pairs[index][0] > extent:
            index -= 1
        
        self.extent, self.factor = pairs[index]
        
    def __str__(self):
        return f"niveau: {self.niveau}, extent: {self.extent}, factor: {self.factor}"

class Series:
    def __init__(self, values, uncertainty=0, niveau="low"):
        values = [value for value in values]
        self.extent = len(values)
        
        self.student = Student(self.extent, niveau)
        
        import math, statistics
        
        self.mean = statistics.mean(values)
        self.single_deviation = statistics.stdev(values)
        self.mean_deviation = self.single_deviation/math.sqrt(self.extent) * self.student.factor
        
        self.uncertainty = uncertainty
        self.delta = self.mean_deviation + self.uncertainty
        
    def __str__(self):
        return (
            f"extent: {self.extent}\n"
            f"student: {{{self.student}}}\n"
            f"mean: {self.mean}\n"
            f"single_deviation: {self.single_deviation}\n"
            f"mean_deviation: {self.mean_deviation}\n"
            f"uncertainty: {self.uncertainty}\n"
            f"delta: {self.delta}"
        )

class Impact:
    def __init__(self, function, variables, constants={}):
        self.function = function
        
        import uncertainties
        
        self.variables = {}
        
        for key, value in variables.items():
            self.variables[key] = uncertainties.ufloat_fromstr(value)
        
        self.constants = constants
        
        from physix.math import evaluate
        
        expression = evaluate(function, self.variables, self.constants)
        derivatives = [expression.derivatives[value] for value in self.variables.values()]
        
        summands = [abs(derivative * variable.s)
            for variable, derivative in zip(self.variables.values(), derivatives)]
        
        self.summands = {name: summand for name, summand in zip(self.variables.keys(), summands)}
        self.result = uncertainties.ufloat(expression.n, sum(self.summands.values()))
        
    def __str__(self):
        return (
            f"function: {self.function}\n"
            f"variables: {self.variables}\n"
            f"constants: {self.constants}\n"
            f"summands: {self.summands}\n"
            f"result: {self.result}"
        )
