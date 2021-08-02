class LeastSquares:
    def __init__(self, function, X, Y, y=None, names=None):
        from scipy.optimize import curve_fit as fit
        import numpy as numeric
        
        parameters, self.covariance = fit(function, X, Y, sigma=y)
        deltas = numeric.sqrt(numeric.diag(self.covariance))
        
        if names is None: names = [*range(len(parameters))]
        
        self.parameters = {}
        
        for name, parameter, delta in zip(names, parameters, deltas):
            self.parameters[name] = {"nominal": parameter, "deviation": delta}
        
        X = numeric.array(X)
        Y = numeric.array(Y)
        
        residuals = Y - function(X, *parameters)
        
        squares_sum = numeric.sum(residuals**2)
        squares_total = numeric.sum((Y - Y.mean())**2)
        
        self.determination_coefficient = 1 - squares_sum/squares_total
        
    def __getattr__(self, name):
        nominal = self.parameters[name]["nominal"]
        deviation = self.parameters[name]["deviation"]
        
        from uncertainties import ufloat as real
        
        return real(nominal, deviation)
        
    def __str__(self):
        return (
            f"determination coefficient: {self.determination_coefficient*100:.3f}\n"
            +"\n".join([f"{name}: {self.__getattr__(name)}" for name in self.parameters.keys()])
        )

class Orthogonal:
    def __init__(self, function, X, x, Y, y, estimate):
        import scipy.odr as regression
        
        model = regression.Model(lambda parameters, x: function(x, *parameters))
        data = regression.RealData(X, Y, x, y)
        
        calculation = regression.ODR(data, model, beta0=list(estimate.values()))
        result = calculation.run()
        
        self.information = result.stopreason
        self.covariance = result.cov_beta
        self.residual_variance = result.res_var
        
        self.parameters = {}
        
        for name, parameter, delta in zip(estimate.keys(), result.beta, result.sd_beta):
            self.parameters[name] = {"nominal": parameter, "deviation": delta}
        
    def __getattr__(self, name):
        nominal = self.parameters[name]["nominal"]
        deviation = self.parameters[name]["deviation"]
        
        from uncertainties import ufloat as real
        
        return real(nominal, deviation)
        
    def __str__(self):
        information = "halt: "+", ".join(self.information)
        
        variance = f"residual variance: {self.residual_variance:.5f}"
        
        variables = "\n".join([f"{name}: {self.__getattr__(name)}"
            for name in self.parameters.keys()])
        
        return "\n".join([information, variance, variables])
