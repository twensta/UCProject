import pyomo.environ as pyo
print(pyo.SolverFactory('highs').available())

