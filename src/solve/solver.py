# src/solve/solver.py
import pyomo.environ as pyo

def solve_model(model: pyo.ConcreteModel, tee: bool = False):
    """
    Solve the UC MILP with HiGHS (via highspy).
    """
    solver = pyo.SolverFactory("highs")
    results = solver.solve(model, tee=tee)

    return results
