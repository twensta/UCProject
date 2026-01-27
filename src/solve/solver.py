# src/solve/solver.py
from __future__ import annotations

import pyomo.environ as pyo


def solve_model(model: pyo.ConcreteModel, tee: bool = False) -> pyo.SolverResults:
    """
    Solve the UC MILP with HiGHS (via highspy).
    Requirements:
      pip install highspy
    """
    solver = pyo.SolverFactory("highs")
    if not solver.available(exception_flag=False):
        raise RuntimeError(
            "HiGHS solver not available. Install it with: pip install highspy"
        )

    results = solver.solve(model, tee=tee)

    term = str(results.solver.termination_condition).lower()
    if "infeasible" in term:
        raise RuntimeError("Solve finished as infeasible (HiGHS). Check your data/model.")

    return results
