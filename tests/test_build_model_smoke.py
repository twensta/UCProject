# tests/test_build_model_smoke.py

"""
Smoke test: verify that `build_model(data)` builds a valid Pyomo model on a tiny instance.

This file builds a minimal but VALID `data` dict compatible with src/model/build_model.py.

Expected `data` format (example used below):

data = {
  "time": {"T": 3, "delta_t": 1.0},
  "SETS": {
    "T": [1, 2, 3],
    "G": ["G1"],
    "V": ["R1", "R2"],
    "A": ["T1"],          # arcs = turbines + pumps
    "A_turb": ["T1"],
    "A_pump": [],
    "J": [1, 2]           # segment indices
  },
  "demand": {1: 20.0, 2: 20.0, 3: 20.0},

  "thermal": {
    "p_min": {("G1", 1): 10.0, ("G1", 2): 10.0, ("G1", 3): 10.0},
    "p_max": {("G1", 1): 100.0, ("G1", 2): 100.0, ("G1", 3): 100.0},
    "cost":  {("G1", 1): 50.0, ("G1", 2): 50.0, ("G1", 3): 50.0},
    "startup_cost": {"G1": 200.0},
    "RU": {"G1": 100.0},
    "RD": {"G1": 100.0},
    "min_up": {"G1": 1},
    "min_down": {"G1": 1},
  },

  "reservoirs": {
    "V0": {"R1": 1000.0, "R2": 1000.0},
    "Vmin": {("R1", 1): 0.0, ("R1", 2): 0.0, ("R1", 3): 0.0,
             ("R2", 1): 0.0, ("R2", 2): 0.0, ("R2", 3): 0.0},
    "Vmax": {("R1", 1): 2000.0, ("R1", 2): 2000.0, ("R1", 3): 2000.0,
             ("R2", 1): 2000.0, ("R2", 2): 2000.0, ("R2", 3): 2000.0},
    "inflow": {("R1", 1): 0.0, ("R1", 2): 0.0, ("R1", 3): 0.0,
               ("R2", 1): 0.0, ("R2", 2): 0.0, ("R2", 3): 0.0},
  },

  "arcs": {
    "from": {"T1": "R1"},
    "to": {"T1": "R2"},
    "f_min": {("T1", 1): 0.0, ("T1", 2): 0.0, ("T1", 3): 0.0},
    "f_max": {("T1", 1): 10.0, ("T1", 2): 10.0, ("T1", 3): 10.0},
    "RU": {"T1": 10.0},
    "RD": {"T1": 10.0},
    # Power bounds for hydro arc (must exist in current build_model.py)
    "p_min": {("T1", 1): 0.0, ("T1", 2): 0.0, ("T1", 3): 0.0},
    "p_max": {("T1", 1): 50.0, ("T1", 2): 50.0, ("T1", 3): 50.0},
  },

  "graph": {
    "In": {"R1": [], "R2": ["T1"]},
    "Out": {"R1": ["T1"], "R2": []},
  },

  "turbine_segments": {
    ("T1", 1): {"f": 0.0,  "p": 0.0,  "rho": 5.0},  # p <= 0 + 5(f-0) => p<=5f
    ("T1", 2): {"f": 10.0, "p": 50.0, "rho": 0.0},  # p <= 50
  },

  # For pumps, you would also provide:
  # "pump_rho": {"P1": some_value}
}

Run:
  python -m tests.test_build_model_smoke
"""

from __future__ import annotations

from typing import Dict, Any

import pyomo.environ as pyo

from src.model.build_model import build_model


def make_minimal_data() -> Dict[str, Any]:
    T = [1, 2, 3]
    G = ["G1"]
    V = ["R1", "R2"]
    A = ["T1"]
    J = [1, 2]

    data: Dict[str, Any] = {
        "time": {"T": len(T), "delta_t": 1.0},
        "SETS": {"T": T, "G": G, "V": V, "A": A, "A_turb": ["T1"], "A_pump": [], "J": J},
        "demand": {1: 20.0, 2: 20.0, 3: 20.0},
        "thermal": {
            "p_min": {("G1", t): 10.0 for t in T},
            "p_max": {("G1", t): 100.0 for t in T},
            "cost": {("G1", t): 50.0 for t in T},
            "startup_cost": {"G1": 200.0},
            "RU": {"G1": 100.0},
            "RD": {"G1": 100.0},
            "min_up": {"G1": 1},
            "min_down": {"G1": 1},
        },
        "reservoirs": {
            "V0": {"R1": 1000.0, "R2": 1000.0},
            "Vmin": {(v, t): 0.0 for v in V for t in T},
            "Vmax": {(v, t): 2000.0 for v in V for t in T},
            "inflow": {(v, t): 0.0 for v in V for t in T},
        },
        "arcs": {
            "from": {"T1": "R1"},
            "to": {"T1": "R2"},
            "f_min": {("T1", t): 0.0 for t in T},
            "f_max": {("T1", t): 10.0 for t in T},
            "RU": {"T1": 10.0},
            "RD": {"T1": 10.0},
            "p_min": {("T1", t): 0.0 for t in T},
            "p_max": {("T1", t): 50.0 for t in T},
        },
        "graph": {"In": {"R1": [], "R2": ["T1"]}, "Out": {"R1": ["T1"], "R2": []}},
        "turbine_segments": {
            ("T1", 1): {"f": 0.0, "p": 0.0, "rho": 5.0},
            ("T1", 2): {"f": 10.0, "p": 50.0, "rho": 0.0},
        },
    }
    return data


def main() -> None:
    data = make_minimal_data()
    m = build_model(data)

    # Basic sanity checks: components exist
    assert hasattr(m, "OBJ")
    assert hasattr(m, "Balance")
    assert hasattr(m, "V")
    assert len(list(m.T)) == 3
    assert len(list(m.G)) == 1
    assert len(list(m.R)) == 2
    assert len(list(m.A)) == 1

    # Check the model can be written to LP (common way to detect build errors)
    m.write(filename="outputs/models/_smoke_test.lp", io_options={"symbolic_solver_labels": True})

    # Optional: try solving if solver is available (doesn't fail test if not)
    solver = pyo.SolverFactory("highs")
    if solver.available():
        res = solver.solve(m, tee=False)
        print("Solve status:", res.solver.status)
        print("Termination:", res.solver.termination_condition)
        print("Objective:", pyo.value(m.OBJ))
    else:
        print("HiGHS not available; build-only smoke test passed.")

    print("✅ build_model smoke test passed.")


if __name__ == "__main__":
    main()
