from __future__ import annotations

# tests/test_build_and_solve_smoke.py
"""
Smoke test end-to-end (minimal):
  - build a tiny valid UC instance (data dict)
  - call build_model(data)
  - call solve_model(model) using HiGHS
  - check solve status is optimal

Run:
  python -m tests.test_build_and_solve_smoke
"""
import sys
from pathlib import Path

# Ajoute la racine du projet au PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parents[1]))




from typing import Dict, Any

import pyomo.environ as pyo

from src.model.build_model import build_model
from src.solve.solver import solve_model


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
            ("T1", 1): {"f": 0.0, "p": 0.0, "rho": 5.0},   # pho * f
            ("T1", 2): {"f": 10.0, "p": 50.0, "rho": 0.0}, # cap at 50
        },
    }
    return data


def main() -> None:
    data = make_minimal_data()

    # Build
    model = build_model(data)

    # Solve (HiGHS)
    results = solve_model(model, tee=False)

    status = str(results.solver.status).lower()
    term = str(results.solver.termination_condition).lower()

    print("Solve status:", results.solver.status)
    print("Termination:", results.solver.termination_condition)
    print("Objective:", pyo.value(model.OBJ))

    if "ok" not in status and "success" not in status:
        raise RuntimeError(f"Unexpected solver status: {results.solver.status}")
    if "optimal" not in term:
        raise RuntimeError(f"Expected optimal termination, got: {results.solver.termination_condition}")

    print("✅ build+solve smoke test passed.")


if __name__ == "__main__":
    main()
