# src/main.py
import sys
from pathlib import Path
import pyomo.environ as pyo

# Ajoute la racine du projet au PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parents[1]))

from tests.test_read_hydro_data import read_hydro_data
from src import config
from src.model.build_hydro_model import build_hydro_model
from src.solve.solver import solve_model

def main():
    print(f"Lecture du fichier SMSPP : {config.HYDRO_FILE}")

    raw_data = read_hydro_data(config.HYDRO_FILE)  
    model = build_hydro_model(raw_data)

    print("Données chargées avec succès")
    print("Horizon T =", raw_data["time"]["T"])


    
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


if __name__ == "__main__":
    main()
