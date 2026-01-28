# src/main.py
from pyexpat import model
import sys
from pathlib import Path
# Ajoute la racine du projet au PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pyomo.environ as pyo

from model import build_thermal_model
from src.io.read_data import read_smspp_file
from src import config
from src.model.build_model import build_model
from src.model.build_thermal_model import build_thermal_model
from src.io.visualize import visualize_results


from src.solve.solver import solve_model

def main():
    print(f"Lecture du fichier thermal : {config.THERMAL_FILE}")

    raw_data = read_smspp_file(config.THERMAL_FILE)  
    model = build_thermal_model(raw_data)

    print("Données chargées avec succès")
    print("Horizon T =", raw_data["time"]["T"])
    print("Nb thermiques =", len(raw_data["SETS"]["G"]))
    print("Nb arcs hydro =", len(raw_data["SETS"]["A"]))
    # Solve (HiGHS)
    for a in model.A:
        for t in model.T:
            model.f[a, t].fix(0.0)
            model.pa[a, t].fix(0.0)
    results = solve_model(model, tee=False)

    outdir = Path("outputs") / "plots"
    visualize_results(model, outdir=outdir, show=False, max_units=10)


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
