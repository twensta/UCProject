# src/main.py
import sys
from pathlib import Path

#Ajoute la racine du projet au PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pyomo.environ as pyo

from src.io.read_HT_data import read_smspp_file
from src.model.build_model import build_model
from src.solve.solver import solve_model
from src.io.visualize import visualize_results

from src.io.animation import animate_hydro_network

from src import config

def main():

    #Lecture des données
    print(f"Lecture des données d'entrées : {config.HT_FILE}...")

    raw_data = read_smspp_file(config.HT_FILE)  

    print("Données chargées avec succès")
    print("Horizon T =", raw_data["time"]["T"])

    #Construction du modèle
    print("Construction du modèle Pyomo...")
    model = build_model(raw_data)

    print("Modèle construit avec succès")

    #Export du modèle (optionnel)
    model.write(f"outputs/models/model_{config.MODEL_NAME}.lp", io_options={"symbolic_solver_labels": True})

    # Solve (HiGHS)
    print(f"Résolution du modèle avec le solveur : {config.SOLVER_NAME}...")

    results = solve_model(model, tee=False)

    status = str(results.solver.status)
    term = str(results.solver.termination_condition)

    print("Solve status:", results.solver.status)
    print("Termination:", results.solver.termination_condition)
    print("Objective:", pyo.value(model.OBJ))

    # Visualisation des résultats
    print("Visualisation des résultats...")
    visualize_results(model, outdir=Path("outputs"), show=False)

    # Si ton modèle a bien model.v[res,t] et model.f[arc,t]
    animate_hydro_network(model, outpath=Path("outputs/hydro.gif"), data=raw_data, fps=10, show=False)
    
if __name__ == "__main__":
    main()
