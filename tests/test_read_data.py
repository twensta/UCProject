# src/test_read_data.py

import sys
from pathlib import Path
from pprint import pprint

# Ajoute la racine du projet au PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.io.read_HT_data import read_smspp_file
from src import config

def main():
    print(f"Lecture du fichier SMSPP : {config.HT_FILE}")

    raw_data = read_smspp_file(config.HT_FILE)

    print("Données chargées avec succès")
    print("Horizon T =", raw_data["time"]["T"])
    print("Nb thermiques =", len(raw_data["SETS"]["G"]))
    print("Nb arcs hydro =", len(raw_data["SETS"]["A"]))

    pprint(raw_data["arcs"])
    pprint(raw_data["arcs"])

    """missing = [a for a, props in raw_data["arcs"].items() if "from" not in props]
    print("Nb arcs sans 'from':", len(missing))
    print("Exemples:", missing[:10])"""


if __name__ == "__main__":
    main()
