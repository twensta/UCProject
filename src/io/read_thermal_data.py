# src/io/read_thermal_data.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

import netCDF4
import numpy as np


def read_thermal_file(filename: Union[str, Path]) -> Dict[str, Any]:
    """
    Lit un fichier netCDF4 thermique (UCBlock + ThermalUnitBlock) et retourne un dict `data`
    avec EXACTEMENT la même structure que `read_smspp_file` (version actuelle dans src/io/read_data.py).

    Structure garantie en sortie :
      - data["time"] = {"T": int, "delta_t": 1.0}
      - data["SETS"] = {"T": [...], "G": [...], "V": [...], "A": [...]}
      - data["demand"] = {t: float}
      - data["thermal"] = {"p_min":{(g,t):...}, "p_max":..., "cost":..., "startup_cost":..., "RU":..., "RD":..., "min_up":..., "min_down":...}
      - data["arcs"] = {}               (vide, car pas d'hydro)
      - data["reservoirs"] = {}         (vide, car pas d'hydro)
      - data["turbine_segments"] = {}   (vide, car pas d'hydro)
    """
    filename = Path(filename)
    if not filename.exists():
        raise FileNotFoundError(f"NetCDF file not found: {filename}")

    data: Dict[str, Any] = {}

    nc = netCDF4.Dataset(str(filename), mode="r")
    try:
        if "Block_0" not in nc.groups:
            raise KeyError("Group 'Block_0' not found in the NetCDF file.")
        block = nc["Block_0"]

        # Temps et ensembles
        if "TimeHorizon" not in block.dimensions:
            raise KeyError("Dimension 'TimeHorizon' not found in Block_0.")
        TimeHorizon = int(block.dimensions["TimeHorizon"].size)

        data["time"] = {"T": TimeHorizon, "delta_t": 1.0}
        data["SETS"] = {"T": list(range(1, TimeHorizon + 1))}

        # Demande
        if "ActivePowerDemand" not in block.variables:
            raise KeyError("Variable 'ActivePowerDemand' not found in Block_0.")
        ActivePowerDemand = np.array(block["ActivePowerDemand"][:])

        # Dans ton read_smspp_file: tu fais sum(ActivePowerDemand[t]) (prévu aussi si multidim)
        # Ici c'est 1D, mais on garde la même logique robuste.
        data["demand"] = {
            t + 1: float(np.sum(ActivePowerDemand[t])) for t in range(TimeHorizon)
        }

        # Thermiques
        data["SETS"]["G"] = []
        data["thermal"] = {
            "p_min": {},
            "p_max": {},
            "cost": {},
            "startup_cost": {},
            "RU": {},
            "RD": {},
            "min_up": {},
            "min_down": {},
        }

        # Hydro (vides mais présentes)
        data["SETS"]["V"] = []
        data["SETS"]["A"] = []
        data["arcs"] = {}
        data["reservoirs"] = {}
        data["turbine_segments"] = {}

        # Parcours des groupes UnitBlock
        for gname, g in block.groups.items():
            gtype = g.getncattr("type") if "type" in g.ncattrs() else ""

            if gtype != "ThermalUnitBlock":
                continue

            unit_name = gname
            data["SETS"]["G"].append(unit_name)

            # Variables thermiques (scalaires)
            p_min_val = float(np.squeeze(g["MinPower"][:]))
            p_max_val = float(np.squeeze(g["MaxPower"][:]))
            startup_cost_val = float(np.squeeze(g["StartUpCost"][:]))
            RU_val = float(np.squeeze(g["DeltaRampUp"][:]))
            RD_val = float(np.squeeze(g["DeltaRampDown"][:]))
            min_up_val = int(np.squeeze(g["MinUpTime"][:]))
            min_down_val = int(np.squeeze(g["MinDownTime"][:]))
            cost_val = float(np.squeeze(g["LinearTerm"][:]))

            # Remplissage indexé temps (comme ton reader)
            for t in range(1, TimeHorizon + 1):
                data["thermal"]["p_min"][(unit_name, t)] = p_min_val
                data["thermal"]["p_max"][(unit_name, t)] = p_max_val
                data["thermal"]["cost"][(unit_name, t)] = cost_val

            # Paramètres unitaires
            data["thermal"]["startup_cost"][unit_name] = startup_cost_val
            data["thermal"]["RU"][unit_name] = RU_val
            data["thermal"]["RD"][unit_name] = RD_val
            data["thermal"]["min_up"][unit_name] = min_up_val
            data["thermal"]["min_down"][unit_name] = min_down_val

        if len(data["SETS"]["G"]) == 0:
            raise ValueError("No ThermalUnitBlock found in Block_0 groups.")

        return data

    finally:
        nc.close()
