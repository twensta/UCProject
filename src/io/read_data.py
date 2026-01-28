# src/io/read_data.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union, Optional

import netCDF4
import numpy as np

def read_smspp_file(filename):
    """
    Lit UN fichier SMSPP netCDF4 (.nc4) et retourne un dict "data".
    """
    
    filename = Path(filename)
    data: Dict[str, Any] = {}

    nc = netCDF4.Dataset(str(filename), mode="r")
    block = nc["Block_0"]

    # Temps et ensembles
    TimeHorizon = block.dimensions["TimeHorizon"].size
    data["time"] = {"T": int(TimeHorizon), "delta_t": 1.0}
    data["SETS"] = {"T": list(range(1, int(TimeHorizon) + 1))}

    # Demande (somme sur toutes les zones si besoin)
    ActivePowerDemand = np.array(block["ActivePowerDemand"][:])
    data["demand"] = {t + 1: float(np.sum(ActivePowerDemand[t])) for t in range(TimeHorizon)}

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

    # Hydro (brut)
    data["SETS"]["V"] = []  # réservoirs
    data["SETS"]["A"] = []  # arcs
    data["arcs"] = {}       # dict brut par arc
    data["reservoirs"] = {} # dict brut par réservoir
    data["turbine_segments"] = {}

    # Parcours des groupes UnitBlock
    for gname, g in block.groups.items():
        gtype = g.getncattr("type") if "type" in g.ncattrs() else ""

        # --- Thermiques ---
        if gtype == "ThermalUnitBlock":
            unit_name = gname
            data["SETS"]["G"].append(unit_name)

            p_min_val = float(np.squeeze(g["MinPower"][:]))
            p_max_val = float(np.squeeze(g["MaxPower"][:]))
            startup_cost_val = float(np.squeeze(g["StartUpCost"][:]))
            RU_val = float(np.squeeze(g["DeltaRampUp"][:]))
            RD_val = float(np.squeeze(g["DeltaRampDown"][:]))
            min_up_val = int(np.squeeze(g["MinUpTime"][:]))
            min_down_val = int(np.squeeze(g["MinDownTime"][:]))
            cost_val = float(np.squeeze(g["LinearTerm"][:]))

            for t in range(1, TimeHorizon + 1):
                data["thermal"]["p_min"][(unit_name, t)] = p_min_val
                data["thermal"]["p_max"][(unit_name, t)] = p_max_val
                data["thermal"]["cost"][(unit_name, t)] = cost_val

            data["thermal"]["startup_cost"][unit_name] = startup_cost_val
            data["thermal"]["RU"][unit_name] = RU_val
            data["thermal"]["RD"][unit_name] = RD_val
            data["thermal"]["min_up"][unit_name] = min_up_val
            data["thermal"]["min_down"][unit_name] = min_down_val

        # --- Hydro ---
        elif gtype == "HydroUnitBlock":

            num_intervals = g.dimensions["NumberIntervals"].size if "NumberIntervals" in g.dimensions else TimeHorizon
            num_res = g.dimensions["NumberReservoirs"].size if "NumberReservoirs" in g.dimensions else 1
            num_arcs = g.dimensions["NumberArcs"].size if "NumberArcs" in g.dimensions else 1

            # =========================
            # INIT STRUCTURES (ONCE)
            # =========================
            if "A_turb" not in data["SETS"]:
                data["SETS"]["A_turb"] = []
            if "J" not in data["SETS"]:
                data["SETS"]["J"] = []

            if not data["reservoirs"]:
                data["reservoirs"] = {
                    "V0": {},
                    "Vmin": {},
                    "Vmax": {},
                    "inflow": {}
                }

            if not data["arcs"]:
                data["arcs"] = {
                    "from": {},
                    "to": {},
                    "f_min": {},
                    "f_max": {},
                    "p_min": {},
                    "p_max": {},
                    "RU": {},
                    "RD": {}
                }

            if "graph" not in data:
                data["graph"] = {
                    "In": {},
                    "Out": {}
                }

            # =========================
            # RESERVOIRS
            # =========================
            for r in range(num_res):
                res_name = f"{gname}_res{r}"
                data["SETS"]["V"].append(res_name)

                V0 = float(np.squeeze(g["InitialVolumetric"][r])) if "InitialVolumetric" in g.variables else 0.0
                inflows = g["Inflows"][r, :].tolist() if "Inflows" in g.variables else [0.0] * num_intervals
                Vmin = g["MinVolumetric"][r, :].tolist() if "MinVolumetric" in g.variables else [0.0] * num_intervals
                Vmax = g["MaxVolumetric"][r, :].tolist() if "MaxVolumetric" in g.variables else [1e9] * num_intervals

                data["reservoirs"]["V0"][res_name] = V0

                for t in range(1, num_intervals + 1):
                    data["reservoirs"]["inflow"][(res_name, t)] = float(inflows[t - 1])
                    data["reservoirs"]["Vmin"][(res_name, t)] = float(Vmin[t - 1])
                    data["reservoirs"]["Vmax"][(res_name, t)] = float(Vmax[t - 1])

                data["graph"]["In"][res_name] = []
                data["graph"]["Out"][res_name] = []

            # =========================
            # ARCS (TURBINES)
            # =========================
            for a in range(num_arcs):
                arc_name = f"{gname}_arc{a}"
                data["SETS"]["A"].append(arc_name)
                data["SETS"]["A_turb"].append(arc_name)

                min_flow = g["MinFlow"][:, a].tolist() if "MinFlow" in g.variables else [0.0] * num_intervals
                max_flow = g["MaxFlow"][:, a].tolist() if "MaxFlow" in g.variables else [1e6] * num_intervals
                min_power = g["MinPower"][:, a].tolist() if "MinPower" in g.variables else [0.0] * num_intervals
                max_power = g["MaxPower"][:, a].tolist() if "MaxPower" in g.variables else [1e6] * num_intervals

                RU_arr = g["DeltaRampUp"][:, a].tolist() if "DeltaRampUp" in g.variables else [0.0] * num_intervals
                RD_arr = g["DeltaRampDown"][:, a].tolist() if "DeltaRampDown" in g.variables else [0.0] * num_intervals

                data["arcs"]["from"][arc_name] = f"{gname}_res0"
                data["arcs"]["to"][arc_name] = f"{gname}_res0"
                data["arcs"]["RU"][arc_name] = float(max(RU_arr))
                data["arcs"]["RD"][arc_name] = float(max(RD_arr))

                for t in range(1, num_intervals + 1):
                    data["arcs"]["f_min"][(arc_name, t)] = float(min_flow[t - 1])
                    data["arcs"]["f_max"][(arc_name, t)] = float(max_flow[t - 1])
                    data["arcs"]["p_min"][(arc_name, t)] = float(min_power[t - 1])
                    data["arcs"]["p_max"][(arc_name, t)] = float(max_power[t - 1])

                # graph
                data["graph"]["Out"][f"{gname}_res0"].append(arc_name)
                data["graph"]["In"][f"{gname}_res0"].append(arc_name)

            # =========================
            # TURBINE PIECES (HPF)
            # =========================
            if "LinearTerm" in g.variables and "ConstantTerm" in g.variables:
                linear = g["LinearTerm"][:].tolist()
                constant = g["ConstantTerm"][:].tolist()

                for j in range(len(linear)):
                    if j not in data["SETS"]["J"]:
                        data["SETS"]["J"].append(j)

                    for a in data["SETS"]["A_turb"]:
                        if a.startswith(gname):
                            data["turbine_segments"][(a, j)] = {
                                "f": 0.0,
                                "p": float(constant[j]),
                                "rho": float(linear[j]),
                            }


    nc.close()
    return data

