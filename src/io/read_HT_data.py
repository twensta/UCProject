# src/io/read_data.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import netCDF4
import numpy as np


def read_smspp_file(filename: str | Path) -> Dict[str, Any]:
    """
    Lit UN fichier SMSPP netCDF4 (.nc / .nc4) et retourne un dict "data"
    contenant:
      - time, SETS (T, G, V, A, A_turb, A_pump, J)
      - demand
      - thermal (p_min, p_max, cost, startup_cost, RU, RD, min_up, min_down)
      - reservoirs (V0, Vmin, Vmax, inflow)
      - arcs (from, to, f_min, f_max, p_min, p_max, RU, RD)
      - graph (In, Out)
      - turbine_segments
      - pump_rho (optionnel, pour pompes si présentes)
    """

    filename = Path(filename)
    data: Dict[str, Any] = {}

    nc = netCDF4.Dataset(str(filename), mode="r")
    try:
        block = nc["Block_0"] if "Block_0" in nc.groups else nc

        # -------------------------
        # TIME + SETS (base)
        # -------------------------
        TimeHorizon = int(block.dimensions["TimeHorizon"].size)
        data["time"] = {"T": TimeHorizon, "delta_t": 1.0}
        data["SETS"] = {"T": list(range(1, TimeHorizon + 1))}

        # -------------------------
        # DEMAND (somme si multi-zones)
        # -------------------------
        if "ActivePowerDemand" not in block.variables:
            raise KeyError("Variable 'ActivePowerDemand' introuvable sous Block_0.")

        ActivePowerDemand = np.array(block["ActivePowerDemand"][:])
        # gère (T,) ou (T, zones) ou (zones, T) etc. => on somme tout sauf l'axe temps
        if ActivePowerDemand.ndim == 1:
            data["demand"] = {t + 1: float(ActivePowerDemand[t]) for t in range(TimeHorizon)}
        else:
            # on suppose que l'axe temps est le premier si shape[0]==T, sinon le dernier si shape[-1]==T
            if ActivePowerDemand.shape[0] == TimeHorizon:
                data["demand"] = {t + 1: float(np.sum(ActivePowerDemand[t, ...])) for t in range(TimeHorizon)}
            elif ActivePowerDemand.shape[-1] == TimeHorizon:
                data["demand"] = {t + 1: float(np.sum(ActivePowerDemand[..., t])) for t in range(TimeHorizon)}
            else:
                # fallback: somme totale par t sur un reshape (moins propre, mais évite crash)
                flat = ActivePowerDemand.reshape(TimeHorizon, -1) if ActivePowerDemand.size % TimeHorizon == 0 else None
                if flat is None:
                    raise ValueError("ActivePowerDemand: impossible d’identifier l’axe temps.")
                data["demand"] = {t + 1: float(np.sum(flat[t])) for t in range(TimeHorizon)}

        # -------------------------
        # THERMAL init
        # -------------------------
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

        # -------------------------
        # HYDRO init (structures complètes, direct)
        # -------------------------
        data["SETS"]["V"] = []
        data["SETS"]["A"] = []
        data["SETS"]["A_turb"] = []
        data["SETS"]["A_pump"] = []
        data["SETS"]["J"] = []

        data["reservoirs"] = {"V0": {}, "Vmin": {}, "Vmax": {}, "inflow": {}}

        data["arcs"] = {
            "from": {},
            "to": {},
            "p": {},
            "r": {},
            "RU": {},
            "RD": {},
            "p_max": {},
            "p_min": {},
            "f_max": {},
            "f_min": {},
        }

        data["graph"] = {"In": {}, "Out": {}}

        # -------------------------
        # helpers
        # -------------------------
        def _attr(group, name: str, default=""):
            return group.getncattr(name) if name in group.ncattrs() else default

        def _squeeze_float(var) -> float:
            return float(np.squeeze(np.array(var[:])))

        def _squeeze_int(var) -> int:
            return int(np.squeeze(np.array(var[:])))

        def _ensure_res(res_name: str):
            if res_name not in data["graph"]["In"]:
                data["graph"]["In"][res_name] = []
            if res_name not in data["graph"]["Out"]:
                data["graph"]["Out"][res_name] = []

        # -------------------------
        # PARCOURS UnitBlocks
        # -------------------------
        for gname, g in block.groups.items():
            gtype = _attr(g, "type", "")

            # ========= THERMAL =========
            if gtype == "ThermalUnitBlock":
                unit_name = f"{gname}_Thermal"
                data["SETS"]["G"].append(unit_name)

                # scalaires
                p_min_val = _squeeze_float(g["MinPower"])
                p_max_val = _squeeze_float(g["MaxPower"])
                startup_cost_val = _squeeze_float(g["StartUpCost"])
                RU_val = _squeeze_float(g["DeltaRampUp"])
                RD_val = _squeeze_float(g["DeltaRampDown"])
                min_up_val = _squeeze_int(g["MinUpTime"])
                min_down_val = _squeeze_int(g["MinDownTime"])
                cost_val = _squeeze_float(g["LinearTerm"])  # coût linéaire utilisé dans ton modèle

                # time-expanded (toutes périodes identiques dans ce format)
                for t in range(1, TimeHorizon + 1):
                    data["thermal"]["p_min"][(unit_name, t)] = p_min_val
                    data["thermal"]["p_max"][(unit_name, t)] = p_max_val
                    data["thermal"]["cost"][(unit_name, t)] = cost_val

                # unit-level
                data["thermal"]["startup_cost"][unit_name] = startup_cost_val
                data["thermal"]["RU"][unit_name] = RU_val
                data["thermal"]["RD"][unit_name] = RD_val
                data["thermal"]["min_up"][unit_name] = min_up_val
                data["thermal"]["min_down"][unit_name] = min_down_val

            # ========= HYDRO =========
            elif gtype == "HydroUnitBlock":
                num_intervals = int(g.dimensions["NumberIntervals"].size) if "NumberIntervals" in g.dimensions else TimeHorizon
                num_res = int(g.dimensions["NumberReservoirs"].size) if "NumberReservoirs" in g.dimensions else 1
                num_arcs = int(g.dimensions["NumberArcs"].size) if "NumberArcs" in g.dimensions else 1

                # ----- RESERVOIRS -----
                for r in range(num_res):
                    res_name = f"{gname}_res{r}"
                    data["SETS"]["V"].append(res_name)
                    _ensure_res(res_name)

                    V0 = float(np.squeeze(g["InitialVolumetric"][r])) if "InitialVolumetric" in g.variables else 0.0

                    inflows = np.squeeze(g["Inflows"][r]) if "Inflows" in g.variables else 0.0
                    Vmin = np.squeeze(g["MinVolumetric"][r]) if "MinVolumetric" in g.variables else 0.0
                    Vmax = np.squeeze(g["MaxVolumetric"][r]) if "MaxVolumetric" in g.variables else 1e12

                    data["reservoirs"]["V0"][res_name] = float(V0)*1e6
                    data["reservoirs"]["Vmin"][res_name] = float(Vmin)*1e6
                    data["reservoirs"]["Vmax"][res_name] = float(Vmax)*1e6

                    for t in range(1, num_intervals + 1):
                        data["reservoirs"]["inflow"][(res_name, t)] = float(inflows)

                # ----- ARCS -----
                # Ici: on lit des arcs "turbines" (et si tu as des pompes, on peut les détecter via un attribut ou variable)
                # Par défaut, on les met dans A_turb.

                linear_terms = g["LinearTerm"][:] if "LinearTerm" in g.variables else None
                constant_terms = g["ConstantTerm"][:] if "ConstantTerm" in g.variables else None

                for a in range(num_arcs):

                    
                    arc_name = f"{gname}_arc{a}"
                    data["SETS"]["A"].append(arc_name)

                    min_flow = float(np.squeeze(g["MinFlow"][0,a])) if "MinFlow" in g.variables else 0.0
                    max_flow = float(np.squeeze(g["MaxFlow"][0,a])) if "MaxFlow" in g.variables else 1e9

                    min_power = float(np.squeeze(g["MinPower"][0,a])) if "MinPower" in g.variables else 0.0
                    max_power = float(np.squeeze(g["MaxPower"][0,a])) if "MaxPower" in g.variables else 1e9

                    RU = float(np.squeeze(g["DeltaRampUp"][0,a])) if "DeltaRampUp" in g.variables else 0.0
                    RD = float(np.squeeze(g["DeltaRampDown"][0,a])) if "DeltaRampDown" in g.variables else 0.0

                    number_pieces = int(np.squeeze(g["NumberPieces"][a])) if "NumberPieces" in g.variables else 0

                    if max_power > 0:
                        data["SETS"]["A_turb"].append(arc_name)
                        u = +1  # turbine
                    else:
                        data["SETS"]["A_pump"].append(arc_name)
                        u = -1  # pompe

                    data["arcs"]["f_min"][arc_name] = float(min_flow)
                    data["arcs"]["f_max"][arc_name] = float(max_flow)
                    data["arcs"]["p_min"][arc_name] = float(min_power)
                    data["arcs"]["p_max"][arc_name] = float(max_power)
                    data["arcs"]["RU"][arc_name] = float(RU)
                    data["arcs"]["RD"][arc_name] = float(RD)


                    for j in range(number_pieces):
                        pj = float(np.squeeze(constant_terms[a+j])) if constant_terms is not None else 0.0
                        rj = float(np.squeeze(u*linear_terms[a+j])) if linear_terms is not None else 0.0

                        data["arcs"]["p"][(arc_name, "p_" + str(j))] = float(pj)
                        data["arcs"]["r"][(arc_name, "r_" + str(j))] = float(rj)


                    from_res = int(np.squeeze(g["StartArc"][a])) if "FromReservoir" in g.variables else -1
                    to_res = int(np.squeeze(g["EndArc"][a])) if "EndArc" in g.variables else -1

                    if to_res < num_res :
                        data["arcs"]["from"][arc_name] = f"{gname}_res{from_res}"
                        data["arcs"]["to"][arc_name] = f"{gname}_res{to_res}"
                    else :
                        data["arcs"]["from"][arc_name] = f"{gname}_res{from_res}"
                        data["arcs"]["to"][arc_name] = "sink"
        return data

    finally:
        nc.close()
