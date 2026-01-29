from typing import Dict, Any
import netCDF4 as nc
import numpy as np


def read_hydro_data(path: str) -> Dict[str, Any]:
    ds = nc.Dataset(path)

    data: Dict[str, Any] = {}

    # =========================
    # BLOCK & TIME
    # =========================
    block = ds["Block_0"]
    T = block.dimensions["TimeHorizon"].size

    data["time"] = {
        "T": T,
        "delta_t": 1.0,
    }

    data["SETS"] = {
        "T": list(range(1, T + 1)),
        "G": [],
        "V": [],
        "A": [],
        "A_turb": [],
        "A_pump": [],
        "J": [],
    }

    # =========================
    # DEMAND
    # =========================
    demand_raw = np.squeeze(block["ActivePowerDemand"][:])

    data["demand"] = {
        t + 1: float(demand_raw[t])
        for t in range(T)
    }

    

    # =========================
    # HYDRO UNIT
    # =========================
    g = block["UnitBlock_0"]

    num_res = g.dimensions["NumberReservoirs"].size
    num_arcs = g.dimensions["NumberArcs"].size
    num_intervals = g.dimensions["NumberIntervals"].size

    # =========================
    # RESERVOIRS
    # =========================
    data["reservoirs"] = {
        "V0": {},
        "Vmin": {},
        "Vmax": {},
        "inflow": {},
    }

    data["graph"] = {"In": {}, "Out": {}}

    for r in range(num_res):
        v = f"res{r}"
        data["SETS"]["V"].append(v)

        data["reservoirs"]["V0"][v] = float(g["InitialVolumetric"][r])

        for t in range(1, T + 1):
            data["reservoirs"]["Vmin"][(v, t)] = float(g["MaxVolumetric"][r, 0] * 0.0)
            data["reservoirs"]["Vmax"][(v, t)] = float(g["MaxVolumetric"][r, 0])
            data["reservoirs"]["inflow"][(v, t)] = float(g["Inflows"][r, 0])

        data["graph"]["In"][v] = []
        data["graph"]["Out"][v] = []

    # =========================
    # ARCS
    # =========================
    data["arcs"] = {
        "from": {},
        "to": {},
        "f_min": {},
        "f_max": {},
        "p_min": {},
        "p_max": {},
        "RU": {},
        "RD": {},
    }

    for a in range(num_arcs):
        arc = f"arc{a}"
        data["SETS"]["A"].append(arc)

        start = int(g["StartArc"][a])
        end = int(g["EndArc"][a])

        v_from = f"res{start}"
        v_to = f"res{end}"

        data["arcs"]["from"][arc] = v_from
        data["arcs"]["to"][arc] = v_to

    # from reservoir (toujours valide dans ton fichier)
        data["graph"]["Out"][v_from].append(arc)

# to reservoir : seulement si interne
        if end < num_res:
            data["graph"]["In"][v_to].append(arc)


        data["arcs"]["RU"][arc] = float(g["DeltaRampUp"][0, a])
        data["arcs"]["RD"][arc] = float(g["DeltaRampDown"][0, a])

        is_turb = False
        is_pump = False

        for t in range(1, T + 1):
            fmin = float(g["MinFlow"][0, a])
            fmax = float(g["MaxFlow"][0, a])
            pmin = float(g["MinPower"][0, a])
            pmax = float(g["MaxPower"][0, a])

            data["arcs"]["f_min"][(arc, t)] = fmin
            data["arcs"]["f_max"][(arc, t)] = fmax
            data["arcs"]["p_min"][(arc, t)] = pmin
            data["arcs"]["p_max"][(arc, t)] = pmax

            if pmax > 0:
                is_turb = True
            if pmin < 0:
                is_pump = True

        if is_turb:
            data["SETS"]["A_turb"].append(arc)
        if is_pump:
            data["SETS"]["A_pump"].append(arc)

    data["graph"]["Adj"] = {r: [] for r in data["SETS"]["V"]}

    for a in data["SETS"]["A"]:
        to_idx = data["arcs"]["to"][a]
        r_from = data["arcs"]["from"][a]
        r_to   = data["arcs"]["to"].get(a)

        data["graph"]["Adj"][r_from].append((a, -1))  # sort
        if to_idx in data["SETS"]["V"]:
            data["graph"]["Adj"][to_idx].append((a, +1))


    # =========================
    # TURBINE SEGMENTS (HPF)
    # =========================
    linear = np.squeeze(g["LinearTerm"][:])
    constant = np.squeeze(g["ConstantTerm"][:])
    num_pieces = np.squeeze(g["NumberPieces"][:]).astype(int)

    j_offset = 0

    data["turbine_segments"] = {}

    for a in range(num_arcs):
        arc = f"arc{a}"
        npieces = num_pieces[a]

        if arc not in data["SETS"]["A_turb"]:
            j_offset += npieces
            continue

        for j in range(npieces):
            J = len(data["SETS"]["J"])
            data["SETS"]["J"].append(J)

            rho = float(linear[j_offset + j])
            cst = float(constant[j_offset + j])

            data["turbine_segments"][(arc, J)] = {
                "rho": rho,
                "p": cst,
            }

        j_offset += npieces

    return data
