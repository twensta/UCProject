# src/model/build_model.py
from __future__ import annotations
from typing import Any, Dict, Iterable, Tuple, List
import pyomo.environ as pyo


def build_model(data: Dict[str, Any]) -> pyo.ConcreteModel:
    """
    Build the full UC MILP model in Pyomo.

    Expected 'data' structure (minimum):
      data["time"] = {"T": int, "delta_t": float}
      data["SETS"] = {"T": [1..T], "G": [...], "V": [...], "A": [...],
                      "A_turb": [...], "A_pump": [...]}
      data["demand"] = {t: float}

      data["thermal"] with keys:
        p_min[(g,t)], p_max[(g,t)], cost[(g,t)],
        startup_cost[g], RU[g], RD[g], min_up[g], min_down[g]

      data["reservoirs"] with keys:
        V0[v], Vmin[(v,t)], Vmax[(v,t)], inflow[(v,t)]

      data["arcs"] with keys:
        from[a], to[a], f_min[(a,t)], f_max[(a,t)], RU[a], RD[a],
        p_min[(a,t)], p_max[(a,t)]   # hydro power bounds for each arc

      data["graph"] with keys:
        In[v]  = list of arcs entering v
        Out[v] = list of arcs leaving v

      Turbine segments:
        data["turbine_segments"][(a,j)] = {"f":..., "p":..., "rho":...}
        and data["SETS"]["J"] = list of all j indices
        (If you prefer per-arc J_a, you can adapt easily.)
    """
    m = pyo.ConcreteModel(name="UC_Pyomo")

    # -----------------------
    # Sets / basic params
    # -----------------------
    T_set: List[int] = list(data["SETS"]["T"])
    G_set: List[str] = list(data["SETS"]["G"])
    R_set: List[str] = list(data["SETS"]["V"])
    A_set: List[str] = list(data["SETS"]["A"])
    A_turb: List[str] = list(data["SETS"].get("A_turb", []))
    A_pump: List[str] = list(data["SETS"].get("A_pump", []))

    m.T = pyo.Set(initialize=T_set, ordered=True)
    m.G = pyo.Set(initialize=G_set, ordered=False)
    m.R = pyo.Set(initialize=R_set, ordered=False)
    m.A = pyo.Set(initialize=A_set, ordered=False)
    m.A_turb = pyo.Set(initialize=A_turb, within=m.A)
    m.A_pump = pyo.Set(initialize=A_pump, within=m.A)

    delta_t = float(data["time"]["delta_t"])  # hours
    m.delta_t = pyo.Param(initialize=delta_t, within=pyo.PositiveReals)

    # Demande
    # ICI on récupère un dictionnaire du type : demand = {1: 120, 2: 130, 3: 125, ...} ie {t : demande(t)}
    demand = data["demand"]
    m.d = pyo.Param(m.T, initialize=lambda _, t: float(demand[t])) #Pour tout t dans T, d[t] = demande(t), lambda _ = "La valeur du paramètre est donnée par une fonction des indices."

    # -----------------------
    # Thermal parameters
    # -----------------------
    th = data["thermal"]

    m.pmin_g = pyo.Param(m.G, m.T, initialize=lambda _, g, t: float(th["p_min"][(g, t)]))
    m.pmax_g = pyo.Param(m.G, m.T, initialize=lambda _, g, t: float(th["p_max"][(g, t)]))
    m.cost_g = pyo.Param(m.G, m.T, initialize=lambda _, g, t: float(th["cost"][(g, t)]))
    m.startup_cost = pyo.Param(m.G, initialize=lambda _, g: float(th["startup_cost"][g]))
    m.RU_g = pyo.Param(m.G, initialize=lambda _, g: float(th["RU"][g]))  # MW/h
    m.RD_g = pyo.Param(m.G, initialize=lambda _, g: float(th["RD"][g]))  # MW/h
    m.min_up = pyo.Param(m.G, initialize=lambda _, g: int(th["min_up"][g]))
    m.min_down = pyo.Param(m.G, initialize=lambda _, g: int(th["min_down"][g]))

    # -----------------------
    # Hydro parameters
    # -----------------------
    res = data["reservoirs"]
    arcs = data["arcs"]
    graph = data["graph"]  # {"In":{v:[...]}, "Out":{v:[...]}}

    m.V0 = pyo.Param(m.R, initialize=lambda _, r: float(res["V0"][r]))
    m.Vmin = pyo.Param(m.R, m.T, initialize=lambda _, r, t: float(res["Vmin"][(r, t)]))
    m.Vmax = pyo.Param(m.R, m.T, initialize=lambda _, r, t: float(res["Vmax"][(r, t)]))
    m.inflow = pyo.Param(m.R, m.T, initialize=lambda _, r, t: float(res["inflow"][(r, t)]))  # m3/s

    m.arc_from = pyo.Param(m.A, initialize=lambda _, a: arcs["from"][a], within=pyo.Any)
    m.arc_to = pyo.Param(m.A, initialize=lambda _, a: arcs["to"][a], within=pyo.Any)

    m.fmin = pyo.Param(m.A, m.T, initialize=lambda _, a, t: float(arcs["f_min"][(a, t)]))
    m.fmax = pyo.Param(m.A, m.T, initialize=lambda _, a, t: float(arcs["f_max"][(a, t)]))
    m.RU_a = pyo.Param(m.A, initialize=lambda _, a: float(arcs["RU"][a]))  # (m3/s)/h
    m.RD_a = pyo.Param(m.A, initialize=lambda _, a: float(arcs["RD"][a]))  # (m3/s)/h

    # Power bounds on arcs (turbines/pumps). If you don't have them, set wide bounds in data.
    m.pmin_a = pyo.Param(m.A, m.T, initialize=lambda _, a, t: float(arcs["p_min"][(a, t)]))
    m.pmax_a = pyo.Param(m.A, m.T, initialize=lambda _, a, t: float(arcs["p_max"][(a, t)]))

    # Turbine segments (HPF)
    turb_segs = data.get("turbine_segments", {})
    J_set = list(data["SETS"].get("J", []))
    m.J = pyo.Set(initialize=J_set, ordered=True)

    # Helper: which (a,j) exist?
    AJ = [(a, j) for (a, j) in turb_segs.keys()]
    m.AJ = pyo.Set(dimen=2, initialize=AJ)

    m.seg_f = pyo.Param(m.AJ, initialize=lambda _, a, j: float(turb_segs[(a, j)]["f"]))
    m.seg_p = pyo.Param(m.AJ, initialize=lambda _, a, j: float(turb_segs[(a, j)]["p"]))
    m.seg_rho = pyo.Param(m.AJ, initialize=lambda _, a, j: float(turb_segs[(a, j)]["rho"]))

    # -----------------------
    # Decision variables
    # -----------------------
    # Thermal
    m.u = pyo.Var(m.G, m.T, within=pyo.Binary)
    m.y = pyo.Var(m.G, m.T, within=pyo.Binary)  # startup
    m.z = pyo.Var(m.G, m.T, within=pyo.Binary)  # shutdown
    m.pg = pyo.Var(m.G, m.T, within=pyo.NonNegativeReals)

    # Hydro
    m.V = pyo.Var(m.R, m.T, within=pyo.Reals)  #Variable des volumes pour chaque réservoirs et chaque instant
    m.f = pyo.Var(m.A, m.T, within=pyo.Reals)
    m.pa = pyo.Var(m.A, m.T, within=pyo.Reals)

    # -----------------------
    # Objective
    # -----------------------
    def obj_rule(mm: pyo.ConcreteModel) -> pyo.Expr:
        return sum(
            mm.cost_g[g, t] * mm.delta_t * mm.pg[g, t] + mm.startup_cost[g] * mm.y[g, t]
            for g in mm.G
            for t in mm.T
        )

    m.OBJ = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # -----------------------
    # Constraints - Thermal
    # -----------------------
    # Power bounds conditional on ON/OFF
    def th_pmax_rule(mm, g, t):
        return mm.pg[g, t] <= mm.pmax_g[g, t] * mm.u[g, t]

    def th_pmin_rule(mm, g, t):
        return mm.pg[g, t] >= mm.pmin_g[g, t] * mm.u[g, t]

    m.ThPmax = pyo.Constraint(m.G, m.T, rule=th_pmax_rule)
    m.ThPmin = pyo.Constraint(m.G, m.T, rule=th_pmin_rule)

    # Transitions u_t - u_{t-1} = y_t - z_t (t >= 2)
    T_list = T_set

    def th_trans_rule(mm, g, t):
        if t == T_list[0]:
            return pyo.Constraint.Skip
        tprev = T_list[T_list.index(t) - 1]
        return mm.u[g, t] - mm.u[g, tprev] == mm.y[g, t] - mm.z[g, t]

    m.ThTrans = pyo.Constraint(m.G, m.T, rule=th_trans_rule)

    # Ramping (simple + startup/shutdown terms as you used)
    def th_ramp_up_rule(mm, g, t):
        if t == T_list[0]:
            return pyo.Constraint.Skip
        tprev = T_list[T_list.index(t) - 1]
        return mm.pg[g, t] - mm.pg[g, tprev] <= mm.RU_g[g] * mm.delta_t * mm.u[g, tprev] + mm.pmin_g[g, t] * mm.y[g, t]

    def th_ramp_dn_rule(mm, g, t):
        if t == T_list[0]:
            return pyo.Constraint.Skip
        tprev = T_list[T_list.index(t) - 1]
        return mm.pg[g, tprev] - mm.pg[g, t] <= mm.RD_g[g] * mm.delta_t * mm.u[g, t] + mm.pmax_g[g, tprev] * mm.z[g, t]

    m.ThRampUp = pyo.Constraint(m.G, m.T, rule=th_ramp_up_rule)
    m.ThRampDn = pyo.Constraint(m.G, m.T, rule=th_ramp_dn_rule)

    # Minimum up-time: sum_{k=t}^{t+tau-1} u_k >= tau * y_t
    def th_min_up_rule(mm, g, t):
        tau = int(pyo.value(mm.min_up[g]))
        # only apply where window fits
        idx = T_list.index(t)
        if idx + tau - 1 >= len(T_list):
            return pyo.Constraint.Skip
        window = T_list[idx : idx + tau]
        return sum(mm.u[g, k] for k in window) >= tau * mm.y[g, t]

    m.ThMinUp = pyo.Constraint(m.G, m.T, rule=th_min_up_rule)

    # Minimum down-time: sum_{k=t}^{t+tau-1} (1-u_k) >= tau * z_t
    def th_min_down_rule(mm, g, t):
        tau = int(pyo.value(mm.min_down[g]))
        idx = T_list.index(t)
        if idx + tau - 1 >= len(T_list):
            return pyo.Constraint.Skip
        window = T_list[idx : idx + tau]
        return sum(1 - mm.u[g, k] for k in window) >= tau * mm.z[g, t]

    m.ThMinDown = pyo.Constraint(m.G, m.T, rule=th_min_down_rule)

    # -----------------------
    # Constraints - Hydro
    # -----------------------
    # Volume bounds
    def V_bounds_rule(mm, r, t):
        return (mm.Vmin[r, t], mm.V[r, t], mm.Vmax[r, t])

    m.VolBounds = pyo.Constraint(m.R, m.T, rule=V_bounds_rule)

    attachment = 3600.0 * delta_t  # seconds in dt hours

    # Initial volume: V[v, first_t] = V0[v]
    first_t = T_list[0]

    def init_vol_rule(mm, r):
        return mm.V[r, first_t] == mm.V0[r]

    m.InitVol = pyo.Constraint(m.R, rule=init_vol_rule)

    # Mass balance: V_{t+1} = V_t + 3600 dt ( inflow + sum_in f - sum_out f )
    def mass_balance_rule(mm, r, t):
        idx = T_list.index(t)
        if idx == len(T_list) - 1:
            return pyo.Constraint.Skip
        tnext = T_list[idx + 1]
        infl = mm.inflow[r, t]
        in_arcs = graph["In"].get(r, [])
        out_arcs = graph["Out"].get(r, [])
        return mm.V[r, tnext] == mm.V[r, t] + attachment * (
            infl + sum(mm.f[a, t] for a in in_arcs) - sum(mm.f[a, t] for a in out_arcs)
        )

    m.MassBalance = pyo.Constraint(m.R, m.T, rule=mass_balance_rule)

    # Flow bounds
    def flow_bounds_rule(mm, a, t):
        return (mm.fmin[a, t], mm.f[a, t], mm.fmax[a, t])

    m.FlowBounds = pyo.Constraint(m.A, m.T, rule=flow_bounds_rule)

    # Flow ramping
    def flow_ramp_up_rule(mm, a, t):
        if t == first_t:
            return pyo.Constraint.Skip
        tprev = T_list[T_list.index(t) - 1]
        return mm.f[a, t] - mm.f[a, tprev] <= mm.RU_a[a] * mm.delta_t

    def flow_ramp_dn_rule(mm, a, t):
        if t == first_t:
            return pyo.Constraint.Skip
        tprev = T_list[T_list.index(t) - 1]
        return mm.f[a, tprev] - mm.f[a, t] <= mm.RD_a[a] * mm.delta_t

    m.FlowRampUp = pyo.Constraint(m.A, m.T, rule=flow_ramp_up_rule)
    m.FlowRampDn = pyo.Constraint(m.A, m.T, rule=flow_ramp_dn_rule)

    # Power bounds on arcs
    def arc_p_bounds_rule(mm, a, t):
        return (mm.pmin_a[a, t], mm.pa[a, t], mm.pmax_a[a, t])

    m.ArcPowerBounds = pyo.Constraint(m.A, m.T, rule=arc_p_bounds_rule)

    # Turbine HPF envelope: p_{a,t} <= p_j + rho_j (f_{a,t} - f_j)
    # Only for existing (a,j) in AJ
    def hpf_rule(mm, a, j, t):
        return mm.pa[a, t] <= mm.seg_p[a, j] + mm.seg_rho[a, j] * (mm.f[a, t] - mm.seg_f[a, j])

    # Build a 3D index (a,j,t) but only for a in A_turb and (a,j) existing
    def hpf_index_init(mm):
        idxs = []
        for (a, j) in AJ:
            if a in A_turb:
                for t in T_list:
                    idxs.append((a, j, t))
        return idxs

    m.HPF_INDEX = pyo.Set(dimen=3, initialize=hpf_index_init)
    m.HPF = pyo.Constraint(m.HPF_INDEX, rule=lambda mm, a, j, t: hpf_rule(mm, a, j, t))

    # Pumps: p = rho f
    pump_rho = data.get("pump_rho", {})  # {a: rho}
    m.pump_rho = pyo.Param(m.A_pump, initialize=lambda _, a: float(pump_rho[a]) if a in pump_rho else 0.0)

    def pump_rule(mm, a, t):
        return mm.pa[a, t] == mm.pump_rho[a] * mm.f[a, t]

    m.PumpLaw = pyo.Constraint(m.A_pump, m.T, rule=pump_rule)

    # -----------------------
    # System balance
    # -----------------------
    def balance_rule(mm, t):
        return sum(mm.pg[g, t] for g in mm.G) + sum(mm.pa[a, t] for a in mm.A) >= mm.d[t]

    m.Balance = pyo.Constraint(m.T, rule=balance_rule)

    return m
