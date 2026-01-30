# src/model/build_model.py
from __future__ import annotations
from typing import Any, Dict, Iterable, Tuple, List
import pyomo.environ as pyo


def build_thermal_model(data: Dict[str, Any]) -> pyo.ConcreteModel:
   
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
    # Decision variables
    # -----------------------
    # Thermal
    m.u = pyo.Var(m.G, m.T, within=pyo.Binary)
    m.y = pyo.Var(m.G, m.T, within=pyo.Binary)  # startup
    m.z = pyo.Var(m.G, m.T, within=pyo.Binary)  # shutdown
    m.pg = pyo.Var(m.G, m.T, within=pyo.NonNegativeReals)

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
    # System balance
    # -----------------------
    def balance_rule(mm, t):
        return sum(mm.pg[g, t] for g in mm.G) + sum(mm.pa[a, t] for a in mm.A) >= mm.d[t]

    m.Balance = pyo.Constraint(m.T, rule=balance_rule)

    return m
