# src/model/build_model.py
from typing import List
import pyomo.environ as pyo


def build_model(data):
    """
    Build le UC MILP model en Pyomo
    """

    m = pyo.ConcreteModel(name="UC_Pyomo")

    #----------------------
    #-| Défintions des SETS
    #----------------------
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

    #------------------------------------
    #-| Défintions du time step
    #------------------------------------
    delta_t = float(data["time"]["delta_t"])  # hours
    m.delta_t = pyo.Param(initialize=delta_t, within=pyo.PositiveReals)

    #------------------------------------
    #-| Défintions de la demande
    #------------------------------------
    demand = data["demand"] # ICI on récupère un dictionnaire du type {t : demande(t)}
    m.d = pyo.Param(m.T, initialize=lambda _, t: float(demand[t])) #Pour tout t dans T, d[t] = demande(t), lambda _ = "La valeur du paramètre est donnée par une fonction des indices."

    #----------------------------------------------------------
    #-| Défintions des paramètres pour les centrales thermiques
    #----------------------------------------------------------
    th = data["thermal"]

    m.pmin_g = pyo.Param(m.G, m.T, initialize=lambda _, g, t: float(th["p_min"][(g, t)])) #MW
    m.pmax_g = pyo.Param(m.G, m.T, initialize=lambda _, g, t: float(th["p_max"][(g, t)])) #MW
    m.const_cost_g = pyo.Param(m.G, m.T, initialize=lambda _, g, t: float(th["const_cost"][(g, t)])) #euros
    m.lin_cost_g = pyo.Param(m.G, m.T, initialize=lambda _, g, t: float(th["lin_cost"][(g, t)])) #euros/MW
    m.quad_cost_g = pyo.Param(m.G, m.T, initialize=lambda _, g, t: float(th["quad_cost"][(g, t)])) #euros/MW^2
    m.startup_cost = pyo.Param(m.G, initialize=lambda _, g: float(th["startup_cost"][g])) #euros
    m.RU_g = pyo.Param(m.G, initialize=lambda _, g: float(th["RU"][g]))  #MW/h
    m.RD_g = pyo.Param(m.G, initialize=lambda _, g: float(th["RD"][g]))  #MW/h
    m.min_up = pyo.Param(m.G, initialize=lambda _, g: int(th["min_up"][g])) #time step h
    m.min_down = pyo.Param(m.G, initialize=lambda _, g: int(th["min_down"][g])) #time step h

    #------------------------------------------------------
    #-| Défintions des paramètres pour les centrales hydro
    #------------------------------------------------------
    res = data["reservoirs"]
    arcs = data["arcs"]
    graph = data["graph"]  # {"In":{res_0:[arc_0, ...]}, "Out":{res_0:[...]}}

    #Volume initial, bornes des réservoirs et inflows
    m.V0 = pyo.Param(m.R, initialize=lambda _, r: float(res["V0"][r])) #hm^3
    m.Vmin = pyo.Param(m.R, initialize=lambda _, r: float(res["Vmin"][r])) #hm^3
    m.Vmax = pyo.Param(m.R, initialize=lambda _, r: float(res["Vmax"][r])) #hm^3
    m.inflow = pyo.Param(m.R, m.T, initialize=lambda _, r, t: float(res["inflow"][(r, t)]))  #hm3/h

    #Défintions des arcs
    m.arc_from = pyo.Param(m.A, initialize=lambda _, a: arcs["from"][a], within=pyo.Any)
    m.arc_to = pyo.Param(m.A, initialize=lambda _, a: arcs["to"][a], within=pyo.Any)

    #Débits min et max, ramping, puissance min et max des arcs
    m.fmin = pyo.Param(m.A, initialize=lambda _, a: float(arcs["f_min"][a])) #hm3/h
    m.fmax = pyo.Param(m.A, initialize=lambda _, a: float(arcs["f_max"][a])) #hm3/h
    m.RU_a = pyo.Param(m.A, initialize=lambda _, a: float(arcs["RU"][a]))  # (hm3/h)/h
    m.RD_a = pyo.Param(m.A, initialize=lambda _, a: float(arcs["RD"][a]))  # (hm3/h)/h
    m.pmin_a = pyo.Param(m.A, initialize=lambda _, a: float(arcs["p_min"][a])) #MW
    m.pmax_a = pyo.Param(m.A, initialize=lambda _, a: float(arcs["p_max"][a])) #MW

    #Modélisations des turbines par fonctions linéaires par segments
    segs = data["segments"]

    #Ensemble des couples valides (a,j) ie tel que j est un segment de a
    AJ = [(a, j) for a, Js in segs["J"].items() for j in Js]
    m.AJ = pyo.Set(initialize=AJ, dimen=2)

    m.seg_p   = pyo.Param(m.AJ, initialize=lambda mm, a, j: segs["p"][(a, j)],   within=pyo.Reals) #MW
    m.seg_rho = pyo.Param(m.AJ, initialize=lambda mm, a, j: segs["rho"][(a, j)], within=pyo.Reals) #MW / (hm^3/h)
    m.seg_f   = pyo.Param(m.AJ, initialize=lambda mm, a, j: segs["f0"][(a, j)],  within=pyo.Reals) #hm^3/h
    m.seg_u   = pyo.Param(m.A,   initialize=lambda mm, a: segs["u"][a],        within=pyo.Reals)   #turbine=1, pompe=0 

    #------------------------------------------------------
    #-| Défintions des variables de décision
    #------------------------------------------------------
    #Thermal
    m.u = pyo.Var(m.G, m.T, within=pyo.Binary)  # ON/OFF
    m.y = pyo.Var(m.G, m.T, within=pyo.Binary)  # Startup
    m.z = pyo.Var(m.G, m.T, within=pyo.Binary)  # Shutdown
    m.pg = pyo.Var(m.G, m.T, within=pyo.NonNegativeReals) #Création des pg_{g,t} (puissance) pour chaque centrale thermique et chaque instant

    #Hydro
    m.V = pyo.Var(m.R, m.T, within=pyo.Reals)  #Création des V_{r,t} (volume) pour chaque réservoirs et chaque instant
    m.f = pyo.Var(m.A, m.T, within=pyo.Reals)  #Création des f_{a,t} (débit) pour chaque arc et chaque instant
    m.pa = pyo.Var(m.A, m.T, within=pyo.Reals) #Creation des p_{a,t} (puissance) pour chaque arc et chaque instant

    #------------------------------------------------------
    #-| Défintions de la fonction objectif (minimisation des coûts)
    #------------------------------------------------------
    def obj_rule(mm: pyo.ConcreteModel):
        return sum(
            (mm.const_cost_g[g, t] + mm.lin_cost_g[g, t] * mm.pg[g, t] )  + mm.startup_cost[g] * mm.y[g, t]
            for g in mm.G
            for t in mm.T
        )
    #+ mm.quad_cost_g[g, t] * mm.pg[g, t]**2

    m.OBJ = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
    
    #------------------------------------------------------
    #-| Défintions des contraintes sur les centrales thermiques
    #------------------------------------------------------
    #Bornes de production
    def th_pmax_rule(mm, g, t):
        return mm.pg[g, t] <= mm.pmax_g[g, t] * mm.u[g, t]

    def th_pmin_rule(mm, g, t):
        return mm.pg[g, t] >= mm.pmin_g[g, t] * mm.u[g, t]

    m.ThPmax = pyo.Constraint(m.G, m.T, rule=th_pmax_rule)
    m.ThPmin = pyo.Constraint(m.G, m.T, rule=th_pmin_rule)

    #Défitions des y_t, z_t tel que u_t - u_{t-1} = y_t - z_t (y_t = 1 si startup à t, z_t = 1 si shutdown à t, 0 sinon)
    T_list = T_set

    def th_trans_rule(mm, g, t):
        if t == T_list[0]:
            return pyo.Constraint.Skip
        tprev = T_list[T_list.index(t) - 1]
        return mm.u[g, t] - mm.u[g, tprev] == mm.y[g, t] - mm.z[g, t]

    m.ThTrans = pyo.Constraint(m.G, m.T, rule=th_trans_rule)

    #Ramping
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

    #Temps minimum en marche : sum_{k=t}^{t+tau-1} u_k >= tau+ * y_t
    def th_min_up_rule(mm, g, t):
        tau = int(pyo.value(mm.min_up[g]))
        
        idx = T_list.index(t)
        if idx + tau - 1 >= len(T_list):
            return pyo.Constraint.Skip
        window = T_list[idx : idx + tau]
        return sum(mm.u[g, k] for k in window) >= tau * mm.y[g, t]

    m.ThMinUp = pyo.Constraint(m.G, m.T, rule=th_min_up_rule)

    #Temps minimum à l'arrêt: sum_{k=t}^{t+tau-1} (1-u_k) >= tau- * z_t
    def th_min_down_rule(mm, g, t):
        tau = int(pyo.value(mm.min_down[g]))
        idx = T_list.index(t)
        if idx + tau - 1 >= len(T_list):
            return pyo.Constraint.Skip
        window = T_list[idx : idx + tau]
        return sum(1 - mm.u[g, k] for k in window) >= tau * mm.z[g, t]

    m.ThMinDown = pyo.Constraint(m.G, m.T, rule=th_min_down_rule)

    #------------------------------------------------------
    #-| Défintions des contraintes sur les centrales hydro
    #------------------------------------------------------
    #Bornes de volume des réservoirs
    def V_bounds_rule(mm, r, t):
        return (mm.Vmin[r], mm.V[r, t], mm.Vmax[r])

    m.VolBounds = pyo.Constraint(m.R, m.T, rule=V_bounds_rule) #La rule doit retourner (borne_inf, variable, borne_sup)

    #Volume initial des réservoirs
    first_t = T_list[0]

    def init_vol_rule(mm, r):
        return mm.V[r, first_t] == mm.V0[r]

    m.InitVol = pyo.Constraint(m.R, rule=init_vol_rule) #La rule doit retourner une égalité ==

    # Mass balance: V_{r,t+1} = V_{r,t} + delta_t * ( inflow_{r,t} + sum_{a in IN } f_{a,t} - sum_{a in OUT} f_{a,t} )
    graph = data["graph"]   # {"In": {r: [...]}, "Out": {r: [...]}}

    def mass_balance_rule(mm, r, t):

        if t == mm.T.last():
            return pyo.Constraint.Skip
        tnext = mm.T.next(t)

        in_arcs = graph["In"].get(r, []) #Arcs entrants au réservoir r
        out_arcs = graph["Out"].get(r, []) #Arcs sortants du réservoir r

        return mm.V[r, tnext] == mm.V[r, t] + delta_t * (
            mm.inflow[r, t]
            + sum(mm.f[a, t] for a in in_arcs)
            - sum(mm.f[a, t] for a in out_arcs)
        )

    m.MassBalance = pyo.Constraint(m.R, m.T, rule=mass_balance_rule)

    #Bornes des débits sur les arcs
    def flow_bounds_rule(mm, a, t):
        return (mm.fmin[a], mm.f[a, t], mm.fmax[a])

    m.FlowBounds = pyo.Constraint(m.A, m.T, rule=flow_bounds_rule)

    #Rampings des arcs
    #Le flow rate à t=1 n'est pas limité, le flow rate pourra donc commencer à n'importe quel valeur (ce n'est pas grave car on ne sait pas comment c'était avant, ou alors on aurait pu mettre un initial flow)
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

    #Bornes de puissances sur les arcs
    def arc_p_bounds_rule(mm, a, t):
        return (mm.pmin_a[a], mm.pa[a, t], mm.pmax_a[a])

    m.ArcPowerBounds = pyo.Constraint(m.A, m.T, rule=arc_p_bounds_rule)

    #Enveloppe hpf des turbines/pompes : p_{a,t} <= p_j + rho_j (f_{a,t} - f_j)
    def hpf_rule(mm, a, j, t):
        
        if mm.seg_u[a] == 1:  # turbine
            return mm.pa[a, t] <= mm.seg_p[a, j] + mm.seg_rho[a, j] * (mm.f[a, t] - mm.seg_f[a, j])
        else:  # pompe
            return mm.pa[a, t] >= mm.seg_p[a, j] + mm.seg_rho[a, j] * (mm.f[a, t] - mm.seg_f[a, j])

    m.HPF = pyo.Constraint(m.AJ, m.T, rule=hpf_rule)

    #------------------------------------------------------
    #-| Définition de la contrainte d'équilibre
    #------------------------------------------------------
    def balance_rule(mm, t):
        return sum(mm.pg[g, t] for g in mm.G) + sum(mm.pa[a, t] for a in mm.A) >= mm.d[t]

    m.Balance = pyo.Constraint(m.T, rule=balance_rule)

    return m
