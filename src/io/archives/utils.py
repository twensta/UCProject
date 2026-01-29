# src/io/utils.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pyomo.environ as pyo


@dataclass
class DispatchTS:
    T: List[int]
    demand: Dict[int, float]              # t -> MW
    supply_total: Dict[int, float]        # t -> MW
    supply_by_unit: Dict[str, Dict[int, float]]  # g -> (t -> MW)


def _get_time_set(model) -> List[int]:
    """Return list of time indices."""
    if hasattr(model, "T"):
        return list(model.T)
    raise AttributeError("Model has no set 'T'.")


def _get_demand(model, t: int) -> float:
    """Return demand at time t."""
    # Most common: model.d[t] is a Param
    if hasattr(model, "d"):
        return float(pyo.value(model.d[t]))
    # fallback: some models store demand differently
    if hasattr(model, "demand"):
        return float(pyo.value(model.demand[t]))
    raise AttributeError("Could not find demand parameter (expected model.d[t] or model.demand[t]).")


def _get_generators(model) -> List[str]:
    """Return list of generator ids."""
    if hasattr(model, "G"):
        return [str(g) for g in model.G]
    # Sometimes the set is called U
    if hasattr(model, "U"):
        return [str(g) for g in model.U]
    raise AttributeError("Model has no generator set 'G' (or 'U').")


def _find_power_var(model) -> Tuple[str, pyo.Var]:
    """
    Find the thermal power decision variable.
    Common names: p[g,t], pg[g,t], P[g,t]
    """
    for name in ("p", "pg", "P"):
        if hasattr(model, name):
            var = getattr(model, name)
            # basic sanity: must be indexed
            if isinstance(var, pyo.Var):
                return name, var
    raise AttributeError("Could not find thermal power variable (expected model.p or model.pg or model.P).")


def _optional_hydro_power_var(model) -> Optional[pyo.Var]:
    """
    Find hydro arc power variable if present.
    Common names: pa[a,t], p_a[a,t], ph[a,t]
    """
    for name in ("pa", "p_a", "ph"):
        if hasattr(model, name):
            var = getattr(model, name)
            if isinstance(var, pyo.Var):
                return var
    return None


def _optional_hydro_arc_set(model):
    """Find arc set if present (A)."""
    if hasattr(model, "A"):
        return model.A
    return None


def extract_dispatch_timeseries(model) -> DispatchTS:
    """
    Extract:
      - demand[t]
      - supply_total[t] = sum thermal + (optional) hydro
      - supply_by_unit[g][t] = thermal generation
    """
    T = _get_time_set(model)
    G = _get_generators(model)
    _, p_var = _find_power_var(model)

    demand: Dict[int, float] = {}
    supply_total: Dict[int, float] = {}
    supply_by_unit: Dict[str, Dict[int, float]] = {g: {} for g in G}

    hydro_p_var = _optional_hydro_power_var(model)
    A = _optional_hydro_arc_set(model)

    for t in T:
        d_t = _get_demand(model, t)
        demand[int(t)] = d_t

        thermal_sum = 0.0
        for g in G:
            val = float(pyo.value(p_var[g, t]))
            supply_by_unit[g][int(t)] = val
            thermal_sum += val

        hydro_sum = 0.0
        if hydro_p_var is not None and A is not None:
            for a in A:
                # hydro can be fixed to 0 in thermal-only runs; still ok
                hydro_sum += float(pyo.value(hydro_p_var[a, t]))

        supply_total[int(t)] = thermal_sum + hydro_sum

    return DispatchTS(T=[int(t) for t in T], demand=demand, supply_total=supply_total, supply_by_unit=supply_by_unit)
