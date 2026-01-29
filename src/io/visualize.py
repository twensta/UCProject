# src/io/visualize.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Dict, Any, List

import matplotlib.pyplot as plt
import pyomo.environ as pyo
import numpy as np
from src.io.utils import extract_dispatch_timeseries, extract_hydro_timeseries


def plot_supply_demand(model, outdir: Path, show: bool = False) -> Path:
    """
    Plot total supply vs demand over time.
    Saves: outdir/supply_vs_demand.png
    (kept for backward compatibility)
    """
    outdir.mkdir(parents=True, exist_ok=True)

    ts = extract_dispatch_timeseries(model)
    x = ts.T
    demand = [ts.demand[t] for t in x]
    supply = [ts.supply_total[t] for t in x]

    plt.figure()
    plt.plot(x, demand, label="Demand")
    plt.plot(x, supply, label="Total supply", linestyle="--")
    plt.xlabel("Time step")
    plt.ylabel("Power (MW)")
    plt.title("Supply vs Demand")
    plt.legend()

    outpath = outdir / "plots" / "supply_vs_demand.png"
    plt.savefig(outpath, bbox_inches="tight", dpi=150)

    if show:
        plt.show()
    plt.close()
    return outpath


def plot_dispatch_with_hydro(model, outdir: Path, show: bool = False) -> Path:
    """
    Plot demand + total supply, with separate thermal and hydro totals if available.
    Saves: outdir/dispatch_with_hydro.png
    """
    outdir.mkdir(parents=True, exist_ok=True)

    ts = extract_dispatch_timeseries(model)
    hydro = extract_hydro_timeseries(model)

    x = ts.T
    demand = [ts.demand[t] for t in x]
    total = [ts.supply_total[t] for t in x]

    # thermal total = sum over units
    thermal_total: List[float] = []
    for t in x:
        thermal_total.append(sum(ts.supply_by_unit[g][t] for g in ts.supply_by_unit))

    hydro_total: Optional[List[float]] = None
    if hydro is not None and hydro.power_by_arc:
        hydro_total = [hydro.power_total[t] for t in x]

    plt.figure()
    plt.plot(x, demand, label="Demand")
    plt.plot(x, total, label="Total supply", linestyle="--")
    plt.plot(x, thermal_total, label="Thermal total", linestyle=":")
    if hydro_total is not None:
        plt.plot(x, hydro_total, label="Hydro total", linestyle="-.")

    plt.xlabel("Time step")
    plt.ylabel("Power (MW)")
    plt.title("Dispatch (thermal + hydro) vs demand")
    plt.legend()

    outpath = outdir / "plots" / "dispatch_with_hydro.png"
    plt.savefig(outpath, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    plt.close()
    return outpath


def plot_unit_dispatch(model, outdir: Path, max_units: Optional[int] = 10, show: bool = False) -> Path:
    """
    Plot thermal dispatch per unit over time.
    Saves: outdir/thermal_dispatch.png
    """
    outdir.mkdir(parents=True, exist_ok=True)

    ts = extract_dispatch_timeseries(model)
    x = ts.T

    units = list(ts.supply_by_unit.keys())
    if max_units is not None:
        units = units[:max_units]

    plt.figure()
    for g in units:
        y = [ts.supply_by_unit[g][t] for t in x]
        plt.plot(x, y, label=g)

    plt.xlabel("Time step")
    plt.ylabel("Power (MW)")
    plt.title("Thermal unit dispatch")
    plt.legend(ncol=2, fontsize=8)

    outpath = outdir / "plots" /"thermal_dispatch.png"
    plt.savefig(outpath, bbox_inches="tight", dpi=150)

    if show:
        plt.show()
    plt.close()
    return outpath


def plot_hydro_flows_by_arc(
    model,
    outdir: Path,
    data: Optional[Dict[str, Any]] = None,
    arcs: Optional[Sequence[str]] = None,
    show: bool = False,
) -> Optional[Path]:
    """
    Plot hydro flows f[a,t] per arc over time.
    If `data` is provided with data["arcs"]["from/to"], legend shows "a: from -> to".

    Saves: outdir/hydro_flow_by_arc.png
    Returns None if model has no hydro flow data.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    hydro = extract_hydro_timeseries(model)
    if hydro is None or not hydro.flow_by_arc:
        return None

    x = hydro.T  # focus on one day

    # choose arcs
    arc_list = list(hydro.flow_by_arc.keys())
    if arcs is not None:
        arc_set = set(arcs)
        arc_list = [a for a in arc_list if a in arc_set]

    arc_from = None
    arc_to = None
    if data is not None and "arcs" in data and "from" in data["arcs"] and "to" in data["arcs"]:
        arc_from = data["arcs"]["from"]
        arc_to = data["arcs"]["to"]

    plt.figure()
    linestyles = ["-", "--", "-.", ":"]
    base_colors = list(plt.get_cmap("tab10").colors) + list(plt.get_cmap("tab20").colors)
    filtered_arcs = [a for a in arc_list if a[11] == "0"]

    for i, a in enumerate(filtered_arcs):
        y = [hydro.flow_by_arc[a][t] for t in x]

        if arc_from and arc_to:
            fr = arc_from.get(a, "?")
            to = arc_to.get(a, "?")
            label = f"{a}: {fr} → {to}"
        else:
            label = a

        color = base_colors[i % len(base_colors)]
        style = linestyles[(i // len(base_colors)) % len(linestyles)]

        plt.plot(
            x,
            y,
            label=label,
            color=color,
            linestyle=style,
            linewidth=1.5,
        )

    plt.xlabel("Time step")
    plt.ylabel("Flow")
    plt.title("Hydro flow by arc")
    plt.legend(fontsize=8, ncol=2)

    outpath = outdir / "plots" /"hydro_flow_by_arc.png"
    plt.savefig(outpath, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    plt.close()
    return outpath


def plot_reservoir_volumes(
    model,
    outdir: Path,
    reservoirs: Optional[Sequence[str]] = None,
    show: bool = False,
) -> Optional[Path]:
    """
    Plot reservoir volumes V[r,t] over time.
    Saves: outdir/reservoir_volumes.png
    Returns None if model has no reservoir volumes.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    hydro = extract_hydro_timeseries(model)
    if hydro is None or not hydro.volume_by_res:
        return None

    x = hydro.T
    res_list = list(hydro.volume_by_res.keys())
    if reservoirs is not None:
        res_set = set(reservoirs)
        res_list = [r for r in res_list if r in res_set]

    plt.figure()
    for r in res_list:
        if r[11] == "0":
            y = np.array([hydro.volume_by_res[r][t] for t in x], dtype=float)
            y = (y - y.min()) / (y.max() - y.min() + 1e-12)
            plt.plot(x, y, label=r)

    plt.xlabel("Time step")
    plt.ylabel("Volume (or level)")
    plt.title("Reservoir volumes over time")
    plt.legend(fontsize=8, ncol=2)

    outpath = outdir / "plots" / "reservoir_volumes.png"
    plt.savefig(outpath, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    plt.close()
    return outpath


def visualize_results(
    model,
    outdir: Path,
    show: bool = False,
    max_units: Optional[int] = 10,
    data: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Convenience wrapper: generates dispatch + thermal + hydro plots.
    If you pass `data`, hydro arcs will be labeled with from/to reservoirs.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    p0 = plot_supply_demand(model, outdir=outdir, show=show)
    p1 = plot_dispatch_with_hydro(model, outdir=outdir, show=show)
    p2 = plot_unit_dispatch(model, outdir=outdir, max_units=max_units, show=show)

    p3 = plot_hydro_flows_by_arc(model, outdir=outdir, data=data, show=show)
    p4 = plot_reservoir_volumes(model, outdir=outdir, show=show)

    try:
        obj = pyo.value(model.OBJ)
    except Exception:
        obj = None

    print("Saved plots:")
    print(" -", p0)
    print(" -", p1)
    print(" -", p2)
    if p3 is not None:
        print(" -", p3)
    if p4 is not None:
        print(" -", p4)
    if obj is not None:
        print("Objective:", obj)
