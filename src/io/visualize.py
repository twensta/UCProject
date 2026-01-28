# src/io/visualize.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import pyomo.environ as pyo

from src.io.utils import extract_dispatch_timeseries


def plot_supply_demand(model, outdir: Path, show: bool = False) -> Path:
    """
    Plot total supply vs demand over time.
    Saves: outdir/supply_vs_demand.png
    """
    outdir.mkdir(parents=True, exist_ok=True)

    ts = extract_dispatch_timeseries(model)
    x = ts.T
    demand = [ts.demand[t] for t in x]
    supply = [ts.supply_total[t] for t in x]

    plt.figure()
    plt.plot(x, demand, label="Demand")
    plt.plot(x, supply, label="Total supply",linestyle="--")
    plt.xlabel("Time step")
    plt.ylabel("Power (MW)")
    plt.title("Supply vs Demand")
    plt.legend()

    outpath = outdir / "supply_vs_demand.png"
    plt.savefig(outpath, bbox_inches="tight", dpi=150)

    if show:
        plt.show()
    plt.close()
    return outpath


def plot_unit_dispatch(model, outdir: Path, max_units: Optional[int] = 10, show: bool = False) -> Path:
    """
    Plot thermal dispatch per unit over time.
    Saves: outdir/thermal_dispatch.png

    max_units: if too many units, we plot only the first N to keep it readable.
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

    outpath = outdir / "thermal_dispatch.png"
    plt.savefig(outpath, bbox_inches="tight", dpi=150)

    if show:
        plt.show()
    plt.close()
    return outpath


def visualize_results(model, outdir: Path, show: bool = False, max_units: Optional[int] = 10) -> None:
    """
    Convenience wrapper: generates both plots.
    """
    p1 = plot_supply_demand(model, outdir=outdir, show=show)
    p2 = plot_unit_dispatch(model, outdir=outdir, max_units=max_units, show=show)

    # Helpful prints for logs
    try:
        obj = pyo.value(model.OBJ)
    except Exception:
        obj = None

    print("Saved plots:")
    print(" -", p1)
    print(" -", p2)
    if obj is not None:
        print("Objective:", obj)
