# src/io/animation.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.animation import FuncAnimation

from src.io.utils import extract_hydro_timeseries  # même style que visualize.py

from collections import defaultdict, deque
import re

@dataclass
class _HydroAnimData:
    T: List[int]
    reservoirs: List[str]
    arcs: List[str]
    vol: Dict[str, Dict[int, float]]          # vol[r][t]
    flow: Dict[str, Dict[int, float]]         # flow[a][t]
    arc_from: Dict[str, str]
    arc_to: Dict[str, str]
    vol_min: Dict[str, float]
    vol_max: Dict[str, float]
    flow_min: Dict[str, float]
    flow_max: Dict[str, float]


def _safe_minmax(values: np.ndarray) -> Tuple[float, float]:
    if values.size == 0:
        return 0.0, 1.0
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if not np.isfinite(vmin):
        vmin = 0.0
    if not np.isfinite(vmax) or abs(vmax - vmin) < 1e-12:
        vmax = vmin + 1.0
    return vmin, vmax


def _normalize(x: float, vmin: float, vmax: float) -> float:
    return float(np.clip((x - vmin) / (vmax - vmin), 0.0, 1.0))


def _build_anim_data(model, data: Optional[Dict[str, Any]] = None) -> Optional[_HydroAnimData]:
    hydro = extract_hydro_timeseries(model)
    if hydro is None or (not hydro.volume_by_res) or (not hydro.flow_by_arc):
        return None

    T = list(hydro.T)
    reservoirs = list(hydro.volume_by_res.keys())
    arcs = list(hydro.flow_by_arc.keys())

    # endpoints from `data` if provided
    arc_from: Dict[str, str] = {}
    arc_to: Dict[str, str] = {}
    if data is not None and "arcs" in data and "from" in data["arcs"] and "to" in data["arcs"]:
        arc_from = dict(data["arcs"]["from"])
        arc_to = dict(data["arcs"]["to"])
    else:
        # fallback: make everything loop on itself (still animates thickness)
        for a in arcs:
            arc_from[a] = reservoirs[0]
            arc_to[a] = reservoirs[0]

    # time series dicts (already in hydro.* as dict-of-dict style)
    vol = hydro.volume_by_res
    flow = hydro.flow_by_arc

    vol_min: Dict[str, float] = {}
    vol_max: Dict[str, float] = {}
    for r in reservoirs:
        arr = np.array([vol[r][t] for t in T], dtype=float)
        vmin, vmax = _safe_minmax(arr)
        vol_min[r], vol_max[r] = vmin, vmax

    flow_min: Dict[str, float] = {}
    flow_max: Dict[str, float] = {}
    for a in arcs:
        arr = np.array([flow[a][t] for t in T], dtype=float)
        vmin, vmax = _safe_minmax(arr)
        flow_min[a], flow_max[a] = vmin, vmax

    return _HydroAnimData(
        T=T,
        reservoirs=reservoirs,
        arcs=arcs,
        vol=vol,
        flow=flow,
        arc_from=arc_from,
        arc_to=arc_to,
        vol_min=vol_min,
        vol_max=vol_max,
        flow_min=flow_min,
        flow_max=flow_max,
    )


def _res_group_name(res_name: str) -> str:
    """
    Regroupe par préfixe avant '_resX'.
    Ex: 'UnitBlock_10_res2' -> 'UnitBlock_10'
    """
    m = re.match(r"^(.*)_res\d+$", res_name)
    return m.group(1) if m else "GLOBAL"


def _layered_layout_for_group(
    reservoirs: List[str],
    arcs: List[str],
    arc_from: Dict[str, str],
    arc_to: Dict[str, str],
    x_offset: float,
    x_step: float = 3,
    y_step: float = 1.4,
) -> Dict[str, Tuple[float, float]]:
    """
    Place les réservoirs par 'layers' selon la direction des arcs.
    Si cycles / ambiguïté, on fait un fallback simple en grille.
    """
    res_set = set(reservoirs)

    # Build adjacency + indegree (dans le sous-graphe)
    adj = {r: [] for r in reservoirs}
    indeg = {r: 0 for r in reservoirs}

    for a in arcs:
        u = arc_from.get(a)
        v = arc_to.get(a)
        if u in res_set and v in res_set and u != v:
            adj[u].append(v)
            indeg[v] += 1

    # Kahn topo (si cycle, topo partiel)
    q = deque([r for r in reservoirs if indeg[r] == 0])
    topo = []
    indeg2 = indeg.copy()
    while q:
        u = q.popleft()
        topo.append(u)
        for v in adj[u]:
            indeg2[v] -= 1
            if indeg2[v] == 0:
                q.append(v)

    # Si topo incomplet => cycle => fallback grille
    if len(topo) < len(reservoirs):
        pos = {}
        cols = max(1, int(np.ceil(np.sqrt(len(reservoirs)))))
        for i, r in enumerate(reservoirs):
            cx = i % cols
            cy = i // cols
            pos[r] = (x_offset + cx * x_step, -cy * y_step)
        return pos

    # Longest-path layers on DAG order
    layer = {r: 0 for r in reservoirs}
    for u in topo:
        for v in adj[u]:
            layer[v] = max(layer[v], layer[u] + 1)

    layers = defaultdict(list)
    for r in reservoirs:
        layers[layer[r]].append(r)

    # Sort nodes inside each layer for stability
    for k in layers:
        layers[k].sort()

    # Build coords: x=layer, y=spread within layer
    pos = {}
    max_layer = max(layers.keys()) if layers else 0
    for L in range(max_layer + 1):
        nodes = layers.get(L, [])
        # centrer verticalement la couche
        if not nodes:
            continue
        y0 = (len(nodes) - 1) * 0.5 * y_step
        for i, r in enumerate(nodes):
            x = x_offset + L * x_step
            y = -(i * y_step - y0)  # inversé pour avoir “en haut” positif si tu veux
            pos[r] = (x, y)
    return pos


def _layout_grouped_layered(
    reservoirs: List[str],
    arcs: List[str],
    arc_from: Dict[str, str],
    arc_to: Dict[str, str],
) -> Dict[str, Tuple[float, float]]:
    """
    1) Regroupe les réservoirs par 'valley' / unitblock via le préfixe.
    2) Dans chaque groupe, layout en layers (cascade).
    3) Place chaque groupe à côté (offset x).
    """
    groups = defaultdict(list)
    for r in reservoirs:
        groups[_res_group_name(r)].append(r)

    # Pour découper les arcs par groupe
    group_of = {r: _res_group_name(r) for r in reservoirs}
    arcs_by_group = defaultdict(list)
    for a in arcs:
        u = arc_from.get(a)
        v = arc_to.get(a)
        if u in group_of and v in group_of and group_of[u] == group_of[v]:
            arcs_by_group[group_of[u]].append(a)

    # Ordonner les groupes (stable)
    group_names = sorted(groups.keys())

    pos: Dict[str, Tuple[float, float]] = {}
    x_offset = 0.0
    group_gap = 3.5  # espace entre vallées

    for gname in group_names:
        res_g = sorted(groups[gname])
        arcs_g = arcs_by_group.get(gname, [])

        # Layout interne par layers
        local_pos = _layered_layout_for_group(
            reservoirs=res_g,
            arcs=arcs_g,
            arc_from=arc_from,
            arc_to=arc_to,
            x_offset=x_offset,
            x_step=3,
            y_step=3,
        )
        pos.update(local_pos)

        # Calcul largeur du groupe pour décaler le suivant
        xs = [local_pos[r][0] for r in res_g]
        width = (max(xs) - min(xs)) if xs else 0.0
        x_offset += width + group_gap

    return pos



def animate_hydro_network(
    model,
    outpath: Optional[Path] = None,
    data: Optional[Dict[str, Any]] = None,
    fps: int = 10,
    show: bool = True,
) -> Optional[Path]:
    """
    Anime le réseau hydro :
      - noeuds: réservoirs (remplissage normalisé par réservoir)
      - arcs: flèches (épaisseur normalisée par arc)

    Si outpath est fourni :
      - .gif -> PillowWriter
      - .mp4 -> FFMpegWriter (si ffmpeg dispo)
    """
    anim = _build_anim_data(model, data=data)
    if anim is None:
        print("[animation] No hydro volumes/flows found in model -> nothing to animate.")
        return None

    # positions
    pos = _layout_grouped_layered(anim.reservoirs, anim.arcs, anim.arc_from, anim.arc_to)

    fig, ax = plt.subplots()
    ax.set_aspect("equal")

    # sizing
    node_radius = 1
    xvals = [pos[r][0] for r in anim.reservoirs]
    yvals = [pos[r][1] for r in anim.reservoirs]
    ax.set_xlim(min(xvals) - 1.5, max(xvals) + 1.5)
    ax.set_ylim(min(yvals) - 2.0, max(yvals) + 2.0)
    ax.axis("off")

    # --- static node outlines + labels ---
    node_outline: Dict[str, patches.Circle] = {}
    node_fill: Dict[str, patches.Wedge] = {}
    for r in anim.reservoirs:
        x, y = pos[r]
        circ = patches.Circle((x, y), radius=node_radius, fill=False, linewidth=2.0)
        ax.add_patch(circ)
        node_outline[r] = circ

        # fill as wedge from -90° upward (like a "level")
        wedge = patches.Wedge((x, y), node_radius * 0.98, -90, -90, width=node_radius * 0.98)
        ax.add_patch(wedge)
        node_fill[r] = wedge

        ax.text(x, y - node_radius - 0.25, r, ha="center", va="top", fontsize=8)

    # --- dynamic arrows (we'll redraw each frame for simplicity/robustness) ---
    arrow_artists: List[Any] = []

    title = ax.text(0.5, 0.98, "", transform=ax.transAxes, ha="center", va="top")

    def _clear_arrows():
        nonlocal arrow_artists
        for art in arrow_artists:
            try:
                art.remove()
            except Exception:
                pass
        arrow_artists = []

    def _draw_arrows(t: int):
        nonlocal arrow_artists

        for a in anim.arcs:
            fr = anim.arc_from.get(a, None)
            to = anim.arc_to.get(a, None)
            if fr is None or to is None or fr not in pos or to not in pos:
                continue

            x0, y0 = pos[fr]
            x1, y1 = pos[to]
            if abs(x1 - x0) < 1e-12 and abs(y1 - y0) < 1e-12:
                # self-loop: draw a small curved arrow above the node
                fval = float(anim.flow[a][t])
                s = _normalize(fval, anim.flow_min[a], anim.flow_max[a])
                lw = 0.5 + 6.0 * s
                alpha = 0.15 + 0.85 * s

                loop = patches.FancyArrowPatch(
                    (x0, y0 + node_radius),
                    (x0 + 0.001, y0 + node_radius),
                    connectionstyle="arc3,rad=0.8",
                    arrowstyle="-|>",
                    mutation_scale=12 + 18 * s,
                    linewidth=lw,
                    alpha=alpha,
                )
                ax.add_patch(loop)
                arrow_artists.append(loop)
                continue

            # normal arrow between nodes
            fval = float(anim.flow[a][t])
            s = _normalize(fval, anim.flow_min[a], anim.flow_max[a])

            lw = 0.5 + 6.0 * s
            alpha = 0.15 + 0.85 * s

            # shrink so it doesn't enter the circle
            vx, vy = (x1 - x0), (y1 - y0)
            L = (vx * vx + vy * vy) ** 0.5
            ux, uy = vx / L, vy / L
            start = (x0 + ux * node_radius, y0 + uy * node_radius)
            end = (x1 - ux * node_radius, y1 - uy * node_radius)

            arr = patches.FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=8 + 18 * s,
                linewidth=lw,
                alpha=alpha,
            )
            ax.add_patch(arr)
            arrow_artists.append(arr)

    def update(frame_idx: int):
        t = anim.T[frame_idx]
        title.set_text(f"Hydro network – t={t}")

        # update node fills
        for r in anim.reservoirs:
            v = float(anim.vol[r][t])
            s = _normalize(v, anim.vol_min[r], anim.vol_max[r])  # 0..1
            # map fill level to angle span (0..360)
            # We fill from bottom (-90°) up to (-90 + 360*s)
            node_fill[r].set_theta1(-90)
            node_fill[r].set_theta2(-90 + 360.0 * s)
            node_fill[r].set_alpha(0.15 + 0.85 * s)

        _clear_arrows()
        _draw_arrows(t)

        # return artists for blit
        return [title, *node_fill.values(), *arrow_artists, *node_outline.values()]

    ani = FuncAnimation(fig, update, frames=len(anim.T), interval=1000 / fps, blit=False)

    saved: Optional[Path] = None
    if outpath is not None:
        outpath = Path(outpath)
        outpath.parent.mkdir(parents=True, exist_ok=True)

        if outpath.suffix.lower() == ".gif":
            ani.save(outpath, writer="pillow", fps=fps)
        elif outpath.suffix.lower() == ".mp4":
            # nécessite ffmpeg installé
            ani.save(outpath, writer="ffmpeg", fps=fps)
        else:
            raise ValueError("outpath must end with .gif or .mp4")

        saved = outpath
        print("[animation] Saved:", saved)

    if show:
        plt.show()
    plt.close(fig)
    return saved


if __name__ == "__main__":
    # Exemple d’usage (à adapter si tu veux tester vite)
    # depuis le projet: python -m src.io.animation
    print("Run this from your main after solving: animate_hydro_network(model, outpath=Path('outputs/hydro.gif'), data=raw_data)")
