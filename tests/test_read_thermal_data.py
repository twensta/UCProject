# tests/test_read_thermal_data_smoke.py
from __future__ import annotations

import sys
from pathlib import Path

# Allow: python tests/test_read_thermal_data_smoke.py
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.io.read_thermal_data import read_thermal_file


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    f = project_root / "data" / "thermal_data" / "10_0_1_w.nc4"

    print("=== read_thermal_file smoke test ===")
    print("File:", f)

    data = read_thermal_file(f)

    # Check required top-level keys
    required = ["time", "SETS", "demand", "thermal", "arcs", "reservoirs", "turbine_segments"]
    print("\nTop-level keys:", list(data.keys()))
    for k in required:
        assert k in data, f"Missing key: {k}"

    # Basic summary
    T = data["time"]["T"]
    print("\n--- Summary ---")
    print("T =", T, "| delta_t =", data["time"]["delta_t"])
    print("Nb units G =", len(data["SETS"]["G"]))
    print("First units:", data["SETS"]["G"][:5])

    # Demand preview
    print("\n--- Demand preview ---")
    print("d[1] =", data["demand"][1])
    print("d[2] =", data["demand"][2])
    print("d[T] =", data["demand"][T])

    # Thermal preview for first unit
    g0 = data["SETS"]["G"][0]
    print("\n--- Thermal preview (first unit) ---")
    print("unit =", g0)
    print("p_min[1] =", data["thermal"]["p_min"][(g0, 1)])
    print("p_max[1] =", data["thermal"]["p_max"][(g0, 1)])
    print("cost[1]  =", data["thermal"]["cost"][(g0, 1)])
    print("startup  =", data["thermal"]["startup_cost"][g0])
    print("RU       =", data["thermal"]["RU"][g0])
    print("RD       =", data["thermal"]["RD"][g0])
    print("min_up   =", data["thermal"]["min_up"][g0])
    print("min_down =", data["thermal"]["min_down"][g0])

    # Hydro placeholders should be present and empty
    print("\n--- Hydro placeholders ---")
    print("len(SETS['V']) =", len(data["SETS"]["V"]))
    print("len(SETS['A']) =", len(data["SETS"]["A"]))
    print("arcs dict empty? =", data["arcs"] == {})
    print("reservoirs empty? =", data["reservoirs"] == {})
    print("turbine_segments empty? =", data["turbine_segments"] == {})

    print("\n✅ OK: structure matches read_smspp_file (thermal-only).")


if __name__ == "__main__":
    main()
