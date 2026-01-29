# src/config.py
from __future__ import annotations
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#DATA à utiliser
SMSPP_FILE = PROJECT_ROOT / "data" / "smspp" / "20090907_extended_pHydro_18_none.nc4"
THERMAL_FILE = PROJECT_ROOT / "data" / "thermal_data" / "10_0_1_w.nc4"
HYDRO_FILE= PROJECT_ROOT / "data" / "Others" / "Aurland_1000.nc4"
HT_FILE= PROJECT_ROOT / "data" / "Homemade" / "case3.nc"

SOLVER_NAME = "highs"

