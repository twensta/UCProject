#!/usr/bin/env python3
import shutil
import subprocess
from pathlib import Path
import sys

def is_cdl_netcdf(text: str) -> bool:
    t = text.lstrip()
    return t.startswith("netcdf ") and "dimensions:" in t and "variables:" in t

def convert_cdl_to_netcdf4(cdl_path: Path, out_nc_path: Path) -> None:
    # Vérifie que ncgen est dispo
    ncgen = shutil.which("ncgen")
    if not ncgen:
        raise RuntimeError(
            "ncgen introuvable. Installe les outils NetCDF.\n"
            "Sur Debian/Ubuntu: sudo apt-get install netcdf-bin\n"
            "Sur macOS (Homebrew): brew install netcdf\n"
        )

    # Compile en NetCDF4 (-4). -o spécifie le fichier de sortie.
    # (optionnel) -k 4 force NetCDF-4 explicitement sur certaines versions.
    cmd = [ncgen, "-4", "-o", str(out_nc_path), str(cdl_path)]
    subprocess.run(cmd, check=True)

def main():
    in_path = Path("case2.txt")  # <- ton fichier
    out_path = Path("case3.nc")  # <- sortie

    text = in_path.read_text(encoding="utf-8", errors="replace")
    if not is_cdl_netcdf(text):
        raise ValueError(
            "Ce fichier ne ressemble pas à du CDL NetCDF (pas de 'netcdf {...}', "
            "'dimensions:' ou 'variables:')."
        )

    convert_cdl_to_netcdf4(in_path, out_path)
    print(f"OK: NetCDF4 généré -> {out_path}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERREUR: {e}", file=sys.stderr)
        sys.exit(1)
