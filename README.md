# Unit Commitment + Hydro Optimization Model

Optimization project solving thermal and hydro unit commitment problem using Pyomo.

---

## Environment setup

Create and activate a Python virtual environment:

```bash
python3 -m venv uc_env
source uc_env/bin/activate
pip install -r requirements.txt
```

---

## What this code does

This project builds and solves an optimization model that:
- Schedules thermal generators (unit commitment)
- Models hydro reservoirs, flows, and storage (cascading system)
- Satisfies time-varying electricity demand
- Minimizes production and startup costs

The model is implemented in Pyomo and solved with the MILP solver HiGHS.

---

## Project structure

```
data/       Input datasets (netCDF4)
src/        Model, solver, and visualization code
outputs/    Results, plots, and exported model files (.lp)
```

---

## Data

All input datasets are stored in:

```
data/
```

They include:
- Thermal generator parameters  
- Hydro system topology and reservoir data  
- Demand time series  

Datasets are in netCDF4 format.

---

## Outputs

Generated files are stored in:

```
outputs/
```

This includes:
- Solver logs  
- Plots  
- Exported optimization models (.lp files)

---

## Running the model

```bash
python src/main.py
```

The script will:
1. Read the input data  
2. Build the optimization model  
3. Call the solver  
4. Save results and visualizations  

---

## Requirements

- Python 3.10+
- Pyomo
- HiGHS
