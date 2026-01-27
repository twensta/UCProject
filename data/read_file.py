import netCDF4
import numpy as np

def read_smspp_file(filename):
    data = {}

    nc = netCDF4.Dataset(filename, mode='r')
    block = nc['Block_0']

    # Temps et ensembles
    TimeHorizon = block.dimensions['TimeHorizon'].size
    data['time'] = {"T": TimeHorizon, "delta_t": 1.0}
    data['SETS'] = {'T': list(range(1, TimeHorizon+1))}

    # Demande
    ActivePowerDemand = np.array(block['ActivePowerDemand'][:])
    data['demand'] = {t+1: float(np.sum(ActivePowerDemand[t])) for t in range(TimeHorizon)}

    # Thermal units
    data['SETS']['G'] = []
    data['thermal'] = {'p_min': {}, 'p_max': {}, 'cost': {}, 'startup_cost': {}, 
                       'RU': {}, 'RD': {}, 'min_up': {}, 'min_down': {}}

    # Hydro
    data['SETS']['V'] = []
    data['SETS']['A'] = []
    data['arcs'] = {}
    data['reservoirs'] = {}
    data['turbine_segments'] = {}

    # Parcours tous les UnitBlock
    for gname, g in block.groups.items():
        gtype = g.getncattr('type') if 'type' in g.ncattrs() else ''

        # --- Thermiques ---
        if gtype == 'ThermalUnitBlock':
            unit_name = gname
            data['SETS']['G'].append(unit_name)

            p_min_val = float(np.squeeze(g['MinPower'][:]))
            p_max_val = float(np.squeeze(g['MaxPower'][:]))
            startup_cost_val = float(np.squeeze(g['StartUpCost'][:]))
            RU_val = float(np.squeeze(g['DeltaRampUp'][:]))
            RD_val = float(np.squeeze(g['DeltaRampDown'][:]))
            min_up_val = int(np.squeeze(g['MinUpTime'][:]))
            min_down_val = int(np.squeeze(g['MinDownTime'][:]))
            cost_val = float(np.squeeze(g['LinearTerm'][:]))

            for t in range(1, TimeHorizon+1):
                data['thermal']['p_min'][(unit_name,t)] = p_min_val
                data['thermal']['p_max'][(unit_name,t)] = p_max_val
                data['thermal']['cost'][(unit_name,t)] = cost_val

            data['thermal']['startup_cost'][unit_name] = startup_cost_val
            data['thermal']['RU'][unit_name] = RU_val
            data['thermal']['RD'][unit_name] = RD_val
            data['thermal']['min_up'][unit_name] = min_up_val
            data['thermal']['min_down'][unit_name] = min_down_val

        # --- Hydro ---
        elif gtype == 'HydroUnitBlock':
            arc_name = gname
            data['SETS']['A'].append(arc_name)

            # Dimensions
            num_intervals = g.dimensions['NumberIntervals'].size if 'NumberIntervals' in g.dimensions else TimeHorizon
            num_res = g.dimensions['NumberReservoirs'].size if 'NumberReservoirs' in g.dimensions else 1
            num_arcs = g.dimensions['NumberArcs'].size if 'NumberArcs' in g.dimensions else 1
            total_pieces = g.dimensions['TotalNumberPieces'].size if 'TotalNumberPieces' in g.dimensions else 1

            # Réservoirs
            for r in range(num_res):
                res_name = f"{arc_name}_res{r}"
                data['SETS']['V'].append(res_name)

                inflows = g['Inflows'][r,:].tolist() if 'Inflows' in g.variables else [0.0]*num_intervals
    nc.close()
    return data


filename = "/home/enzosawaya/ProjetOptiDisc/UCProject/data/smspp-hydro-units-main-Given Data/Given Data/20090907_extended_pHydro_18_none.nc4"
data = read_smspp_file(filename)

print(data.keys())
print(data['thermal'])

