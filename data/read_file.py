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

            num_intervals = g.dimensions['NumberIntervals'].size if 'NumberIntervals' in g.dimensions else TimeHorizon
            num_res = g.dimensions['NumberReservoirs'].size if 'NumberReservoirs' in g.dimensions else 1
            num_arcs = g.dimensions['NumberArcs'].size if 'NumberArcs' in g.dimensions else 1
            total_pieces = g.dimensions['TotalNumberPieces'].size if 'TotalNumberPieces' in g.dimensions else 1

            # --- Réservoirs ---
            for r in range(num_res):
                res_name = f"{arc_name}_res{r}"
                data['SETS']['V'].append(res_name)

                inflows = g['Inflows'][r,:].tolist() if 'Inflows' in g.variables else [0.0]*num_intervals
                V0 = float(np.squeeze(g['InitialVolumetric'][r])) if 'InitialVolumetric' in g.variables else 0.0
                Vmin = g['MinVolumetric'][r,:].tolist() if 'MinVolumetric' in g.variables else [0.0]*num_intervals
                Vmax = g['MaxVolumetric'][r,:].tolist() if 'MaxVolumetric' in g.variables else [1e6]*num_intervals

                # Assigner le dictionnaire complet
                data['reservoirs'][res_name] = {
                    'V0': V0,
                    'inflow': {t+1: inflows[t] for t in range(num_intervals)},
                    'Vmin': {t+1: Vmin[t] for t in range(num_intervals)},
                    'Vmax': {t+1: Vmax[t] for t in range(num_intervals)}
                }

            # --- Arcs ---
            for a in range(num_arcs):
                arc_name = f"{gname}_arc{a}"
                data['SETS']['A'].append(arc_name)

                min_flow = g['MinFlow'][:,a].tolist() if 'MinFlow' in g.variables else [0.0]*num_intervals
                max_flow = g['MaxFlow'][:,a].tolist() if 'MaxFlow' in g.variables else [1e6]*num_intervals
                min_power = g['MinPower'][:,a].tolist() if 'MinPower' in g.variables else [0.0]*num_intervals
                max_power = g['MaxPower'][:,a].tolist() if 'MaxPower' in g.variables else [1e6]*num_intervals
                RU_arr = g['DeltaRampUp'][:,a].tolist() if 'DeltaRampUp' in g.variables else [0.0]*num_intervals
                RD_arr = g['DeltaRampDown'][:,a].tolist() if 'DeltaRampDown' in g.variables else [0.0]*num_intervals

                data['arcs'][arc_name] = {
                    'from': arc_name,
                    'to': arc_name,
                    'f_min': {t+1: min_flow[t] for t in range(num_intervals)},
                    'f_max': {t+1: max_flow[t] for t in range(num_intervals)},
                    'RU': {t+1: RU_arr[t] for t in range(num_intervals)},
                    'RD': {t+1: RD_arr[t] for t in range(num_intervals)},
                    'p_min': {t+1: min_power[t] for t in range(num_intervals)},
                    'p_max': {t+1: max_power[t] for t in range(num_intervals)}
                }

            # --- Segments de turbines ---
            if 'LinearTerm' in g.variables and 'ConstantTerm' in g.variables:
                linear = g['LinearTerm'][:].tolist()
                constant = g['ConstantTerm'][:].tolist()
                for j in range(len(linear)):
                    data['turbine_segments'][(arc_name,j)] = {'p': linear[j], 'rho': constant[j]}            
    nc.close()
    return data


filename = "/home/enzosawaya/ProjetOptiDisc/UCProject/data/smspp-hydro-units-main-Given Data/Given Data/20090907_extended_pHydro_18_none.nc4"
data = read_smspp_file(filename)

print(data.keys())
print(data['arcs']['UnitBlock_159_arc4'])


