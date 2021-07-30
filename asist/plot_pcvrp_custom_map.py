from ipynb.plot_pcvrp import discrete_cmap
from ipynb.plot_pcvrp import plot_vehicle_routes
from asist.get_attention_problem import *

from asist.graph import VictimType

import os
import numpy as np
import torch
import pickle
import json

from torch.utils.data import DataLoader
from utils import load_model
from problems import PCVRP

from matplotlib import pyplot as plt

# 1009 1017 1025 1047 1049
high_value = 1
PRICE_MODE = 1
map_size = 55

with open('data/json/Saturn/Saturn_1.5_3D_sm_with_victimsA.json') as f:
    data = json.load(f)

medic_model, _ = load_model(f'../outputs/2021-6-23/graph_size=55,price_mode={PRICE_MODE},high_value={high_value}') # PCVRP

engineer_model, _ = load_model(f'../outputs/2021-7-28/2')

torch.manual_seed(1000)

graph = MapParser.parse_saturn_map(data)
victim_list_copy = graph.victim_list.copy()

rubble_normal_highvalue_victim = data["blocking_rubble"]
victim_list_reordered = []
for v in rubble_normal_highvalue_victim:
    victim_list_reordered.append(graph[v])

for v in victim_list_copy:
    if v not in victim_list_reordered:
        victim_list_reordered.append(v)

victim_list_copy = victim_list_reordered

node_list = [graph['ew_1']] + victim_list_copy
D = get_distance_matrix_original(graph, node_list)

higher = distance_matrix_to_coordinate(D)

# curr_lowest_cost = -2.564
# for sd in [1136, 1796]:

curr_lowest_cost = 0
length_ratio_list = []
total_time_list = []
# 1-1-1 seed 2037
for sd in range(2037, 2038):
    lower = np.array(jl_transform(higher, 2, seed=sd))


    max_length = 5.5  # set this value the same as that for trained model

    loc = lower.copy()
    loc_all = lower.copy()

    lx = min(loc_all, key=lambda x: x[0])[0]
    lz = min(loc_all, key=lambda x: x[1])[1]
    rx = max(loc_all, key=lambda x: x[0])[0]
    rz = max(loc_all, key=lambda x: x[1])[1]
    span = max(rx - lx, rz - lz)

    for l in loc:
        l[0] = (l[0] - lx) / span
        l[1] = (l[1] - lz) / span

    depot, loc = loc[0].tolist(), loc[1:].tolist()

    demand = [1/15] * map_size

    # for high_value in [0.1, 0.4, 0.7, 1]:
    #     for PRICE_MODE in [1, 2]:

    prize_list = []
    for v in victim_list_copy:
        if v.victim_type == VictimType.Yellow:
            prize_list.append(high_value)
        elif v.victim_type == VictimType.Green:
            prize_list.append(0.1)


    cord2D_obj = [(depot, loc, demand, prize_list)]

    with open('saturn_A_1.5_medic.pkl', 'wb') as f:
        pickle.dump(cord2D_obj, f)


    PCVRP.switch_player_role("medic")
    medic_dataset = PCVRP.make_dataset(size=55, filename='saturn_A_1.5_medic.pkl')

    # Need a dataloader to batch instances
    dataloader = DataLoader(medic_dataset, batch_size=1)

    # Make var works for dicts
    medic_instance = next(iter(dataloader))

    # Run the medic model
    medic_model.eval()
    medic_model.set_decode_type('greedy')
    with torch.no_grad():
        medic_length, medic_log_p, medic_pi = medic_model(medic_instance, return_pi=True)
    medic_tours = medic_pi

    PCVRP.switch_player_role("engineer")
    engineer_instance = medic_instance.copy()
    engineer_instance['medic_tour'] = medic_tours
    engineer_instance['demand'] = (torch.ones(28) / 131).unsqueeze(0)
    engineer_instance['prize'] = torch.zeros(28).unsqueeze(0)
    engineer_instance['victim_loc'] = engineer_instance['loc']
    engineer_instance['loc'] = torch.cat((engineer_instance['loc'][:, :23, :], engineer_instance['loc'][:, -5:, :]), 1)

    # Run the engineer model
    engineer_model.eval()
    engineer_model.set_decode_type('greedy')
    with torch.no_grad():
        engineer_length, engineer_log_p, engineer_pi = engineer_model(engineer_instance, return_pi=True)
    engineer_tours = engineer_pi



    # print(tours.shape)
    # print(tours[0])

    # Plot the results
    # for i, (medic_data, medic_tour) in enumerate(zip(medic_dataset, medic_tours)):
        # print(data)

    medic_data, medic_tour = medic_dataset[0], medic_tours[0]
    fig, ax = plt.subplots(figsize=(10, 10))
    medic_routes, medic_cost, medic_path_length = plot_vehicle_routes(medic_data, medic_tour, ax, visualize_demands=False, demand_scale=50, round_demand=True, return_routes=True)

    engineer_data = engineer_instance.copy()
    for ekey in engineer_data:
        engineer_data[ekey] = engineer_data[ekey].squeeze(0)
    engineer_tour = engineer_tours[0]
    engineer_routes, engineer_cost, engineer_path_length = plot_vehicle_routes(engineer_data, engineer_tour, ax,
                                                                      visualize_demands=False, demand_scale=50,
                                                                      round_demand=True, return_routes=True)

    print(f"{sd}    {medic_cost:.2f}    {engineer_cost:.2f}    {medic_path_length:.2f}    {engineer_path_length:.2f}")

    # print(f"cost: {cost:.4f}  current_lowest: {curr_lowest_cost:.4f}  seed: {sd}")
    # if cost < curr_lowest_cost:
    #     curr_lowest_cost = cost
    # fig.savefig(os.path.join('images', 'cvrp_{}.png'.format(i)))
    # print(routes)
    # fig.savefig(os.path.join('../images', f'pcvrp_price_mode={PRICE_MODE}_high_value={high_value}_seed={sd}.png'))

    # print(sd, len(routes), length)

    # total_length = 0
    # all_routes_json = []
    # medic_speed = 4.32*1.3


    medic_path = []
    for medic_route_id, medic_path_idx in enumerate(medic_routes):
        medic_full_path = []
        for i in range(len(medic_path_idx) - 1):
            # print(node_list[path_idx[i]], node_list[path_idx[i+1]])
            # print(list(map(lambda x:x.id, nx.dijkstra_path(graph, node_list[path_idx[i]], node_list[path_idx[i+1]]))))
            medic_full_path += list(
                map(lambda x: x.id, nx.dijkstra_path(graph, node_list[medic_path_idx[i]], node_list[medic_path_idx[i + 1]])))[1:-1] + [
                             "@" + node_list[medic_path_idx[i + 1]].id]
            medic_full_path = [node_list[0].id] + medic_full_path + list(map(lambda x:x.id, nx.dijkstra_path(graph, node_list[medic_path_idx[-1]], node_list[0])))[1:]



        # for tmp_node in range(len(full_path)-1):
        #     node_str = full_path[tmp_node].replace("@", "")
        #     node_str_p1 = full_path[tmp_node+1].replace("@", "")
        #     total_length += ((graph[node_str].loc[0] - graph[node_str_p1].loc[0]) ** 2 + (graph[node_str].loc[1] - graph[node_str_p1].loc[1]) ** 2) ** 0.5

    ## Investigate time
    # total_time = total_length / medic_speed
    # total_triage_time = 7.5*50 + 15*5
    # print(f"{total_time:.2f}+{total_triage_time:.2f}={total_time+total_triage_time:.2f}")
    # total_time_list.append(total_time+total_triage_time)

# print(np.average(total_time_list), np.min(total_time_list))


    # print(sd, path_length, total_length, path_length/total_length)
    # length_ratio_list.append(path_length/total_length)
# print(np.average(length_ratio_list), np.std(length_ratio_list))

        for path_id, path_node in enumerate(medic_full_path):
            is_triage = False
            _path_node = path_node
            if "@" in path_node:
                is_triage = True
                _path_node = path_node.replace("@", "")
            loc_x, loc_z = graph[_path_node].loc
            medic_path.append({
                "route_idx": medic_route_id,
                "path_idx": path_id,
                "node_id": _path_node,
                "is_action": is_triage,
                "loc_x": loc_x,
                "loc_z": loc_z
            })

    engineer_path = []
    for engineer_route_id, engineer_path_idx in enumerate(engineer_routes):
        engineer_full_path = []
        for i in range(len(engineer_path_idx) - 1):
            # print(node_list[path_idx[i]], node_list[path_idx[i+1]])
            # print(list(map(lambda x:x.id, nx.dijkstra_path(graph, node_list[path_idx[i]], node_list[path_idx[i+1]]))))
            engineer_full_path += list(
                map(lambda x: x.id,
                    nx.dijkstra_path(graph, node_list[engineer_path_idx[i]], node_list[engineer_path_idx[i + 1]])))[1:-1] + [
                                   "@" + node_list[engineer_path_idx[i + 1]].id]
            engineer_full_path = [node_list[0].id] + engineer_full_path + list(
                map(lambda x: x.id, nx.dijkstra_path(graph, node_list[engineer_path_idx[-1]], node_list[0])))[1:]

        for path_id, path_node in enumerate(engineer_full_path):
            is_action = False
            _path_node = path_node
            if "@" in path_node:
                is_action = True
                _path_node = path_node.replace("@", "")
            loc_x, loc_z = graph[_path_node].loc
            engineer_path.append({
                "route_idx": engineer_route_id,
                "path_idx": path_id,
                "node_id": _path_node,
                "is_action": is_action,
                "loc_x": loc_x,
                "loc_z": loc_z
            })

    with open(os.path.join('../images', f'sprint_1-1-1-ratio.json'), 'w') as json_file:
        json.dump({"meta": {
                        "high_value": high_value,
                        "price_mode": PRICE_MODE,
                        "cost_ratio": "1-1-1",
                        "seed": sd,
                    },
                   "data": {
                       "medic": medic_path,
                       "engineer": engineer_path,
                   }}, json_file, indent=2)


