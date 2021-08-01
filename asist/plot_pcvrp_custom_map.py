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

torch.manual_seed(1000)

class RouteGen:
    def __init__(self, sm_data_path, medic_model_path, engineer_model_path):
        self.high_value = 1
        self.PRICE_MODE = 1
        self.medic_speed = 4.32 * 1.3
        self.engineer_speed = 3.02 * 1.3
        self.medic_graph_size = 55
        self.engineer_graph_size = 28
        self.MEDIC_TOOL_DURABILITY = 15
        self.ENGINEER_TOOL_DURABILITY = 131
        self.RUBBLE_GRAPH_SIZE = 23
        self.HIGH_VALUE_VICTIM_SIZE = 5
        self.jl_transform_seed = 2037
        self.depot_room_id = 'ew_1'
        self.loadSMData(sm_data_path)
        self.graph = MapParser.parse_saturn_map(self.sm_data)
        self.medic_model, self.engineer_model = self.loadRouteModel(medic_model_path, engineer_model_path)
        self.rubble_normal_highvalue_victim = self.sm_data["blocking_rubble"]


    def loadSMData(self, sm_json_file):
        with open(sm_data_path) as sm_json_file:
            self.sm_data = json.load(sm_json_file)

    def loadRouteModel(self, medic=None, engineer=None):
        assert medic is not None or engineer is not None, "need to load at least one model"
        medic_model, _ = load_model(medic) if medic is not None else None
        engineer_model, _ = load_model(engineer) if engineer is not None else None
        return medic_model, engineer_model

    def get_ordered_node_list(self):
        victim_list_copy = self.graph.victim_list.copy()
        victim_list_reordered = []
        for v in self.rubble_normal_highvalue_victim:
            victim_list_reordered.append(self.graph[v])
        for v in victim_list_copy:
            if v not in victim_list_reordered:
                victim_list_reordered.append(v)
        self.victim_list = victim_list_reordered
        self.node_list = [self.graph[self.depot_room_id]] + victim_list_reordered

    def translate2D01space(self, dump_pkl=None):
        D = get_distance_matrix_original(self.graph, self.node_list)
        higher = distance_matrix_to_coordinate(D)
        lower = np.array(jl_transform(higher, 2, seed=self.jl_transform_seed))
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
        max_length = 5.5  # set this value the same as that for trained model

        demand = [1 / self.MEDIC_TOOL_DURABILITY] * self.medic_graph_size
        prize_list = []
        for v in self.victim_list:
            if v.victim_type == VictimType.Yellow:
                prize_list.append(self.high_value)
            elif v.victim_type == VictimType.Green:
                prize_list.append(0.1)

        cord2D_obj = [(depot, loc, demand, prize_list)]

        if dump_pkl is not None:
            with open(dump_pkl, 'wb') as f:
                pickle.dump(cord2D_obj, f)

    def get_medic_tour_index(self, medic_pickle='saturn_A_1.5_medic.pkl'):
        PCVRP.switch_player_role("medic")
        medic_dataset = PCVRP.make_dataset(size=self.medic_graph_size, filename='saturn_A_1.5_medic.pkl')

        # Need a dataloader to batch instances
        dataloader = DataLoader(medic_dataset, batch_size=1)

        # Make var works for dicts
        medic_instance = next(iter(dataloader))

        # Run the medic model
        self.medic_model.eval()
        self.medic_model.set_decode_type('greedy')
        with torch.no_grad():
            medic_length, medic_log_p, medic_pi = self.medic_model(medic_instance, return_pi=True)
        medic_tours = medic_pi
        return medic_tours[0], medic_dataset[0]

    def get_engineer_tour_index(self, medic_tour_index, medic_instance):
        PCVRP.switch_player_role("engineer")
        engineer_instance = medic_instance.copy()
        engineer_instance['medic_tour'] = medic_tour_index.unsqueeze(0)
        engineer_instance['depot'] = engineer_instance['depot'].unsqueeze(0)
        engineer_instance['demand'] = (torch.ones(self.engineer_graph_size) / self.ENGINEER_TOOL_DURABILITY).unsqueeze(0)
        engineer_instance['prize'] = torch.zeros(self.engineer_graph_size).unsqueeze(0)
        engineer_instance['victim_loc'] = engineer_instance['loc'].unsqueeze(0)
        engineer_instance['loc'] = torch.cat((engineer_instance['victim_loc'][:, :self.RUBBLE_GRAPH_SIZE, :], engineer_instance['victim_loc'][:, -self.HIGH_VALUE_VICTIM_SIZE:, :]), 1)

        # Run the engineer model
        self.engineer_model.eval()
        self.engineer_model.set_decode_type('greedy')
        with torch.no_grad():
            engineer_length, engineer_log_p, engineer_pi = self.engineer_model(engineer_instance, return_pi=True)
        engineer_tours = engineer_pi
        return engineer_tours[0], engineer_instance

    def get_medic_seperate_tour_index(self, medic_tour, medic_instance, save_fig_path=None):
        fig, ax = plt.subplots(figsize=(10, 10))
        medic_routes, medic_cost, medic_path_length = plot_vehicle_routes(medic_instance, medic_tour, ax,
                                                                          visualize_demands=False, demand_scale=50,
                                                                          round_demand=True, return_routes=True)
        if save_fig_path is not None:
            fig.savefig(os.path.join(save_fig_path))

        return medic_routes, medic_cost, medic_path_length

    def get_engineer_seperate_tour_index(self, engineer_tour, engineer_instance, save_fig_path=None):
        engineer_data = engineer_instance.copy()
        fig, ax = plt.subplots(figsize=(10, 10))
        for ekey in engineer_data:
            engineer_data[ekey] = engineer_data[ekey].squeeze(0)
        engineer_routes, engineer_cost, engineer_path_length = plot_vehicle_routes(engineer_data, engineer_tour, ax,
                                                                                   visualize_demands=False,
                                                                                   demand_scale=50,
                                                                                   round_demand=True,
                                                                                   return_routes=True)
        if save_fig_path is not None:
            fig.savefig(os.path.join(save_fig_path))

        return engineer_routes, engineer_cost, engineer_path_length

    def get_medic_path_readable(self, medic_routes):
        medic_path = []
        for medic_route_id, medic_path_idx in enumerate(medic_routes):
            medic_full_path = []
            for i in range(len(medic_path_idx) - 1):
                # print(node_list[path_idx[i]], node_list[path_idx[i+1]])
                # print(list(map(lambda x:x.id, nx.dijkstra_path(graph, node_list[path_idx[i]], node_list[path_idx[i+1]]))))
                medic_full_path += list(
                    map(lambda x: x.id,
                        nx.dijkstra_path(self.graph, self.node_list[medic_path_idx[i]], self.node_list[medic_path_idx[i + 1]])))[
                                   1:-1] + ["@" + self.node_list[medic_path_idx[i + 1]].id]
            medic_full_path = [self.node_list[0].id] + medic_full_path + list(
                map(lambda x: x.id, nx.dijkstra_path(self.graph, self.node_list[medic_path_idx[-1]], self.node_list[0])))[1:]

            for path_id, path_node in enumerate(medic_full_path):
                is_triage = False
                _path_node = path_node
                if "@" in path_node:
                    is_triage = True
                    _path_node = path_node.replace("@", "")
                loc_x, loc_z = self.graph[_path_node].loc
                medic_path.append({
                    "route_idx": medic_route_id,
                    "path_idx": path_id,
                    "node_id": _path_node,
                    "is_action": is_triage,
                    "loc_x": loc_x,
                    "loc_z": loc_z
                })
        return medic_path

    def get_engineer_path_readable(self, engineer_routes):
        engineer_path = []
        for engineer_route_id, engineer_path_idx in enumerate(engineer_routes):
            engineer_full_path = []
            for i in range(len(engineer_path_idx) - 1):
                # print(node_list[path_idx[i]], node_list[path_idx[i+1]])
                # print(list(map(lambda x:x.id, nx.dijkstra_path(graph, node_list[path_idx[i]], node_list[path_idx[i+1]]))))
                engineer_full_path += list(
                    map(lambda x: x.id,
                        nx.dijkstra_path(self.graph, self.node_list[engineer_path_idx[i]], self.node_list[engineer_path_idx[i + 1]])))[
                                      1:-1] + ["@" + self.node_list[engineer_path_idx[i + 1]].id]
            engineer_full_path = [self.node_list[0].id] + engineer_full_path + list(
                map(lambda x: x.id, nx.dijkstra_path(self.graph, self.node_list[engineer_path_idx[-1]], self.node_list[0])))[1:]

            for path_id, path_node in enumerate(engineer_full_path):
                is_action = False
                _path_node = path_node
                if "@" in path_node:
                    is_action = True
                    _path_node = path_node.replace("@", "")
                loc_x, loc_z = self.graph[_path_node].loc
                engineer_path.append({
                    "route_idx": engineer_route_id,
                    "path_idx": path_id,
                    "node_id": _path_node,
                    "is_action": is_action,
                    "loc_x": loc_x,
                    "loc_z": loc_z
                })

        return engineer_path


    def save_to_path_file(self, medic_path, engineer_path, output_file):
        with open(os.path.join(output_file), 'w') as json_file:
            json.dump({"meta": {
                "high_value": self.high_value,
                "price_mode": self.PRICE_MODE,
                "cost_ratio": "1-1-1",
                "seed": self.jl_transform_seed,
            },
                "data": {
                    "medic": medic_path,
                    "engineer": engineer_path,
                }}, json_file, indent=2)

    def get_event_list(self, path_ids):
        curr_length = 0
        curr_time = 0
        events = []
        action_delta_length = 0
        prev_node = None
        for tmp_node in path_ids:
            is_action_node = True if '@' in tmp_node else False
            node_str = tmp_node.replace("@", "")
            if prev_node is not None:
                if node_str == 'ew_1' and prev_node == 'ew_1':
                    continue
                delta_length = ((self.graph[node_str].loc[0] - self.graph[prev_node].loc[0]) ** 2 + (
                            self.graph[node_str].loc[1] - self.graph[prev_node].loc[1]) ** 2) ** 0.5
                action_delta_length += delta_length
                if is_action_node or node_str == 'ew_1':
                    events.append((node_str, action_delta_length, is_action_node))
                    action_delta_length = 0
            prev_node = node_str
        return events

    def path_time_analysis(self, medic_path, engineer_path, time_log_file=None, print_log=True):
        # self.rubble_normal_highvalue_victim
        # time_log = open(time_log_file, 'w')
        log_list = []
        medic_path_ids = []
        for p in medic_path:
            medic_path_ids.append(p['node_id'] if not p['is_action'] else '@' + p['node_id'])

        engineer_path_ids = []
        for p in engineer_path:
            engineer_path_ids.append(p['node_id'] if not p['is_action'] else '@' + p['node_id'])

        # (node_id, length, is_action)
        medic_events = self.get_event_list(medic_path_ids)
        engineer_events = self.get_event_list(engineer_path_ids)

        action_time = {
            'med_green': 7.5,
            'med_yellow': 15,
            'eng_break': 0.5,
            'med_wait': -1,
            'eng_wait': -1
        }
        from itertools import zip_longest
        for md, eg in zip_longest(medic_events, engineer_events):
            if md is None:
                ttt = eg[1] / self.engineer_speed
                print(f"{''<20}{eg[0]:<10}{ttt:<10.1f}")
            elif eg is None:
                ttt = md[1] / self.medic_speed
                print(f"{md[0]:<10}{ttt:<10.1f}{'':<20}")
            else:
                tttm = md[1] / self.medic_speed
                ttte = eg[1] / self.engineer_speed
                print(f"{md[0]:<10}{tttm:<10.1f}{eg[0]:<10}{ttte:<10.1f}")

#         cleared_rubble_ids = set()

#         medic_total_time, engineer_total_time = 0, 0
#         log_list.append((0, 'INFO', "The game has started."))
#         mei, eei = 0, 0
#         medic_action, engineer_action = None, None
#         medic_wait, engineer_wait = None, None
#         while mei < len(medic_events) or eei < len(engineer_events):
#             medic_time = action_time[medic_action] if medic_action is not None else medic_events[mei][1] / self.medic_speed
#             engineer_time = action_time[engineer_action] if engineer_action is not None else engineer_events[eei][1] / self.engineer_speed
#             if medic_total_time + medic_time < engineer_total_time + engineer_time and medic_wait is None:
#                 medic_total_time += medic_time
#                 if medic_action is None:
#                     log_str = f"Medic has moved to {medic_events[mei].upper()}, cost {medic_time:.1f}s"
#                     if 'vg' in medic_events[mei][0]:
#                         if medic_events[mei][0] in cleared_rubble_ids:
#                             medic_action = 'med_green'
#                         else:
#                             medic_action = 'med_wait'
#                             action_time['med_wait'] = 0
#                             medic_wait = medic_events[mei][0]
#                         mei += 1
#                     if 'vy' in medic_events[mei][0]:
#                         if engineer_wait == medic_events[mei][0]:
#                             medic_action = 'med_yellow'
#                         else:
#                             medic_action = 'med_wait'
#                             action_time['med_wait'] = 0
#                             medic_wait = medic_events[mei][0]
#                 elif medic_action == 'med_green':
#                     log_str = f"Medic has triaged normal victim {medic_events[mei].upper()}, cost {medic_time:.1f}s"
#                 elif medic_action == 'med_yellow':
#                     log_str = f"Medic has triaged high-value victim {medic_events[mei].upper()}, cost {medic_time:.1f}s"
#                 log_list.append((medic_total_time, 'MEDI', log_str))
#             else:
#                 engineer_total_time += engineer_time
#                 if engineer_action is None:
#                     log_str = f"Engineer has moved to {engineer_events[eei].upper()}, cost {engineer_time:.1f}s"
#                     if engineer_events[eei][0] in self.blocking_rubbles:
#                         engineer_action = 'eng_break'
#                     if 'vy' in engineer_events[eei][0]:
#                         engineer_action = 'med_yellow'
#                 elif medic_action == 'med_green':
#                     log_str = f"Medic has triaged normal victim {medic_events[mei].upper()}, cost {medic_time:.1f}s"
#                 elif medic_action == 'med_yellow':
#                     log_str = f"Medic has triaged high-value victim {medic_events[mei].upper()}, cost {medic_time:.1f}s"
#                 log_list.append((medic_total_time, 'MEDI', log_str))


if __name__ == '__main__':
    sm_data_path = os.path.join('data', 'json', 'Saturn', 'Saturn_1.5_3D_sm_with_victimsA.json')
    medic_model_path = os.path.join('..', 'outputs', '2021-6-23', 'graph_size=55,price_mode=1,high_value=1')
    engineer_model_path = os.path.join('..', 'outputs', '2021-7-28', '2')
    medic_pickle_file = "saturn_A_1.5_medic.pkl"
    medic_fig_path = os.path.join('..', 'images', 'medic.png')
    engineer_fig_path = os.path.join('..', 'images', 'engineer.png')
    output_path_json_path = os.path.join('..', 'images', 'sprint_1-1-1-ratio.json')

    rg = RouteGen(sm_data_path, medic_model_path, engineer_model_path)
    rg.get_ordered_node_list()
    rg.translate2D01space(medic_pickle_file)
    medic_tour, medic_data = rg.get_medic_tour_index()
    engineer_tour, engineer_data = rg.get_engineer_tour_index(medic_tour, medic_data)
    medic_routes, medic_cost, medic_path_length = rg.get_medic_seperate_tour_index(medic_tour, medic_data, save_fig_path=medic_fig_path)
    engineer_routes, engineer_cost, engineer_path_length = rg.get_engineer_seperate_tour_index(engineer_tour, engineer_data, save_fig_path=engineer_fig_path)
    medic_path = rg.get_medic_path_readable(medic_routes)
    engineer_path = rg.get_engineer_path_readable(engineer_routes)
    rg.save_to_path_file(medic_path, engineer_path, output_path_json_path)

# curr_lowest_cost = -2.564
# for sd in [1136, 1796]:

# curr_lowest_cost = 0
# length_ratio_list = []
# total_time_list = []
# # 1-1-1 seed 2037
# for sd in range(2037, 2038):
#
#
#
#
#
#     print(f"{sd}    {medic_cost:.2f}    {engineer_cost:.2f}    {medic_path_length:.2f}    {engineer_path_length:.2f}")

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








