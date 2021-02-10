from __future__ import print_function
from ortools.linear_solver import pywraplp
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

import json
from random import shuffle

from environment import MapParser
import visualizer
from graph.Nodes import VictimType

import time

def get_distance_matrix_original(graph, node_list):
    distance_matrix = np.zeros((len(node_list), len(node_list)))
    for n1 in range(len(node_list)):
        for n2 in range(n1+1, len(node_list)):
            length = nx.dijkstra_path_length(graph, node_list[n1], node_list[n2])
            distance_matrix[n1][n2] = length
    distance_matrix = distance_matrix + distance_matrix.transpose()
    distance_matrix[:,0] = 0
    return distance_matrix.tolist()

def get_distance_matrix_triage(graph, node_list):
    distance_matrix = np.zeros((len(node_list), len(node_list)))
    for n1 in range(len(node_list)):
        for n2 in range(n1+1, len(node_list)):
            length = nx.dijkstra_path_length(graph, node_list[n1], node_list[n2])
            triage_time = 15*5.6 if node_list[n2].victim_type == VictimType.Yellow else 7.5*5.6
            distance_matrix[n1][n2] = length + triage_time
    distance_matrix = distance_matrix + distance_matrix.transpose()
    distance_matrix[:,0] = 0
    return distance_matrix.tolist()

def get_distance_matrix_triage_innerdistance(graph, node_list, innerdistance):
    distance_matrix = np.zeros((len(node_list), len(node_list)))
    for n1 in range(len(node_list)):
        for n2 in range(n1+1, len(node_list)):
            length = nx.dijkstra_path_length(graph, graph[node_list[n1]], graph[node_list[n2]])
            triage_time = 15*5.6 if graph[node_list[n2]].victim_type == VictimType.Yellow else 7.5*5.6
            distance_matrix[n1][n2] = length + triage_time + innerdistance[n2]
    distance_matrix = distance_matrix + distance_matrix.transpose()
    distance_matrix[:,0] = 0
    return distance_matrix.tolist()

def scale_down_matrix(matrix, n):
    max_value = max([max(i) for i in matrix])
    for i in range(n):
        for j in range(n):
            matrix[i][j] /= max_value
    return matrix

def sep_yellow_green_victim_list(victim_list):
    yellow_victims = []
    green_victims = []
    for v in victim_list:
        if v.victim_type == VictimType.Yellow:
            yellow_victims.append(v)
        elif v.victim_type == VictimType.Green:
            green_victims.append(v)
    return yellow_victims, green_victims

def initialize(graph_json_data, init_pos="ew", victim_mask=[]):
    """ initialized data required for MIP and clustering
    :param graph_json_data: json data used to construct the map
    :param init_pos: current location of the agent
    :param victim_mask: list of victim ids to ignore (has already been saved or died)
    :return: a dictionary of needed data
    """
    graph = MapParser.parse_json_map_data_new_format(graph_json_data)
    victim_list = graph.victim_list.copy()
    victim_list_sift = []
    for v in victim_list:
        if v.id not in victim_mask:
            victim_list_sift.append(v)
    yellow_victims, green_victims = sep_yellow_green_victim_list(victim_list_sift)
    node_list = [graph[init_pos]] + yellow_victims + green_victims
    # victim_list = yellow_victims + green_victims
    distance_matrix = get_distance_matrix_original(graph, node_list)
    data = {
        "graph": graph,
        "node_list": node_list,
        "distance_matrix": distance_matrix,
        "num_yellow": len(yellow_victims),
        "num_green": len(green_victims),
        "num_all_nodes": len(node_list)
    }
    return data

def toy_example_1():
    distance_matrix = [
        [0, 2, 3, 3, 3],
        [0, 0, 1, 4, 2],
        [0, 1, 0, 1, 2],
        [0, 4, 1, 0, 1],
        [0, 2, 2, 1, 0]
    ]
    data = {
        "distance_matrix": distance_matrix,
        "num_yellow": 2,
        "num_green": 2,
        "num_all_nodes": 5
    }
    return data

def toy_example_2():
    distance_matrix = [
        [0, 2, 3, 3, 3, 3, 3, 3, 3],
        [0, 0, 1, 4, 2, 3, 4, 3, 5],
        [0, 1, 0, 1, 2, 5, 1, 3, 6],
        [0, 4, 1, 0, 1, 2, 6, 2, 7],
        [0, 2, 2, 1, 0, 3, 5, 2, 9],
        [0, 3, 5, 2, 3, 0, 4, 1, 2],
        [0, 4, 1, 6, 5, 4, 0, 7, 4],
        [0, 3, 3, 2, 2, 1, 7, 0, 3],
        [0, 5, 6, 7, 9, 2, 4, 3, 0]
    ]
    data = {
        "distance_matrix": distance_matrix,
        "num_yellow": 3,
        "num_green": 5,
        "num_all_nodes": 9
    }
    return data

def mip_solve(data, verbose=True):
    n = data["num_all_nodes"]
    solver = pywraplp.Solver.CreateSolver('CBC')
    M = 10000
    X = {}
    for s in range(n):
        for i in range(n):
            for j in range(n):
                X[f"{s}_{i}_{j}"] = solver.BoolVar(f"X[{s}_{i}_{j}]")

    for s in range(n):
        constraint_expr = []
        for i in range(n):
            for j in range(n):
                constraint_expr.append(X[f"{s}_{i}_{j}"])
        solver.Add(sum(constraint_expr) == 1)

    for i in range(n):
        constraint_expr = []
        for s in range(n):
            for j in range(n):
                if i != j:
                    constraint_expr.append(X[f"{s}_{i}_{j}"])
        solver.Add(sum(constraint_expr) == 1)

    for j in range(n):
        constraint_expr = []
        for s in range(n):
            for i in range(n):
                if i != j:
                    constraint_expr.append(X[f"{s}_{i}_{j}"])
        solver.Add(sum(constraint_expr) == 1)

    for s in range(n-1):
        for j in range(n):
            prev_node = []
            next_node = []
            for i in range(n):
                prev_node.append(X[f"{s}_{i}_{j}"])
            for k in range(n):
                next_node.append(X[f"{s+1}_{j}_{k}"])
            solver.Add(sum(prev_node) - sum(next_node) == 0)

    constraint_expr = []
    for j in range(n):
        constraint_expr.append(X[f"0_0_{j}"])
    solver.Add(sum(constraint_expr) == 1)

    Y = {}
    for s in range(n):
        for j in range(n):
            Y[f"{s}_{j}"] = solver.BoolVar(f"Y[{s}_{j}]")
    for s in range(n):
        for j in range(n):
            constraint_expr = []
            for i in range(n):
                constraint_expr.append(X[f"{s}_{i}_{j}"])
            solver.Add(Y[f"{s}_{j}"] == sum(constraint_expr))

    D = {}
    for s in range(n):
        for j in range(n):
            D[f"{s}_{j}"] = solver.NumVar(0, solver.infinity(), f'D[{s}_{j}]')
    for s in range(n):
        for j in range(n):
            constraint_expr = []
            for i in range(n):
                constraint_expr.append(X[f"{s}_{i}_{j}"] * data["distance_matrix"][i][j])
            D[f"{s}_{j}"] = sum(constraint_expr)

    ST = {}
    for s in range(n):
        ST[f"{s}"] = solver.NumVar(0, solver.infinity(), f'ST[{s}]')
    for _s in range(n):
        constraint_expr = []
        for s in range(_s):
            for i in range(n):
                constraint_expr.append(D[f"{s}_{i}"])
        solver.Add(ST[f"{_s}"] == sum(constraint_expr))

    T = {}
    for i in range(n):
        T[f"{i}"] = solver.NumVar(0, solver.infinity(), f'T[{i}]')
    for s in range(1, n):
        for i in range(n):
            solver.Add(T[f"{i}"] >= ST[f"{s}"] + D[f"{s}_{i}"] - M*(1 - Y[f"{s}_{i}"]))

    Threshold_yellow = 5 * 60 * 5.6
    Threshold_green = 10 * 60 * 5.6

    V = {}
    for i in range(n):
        V[f"{i}"] = solver.BoolVar(f"V[{i}]")
    # V = {}
    # for i in range(n):
    #     constraint_expr = []
    #     for s in range(n):
    #         constraint_expr.append(Y[f"{s}_{i}"])
    #     V[f"{i}"] = sum(constraint_expr)

    for i in range(1, 1 + data["num_yellow"]):
        solver.Add(T[f"{i}"] - M*(1 - V[f"{i}"]) <= Threshold_yellow)

    for i in range(1 + data["num_yellow"], data["num_all_nodes"]):
        solver.Add(T[f"{i}"] - M*(1 - V[f"{i}"]) <= Threshold_green)

    Reward_yellow = 30
    Reward_green = 10

    obj_expr_1 = sum([T[f"{i}"] for i in range(n)])
    obj_expr_2 = sum([Reward_yellow * V[f"{i}"] for i in range(1 + data["num_yellow"])])
    obj_expr_3 = sum([Reward_green * V[f"{i}"] for i in range(1 + data["num_yellow"], data["num_all_nodes"])])
    # solver.Minimize(obj_expr_1 - obj_expr_2 - obj_expr_3)
    solver.Maximize(obj_expr_2 + obj_expr_3)

    solver.SetTimeLimit(100000000)
    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        if verbose:
            print('Objective value =', solver.Objective().Value())
            print('Problem solved in %f milliseconds' % solver.wall_time())
            print('Problem solved in %d iterations' % solver.iterations())
            print('Problem solved in %d branch-and-bound nodes' % solver.nodes())

            print("below print for X")
            for s in range(n):
                print(f"s={s}")
                for i in range(n):
                    for j in range(n):
                        print(int(X[f"{s}_{i}_{j}"].solution_value()), end=" ")
                    print()
                print()

            print("below print for Y")
            for s in range(n):
                for i in range(n):
                    print(int(Y[f"{s}_{i}"].solution_value()), end=" ")
                print()
            print()

            print("below print for V")
            for i in range(n):
                print(int(V[f"{i}"].solution_value()), end=" ")
            print("\n")

            print("below print for T")
            for i in range(n):
                print(int(T[f"{i}"].solution_value()), end=" ")
            print("\n")

            print("below print for ST")
            for s in range(n):
                print(int(ST[f"{s}"].solution_value()), end=" ")
            print("\n")

            print("below print for D")
            for s in range(n):
                for i in range(n):
                    print(int(D[f"{s}_{i}"].solution_value()), end=" ")
                print()
            print()
    else:
        if verbose:
            print('The problem does not have an optimal solution.')
        return None

    solution = []
    for s in range(n-1):
        for i in range(n):
            if int(Y[f"{s}_{i}"].solution_value()) == 1:
                solution.append(i)
                break
    return solution

def find_path(json_data, init_pos="ew", victim_mask=[], mip_verbose=False):
    """ find path to visit all victims, given current location, and victims already saved
    :param json_data: json data used to construct the map
    :param init_pos: current location of the agent
    :param victim_mask: list of victim ids to ignore (has already been saved or died)
    :return: a list of node ids representing the path, None if no solution is found
    """
    data = initialize(json_data, init_pos, victim_mask)
    graph = data["graph"]
    model = AgglomerativeClustering(distance_threshold=60, n_clusters=None, linkage="complete", affinity='precomputed')
    model = model.fit(data["distance_matrix"])
    clusters = [[] for i in range(max(model.labels_)+1)]
    for idx, n in enumerate(data["node_list"]):
        clusters[model.labels_[idx]].append(n.id)

    sep_cluster_yellow = []
    sep_cluster_green = []
    for l in clusters:
        gvl = []
        yvl = []
        for v in l:
            if 'vg' in v:
                gvl.append(v)
            if 'vy' in v:
                yvl.append(v)
        if len(yvl) > 0:
            sep_cluster_yellow.append(yvl)
        if len(gvl) > 0:
            sep_cluster_green.append(gvl)

    clusters_sep_color = [[init_pos]] + sep_cluster_yellow + sep_cluster_green

    representative_nodes = []
    inner_distance = []

    for c in clusters_sep_color:
        representative_nodes.append(c[0])
        if len(c) == 0:
            inner_distance.append(0)
        else:
            distance = 0
            triage_time = 0
            for n in range(len(c)-1):
                distance += nx.dijkstra_path_length(graph, graph[c[n]], graph[c[n+1]])
                triage_time += 15*5.6 if graph[c[n+1]].victim_type == VictimType.Yellow else 7.5*5.6
            inner_distance.append(distance + triage_time)

    distance_matrix_triage_inner = get_distance_matrix_triage_innerdistance(graph, representative_nodes, inner_distance)

    data_cluster_version = {
        "distance_matrix": distance_matrix_triage_inner,
        "num_yellow": len(sep_cluster_yellow),
        "num_green": len(sep_cluster_green),
        "num_all_nodes": len(clusters_sep_color)
    }
    solution = mip_solve(data_cluster_version, verbose=mip_verbose)
    if solution is None:
        return None
    # print(solution)
    node_path = [init_pos]
    full_path = []
    for c in solution:
        node_path += clusters_sep_color[c]
    # print(node_path)
    for i in range(len(node_path)-1):
        full_path += list(map(lambda x:x.id, nx.dijkstra_path(graph, graph[node_path[i]], graph[node_path[i+1]])))[1:-1] + ["@"+graph[node_path[i+1]].id]
    full_path = [init_pos] + full_path
    # print(full_path)
    return full_path

if __name__ == "__main__":
    with open('data\\json\\Falcon_v1.0_Easy_sm_clean.json') as f:
        graph_json_data = json.load(f)

    full_path = find_path(graph_json_data, init_pos="ew", victim_mask=['vg21', 'vy1'])
    print(full_path)

    # graph = MapParser.parse_json_map_data_new_format(graph_json_data)
    # animate_frame = visualizer.simulate_run(graph, full_path[1:])
    # visualizer.animate_MIP_graph(animate_frame, graph_json_data)