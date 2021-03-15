import json
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from environment import MapParser
from graph import VictimType
import pickle
from numpy import linalg as LA
from numpy.linalg import matrix_rank
import math
import visualizer

def center(pos1, pos2):
    return (pos1[0] + pos2[0])/2, (pos1[1] + pos2[1])/2

def jl_transform(dataset_in,objective_dim,type_transform="basic"):
    """
    This function takes the dataset_in and returns the reduced dataset. The
    output dimension is objective_dim.
    dataset_in -- original dataset, list of Numpy ndarray
    objective_dim -- objective dimension of the reduction
    type_transform -- type of the transformation matrix used.
    If "basic" (default), each component of the transformation matrix
    is taken at random in N(0,1).
    If "discrete", each component of the transformation matrix
    is taken at random in {-1,1}.
    If "circulant", he first row of the transformation matrix
    is taken at random in N(0,1), and each row is obtainedfrom the
    previous one by a one-left shift.
    If "toeplitz", the first row and column of the transformation
    matrix is taken at random in N(0,1), and each diagonal has a
    constant value taken from these first vector.
    """
    if type_transform.lower() == "basic":
        jlt=(1/math.sqrt(objective_dim))*np.random.normal(0,1,size=(objective_dim,
                                                                    len(dataset_in[0])))
    elif type_transform.lower() == "discrete":
        jlt=(1/math.sqrt(objective_dim))*np.random.choice([-1,1],
                                                          size=(objective_dim,len(dataset_in[0])))
    elif type_transform.lower() == "circulant":
        from scipy.linalg import circulant
        first_row=np.random.normal(0,1,size=(1,len(dataset_in[0])))
        jlt=((1/math.sqrt(objective_dim))*circulant(first_row))[:objective_dim]
    elif type_transform.lower() == "toeplitz":
        from scipy.linalg import toeplitz
        first_row=np.random.normal(0,1,size=(1,len(dataset_in[0])))
        first_column=np.random.normal(0,1,size=(1,objective_dim))
        jlt=((1/math.sqrt(objective_dim))*toeplitz(first_column,first_row))
    else:
        print('Wrong transformation type')
        return None

    trans_dataset=[]
    [trans_dataset.append(np.dot(jlt,np.transpose(dataset_in[i])))
     for i in range(len(dataset_in))]
    return trans_dataset

def get_distance_matrix_original(graph, node_list):
    distance_matrix = np.zeros((len(node_list), len(node_list)))
    for n1 in range(len(node_list)):
        for n2 in range(n1+1, len(node_list)):
            length = nx.dijkstra_path_length(graph, node_list[n1], node_list[n2])
            distance_matrix[n1][n2] = length
    distance_matrix = distance_matrix + distance_matrix.transpose()
    return distance_matrix.tolist()

def distance_matrix_to_coordinate(D):
    # method to translate distance matrix to coordinate system
    # https://math.stackexchange.com/questions/156161/finding-the-coordinates-of-points-from-distance-matrix
    M = np.zeros((len(D), len(D)))
    for i in range(len(D)):
        for j in range(len(D)):
            M[i][j] = (D[0][j] ** 2 + D[i][0] ** 2 - D[i][j] ** 2) / 2

    w, v = LA.eig(M)
    X = v * np.sqrt(w)

    # remove zero and NaN columns
    X = X[:,~np.all(np.isnan(X), axis=0)]
    X = X[:,~np.all(X == 0, axis = 0)]

    coordinates = [np.array(i) for i in X]
    return coordinates

if __name__ == '__main__':


    with open('data\\json\\Saturn_1.0_sm_with_victimsA.json') as f:
        data = json.load(f)

    graph = MapParser.parse_saturn_map(data)
    victim_list_copy = graph.victim_list.copy()

    prize_list = []
    for v in victim_list_copy:
        if v.victim_type == VictimType.Yellow:
            prize_list.append(0.5)
        elif v.victim_type == VictimType.Green:
            prize_list.append(0.1)

    node_list = [graph['ew_1']] + victim_list_copy
    D = get_distance_matrix_original(graph, node_list)

    higher = distance_matrix_to_coordinate(D)
    lower = np.array(jl_transform(higher, 2))

    max_length = 5.5 # set this value the same as that for trained model

    loc = lower.copy()
    loc_all = lower.copy()

    lx = min(loc_all, key=lambda x:x[0])[0]
    lz = min(loc_all, key=lambda x:x[1])[1]
    rx = max(loc_all, key=lambda x:x[0])[0]
    rz = max(loc_all, key=lambda x:x[1])[1]
    span = max(rx - lx, rz - lz)

    for l in loc:
        l[0] = (l[0] - lx) / span
        l[1] = (l[1] - lz) / span


    depot, loc = loc[0].tolist(), loc[1:].tolist()
    # cord2D_obj = [(depot, loc, prize, max_length/span)]
    cord2D_obj = [(depot, loc, prize_list, max_length)]

    # with open('saturn_B.pkl', 'wb') as f:
    #     pickle.dump(cord2D_obj, f)

    path_idx = [0, 35, 33, 46, 50, 7, 2, 3, 23, 24, 21, 52, 11, 19, 14, 20, 22, 4, 5, 13, 51, 6, 18, 12, 10, 1, 8, 9, 15, 27, 16, 17, 30, 32, 53, 26, 29, 28, 25, 36, 37, 55, 39, 38, 31, 48, 54, 34, 40, 44, 49, 41, 45, 43, 42, 47]
    # path_idx.reverse()
    full_path = []
    for i in range(len(path_idx)-1):
        # print(node_list[path_idx[i]], node_list[path_idx[i+1]])
        # print(list(map(lambda x:x.id, nx.dijkstra_path(graph, node_list[path_idx[i]], node_list[path_idx[i+1]]))))
        full_path += list(map(lambda x:x.id, nx.dijkstra_path(graph, node_list[path_idx[i]], node_list[path_idx[i+1]])))[1:-1] + ["@"+node_list[path_idx[i+1]].id]
    full_path = ['ew_1'] + full_path
    # print(full_path)

    animate_frame = visualizer.simulate_run(graph, full_path[1:])

    visualizer.animate_MIP_graph(animate_frame, data, with_save="saturn_A_OP")