#!/usr/bin/env python

import os
import numpy as np
import torch
import pickle
import json
from environment import MapParser
from graph import VictimType
from get_attention_problem import get_distance_matrix_original
from get_attention_problem import distance_matrix_to_coordinate
from get_attention_problem import jl_transform

from utils import load_model
import visualizer
import graph

from matplotlib import pyplot as plt
import matplotlib.colors as colors

def plot_tsp(xy, tour, ax1):

    """
    Plot the TSP tour on matplotlib axis ax1.
    """
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    
    xs, ys = xy[tour].transpose()
    xs, ys = xy[tour].transpose()
    dx = np.roll(xs, -1) - xs
    dy = np.roll(ys, -1) - ys
    d = np.sqrt(dx * dx + dy * dy)
    lengths = d.cumsum()
    
    # Scatter nodes
    ax1.scatter(xs, ys, s=40, color='blue')
    # Starting node
    ax1.scatter([xs[0]], [ys[0]], s=100, color='red')
    
    # Arcs
    qv = ax1.quiver(
        xs, ys, dx, dy,
        scale_units='xy',
        angles='xy',
        scale=1,
    )
    print('OPTIMAL PATH: {} nodes, total length {:.2f}'.format(len(tour), lengths[-1]))
    ax1.set_title('{} nodes, total length {:.2f}'.format(len(tour), lengths[-1]))

def get_length(xy, tour):

    """
    get tsp tour length without doing plot
    """
    xs, ys = xy[tour].transpose()
    xs, ys = xy[tour].transpose()
    dx = np.roll(xs, -1) - xs
    dy = np.roll(ys, -1) - ys
    d = np.sqrt(dx * dx + dy * dy)
    lengths = d.cumsum()

    return lengths[-1]

def make_oracle(model, xy, temperature=1.0):
    num_nodes = len(xy)
    xyt = torch.tensor(xy).float()[None]  # Add batch dimension
    with torch.no_grad():  # Inference only
        embeddings, _ = model.embedder(model._init_embed(xyt))

        # Compute keys, values for the glimpse and keys for the logits once as they can be reused in every step
        fixed = model._precompute(embeddings)
    
    def oracle(tour):
        with torch.no_grad():  # Inference only
            # Input tour with 0 based indices
            # Output vector with probabilities for locations not in tour
            tour = torch.tensor(tour).long()
            if len(tour) == 0:
                step_context = model.W_placeholder
            else:
                step_context = torch.cat((embeddings[0, tour[0]], embeddings[0, tour[-1]]), -1)

            # Compute query = context node embedding, add batch and step dimensions (both 1)
            query = fixed.context_node_projected + model.project_step_context(step_context[None, None, :])

            # Create the mask and convert to bool depending on PyTorch version
            mask = torch.zeros(num_nodes, dtype=torch.uint8) > 0
            mask[tour] = 1
            mask = mask[None, None, :]  # Add batch and step dimension

            log_p, _ = model._one_to_many_logits(query, fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key, mask)
            p = torch.softmax(log_p / temperature, -1)[0, 0]
            assert (p[tour] == 0).all()
            assert (p.sum() - 1).abs() < 1e-5
            #assert np.allclose(p.sum().item(), 1)
        return p.numpy()
    
    return oracle

# set max victims in case you want to use a model with fewer nodes than your map
# excludes allows to recreate new graph after victims have been rescued
def get_graph_coords_json(jfile, model, start_node='ew', max_victims=34, excludes=[]):
    with open(jfile) as f:
        data = json.load(f)

    orig_graph = MapParser.parse_json_map_data(data, excludes)
    victim_list_copy = orig_graph.victim_list.copy()
    prize_list = []
    nexcludes = len(excludes)
    #for v in victim_list_copy:
    for i in range(0,max_victims-nexcludes):
        v = victim_list_copy[i]
        if v.victim_type == VictimType.Yellow:
            prize_list.append(0.3)
        elif v.victim_type == VictimType.Green:
            prize_list.append(0.1)

    node_list = [orig_graph[start_node]] + victim_list_copy
    D = get_distance_matrix_original(orig_graph, node_list)
    
    ##### graph original set of victims to use for indexing
    victim_graph = graph.Graph()
    for n in victim_list_copy:
        if str(n.id).find('g') > -1:
            ntype = VictimType.Green
        else:
            ntype = VictimType.Yellow
        victim_graph.add_victim(ntype, str(n.id), n.name, n.loc)

    higher = distance_matrix_to_coordinate(D)
    lower = np.array(jl_transform(higher, 2))

    max_length = 3.6 # set this value the same as that for trained model

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
    some_obj = [(depot, loc, prize_list, max_length)]

    ###### format coordinate array (from jupytr)
    xy = [some_obj[0][0]] + some_obj[0][1]
    xy = np.array(some_obj[0][1])
    #print(xy)

    oracle = make_oracle(model, xy)
    sample = False
    tour = []
    tour_p = []
    while(len(tour) < len(xy)):
        p = oracle(tour)
        
        if sample:
            # Advertising the Gumbel-Max trick
            g = -np.log(-np.log(np.random.rand(*p.shape)))
            i = np.argmax(np.log(p) + g)
        else:
            # Greedy
            i = np.argmax(p)
        tour.append(i)
        tour_p.append(p)
    
    # print(tour)
    # rearrange the tour such that depot is at the beginning
    #  maybe don't need depot if mid game?
    zero = tour.index(0)
    tour = tour[zero:] + tour[:zero]
    tour_p = tour_p[zero:] + tour_p[:zero]
    #print(tour)
    return orig_graph, victim_graph, tour, xy

def get_coords_pkl(pklfile, model):
    with open(pklfile, 'rb') as f:
        loaded_obj = pickle.load(f)
    
    xy = [loaded_obj[0][0]] + loaded_obj[0][1]
    xy = np.array(loaded_obj[0][1])
    print(xy)

    oracle = make_oracle(model, xy)

    sample = False
    tour = []
    tour_p = []
    while(len(tour) < len(xy)):
        p = oracle(tour)
        if sample:
            # Advertising the Gumbel-Max trick
            g = -np.log(-np.log(np.random.rand(*p.shape)))
            i = np.argmax(np.log(p) + g)
        else:
            # Greedy
            i = np.argmax(p)
        tour.append(i)
        tour_p.append(p)
    print(tour)
    # rearrange the tour such that depot is at the beginning
    zero = tour.index(0)
    tour = tour[zero:] + tour[:zero]
    tour_p = tour_p[zero:] + tour_p[:zero]
    print("NEW TOUR")
    print(tour)
    return tour, xy

# plot optimized graph with normalized values to see if new json file looks like plot_tsp
# TODO: get nvals & nxvals & span from graph
def plot_graph_norm(optgraph, tour):
    nxvals = 71
    nzvals = 46
    normxy = []
    regxy = []
    xvals = []
    zvals = []
    minx = optgraph.nodes_list[0].loc[0]
    maxx = optgraph.nodes_list[0].loc[0]
    minz = optgraph.nodes_list[0].loc[1]
    maxz = optgraph.nodes_list[0].loc[1]
    for n in optgraph.nodes_list:
        xvals.append(n.loc[0])
        zvals.append(n.loc[1])
        regxy.append((n.loc[0], n.loc[1]))
        if n.loc[0] < minx:
            minx = n.loc[0]
        if n.loc[0] > maxx:
            maxx = n.loc[0]
        if n.loc[1] < minz:
            minz = n.loc[1]
        if n.loc[1] > maxz:
            maxz = n.loc[1]
        loc1 = (n.loc[0]+2096)/nxvals
        loc2 = (n.loc[1]-145)/nzvals
        normxy.append((loc1,loc2))
        print("orig x = "+str(n.loc[0])+" orig z = "+str(n.loc[1]))
        print("new x  = "+str(loc1)+" new z  = "+str(loc2))
    print(" min x = "+str(minx)+" maxx = "+str(maxx)+" minz = "+str(minz)+" maxz "+str(maxz)+" lenxy = "+str(len(normxy)))    

    fig, ax = plt.subplots(figsize=(10, 10))
    plot_tsp(np.array(normxy), tour, ax)
    plt.show()

##### writing new path to map file -- currently can only run when you have full semantic map
## TODO: update to run with just tour & xy
def write_optimal_json(tour, optgraph, fname):
    idxs_for_json = []
    colors_for_json = []
    for i in range(0,len(tour)-1):
        idx1 = tour[i]
        idx2 = tour[i+1]
        n1 = optgraph.nodes_list[idx1]
        n2 = optgraph.nodes_list[idx2]
        optgraph.add_edge(n1, n2, weight=1)
        idxs_for_json.append((n1.loc))
        if optgraph.nodes_list[idx1].victim_type == VictimType.Yellow:
            c = "yellow"
        else:
            c = "green"
        colors_for_json.append(c+"_victim")

    vcnt = 0
    jout = open(fname, 'w') # new file only has victim locations, do we want to add back the other stuff?
    data = {}
    data['objects'] = []
    for x in idxs_for_json:
        data['objects'].append({
            "id":"vg"+str(vcnt),
            "type": colors_for_json[vcnt],
            "bounds": {
                "type" : "block",
                "coordinates": [
                    {
                        "x": x[0], 
                        "z": x[1]
                        }
                    ]
                }
            })
        vcnt += 1
    json.dump(data,jout,indent=True)
    jout.close()         

def write_optimal_json_no_graph(tour, fname):
    idxs_for_json = []
    colors_for_json = []
    for i in range(0,len(tour)-1):
        idx1 = tour[i]
        idx2 = tour[i+1]
        n1 = optgraph.nodes_list[idx1]
        n2 = optgraph.nodes_list[idx2]
        optgraph.add_edge(n1, n2, weight=1)
        idxs_for_json.append((n1.loc))
        if optgraph.nodes_list[idx1].victim_type == VictimType.Yellow:
            c = "yellow"
        else:
            c = "green"
        colors_for_json.append(c+"_victim")

    vcnt = 0
    jout = open(fname, 'w') # new file only has victim locations, do we want to add back the other stuff?
    data = {}
    data['objects'] = []
    for x in idxs_for_json:
        data['objects'].append({
            "id":"vg"+str(vcnt),
            "type": colors_for_json[vcnt],
            "bounds": {
                "type" : "block",
                "coordinates": [
                    {
                        "x": x[0], 
                        "z": x[1]
                        }
                    ]
                }
            })
        vcnt += 1
    json.dump(data,jout,indent=True)
    jout.close()         
