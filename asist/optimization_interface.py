#!/usr/bin/env python

from interface_utils import *
import os
import sys
import json
from utils import load_model
from matplotlib import pyplot as plt

#######

def proc_map(input_type, fname, modfile):
    if input_type == "" or fname == "":
        print("required arguments: --smap <json file semantic map> or --pkl <coorinates pre-formatted>")
        sys.exit()
    
    ###### use pre-trained model to find optimal path ordering

    model, _ = load_model(modfile)
    model.eval()  # Put in evaluation mode to not track gradients
    
    if input_type == 'json':
        optgraph, tour, xy = get_graph_coords_json(fname, model)
    else:
        tour, xy = get_coords_pkl(fname, model)        

    ##################### show optimized graph
    
    fig, ax = plt.subplots(figsize=(10, 10))
    plot_tsp(xy, tour, ax)
    plt.show()

    ##################### write to new semantic map json file
    if input_type == 'json':
        write_optimal_json(tour, optgraph, outfile)

if __name__ == "__main__":

    # set defaults
    input_type = ""
    infile = ""
    modfile = '../outputs/tsp_34/tsp34_rollout_20210106T144819/epoch-2.pt'
    outfile = 'opt_path_out.json'

    for i in range(1, len(sys.argv)):
        if sys.argv[i] == '--smap':
            input_type = 'json'
            infile = sys.argv[i+1]
        elif sys.argv[i] == '--pkl':
            input_type = 'pickle'
            infile = sys.argv[i+1]
        elif sys.argv[i] == '--modfile':
            modfile = sys.argv[i+1]
        elif sys.argv[i] == '--outfile':
            outfile = sys.argv[i+1]           
    proc_map(input_type, infile, modfile)


