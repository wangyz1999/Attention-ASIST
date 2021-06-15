#!/usr/bin/env python

# check messages from bus, when triage found generate message indicating next victim that should be rescued

import os
import numpy as np
import torch
import graph
import sys
import csv
import json
from utils import load_model
from interface_utils import *

# process triage from message bus
modfile = '/app/Attention-ASIST/asist/epoch-12.pt'
smap = '/app/Attention-ASIST/asist/Saturn_1.0_sm_with_victimsA.json'
#smap = 'data/json/Saturn_1.0_sm_with_victimsB.json'
rooms = []

class room(object):
    def __init__(self, name, x0, z0, x1, z1):
        self.name = name
        # coords in new file using top rt corner, bottom left (unlike previous) so need to adjust for calculating range
        if x0 < x1:
            self.xrange = range(x0,x1+1)
        else:
            self.xrange = range(x1+1,x0)
        if z0 < z1:
            self.zrange = range(z0,z1+1)
        else:
            self.zrange = range(z1+1,z0)

    def in_room(self, _x, _z):
        if _x in self.xrange and _z in self.zrange:
            return True   
        else:
            return False


def load_rooms():
    rfile = open(smap, 'rt')
    rdict = json.load(rfile)
    rloc = rdict['locations']
    coords = ''
    rid = ''
    x0 = 0
    z0 = 0
    x1 = 0
    z1 = 0
    for r in rloc:
        if 'bounds' in r.keys():
            try:
                coords = (r['bounds']['coordinates'])
            except:
                coords = ''
            if coords != '':
                rid = r['id']
                x0 = coords[0]['x']
                z0 = coords[0]['z']
                x1 = coords[1]['x']
                z1 = coords[1]['z']
                rm = room(rid, x0, z0, x1, z1)
                rooms.append(rm)
    rfile.close()



# ends with removal so can be called over and over in agnet loop? excludes will need to be stored in wrapper
def get_next_victim_load(vx=0, vz=0, excludes=[]):
    load_rooms()
    model, _ = load_model(modfile)
    model.eval()

    nvictims = 55
    triage_room = ''
    vid = ''
    # find victim room
    for r in rooms:
        if r.in_room(vx,vz):
            triage_room = r.name
            break
    # make curr triage location the start node
    start_node = triage_room
    # run opt to get graph info and determine optimal path
    orig_graph, optgraph, tour, xy = get_graph_coords_json(smap, model, start_node, nvictims, excludes)
    # get id for victim so can add to excludes
    rescuedidx = 0
    for n in optgraph.nodes_list:
        x,z = n.loc
        if x == vx and z == vz:
            rescuedid = optgraph.victim_list[rescuedidx].id
            vid = rescuedid
            # got victim id
            found = True
            break
        rescuedidx += 1
    excludes.append(vid)
    
    # get next victim from optimal path---should it be tour[1], tour[0] is current room
    vx = optgraph.victim_list[tour[1]].loc[0]
    vz = optgraph.victim_list[tour[1]].loc[1]
    vid = optgraph.victim_list[tour[1]].id

    # find room for new victim where agent should go next
    next_room = ''
    for r in rooms:
        if r.in_room(vx,vz):
            next_room = r.name
            break
    return next_room, vx, vz, excludes

def get_remaining_victims_load(vx=0, vz=0, excludes=[]):
    load_rooms()
    model, _ = load_model(modfile)
    model.eval()

    nvictims = 55
    triage_room = ''
    vid = ''
    # find victim room
    for r in rooms:
        if r.in_room(vx,vz):
            triage_room = r.name
            break
    # make curr triage location the start node
    start_node = triage_room
    # run opt to get graph info and determine optimal path
    orig_graph, optgraph, tour, xy = get_graph_coords_json(smap, model, start_node, nvictims, excludes)
    # get id for victim so can add to excludes
    rescuedidx = 0
    for n in optgraph.nodes_list:
        x,z = n.loc
        if x == vx and z == vz:
            rescuedid = optgraph.victim_list[rescuedidx].id
            vid = rescuedid
            # got victim id
            found = True
            break
        rescuedidx += 1
    excludes.append(vid)
    
    # get next victim from optimal path---should it be tour[1], tour[0] is current room
    vx = optgraph.victim_list[tour[1]].loc[0]
    vz = optgraph.victim_list[tour[1]].loc[1]
    vid = optgraph.victim_list[tour[1]].id

    vxs = []
    vzs = []
    nvictims = len(optgraph.victim_list)
    for i in range(1,nvictims):
        vxs.append(optgraph.victim_list[tour[i]].loc[0])
        vzs.append(optgraph.victim_list[tour[i]].loc[1])                   
    
    # find room for new victim where agent should go next
    next_room = ''
    for r in rooms:
        if r.in_room(vx,vz):
            next_room = r.name
            break
    return next_room, vxs, vzs, excludes # wont need room leaving for now

# test
if __name__ == '__main__':
    rm, x, z, excl = get_next_victim_load(-2136, -1) 
    print("next victim is at "+str(x)+","+str(z)+" in room "+str(rm))
    print("excludes: "+str(excl))
