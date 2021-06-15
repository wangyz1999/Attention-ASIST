#!/usr/bin/env python

# check messages from bus, when triage found generate message indicating next victim that should be rescued
# TODO: start node shouldn't be ew_1, also when getting triage, remove that victim right away then determine start room/coords

import os
from optimization_interface import *
import numpy as np
import torch
import graph
import sys
import csv
import json

# reads triage from file (eventually read directly from bus)
# using saturnA (50 victims) to match the csv file, but using 55 model
msgfile = open('data/csv/test_messagebus_saturnA.csv')
modfile = '../outputs/tsp_55/tsp55_20210312T130700/epoch-12.pt'
smap = 'data/json/Saturn_1.0_sm_with_victimsA.json'
#smap = '../../saturn/Saturn_1.0_sm_with_victimsA.json'

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

# load room list
rooms = []
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

# find first triage from message bus output & get coords
vx = 0
vz = 0
for line in msgfile.readlines():
    if line.find('Event:Triage') > -1:
        fcnt = 0 
        fields = line.split(',')
        vx = int(fields[98].replace('"',''))
        vz = int(fields[100]) # will switch to json proc on real bus
        break
msgfile.close()

# load model & set defaults

model, _ = load_model(modfile)
model.eval()
excludes = []
start_node = 'ew_1'
nvictims = 55
i = 0
vid = ''
triage_room = ''

###### main loop

for i in range(0,nvictims):

    # initial graph will contain all victims
    orig_graph, optgraph, tour, xy = get_graph_coords_json(smap, model, start_node, nvictims, excludes)
    print("path length: {:.2f}".format(get_length(xy, tour)))
    print("tour len: "+str(len(tour)))
    # get id & room for start victim found triaged in message bus
    if i == 0:
        rescuedidx = 0
        for n in optgraph.nodes_list:
            x,z = n.loc
            if x == vx and z == vz:
                rescuedid = optgraph.victim_list[rescuedidx].id
                vid = rescuedid
                print("FOUND VICTIM "+str(rescuedid))
                found = True
                break
            rescuedidx += 1

        for r in rooms:
            if r.in_room(vx,vz):
                triage_room = r.name
                break
        excludes.append(vid)
         
    # make curr triage location the start node
    start_node = triage_room
        
    # triage the victim: exclude from next iteration


    # get next victim from optimal path
    vx = optgraph.victim_list[tour[0]].loc[0]
    vz = optgraph.victim_list[tour[0]].loc[1]
    vid = optgraph.victim_list[tour[0]].id

    # find room for new victim where agent should go next
    triage_room = ''
    for r in rooms:
        if r.in_room(vx,vz):
            triage_room = r.name
            break
    print("NEXT victim to be triaged: "+str(optgraph.victim_list[tour[0]].id)+" in room: "+triage_room)
    if i > 0:
        excludes.append(vid)
    print("number of victims triaged: "+str(len(excludes)))
