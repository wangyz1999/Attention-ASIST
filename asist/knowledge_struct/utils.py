#!/usr/bin/env python3

import csv
import json
import sys

# only adding room PARTS to regions...coords for parent rooms not in orig saturn file
# can map back to parent room during processing if needed?
class region(object):
    def __init__(self, rdict):
        self.name = rdict['id']
        x0 = rdict['x0']
        x1 = rdict['x1']
        z0 = rdict['z0']
        z1 = rdict['z1']
        self.x0= min(x0, x1)
        self.x1= max(x0, x1)
        self.z0= min(z0, z1)
        self.z1= max(z0, z1)
        self.rdict = rdict

    def in_region(self, _x, _z, epsilon = 0):
        return (self.x0 - epsilon <= _x) and (_x <= self.x1 + epsilon) and (self.z0 - epsilon <= _z) and (_z <= self.z1 + epsilon)
    
    def __repr__(self):
        return '%s %.0f %.0f %.0f %.0f' % (self.name, self.x0, self.x1, self.z0, self.z1)

# using same as region....should combine these???
class room(object):
    def __init__(self, rdict):
        self.name = rdict['id']
        x0 = rdict['bounds']['coordinates'][0]['x']
        x1 = rdict['bounds']['coordinates'][1]['x']
        z0 = rdict['bounds']['coordinates'][0]['z']
        z1 = rdict['bounds']['coordinates'][1]['z']
        self.x0= min(x0, x1)
        self.x1= max(x0, x1)
        self.z0= min(z0, z1)
        self.z1= max(z0, z1)
        self.rdict = rdict

    def in_room(self, _x, _z, epsilon = 0):
        return (self.x0 - epsilon <= _x) and (_x <= self.x1 + epsilon) and (self.z0 - epsilon <= _z) and (_z <= self.z1 + epsilon)
    
    def __repr__(self):
        return '%s %.0f %.0f %.0f %.0f' % (self.name, self.x0, self.x1, self.z0, self.z1) 

def add_victims_to_room(rdict,mstruct):
    vroom = room(rdict)
    for (k,v) in mstruct.items():
        if k == 'objects':
            for obj in v:
                if obj['type'] == 'green_victim' or obj['type'] == 'yellow_victim':
                    vx = obj['bounds']['coordinates'][0]['x']
                    vz = obj['bounds']['coordinates'][0]['z']
                    obj['seen'] = False
                    if vroom.in_room(vx,vz):
                        rdict['victims'].append(obj)
                        break

def add_rubble_to_room(rdict,mstruct):
    rroom = room(rdict)
    for (k,v) in mstruct.items():
        if k == 'objects':
            for obj in v:
                if obj['type'] == 'rubble':
                    rx = int(obj['bounds']['coordinates'][0]['x']) # are strings due to bug in my script that generated main Saturn file
                    rz = int(obj['bounds']['coordinates'][0]['z'])
                    obj['seen'] = False
                    if rroom.in_room(rx,rz):
                        rdict['rubble'].append(obj)
                        break

# assumes saturnB
# def add_fpanes_to_room(rdict,mstruct):
# NOTE: room parts overlap, pane may show twice
def add_fpanes_to_room(rdict):
    freeze_panes_dicts = [{"x":-2195,"y":59,"z":3},{"x":-2162,"y":59,"z":8},{"x":-2130,"y":59,"z":15},{"x":-2093,"y":59,"z":42},{"x":-2181,"y":59,"z":45}]
    froom = room(rdict)
    #print(list(fpdict.keys())[list(fpdict.values()).index('freeze_panes')])
    for fp in freeze_panes_dicts:
        fx = fp['x']
        fz = fp['z']
        frz_obj = {}
        if froom.in_room(fx,fz):
            frz_obj['x'] = fx
            frz_obj['z'] = fz
            frz_obj['seen'] = True
            rdict['freeze_panes'].append(frz_obj)

# may need to add 'name' since all other objs have one?
# only need to calc time for subrooms (can total for others based on that)
def make_regions(reg_dicts):
    reg_dict1 = {"id": "reg_1",
                 "type": "region",
                 "x0": -2225,
                 "z0": -11,
                 "x1": -2175,
                 "z1": 26,
                 "time_in": 0,
                 "rooms":[]
                 }
    reg_dict2 = {"id": "reg_2",
                 "type": "region",
                 "x0": -2225,
                 "z0": 27,
                 "x1": -2175,
                 "z1": 61,
                 "time_in": 0,
                 "rooms":[]
                 }
    reg_dict3 = {"id": "reg_3",
                 "type": "region",
                 "x0": -2174,
                 "z0": -11,
                 "x1": -2135,
                 "z1": 26,
                 "time_in": 0,
                 "rooms":[]
                 }
    reg_dict4 = {"id": "reg_4",
                 "type": "region",
                 "x0": -2174,
                 "z0": 27,
                 "x1": -2135,
                 "z1": 61,
                 "time_in": 0,
                 "rooms":[]
                 }
    reg_dict5 = {"id": "reg_5",
                 "type": "region",
                 "x0": -2134,
                 "z0": -11,
                 "x1": -2087,
                 "z1": 26,
                 "time_in": 0,
                 "rooms":[]
                 }
    reg_dict6 = {"id": "reg_6",
                 "type": "region",
                 "x0": -2134,
                 "z0": 27,
                 "x1": -2087,
                 "z1": 61,
                 "time_in": 0,
                 "rooms":[]
                 }

    reg_dicts['reg_locations'].append(reg_dict1)
    reg_dicts['reg_locations'].append(reg_dict2)
    reg_dicts['reg_locations'].append(reg_dict3)
    reg_dicts['reg_locations'].append(reg_dict4)
    reg_dicts['reg_locations'].append(reg_dict5)
    reg_dicts['reg_locations'].append(reg_dict6)
    


