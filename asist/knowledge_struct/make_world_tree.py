#!/usr/bin/env python3

import csv
import json
import sys
from utils import *
from proc_knowledge_msgs import process_json_file

# takes main Saturn map as input & generates nested data struct for tracking player knowledge
# throughtout game play

# makes template for tracking elements of interest
def make_initial_struct(fname):
    jsonfile = open(fname, 'rt')
    kstruct = json.load(jsonfile)
    jsonfile.close()            
    make_regions(reg_dicts)
    # now add room/room_parts to regions
    template = open(template_name,'w')
    for (k,v) in kstruct.items():
        if k == 'locations':
            for k1 in kstruct['locations']:
                if k1['type'] == 'room_part':
                    k1['time_in'] = 0
                    k1['victims'] = []
                    k1['rubble'] = []
                    k1['freeze_panes'] = []
                    add_victims_to_room(k1, kstruct)
                    add_rubble_to_room(k1, kstruct)
                    add_fpanes_to_room(k1)
                    for rd in reg_dicts['reg_locations']: # determines which region this room_part is in
                        coords = k1['bounds']['coordinates']
                        reg = region(rd)
                        if reg.in_region(coords[0]['x'],coords[0]['z']): #room pt is in reg, add to reg dict
                            rd['rooms'].append(k1)
        elif k == 'connections': #add all portals no need to alter since will span multiple regs/rooms
            reg_dicts['connections'].append(v)

    # add victims to subrooms

    json.dump(reg_dicts,template,indent=True)
    template.close()

#------------------MAIN

#msgfile = '/home/skenny/usc/asist/data/opt_world_test/p2_hsr_t508_tm154_saturnb_vers4.meta'
msgfile = '/home/skenny/usc/asist/data/study_2/t000416.metadata'
template_name = './saturn_b.template'
#orig_map = './Saturn_1.0_sm_with_victimsA.json'
orig_map = './Saturn_1.0_sm_with_victimsB.json'
reg_dicts = {}
reg_dicts['reg_locations'] = []
reg_dicts['connections'] = []

make_initial_struct(orig_map)
# make sure new file can be loaded/read
print("....checking resulting file....")
jsonfile = open(template_name, 'rt')
kstruct = json.load(jsonfile)

# TODO: pull players from metadata header
players = ['E000324', 'E000322', 'E000323']
# default time_of_interest is whole run, 15min (900000ms)
time_of_interest = 900000
for player in players:
    new_kstrct = process_json_file(msgfile, kstruct, player, time_of_interest)
