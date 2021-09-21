#!/usr/bin/env python3

# read necessary portions of message buffer for psychsim
# can call on entire file or request latest event
# adapted from original message_reader to process engineered subjects data/AI/non-human which don't have triage event

from builtins import range
import os
import sys
import time
import functools
import json
import math
import csv
import time
print = functools.partial(print, flush=True)
from utils import *

# find reg & increment time
def find_region(x,z,kstruct):
    for (k,v) in kstruct.items():
        if k == 'reg_locations':
            for r in v:
                reg = region(r)
                # check if in region
                inreg = reg.in_region(x,z)
                if inreg:
                    reg.rdict['time_in'] += 1
                    return reg.rdict
    return "NONE"

def find_room(x,z,region):
    for rdict in region['rooms']:
        r = room(rdict)
        if r.in_room(x,z):
            rdict['time_in'] += 1
            return rdict
    return "NONE"

                

def process_message(jmsg, kstruct, toi=0):
    # first deterimine which region so can narrow down room search? and in intermim increment 'time_in' for regions (to make sure update to struct working)
    # what room are we in, need to increment time
    data = jmsg[u'data']
    x = 0
    z = 0
 #  determine region & room
    if 'x' in data.keys():
        x = data['x']
        z = data['z']
        reg = find_region(x,z,kstruct) 
        if reg != "NONE":
            rm = find_room(x,z,reg)
            # determine victims in room & mark as seen (will update to use proximity once decide how close)
            if rm != "NONE":
                for v in rm['victims']:
                    v['seen'] = True
                # determine rubble in room & mark as seen (will update to use proximity once decide how close)
                for rb in rm['rubble']:
                    rb['seen'] = True
                for fp in rm['freeze_panes']:
                    fp['seen'] = True

# --- used for main

def process_json_file(fname, kstruct, player, toi):
        jsonfile = open(fname, 'rt')
        #jsonMsgs = [json.loads(line) for line in jsonfile.readlines()]
        #jsonfile.close()            
        #for jmsg in jsonMsgs:
        nlines = 0
        for line in jsonfile.readlines():
            if line.find('not initialized') == -1 and line.find('data') > -1 and line.find('mission_timer') > -1 and line.find(player) > -1:
                jmsg = json.loads(line)
                data = jmsg[u'data']
                elap_ms = data['elapsed_milliseconds']
                if elap_ms >= toi: # have all data we care
                    break
                else:
                    process_message(jmsg, kstruct) # add to msglist will then iter list to calc times/items viewed, etc?
                # if at time of interest: break
        #dump new struct 
        template = open(player,'w')
        print("\nstats for "+player+" ::")
        victims = check_victims_seen(kstruct)
        print("victims seen:: "+str(victims))
        rubble = check_rubble_seen(kstruct)
        print("rubble seen:: "+str(rubble))
        json.dump(kstruct,template,indent=True)
