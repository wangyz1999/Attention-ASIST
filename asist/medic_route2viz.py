#!/usr/bin/env python3

# given optimized route json file, simulates a message buffer metadata file for that route to be used with visualizer

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
import copy

class msgreader(object):
    def __init__(self, expid, trialid, mapname, nplayers, pcount, pname):
        self.playername = pname
        self.mapname = mapname
        self.expid = expid
        self.trialid = trialid
        self.elapsed_ms = 0
        self.mission_timer = "15:00" # shouldn't really need to init, will reset based on ms
        self.total_millisec = 900000
        self.players = []
        self.n_low_victims = 0
        self.n_hi_victims = 0
        self.nvictims = 0
        self.nroutes = nroutes
        self.routes = [] #batches messages by route?
        self.victims = []
        self.start_x = -2154.5
        self.start_z = 65
        self.messages = []
        self.messages_adjusted = []
        self.time_points = []
        self.missing_time_points = []
        self.role = 'None'
        if pcount == 0: # first player so make headers
            self.make_header()
            self.make_victim_list()

    # every time new message added, increment timer
    # do mission timer here too
    def add_new_message(self, msg, route=99):
        #print("CURRENT MS = "+str(self.elapsed_ms)+" CURRENT MTIMER "+str(self.mission_timer))
        self.convert_ms() #update mission timer to match curr elap ms MAKE THIS SINGLE CALL
        self.messages.append(msg)
        #print("msg added CURRENT MS = "+str(self.elapsed_ms)+" CURRENT MTIMER "+str(self.mission_timer))
        #self.elapsed_ms += 100
        return msg
        
    def add_filler_msg(self,msg):
        for p in self.players:
            if p.name != self.playername: #don't add for curr route
                xesp = 0
                mcnt = len(self.messages) 
                if (mcnt % 2) == 0:
                    xesp = 1
                fake_msg = copy.deepcopy(msg)
                fake_msg['data']['x'] = p.curr_x+xesp
                fake_msg['data']['z'] = p.curr_z+xesp
                fake_msg['data']['playername'] = p.name
                fake_msg['data']['route'] = 0
                fake_msg['header']['version'] = 'FAKE'
                self.messages.append(fake_msg)

    # might have to also make mission start msg!
    def make_header(self):
        head = {"data":{"experiment_mission":self.mapname, "elapsed_milliseconds":0},"msg":{"experiment_id":self.expid,"trial_id":self.trialid,"sub_type":"start"},"topic":"trial"}
        self.add_new_message(head) # maybe add ddirectly if messes up time/need sep start msg

    # make empty victim template here, will update with victims once all messages read
    def make_victim_list(self):
        victim_list = {"header":{"version":"sim","message_type":"groundtruth"},"data":{"elapsed_milliseconds": 0}}
        self.add_new_message(victim_list)
        
    def init_role(self, role):
        rmsg = {"topic": "observations/events/player/role_selected", "msg": {"trial_id": self.trialid, "sub_type": "Event:RoleSelected", "experiment_id": self.expid}, "data": {"new_role": role, "elapsed_milliseconds": 0, "mission_timer": self.mission_timer, "prev_role": "None", "playername": self.playername}}
        self.role = role
        self.add_new_message(rmsg)
        
    # any id with | is a portal (maybe doesn't matter), use to gen any portal or room msgs
    # probly don't need y in msg
    def make_state_message(self, jmsg):
        pname = self.playername
        px = jmsg['loc_x']
        pz = jmsg['loc_z']
        pidx = jmsg['path_idx']
        ridx = str(jmsg['route_idx'])
        msg = {"topic": "observations/state", "path_idx": pidx,"header": {"version": "1.1", "message_type": "observation"}, "data": {"elapsed_milliseconds": self.elapsed_ms,"x": px, "mission_timer": self.mission_timer, "z": pz, "playername": pname, "y": 60.0, "route": ridx}, "msg": {"trial_id": self.trialid, "sub_type": "state", "experiment_id": self.expid}}
        self.add_new_message(msg) # TODO: nix player since makeing new reader for each player

    def make_triage_message(self, jmsg):
        pname = self.playername
        self.nvictims += 1
        vx = jmsg['loc_x']
        vz = jmsg['loc_z']
        vtype = jmsg['node_id']
        self.victims.append([vx,vz,vtype])
        vname = jmsg['node_id']
        if vname.find('g') > -1:
            self.n_low_victims += 1
            vid = "VL"+str(self.n_low_victims)+"_"+str(self.nvictims)
        else:
            self.n_hi_victims += 1
            vid = "VH"+str(self.n_hi_victims)+"_"+str(self.nvictims)
        ridx = jmsg['route_idx']
        pidx = str(jmsg['path_idx'])
        #label = str(vid)
        label = jmsg['node_id'] # going back to no route or hi/low for now
        msg = {"topic": "observations/events/player/triage", "@version": "1", "header": {"version": "1.1", "message_type": "event"}, "data": {"victim_x": vx, "triage_state": "SUCCESSFUL", "victim_z": vz, "elapsed_milliseconds": self.elapsed_ms, "comment": "victim to be triaged", "playername": pname, "color": label, "mission_timer": self.mission_timer, "route":ridx}, "msg": {"sub_type": "Event:Triage", "trial_id": self.trialid, "experiment_id": self.expid, "version": "1.0"}}
        #self.make_fake_state_message(jmsg)
        self.add_new_message(msg)

    def make_rubble_message(self, jmsg):
        pname = self.playername
        rx = jmsg['loc_x']
        rz = jmsg['loc_z']
        ridx = jmsg['route_idx']
        pidx = str(jmsg['path_idx'])
        msg = {"topic": "observations/events/player/rubble_destroyed", "@version": "1", "header": {"version": "1.1", "message_type": "event"}, "data": {"rubble_x": rx, "rubble_z": rz, "elapsed_milliseconds": self.elapsed_ms, "playername": pname, "mission_timer": self.mission_timer, "route":ridx}, "msg": {"sub_type": "Event:RubbleDestroyed", "trial_id": self.trialid, "experiment_id": self.expid, "version": "1.0"}}
        #self.make_fake_state_message(jmsg)
        self.add_new_message(msg)

    def make_role_message(self, jmsg, curr_role, new_role):
        pname = self.playername
        ridx = jmsg['route_idx']
        pidx = str(jmsg['path_idx'])
        msg = {"topic": "observations/events/player/role_selected", "@version": "1", "header": {"version": "1.1", "message_type": "event"}, "data": {"new_role": new_role, "prev_role": curr_role, "elapsed_milliseconds": self.elapsed_ms, "playername": pname, "mission_timer": self.mission_timer, "route":ridx}, "msg": {"sub_type": "Event:RoleSelected", "trial_id": self.trialid, "experiment_id": self.expid, "version": "1.0"}}
        self.add_new_message(msg)

    # need to make additional state msg at triage location--SAME TIME as prev state, same x/z as triage 
    def make_fake_state_message(self,triagemsg):
        pname = self.playername
        x = triagemsg['loc_x']
        z = triagemsg['loc_z']
        ridx = triagemsg['route_idx']
        pidx = triagemsg['path_idx'] #don't need?
        msg = {"topic": "observations/state", "path_idx": pidx,"header": {"version": "FILLER", "message_type": "observation"}, "data": {"elapsed_milliseconds": self.elapsed_ms,"x": x, "mission_timer": self.mission_timer, "route":ridx, "z": z, "playername": pname, "y": 60.0}, "msg": {"trial_id": self.trialid, "sub_type": "state", "experiment_id": self.expid, "version": "0.5"}}
        self.add_new_message(msg, pname)
        self.add_filler_msg(msg) 

    def convert_ms(self):
        seconds=(self.elapsed_ms/1000)%60
        seconds = int(seconds)
        minutes=(self.elapsed_ms/(1000*60))%60
        minutes = int(minutes)
        hours=(self.elapsed_ms/(1000*60*60))%24 # won't need this
        new_mt_millisec = self.total_millisec - self.elapsed_ms
        seconds=(new_mt_millisec/1000)%60
        seconds = int(seconds)
        minutes=(new_mt_millisec/(1000*60))%60
        minutes = int(minutes)
        hours=(new_mt_millisec/(1000*60*60))%24 # won't need this
        self.mission_timer = ("%d:%d" % (minutes, seconds))

    # can maybe do actual node_id instead of block type?
    def add_victims_to_list(self):
        fullvdict_data = self.messages[1]['data']
        vlist = []
        for v in self.victims:
            vdict = {'x': v[0],'y':60, 'z':v[1],'victim_id':v[2]}
            vlist.append(vdict)
        fullvdict_data.update({"mission_victim_list":vlist})

# TODO: tell yunzshe to use right roles!
def load_msgs(reader,player_nodes,pname):
    #reader.elapsed_ms = pcnt #so will be slight diff---maybe don't need, just reset to 0?
    reader.elapsed_ms = 0
    reader.start_x = -2154.5
    reader.start_z = 65
    reader.mission_timer = "15:00"
    plist = player_nodes[pname]
    curr_role = plist[0]['role']
    if curr_role == 'medic':
        curr_role = 'Medical_Specialist'
    elif curr_role == 'engineer':
        curr_role = 'Hazardous_Material_Specialist'
    reader.init_role(curr_role)
    for pdict in plist:
        reader.elapsed_ms = round(pdict['time']*1000)
        reader.convert_ms()
        reader.time_points.append(round(pdict['time']*1000))
        if pdict['role'] == 'medic':
            pdict['role'] = 'Medical_Specialist'
        elif pdict['role'] == 'engineer':
            pdict['role'] = 'Hazardous_Material_Specialist'
        if pdict['role'] != curr_role:
            reader.make_role_message(pdict, curr_role, pdict['role'])
            curr_role = pdict['role']
        if pdict['is_action']:
            if pdict['role'] == 'Medical_Specialist':
                reader.make_triage_message(pdict)
            else:
                reader.make_rubble_message(pdict)
        else:
            reader.make_state_message(pdict)
            # make filler msg for each of the other routes
    reader.add_victims_to_list()
    

def new_convert_ms(elapsed_ms, total_millisec):
    # subtract elapsed_millisec from total millisec (15:00) & convert back to minutes
    seconds=(elapsed_ms/1000)%60
    seconds = int(seconds)
    minutes=(elapsed_ms/(1000*60))%60
    minutes = int(minutes)
    hours=(elapsed_ms/(1000*60*60))%24 # won't need this
    new_mt_millisec = total_millisec - elapsed_ms
    seconds=(new_mt_millisec/1000)%60
    seconds = int(seconds)
    minutes=(new_mt_millisec/(1000*60))%60
    minutes = int(minutes)
    hours=(new_mt_millisec/(1000*60*60))%24 # won't need this
    mission_timer = ("%d:%d" % (minutes, seconds))
    return mission_timer

#TODO: need to add victim list for saturn B, or allow world map file as input

if __name__ == "__main__":

    # set defaults
    exp = "exp00"
    trial = "trial00"
    playernames = ["player_1", "player_2", "player_3"] # TODO: get from file?
    msg_readers = [] # make reader for each player
    all_player_msgs = []
    all_player_time_points = []
    alltime = [] # will have msg for each player for each time point
    world = "Saturn_B"
    smap = ""
    nroutes = 1 # skip routes for now, will iterative call it once players done
    nplayers = 1
    for i in range(1, len(sys.argv)):
        if sys.argv[i] == '--optfile':
            smap = sys.argv[i+1]
        elif sys.argv[i] == '--nroutes':
            nroutes = int(sys.argv[i+1])
        elif sys.argv[i] == '--help':
            print("USAGE:")
            print("--optfile: json file containing optimized route")
            print("--nroutes: number of routes in file")
            sys.exit()
    if smap == "": # user did not enter mapfile
        print("exiting, no route file specified! --optfile <filename>")
        sys.exit()
    # reader for each player
    mfile = open(smap, 'rt')
    mdict = json.load(mfile)
    all_nodes = mdict['data']

    pcount = 0
    for pname in playernames:
        reader = msgreader(exp, trial, world, nplayers, pcount, pname)
        load_msgs(reader, all_nodes,pname)
        msg_readers.append(reader)
        pcount += 1

    # put msgs from all player readers in single array then sort
    for r in msg_readers:
        for msg in r.messages:
            all_player_msgs.append(msg)
        for t in r.time_points:
            all_player_time_points.append(t)

    # for each t for each plr if they have msg use it otherwise generate one w/that time & use most recent loc--fake state
    for tp in all_player_time_points:
        for r in msg_readers:
            has_match = False
            for t in r.time_points:
                if t == tp: # has match no need to contine
                    has_match = True
                    break
            if not has_match and tp not in r.missing_time_points:
                r.missing_time_points.append(tp)

    # now for each reader for each time point see if there are any missing ones that fall btwn it & the one after it
    for r in msg_readers:
        for i in range(len(r.messages)-1):
            # add real msg to new array--WILL NEED TO AT LAST AT END
            curr_msg = r.messages[i]
            next_msg = r.messages[i+1]
            r.messages_adjusted.append(curr_msg) #maybe skip & just add to alltimes?? will need to fix for headers
            alltime.append(curr_msg)
            real_t1 = curr_msg['data']['elapsed_milliseconds']
            real_t2 = next_msg['data']['elapsed_milliseconds']
            for mstp in r.missing_time_points:
                if mstp > real_t1 and mstp < real_t2: # goes btwn these messages so should mimmic the 1st
                    fake_msg = copy.deepcopy(curr_msg)
                    fake_msg['data']['elapsed_milliseconds'] = mstp
                    fake_msg['data']['mission_timer'] = new_convert_ms(mstp,900000)
                    fake_msg['header']['version'] = 'FAKEE'
                    if 'triage_state' not in fake_msg['data'].keys():
                        alltime.append(fake_msg)

    alltime_msgs = sorted(alltime, key = lambda i: (i['data']['elapsed_milliseconds']))

    for msg in alltime_msgs:
        #if 'mission_timer' in msg['data']:
        #   print(" ms "+str(msg['data']['mission_timer']))
        m = str(msg).replace("'", "\"")
        m = m.replace('player_1','sim_324')
        m = m.replace('player_2','sim_322')
        m = m.replace('player_3','sim_323')

        print(m)
