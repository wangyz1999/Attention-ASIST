#!/usr/bin/env python3

# similar to route2viz but mixes in data from a real subj
# given optimized route json file, simulates a message buffer metadata file for that route
# attempts to add enough messages to match the # of msgs in the human subject data (for smoother movements)
# to be used with visualizer
# default hsr file here using a specific chunk of time (>4:20)
# TODO: need to make all default vars params/args

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
    def __init__(self, expid, trialid, mapname, nplayers, pcount, pname, tot_ms):
        self.playername = pname
        self.mapname = mapname
        self.expid = expid
        self.trialid = trialid
        self.elapsed_ms = 0
        #self.mission_timer = "15:00" # shouldn't really need to init, will reset based on ms
        #self.total_millisec = 900000
        #self.total_millisec = 256000 
        self.total_millisec = tot_ms
        self.mission_timer = self.convert_ms()
        #self.mission_timer = "4:16" # shouldn't really need to init, will reset based on ms
        self.players = []
        self.n_low_victims = 0
        self.n_hi_victims = 0
        self.nvictims = 0
        self.nrubble = 0
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
        self.init_role_msg = ''
        self.header_msg = ''
        self.init_obs_msg = ''
        self.victims_msg = {}
        #if pcount == 2: # first player so make headers, do for last instead will get sorted?
         #   self.make_victim_list() # TODO allow to specify which is hsr & do for THAT player
        if pcount == 0:
            self.make_header()
            

    # every time new message added, increment timer
    # do mission timer here too
    def add_new_message(self, msg, incr=False):
        self.convert_ms() #update mission timer to match curr elap ms MAKE THIS SINGLE CALL
        self.messages.append(msg)
        self.time_points.append(self.elapsed_ms) # now timepoints & messages should matchup
        #if incr:
         #   self.elapsed_ms += 250
        return msg
        
    # might have to also make mission start msg!
    def make_header(self):
        self.head_msg = {"data":{"experiment_mission":self.mapname, "elapsed_milliseconds":0},"msg":{"experiment_id":self.expid,"trial_id":self.trialid,"sub_type":"start"},"topic":"trial"}
        
    def init_role(self, role):
        self.convert_ms()
        self.init_role_msg = {"topic": "observations/events/player/role_selected", "msg": {"trial_id": self.trialid, "sub_type": "Event:RoleSelected", "experiment_id": self.expid}, "data": {"new_role": role, "elapsed_milliseconds": 0, "mission_timer": self.mission_timer, "prev_role": "None", "playername": self.playername}}     
        self.role = role
        
    def init_obs(self, x,z):
        self.init_obs_msg = {"topic": "observations/state","header": {"version": "1.1", "message_type": "observation"}, "data": {"elapsed_milliseconds": 0,"x": x, "mission_timer": self.mission_timer, "z": z, "playername": self.playername, "y": 60.0}, "msg": {"trial_id": self.trialid, "sub_type": "state", "experiment_id": self.expid}}

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

    def make_state_message_hsr(self, jmsg):
        data = jmsg['data']
        pname = data['playername']
        px = data['x']
        pz = data['z']
        ridx = 0
        msg = {"topic": "observations/state","header": {"version": "1.1", "message_type": "observation"}, "data": {"elapsed_milliseconds": self.elapsed_ms,"x": px, "mission_timer": data['mission_timer'], "z": pz, "playername": pname, "y": 60.0, "route": ridx}, "msg": {"trial_id": self.trialid, "sub_type": "state", "experiment_id": self.expid}}
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
        msg = {"topic": "observations/events/player/triage", "@version": "1", "header": {"version": "REGTRIAGE", "message_type": "event"}, "data": {"victim_x": vx, "triage_state": "SUCCESSFUL", "victim_z": vz, "elapsed_milliseconds": self.elapsed_ms, "comment": "victim to be triaged", "playername": pname, "color": label, "mission_timer": self.mission_timer, "route":ridx, "x":vx, "z":vz}, "msg": {"sub_type": "Event:Triage", "trial_id": self.trialid, "experiment_id": self.expid, "version": "1.0"}}
        #self.make_fake_state_message(jmsg) # don't need fakes since lining up with hsr??
        #print("adding new triage tp == "+str(self.elapsed_ms))
        self.add_new_message(msg,True)
    
    # TODO: hardcode victims here, do later cuz huge
    # TODO: get victi_miss_list from hsr file (or add missing victims to csv)
    def make_triage_message_hsr(self, jmsg):
        vic_list_dat = self.victims_msg['data']
        victim_list = vic_list_dat['mission_victim_list']
        data = jmsg['data']
        pname = data['playername']
        self.nvictims += 1
        vx = data['victim_x']
        vz = data['victim_z']
        vid = 'hsr'
        # get real label for victim
        for v in victim_list:
            if v['x'] == vx and v['z'] == vz:
                if v['block_type'] == 'block_victim_1':
                    color = 'g'
                else:
                    color = 'y'
                vid = 'v'+color+str(v['unique_id'])
        #if vid == 'hsr':
         #   print("NO MATCH AT: "+str(vx)+" "+str(vz)+" msg = "+str(jmsg))
        ridx = 0 # no real route for now

        msg = {"topic": "observations/events/player/triage", "@version": "1", "header": {"version": "1.1", "message_type": "event"}, "data": {"victim_x": vx, "triage_state": "SUCCESSFUL", "victim_z": vz, "elapsed_milliseconds": self.elapsed_ms, "comment": "victim to be triaged", "playername": pname, "color": vid, "mission_timer": self.mission_timer, "route":ridx, "x":vx, "z":vz}, "msg": {"sub_type": "Event:Triage", "trial_id": self.trialid, "experiment_id": self.expid, "version": "1.0"}}
        self.add_new_message(msg)

    def make_rubble_message(self, jmsg):
        self.nrubble += 1
        pname = self.playername
        rx = jmsg['loc_x']
        rz = jmsg['loc_z']
        ridx = jmsg['route_idx']
        pidx = str(jmsg['path_idx'])
        msg = {"topic": "observations/events/player/rubble_destroyed", "@version": "1", "header": {"version": "1.1", "message_type": "event"}, "data": {"rubble_x": rx, "rubble_z": rz, "elapsed_milliseconds": self.elapsed_ms, "playername": pname, "mission_timer": self.mission_timer, "route":ridx, "x":rx, "z":rz}, "msg": {"sub_type": "Event:RubbleDestroyed", "trial_id": self.trialid, "experiment_id": self.expid, "version": "1.0"}}
        #self.make_fake_state_message(jmsg)
        self.add_new_message(msg)

    def make_rubble_message_hsr(self, jmsg):
        data = jmsg['data']
        pname = data['playername']
        rx = data['rubble_x']
        rz = data['rubble_z']
        ridx = 0 # doing single route for now
        msg = {"topic": "observations/events/player/rubble_destroyed", "@version": "1", "header": {"version": "1.1", "message_type": "event"}, "data": {"rubble_x": rx, "rubble_z": rz, "elapsed_milliseconds": self.elapsed_ms, "playername": pname, "mission_timer": self.mission_timer, "route":ridx, "x":rx, "z":rz}, "msg": {"sub_type": "Event:RubbleDestroyed", "trial_id": self.trialid, "experiment_id": self.expid, "version": "1.0"}}
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
        msg = {"topic": "observations/state", "path_idx": pidx,"header": {"version": "FAKE STATE", "message_type": "observation"}, "data": {"elapsed_milliseconds": self.elapsed_ms,"x": x, "mission_timer": self.mission_timer, "route":ridx, "z": z, "playername": pname, "y": 60.0}, "msg": {"trial_id": self.trialid, "sub_type": "state", "experiment_id": self.expid, "version": "0.5"}}
        
        self.add_new_message(msg, pname)
        #self.add_filler_msg(msg) 

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
        #print("DOING INITIAL CONVERT: "+str(minutes)+" : "+str(seconds))
        self.mission_timer = ("%d:%d" % (minutes, seconds))

    # can maybe do actual node_id instead of block type?
    def add_victims_to_list(self):
        #fullvdict_data = self.messages[1]['data']
        fullvdict_data = self.victims_msg
        vlist = []
        for v in self.victims:
            vdict = {'x': v[0],'y':60, 'z':v[1],'victim_id':v[2]}
            vlist.append(vdict)
        self.victims = vlist
        fullvdict_data.update({"mission_victim_list":vlist})


# if a player has no message for a given time point, create one
def make_filler_msg(tp,pname, expid, trialid, tot_ms):
    x = 9999999
    z = 9999999
    mt = new_convert_ms(tp, tot_ms)
    fake_msg = {"topic": "observations/state", "header": {"version": "fake_msg", "message_type": "observation"}, "data": {"elapsed_milliseconds": tp,"x": x, "mission_timer": mt, "z": z, "playername": pname, "y": 60.0}, "msg": {"trial_id": trialid, "sub_type": "state", "experiment_id": expid, "version": "0.5"}}
    return fake_msg

def load_msgs(reader,player_nodes,pname):
    reader.elapsed_ms = 0
    reader.start_x = -2154.5 # probly don't need
    reader.start_z = 65
    # reader.mission_timer = "15:00"
    plist = player_nodes[pname]
    curr_role = plist[0]['role']
    if curr_role == 'medic':
        curr_role = 'Medical_Specialist'
    elif curr_role == 'engineer':
        curr_role = 'Hazardous_Material_Specialist'
    reader.init_role(curr_role)
    for pdict in plist:
        reader.elapsed_ms = round(pdict['time']*1000)
        #reader.time_points.append(round(pdict['time']*1000))
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

# TODO: handle rubble (even tho i think no rubble in this initial player)
def load_msgs_hsr(reader,player_msgs,pname, victim_list):
    reader.elapsed_ms = 0
    curr_role = 'Medical_Specialist' # MANUALLY ADD ROLE HERE!!
    reader.init_role(curr_role)
    for pdict in player_msgs:
        reader.elapsed_ms = pdict['data']['elapsed_milliseconds']
        #reader.time_points.append(reader.elapsed_ms)
        #print("hereee adding timepiont: "+str(reader.elapsed_ms))
        if 'triage_state' in pdict['data'].keys():
            if pdict['data']['triage_state'] == 'SUCCESSFUL':
                reader.make_triage_message_hsr(pdict)
        elif 'rubble_destroyed' in pdict['msg'].keys():
                reader.make_rubble_message_hsr(pdict)
        else:
            reader.make_state_message_hsr(pdict)
            # make filler msg for each of the other routes
    #reader.add_victims_to_list()

def load_victims_file(world):
    vlist = []
    if world == 'Saturn_A':
        fname = '/home/skenny/usc/asist/saturn/MapBlocks_SaturnA_Mission_1.csv'
    else:
        fname = '/home/skenny/usc/asist/saturn/MapBlocks_SaturnB_Mission_2.csv'
        #fname = 'data/new_saturn/saturn_b_victims_adjusted.csv'
    with open(fname) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            elif row[1] != 'block_victim_1' and row[1] != 'block_victim_2': # done with victims, skip rest (may need rubble later)
                break
            else:
                coords = row[0].split(' ')
                vtype = row[1]
                vx = coords[0]
                vz = coords[2]
                vlist.append((vx,vz,vtype))
            line_count += 1
    return vlist

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
    # handle if we go over the alloted time
    #if elapsed_ms > total_millisec:
     #   return ("%d:%d" % (0, 0))
    #print("CONVERTING: elapsed = "+str(elapsed_ms)+" total = "+str(total_millisec)+" returning = "+str(mission_timer))
    return mission_timer

#TODO: need to add victim list for saturn B, or allow world map file as input
#TODO need to add init state msgs after role?

if __name__ == "__main__":

    # set defaults
    exp = "exp00"
    trial = "trial00"
    playernames = ["player_1", "player_2"] # TODO: get from file?
    hsr_players = ['E000323'] # evenutually use in full pipeline??
    msg_readers = [] # make reader for each player
    all_player_msgs = []
    all_player_time_points = []
    total_milliseconds = 261000  #adding 8s to get to end of HSR player's run
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
        reader = msgreader(exp, trial, world, nplayers, pcount, pname, total_milliseconds)
        load_msgs(reader, all_nodes,pname)
        msg_readers.append(reader)
        pcount += 1

    # add reader for hsr player
    # TODO: add handling for door, chat etc? (just make them states or make them node_idx's ?
    # TODO: allow crop to happen here rather than separate file
    subjmsgs = '/home/skenny/usc/asist/sk_parser/E000323_cropped.meta' 
    smfile = open(subjmsgs, 'rt')
    msgcnt = 0
    reader = msgreader(exp, trial, world, nplayers, pcount, 'E000323', total_milliseconds)
    hsr_msgs = []
    for line in smfile.readlines(): # *might* wanna make door/swinging into state msgs?
        if line.find('{') > -1 and line.find('swinging') == -1 and line.find('Event:Door') == -1 and line.find('Event:Chat') == -1 and line.find('Event:location') == -1 and line.find('Event:ToolUsed') == -1 and line.find('scoreboard') == -1 and line.find('proximity') == -1:
            fullmsg = json.loads(line)
            t = fullmsg['data']['elapsed_milliseconds']
            #t2 = t - 643850
            t2 = t - 643852
            #print("hereee t2 = "+str(t2))
            fullmsg['data']['elapsed_milliseconds'] = t2
            hsr_msgs.append(fullmsg)
            msgcnt += 1
        elif line.find('mission_victim_list') > -1:
            reader.victims_msg = json.loads(line)
            
    smfile.close()
    all_victims = load_victims_file(world)
    load_msgs_hsr(reader, hsr_msgs, 'E000323', all_victims) # passing victim list from player 1 so can have right label
    msg_readers.append(reader)
        
    # put msgs from all player readers in single array then sort
    for r in msg_readers:
        for msg in r.messages:
            all_player_msgs.append(msg)
        for t in r.time_points:
            if t not in all_player_time_points:
                all_player_time_points.append(t)

    player_msgs_for_tp = []
    # for each t for each plr if they have msg use it otherwise generate one w/that time & use most recent loc--fake state
    # sort tp first????
    all_player_time_points = sorted(all_player_time_points, key = lambda i: (i))
    for tp in all_player_time_points:
        #print("IN ALL TP, TP = "+str(tp))
        players_at_time = []
        for r in msg_readers: # for each player
            if tp in r.time_points: # we have a msg for time
                tidx = r.time_points.index(tp)
                pmsg = r.messages[tidx]
            else:
                pmsg = make_filler_msg(tp,r.playername, exp, trial, total_milliseconds) # make msg for this player for this time
            players_at_time.append(pmsg)
        player_msgs_for_tp.append(players_at_time)

    # get headers & initial roles
    aaaall_msgs = []
    #aaaall_msgs.append(msg_readers[0].header_msg)
    tmp_head = {"data": {"experiment_mission": world, "elapsed_milliseconds": 0}, "msg": {"experiment_id": "exp00", "trial_id": "trial00", "sub_type": "start"}, "topic": "trial"}
    aaaall_msgs.append(tmp_head)
    aaaall_msgs.append(msg_readers[0].victims_msg)

    # skip first role selected msg
    # set start positions--TODO: automate this or make global vars??
    # need array of all coords
    # need some 'player_stats' object
    p1x = -2153
    p1z = 64
    p2x = -2217.5
    p2z = 32
    p3x = -2212.7
    p3z = 11.1
    # add init roles here manually
    rcnt = 0
    for rdr in msg_readers:
        if rcnt == 0:
            rdr.init_obs(p1x,p1z)
        elif rcnt == 1:
            rdr.init_obs(p2x,p2z)
        elif rcnt == 2:
            rdr.init_obs(p3x,p3z)

        aaaall_msgs.append(rdr.init_role_msg)
        aaaall_msgs.append(rdr.init_obs_msg)
        rcnt += 1
         

    for i in range(1,len(player_msgs_for_tp)-1):
        xesp = 0 #-1
        if (i % 2) == 0:
            xesp = 0 #1
        pmsgs = player_msgs_for_tp[i]
        pm1 = pmsgs[0] 
        if pm1['data']['x'] == 9999999: # found fake fill in w/most recent loc
            pm1['data']['x'] = p1x+xesp
            pm1['data']['z'] = p1z+xesp
        else:
            p1x = pm1['data']['x'] # reset to new loc
            p1z = pm1['data']['z'] # reset to new loc
            
        pm2 = pmsgs[1]
        if pm2['data']['x'] == 9999999:
            pm2['data']['x'] = p2x+xesp
            pm2['data']['z'] = p2z+xesp
        else:
            p2x = pm2['data']['x'] # reset to new loc
            p2z = pm2['data']['z'] # reset to new loc
        
        pm3 = pmsgs[2]
        if pm3['data']['x'] == 9999999:
            pm3['data']['x'] = p3x+xesp
            pm3['data']['z'] = p3z+xesp
        else:
            p3x = pm3['data']['x'] # reset to new loc
            p3z = pm3['data']['z'] # reset to new loc

        aaaall_msgs.append(pm1)
        aaaall_msgs.append(pm2)
        aaaall_msgs.append(pm3)

    #print("len player_msgs_for_tp = "+str(len(player_msgs_for_tp)))
    team_score = 330 # where it was at fist reup, do we want to start at 0??
    for tmpm in aaaall_msgs:
        
        
        m = str(tmpm).replace("'", "\"")
        m = m.replace('player_1','sim_324')
        m = m.replace('player_2','sim_322')
        print(str(m))
        if m.find('SUCCESSFUL') > -1: # if triage add score message
            data = tmpm['data']
            ems = data['elapsed_milliseconds']
            colorid = data['color']
            if colorid.find('y'):
                team_score += 50
            else:
                team_score += 10
            mtimer = data['mission_timer']
            score_m = {"topic":"observations/events/scoreboard","header":{"version":"0.6","message_type":"observation"},"data":{"scoreboard":{"TeamScore":team_score},"mission_timer":mtimer,"elapsed_milliseconds":ems},"@version":"1","msg":{"source":"simulator","experiment_id":"e1c4e9b5-2bf9-44fd-9779-c99f0498892a","version":"0.5","trial_id":trial,"sub_type":"Event:Scoreboard"}}
            score_m = str(score_m).replace("'", "\"")
            print(score_m)
        
          
#TODO::: check player_2 getting all timpoints, seems to stop around 1:52 (compare with viz of just opt?)
# p1 & hsr look relatively normal
# p2's last move is at ~2m 20sec, p1's is at 4:20....so probly p2 is fine time-wise but path still odd...COMPARE WITH ORIG OPT NO HSR
# WHY DOES PLAYER 2 DO A JUMP AT 3:54...then stays there for a bit
# add a scoreboard msg during final print out, tally triages along way & add one in each time:
# ? for volkan want score for 2nd half or start at existing score? starts at: 


