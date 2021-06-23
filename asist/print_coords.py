#!/usr/bin/env python

"""
Simple ASIST Agent

Author: Roger Carff
email:rcarff@ihmc.us
"""
 
import os
import json
from helpers.ASISTAgentHelper import ASISTAgentHelper
import time

import single_bus_proc

__author__ = 'rcarff'

trial_infos = {}

# This is the function which is called when a message is received for a to
# topic which this Agent is subscribed to.
def on_message(message):
    global extra_info

    topic = message['topic']
    header = message['message']['header'] if 'header' in message['message'] else {}
    msg = message['message']['msg'] if 'msg' in message['message'] else {}
    data = message['message']['data'] if 'data' in message['message'] else {}

    print("Received a message on the topic: ", topic)

    # Now handle the message based on the topic.  Refer to Message Specs for the contents of header, msg, and data
    if topic == 'trial':
        if msg['sub_type'] == 'start':
            print("STARTING................")
            # handle the start of a trial!!
            trial_info = data
            trial_info['experiment_id'] = msg['experiment_id']
            trial_info['trial_id'] = msg['trial_id']
            trial_info['replay_id'] = msg['replay_id'] if 'replay_id' in msg.keys() else None
            trial_info['replay_root_id'] = msg['replay_root_id'] if 'replay_root_id' in msg.keys() else None

            trial_key = trial_info['trial_id']
            if 'replay_id' in trial_info and not trial_info['replay_id'] is None:
                trial_key = trial_key + ":" + trial_info['replay_id']
            trial_infos[trial_key] = trial_info

            print(" - Trial Started with Mission set to: " + trial_info['experiment_mission'])
    elif topic == 'observations/events/player/triage':
        vx = int(data['victim_x'])
        vz = int(data['victim_z'])
        
        print("VICTIM FOUND....calling NEW opt code..."+str(vx)+", "+str(vz))
        next_room, newxs, newzs,_ = single_bus_proc.get_remaining_victims_load(vx,vz)
        print("NEXT VICTIMS SHOULD BE: "+str(newxs)+", "+str(newzs)+" in room "+str(next_room))

        trial_info = data
        trial_info['experiment_id'] = msg['experiment_id']
        trial_info['trial_id'] = msg['trial_id']
        trial_info['replay_root_id'] = None
        trial_info['replay_id'] = None
        
        comment = "victim to be triaged"
        # build up the message's data and publish it
        msg_data = {
            "playername": data['playername'],
            "comment": comment,
            "victim_x": newxs,
            "victim_z": newzs,
            "color": "NEXT",
            "triage_state": "SUCCESSFUL", # for use with viz
            "elapsed_milliseconds": data['elapsed_milliseconds'],
            "mission_timer": data['mission_timer']
            #"comment": "coords "+str(vx)+", "+str(vz)
        }
        #print("msg data = "+str(msg_data))
        #print("trial info = "+str(trial_info))
        #helper.subscribe('observations/events/player/tool_used')            
        helper.send_msg_with_timestamp("agent/opt_triage",
                                       "event",
                                       "OPTPATH",
                                       "1.0",
                                       trial_info,
                                       msg['timestamp'],
                                       msg_data)

        print("ADDED MESSAGE TO BUS....."+str(topic))
# Initialization
helper = ASISTAgentHelper(on_message)

# examples of manually subscribing and unsubscribing to topics

#helper.unsubscribe('observations/events/player/triage')

# load extra info from the ConfigFolder for use later
prefs = helper.get_preferences()
extra_path = os.path.join(prefs['config_folder'], 'extraInfo.json')
extra_info = {"default": "I guess {0} is an okay role."}
if os.path.exists(extra_path):
    with open(extra_path) as extra_file:
        extra_info = json.load(extra_file)

# Main loop
while True:
    helper.check_and_reconnect_if_needed()
