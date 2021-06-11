import json
import pandas as pd

with open('Saturn_1.5_3D_sm_v1.0.json') as f:
    data = json.load(f)

block_data = pd.read_csv('MapBlocks_SaturnA_Mission_1.csv')

green_victim_count = 1
yellow_victim_count = 1
rubble_count = 1

obj_list = []
for index, row in block_data.iterrows():
    loc_x, loc_y, loc_z = list(map(int, row['LocationXYZ'].split()))
    obj_type = row['FeatureType']

    if obj_type in ["Victim", "victim"]:
        new_id = "vg" + str(green_victim_count)
        obj_type = "green_victim"
        block_type = "block"
        green_victim_count += 1
    elif obj_type == "High Value Victim":
        new_id = "vy" + str(yellow_victim_count)
        obj_type = "yellow_victim"
        block_type = "block"
        yellow_victim_count += 1
    elif obj_type in ["Rubble", "rubble"]:
        new_id = "rb" + str(rubble_count)
        obj_type = "rubble"
        block_type = "gravel"
        rubble_count += 1


    new_entry = {
        "id": new_id,
        "type": obj_type,
        "bounds": {
            "type": block_type,
            "coordinates": [
                {
                    "x": loc_x,
                    "y": loc_y,
                    "z": loc_z
                }
            ]
        }
    }

    obj_list.append(new_entry)


data['objects'] += obj_list

a_file = open("Saturn_1.5_3D_sm_with_victimsA.json", "w")
json.dump(data, a_file, indent=2)
a_file.close()