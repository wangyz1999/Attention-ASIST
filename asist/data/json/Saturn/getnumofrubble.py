import json
import pandas as pd
from asist.mapparser import MapParser
from asist.graph import NodeType

def locstr(loc):
    return f"{loc[0]},{loc[1]},{loc[2]}"


with open('Saturn_1.5_3D_basemap_v1.0.json') as f:
    data = json.load(f)

with open('Saturn_1.5_3D_sm_with_victimsA.json') as f:
    data_graph = json.load(f)

graph = MapParser.parse_saturn_map(data_graph)

block_data = pd.read_csv('MapBlocks_SaturnA_Mission_1.csv')

green_victim_count = 1
yellow_victim_count = 1
rubble_count = 1

green_victim_list = dict()
yellow_victim_list = dict()

rubble_location_dict = dict()
for index, row in block_data.iterrows():
    loc_x, loc_y, loc_z = list(map(int, row['LocationXYZ'].split()))
    obj_type = row['FeatureType']

    if obj_type in ["Victim", "victim"]:
        new_id = "vg" + str(green_victim_count)
        green_victim_list[new_id] = f"{loc_x},{loc_y},{loc_z}"
        green_victim_count += 1
    elif obj_type == "High Value Victim":
        new_id = "vy" + str(yellow_victim_count)
        yellow_victim_list[new_id] = f"{loc_x},{loc_y},{loc_z}"
        yellow_victim_count += 1
    elif obj_type in ["Rubble", "rubble"]:
        new_id = "rb" + str(rubble_count)
        rubble_location_dict[f"{loc_x},{loc_y},{loc_z}"] = new_id
        rubble_count += 1

min_x = 99999
min_z = 99999
max_x = -99999
max_z = -99999
solid = set()
for i in data['data']:
    cord_x = i[0][0]
    cord_y = i[0][1]
    cord_z = i[0][2]
    solid.add(f"{cord_x},{cord_y},{cord_z}")
    if cord_x > max_x:
        max_x = cord_x
    if cord_z > max_z:
        max_z = cord_z
    if cord_x < min_x:
        min_x = cord_x
    if cord_z < min_z:
        min_z = cord_z

y_level = 60
for gv in green_victim_list:
    vx, vy, vz = map(int, green_victim_list[gv].split(","))

    print(vx, vy, vz)

    min_rubble = 0
    reached_goal = False

    for i in graph.get_neighbors(graph[gv]):
        # if i.type == NodeType.Portal:
        #     print("jjj")
            goal_x, goal_z = map(int, i.loc)
            # print(goal_x, goal_z)
            queue = []
            visited = set()
            queue.append((vx, vy, vz))
            visited.add(locstr((vx, vy, vz)))

            while not len(queue) == 0:
                curr_x, curr_y, curr_z = queue.pop(0)
                # print(curr_x, curr_y, curr_z)
                # if curr_x == goal_x and curr_y == y_level and curr_z == goal_z:
                #     print("hh")
                #     break

                loc_right = (curr_x + 1, curr_y, curr_z)
                loc_left = (curr_x-1, curr_y, curr_z)
                loc_up = (curr_x, curr_y+1, curr_z)
                loc_down = (curr_x, curr_y-1, curr_z)
                loc_front = (curr_x, curr_y, curr_z+1)
                loc_back = (curr_x, curr_y, curr_z - 1)

                for loc in [loc_right, loc_left, loc_up, loc_down, loc_front, loc_back]:
                    # print(loc, (goal_x, y_level, goal_z), ((loc[0] - goal_x)**2 + (loc[1] - y_level)**2 + (loc[2] - goal_z)**2) ** 0.5)
                    if loc[0] == goal_x and loc[1] == y_level and loc[2] == goal_z:
                        reached_goal = True
                        break

                    if locstr(loc) not in visited and locstr(loc) not in solid and locstr(loc) not in rubble_location_dict and loc[1] > 58 and loc[1] < 62:
                        visited.add(locstr(loc))
                        queue.append(loc)

                if reached_goal:
                    break
            print(reached_goal)
            if reached_goal:
                break
            else:
                queue_norubble = []
                visited_norubble = set()
                queue_norubble.append((vx, vy, vz))
                visited_norubble.add(locstr((vx, vy, vz)))
                path_dict = dict()
                while not len(queue_norubble) == 0:
                    curr_x, curr_y, curr_z = queue_norubble.pop(0)

                    loc_right = (curr_x + 1, curr_y, curr_z)
                    loc_left = (curr_x - 1, curr_y, curr_z)
                    loc_up = (curr_x, curr_y + 1, curr_z)
                    loc_down = (curr_x, curr_y - 1, curr_z)
                    loc_front = (curr_x, curr_y, curr_z + 1)
                    loc_back = (curr_x, curr_y, curr_z - 1)

                    for loc in [loc_right, loc_left, loc_up, loc_down, loc_front, loc_back]:
                        if loc[0] == goal_x and loc[1] == y_level and loc[2] == goal_z:
                            reached_goal = True
                            path_dict[locstr(loc)] = locstr((curr_x, curr_y, curr_z))
                            break

                        if locstr(loc) not in visited and locstr(loc) not in solid and loc[1] > 57 and loc[1] < 64:
                            visited_norubble.add(locstr(loc))
                            queue_norubble.append(loc)
                            path_dict[locstr(loc)] = locstr((curr_x, curr_y, curr_z))

                path = []
                trace_loc = locstr((goal_x, y_level, goal_z))
                while trace_loc in path_dict:
                    path.append(trace_loc)
                    trace_loc = path_dict[trace_loc]

                print(path)


        # print(gv, min_rubble)














