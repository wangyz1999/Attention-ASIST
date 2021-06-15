import graph
import pandas as pd
import numpy as np
from pathlib import Path
import gym
from gym import spaces

class MapParser:

    @classmethod
    def victim_type_str_to_type(cls, type_str):
        if type_str in ["Green", "green"]:
            return graph.VictimType.Green
        if type_str in ["Gold", "yellow"]:
            return graph.VictimType.Yellow

    @classmethod
    def parse_map_data(cls, portal_data, room_data, victim_data):
        """ Given Map Data, construct a Graph
        :param portal_data: pandas data-frame for portal_data
        :param room_data: pandas data-frame for room_data
        :param victim_data: pandas data-frame for victim_data
        :return: the graph
        """
        g = graph.Graph()
        for index, row in room_data.iterrows():
            g.add_room(id=row["id"], location=eval(row["loc"]), victims=eval(row["connections"]))

        for index, row in victim_data.iterrows():
            g.add_victim(cls.victim_type_str_to_type(row["type"]), id=row["id"], location=eval(row["loc"]))
        for index, row in portal_data.iterrows():
            # is_open = row['isOpen'] == "TRUE"
            # g.add_portal(tuple(eval(row["connections"])), is_open, id=row["id"], location=eval(row["loc"]))
            g.add_portal(tuple(eval(row["connections"])), id=row["id"], location=eval(row["loc"]))

        for room in g.room_list:
            g.link_victims_in_room(room, room.victim_list)

        for portal_pair in g.portal_list:
            g.connect_portal_to_rooms(portal_pair)

        for portal_pair in g.portal_list:
            g.connected_portals_to_portals(portal_pair)

        g.make_ordered_node_list()

        return g

    @classmethod
    def parse_json_map_data(cls, data, excludes=[]):
        """ Given Json Map Data, construct a Graph
        :param json_data: The json that contains information about the map
        :return: the graph
        """
        def center(pos1, pos2):
            return (pos1[0] + pos2[0])/2, (pos1[1] + pos2[1])/2

        def inside(pos, pos1, pos2):
            if pos1[0] <= pos[0] <= pos2[0] and pos1[1] <= pos[1] <= pos2[1]:
                return True

        green_victim_list = []
        yellow_victim_list = []

        for i in data["objects"]:
       # for i in data["path"]:
            if i["id"] not in excludes:
                if i["type"] == "green_victim":
                    loc = i["bounds"]["coordinates"][0]
                    x = loc["x"]
                    z = loc["z"]
                    green_victim_list.append((i["id"], (x, z), "green"))
                if i["type"] == "yellow_victim":
                    loc = i["bounds"]["coordinates"][0]
                    x = loc["x"]
                    z = loc["z"]
                    green_victim_list.append((i["id"], (x, z), "yellow"))

        room_data = []
        for j in data["locations"]:
            if "bounds" in j:
                cord = j["bounds"]["coordinates"]
                pos1 = (cord[0]["x"], cord[0]["z"])
                pos2 = (cord[1]["x"], cord[1]["z"])
                inside_victim = []
                for gv in green_victim_list:
                    if inside(gv[1], pos1, pos2):
                        inside_victim.append(gv[0])
                for yv in yellow_victim_list:
                    if inside(yv[1], pos1, pos2):
                        inside_victim.append(yv[0])
                room_data.append((j["id"], center(pos1, pos2), inside_victim))

        portal_data = []
        extension_data = []
        for k in data["connections"]:
            cord = k["bounds"]["coordinates"]
            pos1 = (cord[0]["x"], cord[0]["z"])
            pos2 = (cord[1]["x"], cord[1]["z"])
            portal_data.append(("", center(pos1, pos2), tuple(k["connected_locations"])))

        g = graph.Graph()

        for r in room_data:
            g.add_room(r[0], location=r[1], victims=r[2])

        for gv in green_victim_list:
            g.add_victim(cls.victim_type_str_to_type(gv[2]), id=gv[0], location=gv[1])
        for yv in yellow_victim_list:
            g.add_victim(cls.victim_type_str_to_type(yv[2]), id=yv[0], location=yv[1])

        for p in portal_data:
            g.add_portal(p[2], id=p[0], location=p[1])

        for room in g.room_list:
            g.link_victims_in_room(room, room.victim_list)

        for portal_pair in g.portal_list:
            g.connect_portal_to_rooms(portal_pair)

        for portal_pair in g.portal_list:
            g.connected_portals_to_portals(portal_pair)

        g.make_ordered_node_list()

        return g


    @classmethod
    def parse_json_map_data_new_format(cls, data):
        """ Given Json Map Data, construct a Graph
        :param json_data: The json that contains information about the map
        :return: the graph
        """
        def center(pos1, pos2):
            return (pos1[0] + pos2[0])/2, (pos1[1] + pos2[1])/2

        def inside(pos, pos1, pos2):
            if pos1[0] <= pos[0] <= pos2[0] and pos1[1] <= pos[1] <= pos2[1]:
                return True

        green_victim_list = []
        yellow_victim_list = []

        for i in data["objects"]:
            if i["type"] == "green_victim":
                loc = i["bounds"]["coordinates"][0]
                x = loc["x"]
                z = loc["z"]
                green_victim_list.append((i["id"], (x, z), "green"))
            if i["type"] == "yellow_victim":
                loc = i["bounds"]["coordinates"][0]
                x = loc["x"]
                z = loc["z"]
                yellow_victim_list.append((i["id"], (x, z), "yellow"))

        room_data = []
        for j in data["locations"]:
            if "bounds" in j:
                cord = j["bounds"]["coordinates"]
                pos1 = (cord[0]["x"], cord[0]["z"])
                pos2 = (cord[1]["x"], cord[1]["z"])
                inside_victim = []
                for gv in green_victim_list:
                    if inside(gv[1], pos1, pos2):
                        inside_victim.append(gv[0])
                for yv in yellow_victim_list:
                    if inside(yv[1], pos1, pos2):
                        inside_victim.append(yv[0])
                room_data.append((j["id"], center(pos1, pos2), inside_victim))

        portal_data = []
        extension_data = []
        for k in data["connections"]:
            cord = k["bounds"]["coordinates"]
            pos1 = (cord[0]["x"], cord[0]["z"])
            pos2 = (cord[1]["x"], cord[1]["z"])
            if k["type"] == "extension":
                extension_data.append((k["id"], center(pos1, pos2), tuple(k["connected_locations"])))
            else:
                portal_data.append((k["id"], center(pos1, pos2), tuple(k["connected_locations"])))

        g = graph.Graph()

        for r in room_data:
            g.add_room(r[0], location=r[1], victims=r[2])

        for gv in green_victim_list:
            g.add_victim(cls.victim_type_str_to_type(gv[2]), id=gv[0], location=gv[1])
        for yv in yellow_victim_list:
            g.add_victim(cls.victim_type_str_to_type(yv[2]), id=yv[0], location=yv[1])

        for p in portal_data:
            g.add_portal(p[2], id=p[0], location=p[1])

        for room in g.room_list:
            g.link_victims_in_room(room, room.victim_list)

        for extension in extension_data:
            g.connect_rooms_by_extension(extension)

        for portal_pair in g.portal_list:
            g.connect_portal_to_rooms(portal_pair)

        for portal_pair in g.portal_list:
            g.connected_portals_to_portals(portal_pair)

        g.make_ordered_node_list()

        return g

    @classmethod
    def parse_saturn_map(cls, data):
        """ Given Json Map Data, construct a Graph
        :param json_data: The json that contains information about the map
        :return: the graph
        """
        def center(pos1, pos2):
            return (pos1[0] + pos2[0])/2, (pos1[1] + pos2[1])/2

        def inside(pos, pos1, pos2):
            if pos1[0] <= pos[0] <= pos2[0] and pos1[1] <= pos[1] <= pos2[1]:
                return True

        green_victim_list = []
        yellow_victim_list = []

        for i in data["objects"]:
            if i["type"] == "green_victim":
                loc = i["bounds"]["coordinates"][0]
                x = loc["x"]
                z = loc["z"]
                green_victim_list.append((i["id"], (x, z), "green"))
            if i["type"] == "yellow_victim":
                loc = i["bounds"]["coordinates"][0]
                x = loc["x"]
                z = loc["z"]
                yellow_victim_list.append((i["id"], (x, z), "yellow"))

        room_data = []
        for j in data["locations"]:
            if "bounds" in j:
                cord = j["bounds"]["coordinates"]
                pos1 = (cord[0]["x"], cord[0]["z"])
                pos2 = (cord[1]["x"], cord[1]["z"])
                inside_victim = []
                for gv in green_victim_list:
                    if inside(gv[1], pos1, pos2):
                        inside_victim.append(gv[0])
                for yv in yellow_victim_list:
                    if inside(yv[1], pos1, pos2):
                        inside_victim.append(yv[0])
                room_data.append((j["id"], center(pos1, pos2), inside_victim))

        portal_data = []
        extension_data = []
        for k in data["connections"]:
            cord = k["bounds"]["coordinates"]
            pos1 = (cord[0]["x"], cord[0]["z"])
            pos2 = (cord[1]["x"], cord[1]["z"])
            if k["type"] == "extension":
                extension_data.append((k["id"], center(pos1, pos2), tuple(k["connected_locations"])))
            else:
                portal_data.append((k["id"], center(pos1, pos2), tuple(k["connected_locations"])))

        g = graph.Graph()

        for r in room_data:
            g.add_room(r[0], location=r[1], victims=r[2])

        for gv in green_victim_list:
            g.add_victim(cls.victim_type_str_to_type(gv[2]), id=gv[0], location=gv[1])
        for yv in yellow_victim_list:
            g.add_victim(cls.victim_type_str_to_type(yv[2]), id=yv[0], location=yv[1])

        for p in portal_data:
            # print(p)
            g.add_portal(p[2], id=p[0], location=p[1])
        # input()

        for room in g.room_list:
            g.link_victims_in_room(room, room.victim_list)

        for extension in extension_data:
            g.connect_rooms_by_extension(extension)

        for portal_pair in g.portal_list:
            g.connect_portal_to_rooms(portal_pair)

        for portal_pair in g.portal_list:
            g.connected_portals_to_portals(portal_pair)

        g.make_ordered_node_list()

        return g

    @classmethod
    def parse_json_map_data_new_format_random(cls, data):
        """ Given Json Map Data, construct a Graph
        :param json_data: The json that contains information about the map
        :return: the graph
        """
        def center(pos1, pos2):
            return (pos1[0] + pos2[0])/2, (pos1[1] + pos2[1])/2

        def inside(pos, pos1, pos2):
            if pos1[0] <= pos[0] <= pos2[0] and pos1[1] <= pos[1] <= pos2[1]:
                return True

        green_victim_list = []
        yellow_victim_list = []

        for i in data["objects"]:
            if i["type"] == "green_victim":
                loc = i["bounds"]["coordinates"][0]
                x = loc["x"]
                z = loc["z"]
                green_victim_list.append((i["id"], (x, z), "green"))
            if i["type"] == "yellow_victim":
                loc = i["bounds"]["coordinates"][0]
                x = loc["x"]
                z = loc["z"]
                yellow_victim_list.append((i["id"], (x, z), "yellow"))

        room_data = []
        for j in data["locations"]:
            if "bounds" in j:
                cord = j["bounds"]["coordinates"]
                pos1 = (cord[0]["x"], cord[0]["z"])
                pos2 = (cord[1]["x"], cord[1]["z"])
                inside_victim = []
                for gv in green_victim_list:
                    if inside(gv[1], pos1, pos2):
                        inside_victim.append(gv[0])
                for yv in yellow_victim_list:
                    if inside(yv[1], pos1, pos2):
                        inside_victim.append(yv[0])
                room_data.append((j["id"], center(pos1, pos2), inside_victim))

        portal_data = []
        extension_data = []
        for k in data["connections"]:
            cord = k["bounds"]["coordinates"]
            pos1 = (cord[0]["x"], cord[0]["z"])
            pos2 = (cord[1]["x"], cord[1]["z"])
            if k["type"] == "extension":
                extension_data.append((k["id"], center(pos1, pos2), tuple(k["connected_locations"])))
            else:
                portal_data.append((k["id"], center(pos1, pos2), tuple(k["connected_locations"])))

        g = graph.Graph()

        for r in room_data:
            g.add_room(r[0], location=r[1], victims=r[2])

        for gv in green_victim_list:
            g.add_victim(cls.victim_type_str_to_type(gv[2]), id=gv[0], location=gv[1])
        for yv in yellow_victim_list:
            g.add_victim(cls.victim_type_str_to_type(yv[2]), id=yv[0], location=yv[1])

        for p in portal_data:
            g.add_portal(p[2], id=p[0], location=p[1])

        for room in g.room_list:
            g.link_victims_in_room(room, room.victim_list)

        for extension in extension_data:
            g.connect_rooms_by_extension(extension)

        for portal_pair in g.portal_list:
            g.connect_portal_to_rooms(portal_pair)

        for portal_pair in g.portal_list:
            g.connected_portals_to_portals(portal_pair)

        g.make_ordered_node_list()

        return g


    @classmethod
    def no_victim_map(cls, portal_data, room_data):
        """ Given Map Data, construct a Graph
        :param portal_data: pandas data-frame for portal_data
        :param room_data: pandas data-frame for room_data
        :param victim_data: pandas data-frame for victim_data
        :return: the graph
        """
        g = graph.Graph()
        for index, row in room_data.iterrows():
            g.add_room(id=row["id"], location=eval(row["loc"]))

        for index, row in portal_data.iterrows():
            # is_open = row['isOpen'] == "TRUE"
            # g.add_portal(tuple(eval(row["connections"])), is_open, id=row["id"], location=eval(row["loc"]))
            g.add_portal(tuple(eval(row["connections"])), id=row["id"], location=eval(row["loc"]))

        g.make_ordered_node_list()

        return g
from mapparser import MapParser

class AsistEnvRandGen:
    def __init__(self):
        pass

class AsistEnvGym(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, json_data, portal_data=None, room_data=None, victim_data=None, start_node_id='as', random_victim=False):
    # def __init__(self, data, start_node_id, random_victim=False):
        super(AsistEnvGym, self).__init__()
        if random_victim:
            self.graph = MapParser.no_victim_map(portal_data, room_data)
        else:
            # self.graph = MapParser.parse_map_data(portal_data, room_data, victim_data)
            self.graph = MapParser.parse_json_map_data_new_format(json_data)
            # self.graph = MapParser.parse_json_map_data(data)
        self.start_node_id = start_node_id
        self.curr_pos = self.graph[start_node_id]
        self.prev_pos = None
        self.total_cost = 0
        self.score = 0
        self.positive_reward_multiplier = 10
        self.edge_cost_multiplier = 8
        self.visit_node_sequence = []
        self.victim_data = victim_data
        self.stop_cost = 600
        self.yellow_decease_cost = 300
        self.cost_bits = 6
        self.agent_speed = [1, 4.3, 5.6][2]
        # self.room_visited = set()
        self.no_victim_rooms = {"achl", "alha", "alhb", "ach", "arha", "arhb", "as"}
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:

        ### Experiment #########################
        # max_nei_length = 0
        # for node in self.graph.nodes_list:
        #     neighbor_length = len([n for n in self.graph.neighbors(node)])
        #     max_nei_length = max(max_nei_length, neighbor_length)
        # self.action_space = spaces.Discrete(max_nei_length)

        # self.action_space = spaces.Discrete(len(self.graph.nodes_list))
        self.action_space = spaces.Discrete(1+1+1+6)
        # self.observation_space = spaces.Box(low=0, high=1, shape=(len(self.graph.nodes_list)+2+self.cost_bits,), dtype=np.int)
        # self.observation_space = spaces.Box(low=0, high=1, shape=(3*5+2+self.cost_bits,), dtype=np.int)
        self.observation_space = spaces.Box(low=0, high=1, shape=(9+2+2+1+2+1+3+3,), dtype=np.int)
        # self.observation_space = spaces.Box(low=0, high=1, shape=(7+2+2+1+1+3,), dtype=np.int)

    def _next_observation_old(self):
        room_observation = np.zeros(len(self.graph.room_list))
        portal_observation = np.zeros(len(self.graph.portal_list) * 2)
        victim_observation = np.zeros(len(self.graph.victim_list))

        for n in self.graph.get_neighbors(self.curr_pos):
            if n.type == graph.NodeType.Room:
                room_observation[self.graph.room_list.index(n)] = 1
            if n.type == graph.NodeType.Portal:
                # need to find the exact portal index since the portal list stores tuples of portals
                idx = None
                for pl_idx, pt in enumerate(self.graph.portal_list):
                    if n.id == pt[0].id:
                        idx = pl_idx * 2
                        break
                    elif n.id == pt[1].id:
                        idx = pl_idx * 2 + 1
                        break
                portal_observation[idx] = 1
            if n.type == graph.NodeType.Victim and \
                    (n.victim_type == graph.VictimType.Green or n.victim_type == graph.VictimType.Yellow):
                victim_observation[self.graph.victim_list.index(n)] = 1
        device_num = self.get_device_info()
        if device_num == 0:
            device_info = [0, 0]
        elif device_num == 1:
            device_info = [1, 0]
        else:
            device_info = [1, 1]

        device_info = np.array(device_info)

        time_bins = self.split(range(self.stop_cost+1), 2 ** self.cost_bits)
        bin_idx = None
        for idx, ran in enumerate(time_bins):
            if min(self.total_cost, self.stop_cost) in ran:
                bin_idx = idx
                break

        # print(bin_idx, self.total_cost)
        bin_str = f"{bin_idx:0{self.cost_bits}b}"
        bin_list = list(map(int, list(bin_str)))
        # print(bin_list)

        return np.concatenate([room_observation, portal_observation, victim_observation, device_info, bin_list])

    def _next_observation_yunzhe_narrow(self):
        num_room = 0
        num_portal = 0
        num_green_victim = 0
        num_yellow_victim = 0
        num_other_victim = 0

        for n in self.graph.get_neighbors(self.curr_pos):
            if n.type == graph.NodeType.Room:
                num_room += 1
            if n.type == graph.NodeType.Portal:
                num_portal += 1
            if n.type == graph.NodeType.Victim:
                if n.victim_type == graph.VictimType.Yellow:
                    num_yellow_victim += 1
                elif n.victim_type == graph.VictimType.Green:
                    num_green_victim += 1
                else:
                    num_other_victim += 1

        green_victim_str = f"{num_green_victim:03b}"
        yellow_victim_str = f"{num_yellow_victim:03b}"
        other_victim_str = f"{num_other_victim:03b}"
        room_str = f"{num_room:03b}"
        portal_str = f"{num_portal:03b}"

        green_victim_list = list(map(int, list(green_victim_str)))
        yellow_victim_list = list(map(int, list(yellow_victim_str)))
        other_victim_list = list(map(int, list(other_victim_str)))
        room_list = list(map(int, list(room_str)))
        portal_list = list(map(int, list(portal_str)))

        device_num = self.get_device_info()
        if device_num == 0:
            device_info = [0, 0]
        elif device_num == 1:
            device_info = [1, 0]
        else:
            device_info = [1, 1]

        device_info = np.array(device_info)

        time_bins = self.split(range(self.stop_cost+1), 2 ** self.cost_bits)
        bin_idx = None
        for idx, ran in enumerate(time_bins):
            if min(self.total_cost, self.stop_cost) in ran:
                bin_idx = idx
                break

        # print(bin_idx, self.total_cost)
        bin_str = f"{bin_idx:0{self.cost_bits}b}"
        bin_list = list(map(int, list(bin_str)))
        # print(bin_list)

        return np.concatenate([room_list, portal_list, yellow_victim_list, green_victim_list, other_victim_list, device_info, bin_list])

    def _two_encoding(self, num):
        assert isinstance(num, int) and 0 <= num <= 3
        if num == 0:
            return np.array([0, 0])
        elif num == 1:
            return np.array([0, 1])
        elif num == 2:
            return np.array([1, 0])
        else:
            return np.array([1, 1])

    def _next_observation(self):
        # curr_node_idx = self.graph.ordered_node_list.index(self.curr_pos)
        curr_node_idx = self.graph.nodes_list.index(self.curr_pos)
        curr_node_idx_str = f"{curr_node_idx:09b}"
        curr_node_idx_list = np.array(list(map(int, list(curr_node_idx_str))))

        num_connecting_portal = 0
        num_green_victim = 0
        num_yellow_victim = 0
        has_mirroring_portal = False

        num_connecting_room = 0

        for n in self.graph.get_neighbors(self.curr_pos):
            if n.type == graph.NodeType.Portal:
                num_connecting_portal += 1
                if self.curr_pos.type == graph.NodeType.Portal and self.curr_pos.is_same_portal(n):
                    has_mirroring_portal = True
                    num_connecting_portal -= 1
            if n.type == graph.NodeType.Room:
                num_connecting_room += 1
            if n.type == graph.NodeType.Victim:
                if n.victim_type == graph.VictimType.Yellow:
                    num_yellow_victim += 1
                elif n.victim_type == graph.VictimType.Green:
                    num_green_victim += 1

        connecting_portal_str = f"{num_connecting_portal:03b}"
        connecting_portal_list = np.array(list(map(int, list(connecting_portal_str))))

        connecting_room_str = f"{num_connecting_room:03b}"
        connecting_room_list = np.array(list(map(int, list(connecting_room_str))))

        device_num = self.get_device_info()
        device_info = self._two_encoding(device_num)
        green_victim_list = self._two_encoding(num_green_victim)
        yellow_victim_list = self._two_encoding(num_yellow_victim)

        mirroring_portal_slot = np.array([1]) if has_mirroring_portal else np.array([0])

        time_slot = np.array([min(self.total_cost, self.stop_cost) / self.stop_cost])
        return np.concatenate([curr_node_idx_list, yellow_victim_list, green_victim_list, mirroring_portal_slot, device_info, time_slot, connecting_portal_list, connecting_room_list])
        # return np.concatenate([curr_node_idx_list, yellow_victim_list, green_victim_list, mirroring_portal_slot, time_slot, connecting_portal_list])

    def _node_observation_debug(self):
        room_observation = [(0, node.id) for node in self.graph.room_list]
        portal_observation = [(0, portal.id) for node in self.graph.portal_list for portal in node]
        victim_observation = [(0, node.id) for node in self.graph.victim_list]

        for n in self.graph.get_neighbors(self.curr_pos):
            if n.type == graph.NodeType.Room:
                room_observation[self.graph.room_list.index(n)] = (1, n.id)
            if n.type == graph.NodeType.Portal:
                idx = None
                for pl_idx, pt in enumerate(self.graph.portal_list):
                    if n.id == pt[0].id:
                        idx = pl_idx * 2
                        break
                    elif n.id == pt[1].id:
                        idx = pl_idx * 2 + 1
                        break
                portal_observation[idx] = (1, n.id)
            if n.type == graph.NodeType.Victim:
                victim_observation[self.graph.victim_list.index(n)] = (1, n.id)

        return room_observation + portal_observation + victim_observation


    def get_device_info(self):
        # Nothing: 0,  Green: 1, Yellow: 2,
        if self.curr_pos.type == graph.NodeType.Portal:
            connected_room = self.graph.id2node[self.curr_pos.linked_portal.get_connected_room_id()]
            if self.graph.has_yellow_victim_in(connected_room):
                return 2
            elif self.graph.has_green_victim_in(connected_room):
                return 1
        return 0

    def split(self, a, n):
        k, m = divmod(len(a), n)
        return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


    def step_old(self, action):
        # action_node = self.graph.ordered_node_list[action]
        # neighbors = [n for n in self.graph.neighbors(self.curr_pos)]

        # Room, Victim, Portals
        neighbor_room = []
        neighbor_victim = []
        neighbor_portal = []
        for n in self.graph.neighbors(self.curr_pos):
            if n.type == graph.NodeType.Room:
                neighbor_room.append(n)
            elif n.type == graph.NodeType.Victim:
                neighbor_victim.append(n)
            elif n.type == graph.NodeType.Portal:
                neighbor_portal.append(n)

        neighbors = neighbor_room + neighbor_victim + neighbor_portal
        range_list = self.split(range(self.action_space.n), len(neighbors))

        action_node = None
        for idx, ran in enumerate(range_list):
            if action in ran:
                # print(idx, action, len(neighbors), range_list)
                action_node = neighbors[idx]
                break


        # num_of_bins = len(self.graph.ordered_node_list) // len(neighbors)
        # action_node = neighbors[min(action // num_of_bins, len(neighbors)-1)]

        # print(action, action_node.id)
        # print(str([(0, n.id) for n in self.graph.ordered_node_list]))
        # print(str(self._node_observation_debug()))

        reward = 0
        done = False
        # print(action_node.id)

        # neighbors = [n for n in self.graph.neighbors(self.curr_pos)]
        # if action >= len(neighbors):
        if not any(action_node.id == n.id for n in self.graph.neighbors(self.curr_pos)):
            reward -= 5
            # done = True
            # print(self.curr_pos.id, action_node.id)
            # print(self._next_observation())
            # print(action_node.id)
            print("he")
        else:
            # action_node = neighbors[action]
            # print(action)
            self.visit_node_sequence.append(action_node.id)
            edge_cost = self.graph.get_edge_cost(self.curr_pos, action_node)
            self.prev_pos = self.curr_pos
            self.curr_pos = action_node

            action_node.visited_count += 1

            # if action_node.visited_count == 1:
            #     reward += 20

            reward -= edge_cost
            self.total_cost += edge_cost
            if action_node.type == graph.NodeType.Victim:
                triage_cost, triage_score = self.graph.triage(action_node)
                reward -= triage_cost
                self.total_cost += triage_cost
                self.score += triage_score
                reward += triage_score * self.positive_reward_multiplier

        if self.graph.no_more_victims() or self.total_cost > self.stop_cost:
            extra_reward = sum(10 if n.visited_count > 0 else 0 for n in self.graph.ordered_node_list )
            # all_nodes = [n.visited_count for n in self.graph.ordered_node_list]
            reward += extra_reward
            # print(extra_reward)
            # steps = len(self.visit_node_sequence)
            # reward -= steps * 5

            if self.graph.no_more_victims():
                reward += 100

            done = True

        # return self._next_observation(), reward, done, {"node_debug": self._node_observation_debug()}
        return self._next_observation(), reward, done, {}

    def step(self, action):
        assert isinstance(action, int)
        yellow_victim = None
        green_victim = None
        yellow_dist = 999
        green_dist = 999
        mirror_portal = None
        connecting_portals = list()
        connecting_rooms = list()

        for n in self.graph.neighbors(self.curr_pos):
            if n.type == graph.NodeType.Victim:
                if n.victim_type == graph.VictimType.Yellow:
                    if self.graph.get_edge_cost(self.curr_pos, n) < yellow_dist:
                        yellow_victim = n
                        yellow_dist = self.graph.get_edge_cost(self.curr_pos, n)
                if n.victim_type == graph.VictimType.Green:
                    if self.graph.get_edge_cost(self.curr_pos, n) < green_dist:
                        green_victim = n
                        green_dist = self.graph.get_edge_cost(self.curr_pos, n)
            if n.type == graph.NodeType.Portal:
                if self.curr_pos.type == graph.NodeType.Portal and self.curr_pos.is_same_portal(n):
                    assert mirror_portal is None
                    mirror_portal = n
                else:
                    connecting_portals.append(n)
            if n.type == graph.NodeType.Room:
                connecting_rooms.append(n)

        reward = 0
        done = False
        action_node = None

        if action == 0:
            action_node = yellow_victim
        elif action == 1:
            action_node = green_victim
        elif action == 2:
            action_node = mirror_portal
        elif 3 <= action <= 8:
            if len(connecting_portals) != 0:
                range_list = self.split(range(6), len(connecting_portals))
                for idx, ran in enumerate(range_list):
                    if action-3 in ran:
                        # print(idx, action, len(neighbors), range_list)
                        action_node = connecting_portals[idx]
                        break
        else:
            if len(connecting_rooms) != 0:
                range_list = self.split(range(6), len(connecting_rooms))
                for idx, ran in enumerate(range_list):
                    if action-9 in ran:
                        # print(idx, action, len(neighbors), range_list)
                        action_node = connecting_rooms[idx]
                        break

        if action_node is None:
            reward -= 10
        else:
            self.visit_node_sequence.append(action_node.id)
            edge_dist = self.graph.get_edge_cost(self.curr_pos, action_node)
            edge_cost = edge_dist / self.agent_speed
            self.prev_pos = self.curr_pos
            self.curr_pos = action_node

            action_node.visited_count += 1
            # for n in self.graph.neighbors(self.curr_pos):
            #     if n.type == graph.NodeType.Room and n.id not in self.no_victim_rooms and n.id not in self.room_visited:
            #         self.room_visited.add(n.id)


            reward -= edge_cost * self.edge_cost_multiplier
            self.total_cost += edge_cost
            if action_node.type == graph.NodeType.Victim:
                triage_cost, triage_score = self.graph.triage(action_node)
                reward -= triage_cost
                self.total_cost += triage_cost
                self.score += triage_score

                reward += triage_score * self.positive_reward_multiplier

                # if action_node.type == graph.VictimType.Yellow and self.total_cost < self.yellow_decease_cost:
                #     reward += 100

            # if action_node.visited_count == 1:
            #     reward += 100

            if action_node.type == graph.NodeType.Portal and action_node.visited_count == 1:
                reward += 100

        if self.graph.no_more_victims() or self.total_cost > self.stop_cost:
        # if self.total_cost > self.stop_cost:
            extra_reward = sum(10 if n.visited_count > 0 else 0 for n in self.graph.ordered_node_list )
            # extra_reward = 30 * len(self.graph.safe_victim_list)

            # all_nodes = [n.visited_count for n in self.graph.ordered_node_list]
            reward += extra_reward
            # print(extra_reward)
            # steps = len(self.visit_node_sequence)
            # reward -= steps * 5

            if self.graph.no_more_victims():
                reward += 100

            # if len(self.room_visited) == 24:
            #     reward += 100

            done = True

        # return self._next_observation(), reward, done, {"node_debug": self._node_observation_debug()}
        return self._next_observation(), reward, done, {}

    def step_two_portal(self, action):
        yellow_victim = None
        green_victim = None
        yellow_dist = 999
        green_dist = 999
        mirror_portal = None
        connecting_portals = list()

        for n in self.graph.neighbors(self.curr_pos):
            if n.type == graph.NodeType.Victim:
                if n.victim_type == graph.VictimType.Yellow:
                    if self.graph.get_edge_cost(self.curr_pos, n) < yellow_dist:
                        yellow_victim = n
                        yellow_dist = self.graph.get_edge_cost(self.curr_pos, n)
                if n.victim_type == graph.VictimType.Green:
                    if self.graph.get_edge_cost(self.curr_pos, n) < green_dist:
                        green_victim = n
                        green_dist = self.graph.get_edge_cost(self.curr_pos, n)
            if n.type == graph.NodeType.Portal:
                if self.curr_pos.type == graph.NodeType.Portal and self.curr_pos.is_same_portal(n):
                    assert mirror_portal is None
                    mirror_portal = n
                else:
                    connecting_portals.append(n)

        sorted_connecting_portals = sorted(connecting_portals, key=lambda n: self.graph.get_edge_cost(self.curr_pos, n))

        two_portals = sorted_connecting_portals[:2]
        # if len(two_portals) == 2:
        #     print(self.graph.get_edge_cost(self.curr_pos, two_portals[0]), self.graph.get_edge_cost(self.curr_pos, two_portals[1]))

        reward = 0
        done = False
        action_node = None

        if action == 0:
            action_node = yellow_victim
        elif action == 1:
            action_node = green_victim
        elif action == 2:
            action_node = mirror_portal
        elif action == 3 and len(two_portals) >= 1:
            action_node = two_portals[0]
        elif action == 4 and len(two_portals) == 2:
            action_node = two_portals[1]

        if action_node is None:
            reward -= 10
        else:
            self.visit_node_sequence.append(action_node.id)
            edge_dist = self.graph.get_edge_cost(self.curr_pos, action_node)
            edge_cost = edge_dist / self.agent_speed
            self.prev_pos = self.curr_pos
            self.curr_pos = action_node

            action_node.visited_count += 1

            reward -= edge_cost * self.edge_cost_multiplier
            self.total_cost += edge_cost
            if action_node.type == graph.NodeType.Victim:
                triage_cost, triage_score = self.graph.triage(action_node)
                reward -= triage_cost
                self.total_cost += triage_cost
                self.score += triage_score
                reward += triage_score * self.positive_reward_multiplier

                if action_node.type == graph.VictimType.Yellow and self.total_cost < self.yellow_decease_cost:
                    reward += 100

        if self.graph.no_more_victims() or self.total_cost > self.stop_cost:
            extra_reward = sum(10 if n.visited_count > 0 else 0 for n in self.graph.ordered_node_list )
            # all_nodes = [n.visited_count for n in self.graph.ordered_node_list]
            reward += extra_reward
            # print(extra_reward)
            # steps = len(self.visit_node_sequence)
            # reward -= steps * 5

            if self.graph.no_more_victims():
                reward += 100

            done = True

        # return self._next_observation(), reward, done, {"node_debug": self._node_observation_debug()}
        return self._next_observation(), reward, done, {}


    def reset(self):
        # Reset the state of the environment to an initial state
        self.curr_pos = self.graph[self.start_node_id]
        self.graph.reset()
        # self.graph = self.graph_copy.copy()
        self.total_cost = 0
        # self.room_visited.clear()
        self.score = 0
        self.prev_pos = None
        self.visit_node_sequence.clear()
        # return self.get_observation()
        return self._next_observation()

    def reset_victims(self):
        # Reset the state of the environment to an initial state
        self.curr_pos = self.graph[self.start_node_id]
        self.graph.remove_all_victims()
        no_victim_rooms = {"achl", "alha", "alhb", "ach", "arha", "arhb", "as"}
        self.graph = graph.RandomGraphGenerator.add_random_victims(self.graph, no_victim_rooms)
        # self.graph = self.graph_copy.copy()
        self.total_cost = 0
        self.score = 0
        # self.room_visited.clear()
        self.prev_pos = None
        self.visit_node_sequence.clear()
        # return self.get_observation()

        return self._next_observation()

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        pass


class AsistEnv:
    def __init__(self, portal_data, room_data, victim_data, start_node_id):
        self.graph = MapParser.parse_map_data(portal_data, room_data, victim_data)
        # self.graph_copy = self.graph.copy()
        self.start_node_id = start_node_id
        self.curr_pos = self.graph[start_node_id]
        self.prev_pos = None
        self.total_cost = 0
        self.score = 0
        self.positive_reward_multiplier = 10
        self.visit_node_sequence = []
        self.victim_data = victim_data

    def reset(self):
        self.curr_pos = self.graph[self.start_node_id]
        self.graph.reset()
        # self.graph = self.graph_copy.copy()
        self.total_cost = 0
        self.score = 0
        self.prev_pos = None
        self.visit_node_sequence.clear()
        # return self.get_observation()
        return self.get_unorganized_observation_simplified()

    def get_victim_list_size(self):
        return len(self.graph.victim_list)

    def step(self, action):
        """ The global view
        :param performance: 0: navigate, 1: enter, 2: triage
        :param action_node: the index of the node for your performance
        :return: (observation, reward)
        """
        # assert performance in [0, 1, 2]
        raise NotImplementedError
        performance, action_node_idx = action

        reward = 0
        action_node = self.graph.nodes_list[action_node_idx]

        triage_set = set()
        navigation_set = set()
        portal_enter_set = set()
        for n in self.graph.neighbors(self.curr_pos):
            if n.type == graph.NodeType.Portal:
                if self.curr_pos.type == graph.NodeType.Portal and \
                        n.is_same_portal(self.curr_pos):
                    portal_enter_set.add(n)
                else:
                    navigation_set.add(n)
            elif n.type == graph.NodeType.Victim:
                triage_set.add(n)
            elif n.type == graph.NodeType.Room:
                navigation_set.add(n)

        valid_action = True
        if performance == 0 and action_node not in navigation_set:
            valid_action = False
        if performance == 1 and action_node not in portal_enter_set:
            valid_action = False
        if performance == 2 and action_node not in triage_set:
            valid_action = False

        if not valid_action:
            reward -= 100
            # print("ha")
        else:
            # print(action)
            edge_cost = self.graph.get_edge_cost(self.curr_pos, action_node)
            self.prev_pos = self.curr_pos
            self.curr_pos = action_node
            reward -= edge_cost
            self.total_cost += edge_cost
            if action_node.type == graph.NodeType.Victim:
                triage_cost, triage_score = self.graph.triage(action_node)
                reward -= triage_cost
                self.total_cost += triage_cost
                self.score += triage_score
                reward += triage_score * self.positive_reward_multiplier

        done = False
        if self.graph.no_more_victims() or self.total_cost > 300:
            done = True
        return self.get_observation(), reward, done

    def step_unorganized(self, action):
        action_node = self.graph.nodes_list[action]
        reward = 0
        # print(action_node.id)
        if not any(action_node.id == n.id for n in self.graph.neighbors(self.curr_pos)):
            reward -= 100
            # print(action_node.id)
            print("he")
        else:
            # print(action)
            self.visit_node_sequence.append(action_node.id)
            edge_cost = self.graph.get_edge_cost(self.curr_pos, action_node)
            self.prev_pos = self.curr_pos
            self.curr_pos = action_node
            reward -= edge_cost
            self.total_cost += edge_cost
            if action_node.type == graph.NodeType.Victim:
                triage_cost, triage_score = self.graph.triage(action_node)
                reward -= triage_cost
                self.total_cost += triage_cost
                self.score += triage_score
                reward += triage_score * self.positive_reward_multiplier

        done = False
        if self.graph.no_more_victims() or self.total_cost > 1000:
            done = True
        return self.get_unorganized_observation_simplified(), reward, done

    def step_old(self, action):
        """ The global view
        :param performance: 0: navigate, 1: enter, 2: triage
        :param action_node: the index of the node for your performance
        :return: (observation, reward)
        """
        # assert performance in [0, 1, 2]

        performance, action_node_idx = action

        reward = 0
        action_node = self.graph.nodes_list[action_node_idx]

        triage_set = set()
        navigation_set = set()
        portal_enter_set = set()
        for n in self.graph.neighbors(self.curr_pos):
            if n.type == graph.NodeType.Portal:
                if self.curr_pos.type == graph.NodeType.Portal and \
                        n.is_same_portal(self.curr_pos):
                    portal_enter_set.add(n)
                else:
                    navigation_set.add(n)
            elif n.type == graph.NodeType.Victim:
                triage_set.add(n)
            elif n.type == graph.NodeType.Room:
                navigation_set.add(n)

        valid_action = True
        if performance == 0 and action_node not in navigation_set:
            valid_action = False
        if performance == 1 and action_node not in portal_enter_set:
            valid_action = False
        if performance == 2 and action_node not in triage_set:
            valid_action = False

        if not valid_action:
            reward -= 100
            # print("ha")
        else:
            # print(action)
            edge_cost = self.graph.get_edge_cost(self.curr_pos, action_node)
            self.prev_pos = self.curr_pos
            self.curr_pos = action_node
            reward -= edge_cost
            self.total_cost += edge_cost
            if action_node.type == graph.NodeType.Victim:
                triage_cost, triage_score = self.graph.triage(action_node)
                reward -= triage_cost
                self.total_cost += triage_cost
                self.score += triage_score
                reward += triage_score * self.positive_reward_multiplier

        done = False
        if self.graph.no_more_victims() or self.total_cost > 1000:
            done = True
        return self.get_observation(), reward, done

    def step_for_console_play(self, action):
        action_cost = self.graph.get_edge_cost(self.curr_pos, action)
        action_score = 0
        if action.type == graph.NodeType.Victim:
            triage_cost, triage_score = action.triage()
            action_cost += triage_cost
            action_score += triage_score
        self.total_cost += action_cost
        self.score += action_score
        self.prev_pos = self.curr_pos
        self.curr_pos = action

    def get_observation(self):
        """ Observation is an array of the following:
        [room_observed, portal_observed, victim_observed, device_info]
        all nodes are listed and set to 0 initially, if the agent is in neighbor to
        any of those nodes, the value is set to 1, for victims, types are indicated
        as (1, 2, 3, 4) being (green, yellow, safe, dead) respectively
        lets say the map has one portal_pair, two rooms, and one yellow victim
        [start, r1 | p0-start, p0-r1 | vy1] + [device] could be [0, 1, 1, 0, 1] + [0]
        :return: the observation array
        """
        room_observation = np.zeros(len(self.graph.room_list))
        portal_observation = np.zeros(len(self.graph.portal_list) * 2)
        victim_observation = np.zeros(len(self.graph.victim_list))

        for n in self.graph.get_neighbors(self.curr_pos):
            if n.type == graph.NodeType.Room:
                room_observation[self.graph.room_list.index(n)] = 1
            if n.type == graph.NodeType.Portal:
                # need to find the exact portal index since the portal list stores tuples of portals
                idx = None
                for pl_idx, pt in enumerate(self.graph.portal_list):
                    if n.id == pt[0].id:
                        idx = pl_idx * 2
                        break
                    elif n.id == pt[1].id:
                        idx = pl_idx * 2 + 1
                        break
                portal_observation[idx] = 1
            if n.type == graph.NodeType.Victim:
                victim_observation[self.graph.victim_list.index(n)] = 1 + n.victim_type.value
        # other_info = np.array([self.get_device_info(), self.total_cost, self.score])
        other_info = np.array([self.get_device_info()])

        return np.concatenate([room_observation, portal_observation, victim_observation, other_info])

    def get_unorganized_observation(self):
        node_observation = np.zeros(len(self.graph.nodes_list))
        for n in self.graph.get_neighbors(self.curr_pos):
            if n.type == graph.NodeType.Room or n.type == graph.NodeType.Portal:
                node_observation[self.graph.nodes_list.index(n)] = 1
            if n.type == graph.NodeType.Victim:
                node_observation[self.graph.nodes_list.index(n)] = 1 + n.victim_type.value
        # other_info = np.array([self.get_device_info(), self.total_cost, self.score])
        other_info = np.array([self.get_device_info()])
        return np.concatenate([node_observation, other_info])

    def get_unorganized_observation_simplified(self):
        node_observation = np.zeros(len(self.graph.nodes_list))
        for n in self.graph.get_neighbors(self.curr_pos):
            if n.type == graph.NodeType.Room or n.type == graph.NodeType.Portal:
                node_observation[self.graph.nodes_list.index(n)] = 1
            if n.type == graph.NodeType.Victim and \
                    (n.victim_type == graph.VictimType.Green or n.victim_type == graph.VictimType.Yellow):
                node_observation[self.graph.nodes_list.index(n)] = 1
        # other_info = np.array([self.get_device_info(), self.total_cost, self.score])
        device_num = self.get_device_info()
        if device_num == 0:
            device_info = [0, 0]
        elif device_num == 1:
            device_info = [1, 0]
        else:
            device_info = [1, 1]

        device_info = np.array(device_info)
        return np.concatenate([node_observation, device_info])

    def get_unorganized_observation_debug(self):
        node_observation = [(0, haha.id) for haha in self.graph.nodes_list]
        for n in self.graph.get_neighbors(self.curr_pos):
            if n.type == graph.NodeType.Room or n.type == graph.NodeType.Portal:
                node_observation[self.graph.nodes_list.index(n)] = (1, n.id)
            if n.type == graph.NodeType.Victim:
                node_observation[self.graph.nodes_list.index(n)] = (1 + n.victim_type.value, n.id)
        return node_observation


    def get_observation_debug(self):
        """ Debug observation, log out nodes name and value
        :return: the observation array with value and node id as tuple
        """
        room_observation = [(0, haha.id) for haha in self.graph.room_list]
        portal_observation = []
        for haha in self.graph.portal_list:
            portal_observation.append((0, haha[0].id))
            portal_observation.append((0, haha[1].id))
        victim_observation = [(0, haha.id) for haha in self.graph.victim_list]
        for n in self.graph.get_neighbors(self.curr_pos):
            if n.type == graph.NodeType.Room:
                room_observation[self.graph.room_list.index(n)] = (1, n.id)
            if n.type == graph.NodeType.Portal:
                # need to find the exact portal index since the portal list stores tuples of portals
                idx = None
                for pl_idx, pt in enumerate(self.graph.portal_list):
                    if n.id == pt[0].id:
                        idx = pl_idx * 2
                        break
                    elif n.id == pt[1].id:
                        idx = pl_idx * 2 + 1
                        break
                portal_observation[idx] = (1, n.id)
            if n.type == graph.NodeType.Victim:
                victim_observation[self.graph.victim_list.index(n)] = (1 + n.victim_type.value, n.id)
        device_info = np.array([self.get_device_info()])
        return room_observation + portal_observation + victim_observation

    def get_observation_old(self):
        """ (Discarded) Observation is an array of the following:
        [cur_pos, device_info, victim_1_state, victim_2_state, victim_3_state, ...]
        :return: the above array
        """
        cur_pos = self.graph.nodes_list.index(self.curr_pos)
        device_info = self.get_device_info()
        victim_states = [n.victim_type.value for n in self.graph.victim_list]
        return tuple(np.array([cur_pos] + [device_info] + victim_states))

    def get_action_space_for_console_play(self):
        victim_list = []
        portal_navigation_list = []
        portal_enter_list = []
        room_list = []

        victim_list_str = []
        portal_navigation_list_str = []
        portal_enter_list_str = []
        room_list_str = []
        for n in self.graph.neighbors(self.curr_pos):
            if n.type == graph.NodeType.Portal:
                if self.curr_pos.type == graph.NodeType.Portal and \
                        n.is_same_portal(self.curr_pos):
                    portal_enter_list.append(n)
                    act_str = "Enter Portal {} to Portal {}".format(str(self.curr_pos), str(n))
                    if n == self.prev_pos:
                        act_str += " [Go Back]"
                    portal_enter_list_str.append(act_str)
                else:
                    portal_navigation_list.append(n)
                    act_str = "Navigate to Portal {}".format(str(n))
                    if n == self.prev_pos:
                        act_str += " [Go Back]"
                    portal_navigation_list_str.append(act_str)
            elif n.type == graph.NodeType.Victim:
                victim_list.append(n)
                act_str = "Triage Victim {} ({})".format(str(n), n.get_type_str())
                if n == self.prev_pos:
                    act_str += " [Go Back]"
                victim_list_str.append(act_str)
            elif n.type == graph.NodeType.Room:
                room_list.append(n)
                act_str = "Enter Room Center {}".format(str(n))
                if n == self.prev_pos:
                    act_str += " [Go Back]"
                room_list_str.append(act_str)
        action_space = portal_navigation_list + portal_enter_list + victim_list + room_list
        action_space_str = portal_navigation_list_str + portal_enter_list_str + victim_list_str + room_list_str

        return action_space, action_space_str

    def get_device_info(self):
        # Nothing: 0,  Green: 1, Yellow: 2,
        if self.curr_pos.type == graph.NodeType.Portal:
            connected_room = self.graph.id2node[self.curr_pos.linked_portal.get_connected_room_id()]
            if self.graph.has_yellow_victim_in(connected_room):
                return 2
            elif self.graph.has_green_victim_in(connected_room):
                return 1
        return 0

    def get_device_info_for_console_play(self):
        if self.curr_pos.type == graph.NodeType.Portal:
            connected_room = self.graph.id2node[self.curr_pos.linked_portal.get_connected_room_id()]
            if self.graph.has_yellow_victim_in(connected_room):
                return "Beep-Beep"
            elif self.graph.has_green_victim_in(connected_room):
                return "Beep"
        return "Nothing"

    def console_play(self):
        while True:
            print("============================================\n")
            print("Your Current Position:", str(self.curr_pos))
            print("Your Previous Position:", str(self.prev_pos))
            print("Total Cost:", str(self.total_cost))
            print("Total Reward:", str(self.score))
            print("Device Info:", self.get_device_info_for_console_play())
            print()

            print(self.get_unorganized_observation())

            # for idx, obs in enumerate(self.get_observation()):
            #     print(str(obs), end=" ")
            #     if idx %5 == 0:
            #         print()

            action_space, action_space_str = self.get_action_space_for_console_play()
            print("Possible Actions:")
            print("\n".join(str(idx) + ": " + act_str for idx, act_str in enumerate(action_space_str)))
            act = input("Choose an Action: ")
            if act == "q":
                break
            print()
            chosen_action = action_space[int(act)]
            self.step_for_console_play(chosen_action)

if __name__ == '__main__':
    data_folder = Path("data")

    portals_csv = data_folder / "sparky_portals.csv"
    rooms_csv = data_folder / "sparky_rooms.csv"
    victims_csv = data_folder / "sparky_victims.csv"

    portal_data = pd.read_csv(portals_csv)
    room_data = pd.read_csv(rooms_csv)
    victim_data = pd.read_csv(victims_csv)

    env = AsistEnv(portal_data, room_data, victim_data, "as")
    env.console_play()
    # print(env.get_observation())
