import networkx as nx
from .Nodes import *
from collections.abc import Iterable
import math
import numpy as np
import random


class Graph(nx.Graph):
    """
        A networkx Graph
    """
    def __init__(self):
        super(Graph, self).__init__()
        self.nodes_list = []
        self.ordered_node_list = []

        self.room_list = []
        self.portal_list = []
        self.victim_list = []

        self.green_victim_list = []
        self.yellow_victim_list = []
        self.safe_victim_list = []
        self.dead_victim_list = []

        self.id2node = {}

        self.victimType2list = {
            VictimType.Green: self.green_victim_list,
            VictimType.Yellow: self.yellow_victim_list,
            VictimType.Safe: self.safe_victim_list,
            VictimType.Dead: self.dead_victim_list,
        }

    def __getitem__(self, id_key):
        """ Get the node based on id.
            Usage: graph["R202"] returns the node Room-202
        :param id_key: the id of the node you want
        :return: the actual node
        """
        return self.id2node[id_key]

    def make_ordered_node_list(self):
        for n in self.room_list:
            self.ordered_node_list.append(n)
        for n in self.portal_list:
            self.ordered_node_list.append(n[0])
            self.ordered_node_list.append(n[1])
        for n in self.victim_list:
            self.ordered_node_list.append(n)

    def add_victim_to_ordered_node_list(self):
        for n in self.victim_list:
            self.ordered_node_list.append(n)

    def reset(self):
        for vn in self.victim_list:
            vn.victim_type = vn.victim_type_original

        self.green_victim_list.clear()
        self.yellow_victim_list.clear()
        self.safe_victim_list.clear()
        self.dead_victim_list.clear()

        for victim in self.victim_list:
            self.victimType2list[victim.victim_type].append(victim)

        for node in self.nodes_list:
            node.visited_count = 0

    def add_victim(self, victim_type, id=None, name=None, location=None):
        """ Register a victim node to graph and append the corresponding lists

        :param id: the victim id, if id not give, the method will auto generate one
        :param name: the name of the Victim such as Jason. (Default None)
        :param victim_type: Must be one of [Yellow, Green, Dead, Safe]
        :param location: location of the victim, tuple of x,z coordinate
        :return: the victim node constructed
        """
        assert id is None or isinstance(id, str)
        assert name is None or isinstance(name, str)
        assert isinstance(victim_type, VictimType)
        assert location is None or isinstance(location, tuple) and len(location) == 2 \
               and all(isinstance(l, float) or isinstance(l, int) for l in location)

        node_id = "V"

        if victim_type == VictimType.Green:
            node_id = "G" + str(len(self.green_victim_list))
        elif victim_type == VictimType.Yellow:
            node_id = "Y" + str(len(self.yellow_victim_list))
        elif victim_type == VictimType.Safe:
            node_id = "S" + str(len(self.safe_victim_list))
        elif victim_type == VictimType.Dead:
            node_id = "D" + str(len(self.dead_victim_list))

        if id is not None:
            node_id = id

        node = VictimNode(node_id, name, victim_type, location)
        self.victimType2list[victim_type].append(node)
        self.victim_list.append(node)
        self.nodes_list.append(node)
        self.id2node[node_id] = node

        self.add_node(node)

        return node

    def add_portal(self, connected_room_ids, is_open=False, id=None, name=None, location=None):
        """ Add portal (pair)

        :param id: the portal id, if id not give, the method will auto generate one
        :param name: name of the portal, if any
        :param connected_room: the room that the portal is connected to
        :param location: location of the portal, tuple of x,z coordinate
        :return: the created portal node
        """
        assert id is None or isinstance(id, str)
        assert name is None or isinstance(name, str)
        assert location is None or isinstance(location, tuple) and len(location) == 2 \
               and all(isinstance(l, float) or isinstance(l, int) for l in location)
        assert isinstance(connected_room_ids, tuple) and all(isinstance(r, str) for r in connected_room_ids)

        node_id = id if id is not None else "P" + str(len(self.portal_list))

        node_id_1 = node_id + "|" + connected_room_ids[0]
        node_id_2 = node_id + "|" + connected_room_ids[1]

        node_1 = PortalNode(node_id_1, name, location, is_open)
        node_2 = PortalNode(node_id_2, name, location, is_open)
        node_1.link_portal(node_2)
        node_2.link_portal(node_1)
        self.add_edge(node_1, node_2, weight=1)

        self.portal_list.append((node_1, node_2))
        self.nodes_list.append(node_1)
        self.nodes_list.append(node_2)
        self.id2node[node_id_1] = node_1
        self.id2node[node_id_2] = node_2

        return node_1, node_2


    def add_room(self, id=None, name=None, location=None, victims=None):
        """ Add Room Node
        :param id: the room id, if id not give, the method will auto generate one
        :param name: name of the room, if any
        :param location: location of the center of the room, tuple of x,z coordinate
        :param victims: None or a list of victim id string
        :return: the created room node
        """
        assert id is None or isinstance(id, str)
        assert name is None or isinstance(name, str)
        assert victims is None or isinstance(victims, list) and \
               all(v is None or isinstance(v, str) for v in victims)
        assert location is None or isinstance(location, tuple) and len(location) == 2 \
               and all(isinstance(l, float) or isinstance(l, int) for l in location)

        node_id = id if id is not None else "R" + str(len(self.room_list))
        node = RoomNode(node_id, name, location, victims)

        self.room_list.append(node)
        self.nodes_list.append(node)
        self.id2node[node_id] = node

        return node

    def link_victims_in_room(self, room, list_of_victim_id, random_cost=None):
        """ The First Linkage Function to run
        Make a fully connected sub-graph of room nodes and victims node inside that room

        :param room: the room Node
        :param list_of_victim_id: the list of victim ids inside the room
        :param random_cost: None if cost based on loc, (min, max) if random
        :return: the room node
        """
        assert isinstance(room, RoomNode)
        assert isinstance(list_of_victim_id, list) and all(isinstance(v, str) for v in list_of_victim_id)
        assert random_cost is None or isinstance(random_cost, tuple) and len(random_cost) == 2 and \
               all(isinstance(n, int) for n in random_cost) and random_cost[0] >= 1

        for v_id in list_of_victim_id:
            victim = self.id2node[v_id]
            if random_cost is None:
                self.add_edge(room, victim, weight=self.euclidean_distances(room.loc, victim.loc))
            else:
                self.add_edge(room, victim, weight=random.randint(random_cost[0], random_cost[1]))

        for i in range(len(list_of_victim_id)):
            for j in range(i+1, len(list_of_victim_id)):
                victim_1 = self.id2node[list_of_victim_id[i]]
                victim_2 = self.id2node[list_of_victim_id[j]]
                if random_cost is None:
                    self.add_edge(victim_1, victim_2, weight=self.euclidean_distances(victim_1.loc, victim_2.loc))
                else:
                    self.add_edge(victim_1, victim_2, weight=random.randint(random_cost[0], random_cost[1]))
        return room

    def connect_rooms_by_extension(self, extension):
        rooms = extension[2]
        for i in range(len(rooms)-1):
            for j in range(i+1, len(rooms)):
                room_1 = self.id2node[rooms[i]]
                room_2 = self.id2node[rooms[j]]
                dist1 = self.euclidean_distances(room_1.loc, extension[1])
                dist2 = self.euclidean_distances(room_2.loc, extension[1])
                self.add_edge(room_1, room_2, weight=dist1+dist2)


    def connect_portal_to_rooms(self, portal_tuple, random_cost=None):
        """ The second Linkage Function to run
        Connect the portal to the two rooms it is adjacent to
        :param portal_tuple: the two portals indicate two sides of the door
        """
        assert isinstance(portal_tuple, tuple) and len(portal_tuple) == 2 and \
               all(isinstance(p, PortalNode) for p in portal_tuple)
        assert random_cost is None or isinstance(random_cost, tuple) and len(random_cost) == 2 and \
               all(isinstance(n, int) for n in random_cost) and random_cost[0] >= 1

        portal_1, portal_2 = portal_tuple

        # connecting portal with the two adjacent rooms
        room_1 = self.id2node[portal_1.get_connected_room_id()]
        room_2 = self.id2node[portal_2.get_connected_room_id()]
        if random_cost is None:
            self.add_edge(portal_1, room_1, weight=self.euclidean_distances(room_1.loc, portal_1.loc))
            self.add_edge(portal_2, room_2, weight=self.euclidean_distances(room_2.loc, portal_2.loc))
        else:
            self.add_edge(portal_1, room_1, weight=random.randint(random_cost[0], random_cost[1]))
            self.add_edge(portal_2, room_2, weight=random.randint(random_cost[0], random_cost[1]))

        # connecting portal with all the victims in side the adjacent room
        for v_id in room_1.victim_list:
            victim = self.id2node[v_id]
            if random_cost is None:
                self.add_edge(portal_1, victim, weight=self.euclidean_distances(room_1.loc, victim.loc))
            else:
                self.add_edge(portal_1, victim, weight=random.randint(random_cost[0], random_cost[1]))

        for v_id in room_2.victim_list:
            victim = self.id2node[v_id]
            if random_cost is None:
                self.add_edge(portal_2, victim, weight=self.euclidean_distances(room_2.loc, victim.loc))
            else:
                self.add_edge(portal_2, victim, weight=random.randint(random_cost[0], random_cost[1]))

    def connected_portals_to_portals(self, portal_tuple, random_cost=None):
        """ The third Linkage Function to run
        Connect the portal to portals that is connected with the room it is adjacent to
        :param portal_tuple: the two portals indicate two sides of the door
        """
        assert isinstance(portal_tuple, tuple) and len(portal_tuple) == 2 and \
               all(isinstance(p, PortalNode) for p in portal_tuple)
        assert random_cost is None or isinstance(random_cost, tuple) and len(random_cost) == 2 and \
               all(isinstance(n, int) for n in random_cost) and random_cost[0] >= 1

        portal_1, portal_2 = portal_tuple

        # Get the two adjacent rooms
        room_1 = self.id2node[portal_1.get_connected_room_id()]
        room_2 = self.id2node[portal_2.get_connected_room_id()]

        for n in self.get_neighbors(room_1):
            if n.type == NodeType.Portal and n != portal_1 and not self.has_edge(n, portal_1):
                if random_cost is None:
                    self.add_edge(portal_1, n, weight=self.euclidean_distances(portal_1.loc, n.loc))
                else:
                    self.add_edge(portal_1, n, weight=random.randint(random_cost[0], random_cost[1]))

        for n in self.get_neighbors(room_2):
            if n.type == NodeType.Portal and n != portal_2 and not self.has_edge(n, portal_2):
                if random_cost is None:
                    self.add_edge(portal_2, n, weight=self.euclidean_distances(portal_2.loc, n.loc))
                else:
                    self.add_edge(portal_2, n, weight=random.randint(random_cost[0], random_cost[1]))

    def get_neighbors(self, node):
        # get neighbor nodes of a node
        return self.neighbors(node)

    def get_edge_cost(self, node1, node2):
        # get the edge weight between two nodes
        return self.get_edge_data(node1, node2)["weight"]

    def close_all_portal(self):
        # make all portals close
        for portal_pair in self.portal_list:
            portal_pair[0].close_portal()

    def open_all_portal(self):
        # make all portals open
        for portal_pair in self.portal_list:
            portal_pair[0].open_portal()

    def triage(self, victim):
        assert isinstance(victim, VictimNode)
        if victim.victim_type == VictimType.Safe or victim.victim_type == VictimType.Dead:
            return 0, 0
        # return cost, reward
        elif victim.victim_type == VictimType.Green:
            self.green_victim_list.remove(victim)
            victim.victim_type = VictimType.Safe
            self.safe_victim_list.append(victim)
            return 7.5, 10
        else: # A Yellow Victim
            self.yellow_victim_list.remove(victim)
            victim.victim_type = VictimType.Safe
            self.safe_victim_list.append(victim)
            return 15, 30

    def no_more_victims(self):
        if len(self.yellow_victim_list) == 0 and len(self.green_victim_list) == 0:
            return True
        return False

    def remove_all_victims(self):
        for node in self.victim_list:
            self.remove_node(node)
            self.nodes_list.remove(node)
            self.ordered_node_list.remove(node)
        self.victim_list.clear()
        self.yellow_victim_list.clear()
        self.green_victim_list.clear()
        self.safe_victim_list.clear()
        self.dead_victim_list.clear()
        for node in self.room_list:
            node.victim_list.clear()
        for node in self.nodes_list:
            node.visited_count = 0

    def kill_all_yellow_victims(self):
        # kill all yellow victims by turning them VictimType.Dead
        for yellow_victim in self.yellow_victim_list:
            yellow_victim.yellow_death()
        self.dead_victim_list += self.yellow_victim_list
        self.yellow_victim_list = []

    def has_yellow_victim_in(self, room):
        # whether a room contains any yellow victim, method used for the device
        assert isinstance(room, RoomNode)
        return any(self.id2node[n].victim_type == VictimType.Yellow for n in room.victim_list)

    def has_green_victim_in(self, room):
        # whether a room contains any green victim, method used for the device
        assert isinstance(room, RoomNode)
        return any(self.id2node[n].victim_type == VictimType.Green for n in room.victim_list)

    def add_victim_blocking_rubble(self, victim_id, rubbles):
        for rubble in rubbles:
            assert isinstance(rubble, tuple)
            self[victim_id].blocking_rubbles.append(rubble)

    def add_room_blocking_rubble(self, room_id, rubbles):
        for rubble in rubbles:
            assert isinstance(rubble, tuple)
            self[room_id].blocking_rubbles.append(rubble)

    def better_layout(self, with_spring=False, portal_sep=1.5, fix_portal=True, expand_iteration=20, expand_radius=2.5, shift_dist=0.1):
        """ Make the map layout adhere to the original coordinate layout
        :param with_spring: Experimental, whether to use the networkx spring layout
        :param portal_sep: the separation distance for portal pairs
        :param expand_iteration: the number of iterations to perform the node expansion, set 0 if no expansion needed
        :param expand_radius: the radius of judging how compact the node is
        :param shift_dist: the distance to shift if nodes are too compact
        :return: the graph layout dictionary, and the nodes that are fixed if with_spring is True
        """
        assert isinstance(with_spring, bool)
        assert isinstance(portal_sep, int) or isinstance(portal_sep, float)
        assert isinstance(expand_iteration, int) and expand_iteration >= 0
        assert isinstance(expand_radius, int) or isinstance(expand_radius, float)
        assert isinstance(shift_dist, int) or isinstance(shift_dist, float)
        layout_dict = dict()
        fix_node = list()
        for node in self.nodes_list:
            loc = np.array([node.loc[0], node.loc[1]],dtype=np.float64)
            layout_dict[node] = loc

        if with_spring:
            for node in self.nodes_list:
                fix = True
                for nei in self.neighbors(node):
                    if self.euclidean_distances(node.loc, nei.loc) < 2:
                        fix = False
                if fix:
                    fix_node.append(node)

        # expand the collided and compacted nodes
        for iter in range(expand_iteration):
            for node in self.nodes_list:
                if fix_portal and node.type == NodeType.Portal:
                    continue
                nodes_in_range = []
                for nb in self.get_neighbors(node):
                    if self.euclidean_distances(tuple(layout_dict[nb]), tuple(layout_dict[node])) < expand_radius:
                        nodes_in_range.append(nb)
                # find the centroid of all the neighbor nodes in range
                x = [n.loc[0] for n in nodes_in_range]
                z = [n.loc[1] for n in nodes_in_range]
                if len(nodes_in_range) == 0:
                    continue
                else:
                    centroid = (sum(x) / len(nodes_in_range), sum(z) / len(nodes_in_range))
                new_pos = self.shift_distance(-shift_dist, tuple(layout_dict[node]), centroid)
                layout_dict[node] = np.array(new_pos)


        # separate overlapping portal
        for portal_pair in self.portal_list:
            portal_1, portal_2 = portal_pair
            room_1 = self.id2node[portal_1.get_connected_room_id()]
            room_2 = self.id2node[portal_2.get_connected_room_id()]
            dist_1 = self.euclidean_distances(portal_1.loc, room_1.loc)
            dist_2 = self.euclidean_distances(portal_2.loc, room_2.loc)
            if dist_1 > dist_2:
                pos_1 = self.shift_distance(portal_sep, portal_1.loc, room_1.loc)
                layout_dict[portal_1] = np.array(pos_1)
            else:
                pos_2 = self.shift_distance(portal_sep, portal_2.loc, room_2.loc)
                layout_dict[portal_2] = np.array(pos_2)


        return layout_dict, fix_node

    def better_color(self, curr_node=None):
        """ Color the nodes based on their types
        :return: the color map used for plotting
        """
        color_map = []
        for node in self:
            if node.id == curr_node:
                color_map.append('red')
                continue
            if node.type == NodeType.Victim:
                if node.victim_type == VictimType.Green:
                    color_map.append('limegreen')
                if node.victim_type == VictimType.Yellow:
                    color_map.append('yellow')
                if node.victim_type == VictimType.Dead:
                    color_map.append('tomato')
                if node.victim_type == VictimType.Safe:
                    color_map.append('silver')
            if node.type == NodeType.Portal:
                color_map.append('lightskyblue')
            if node.type == NodeType.Room:
                color_map.append('violet')
        return color_map

    def flip_z(self, pos):
        # flip the graph along z-axis
        assert isinstance(pos, dict)
        for p in pos:
            pos[p][0] *= -1
        return pos

    def flip_x(self, pos):
        # flip the graph along x-axis
        assert isinstance(pos, dict)
        for p in pos:
            pos[p][1] *= -1
        return pos

    def clockwise90(self, pos):
        # rotate the graph clock-wise 90 degrees
        assert isinstance(pos, dict)
        for p in pos:
            pos[p][0], pos[p][1] = pos[p][1], -pos[p][0]
        return pos

    @staticmethod
    def euclidean_distances(pos1, pos2):
        # calculate the euclidean distance between two nodes based the coordinate position
        # the distance is at least 1
        assert isinstance(pos1, tuple) and isinstance(pos2, tuple)
        return max(1, math.ceil(math.sqrt((pos2[0] - pos1[0]) ** 2 + (pos2[1] - pos1[1]) ** 2)))

    def shift_distance(self, shift, pos1, pos2):
        """ Shift pos1 the distance "shift" to pos2
        :param shift: the distance want to shift
        :return: pos1 after shift
        """
        assert isinstance(shift, int) or isinstance(shift, float)
        assert isinstance(pos1, tuple) and isinstance(pos2, tuple)

        dist = self.euclidean_distances(pos1, pos2)
        if shift >= dist:
            return pos1

        ratio = shift / dist
        new_x = pos1[0] + ratio * (pos2[0] - pos1[0])
        new_z = pos1[1] + ratio * (pos2[1] - pos1[1])
        return new_x, new_z

class RandomGraphGenerator:
    @staticmethod
    def circle_location(radius, loc):
        x, y = loc[0], loc[1]
        alpha = 2 * math.pi * random.random()
        new_x = radius * math.cos(alpha) + x
        new_y = radius * math.sin(alpha) + y
        return new_x, new_y

    @classmethod
    def add_random_victims(cls, G, no_victim_rooms, total_green=19, total_yellow=7, room_num_limit=2, rand_distance=5):
        assert isinstance(G, Graph)

        room_green_count = dict()
        room_yellow_count = dict()
        for r in G.room_list:
            if r.id not in no_victim_rooms:
                room_green_count[r.id] = 0
                room_yellow_count[r.id] = 0
        for i in range(total_green):
            room = random.choice(list(room_green_count.keys()))
            while room_green_count[room] == room_num_limit:
                room = random.choice(list(room_green_count.keys()))
            room_green_count[room] += 1
        for i in range(total_yellow):
            room = random.choice(list(room_yellow_count.keys()))
            while room_yellow_count[room] == room_num_limit:
                room = random.choice(list(room_yellow_count.keys()))
            room_yellow_count[room] += 1

        for green_room in room_green_count:
            for num_in_room in range(room_green_count[green_room]):
                random_distance = random.randint(1, rand_distance)
                green_victim = G.add_victim(VictimType.Green, location=RandomGraphGenerator.circle_location(random_distance, G[green_room].loc))
                G[green_room].add_victim(green_victim.id)

        for yellow_room in room_yellow_count:
            for num_in_room in range(room_yellow_count[yellow_room]):
                random_distance = random.randint(1, rand_distance)
                yellow_victim = G.add_victim(VictimType.Yellow, location=RandomGraphGenerator.circle_location(random_distance, G[yellow_room].loc))
                G[yellow_room].add_victim(yellow_victim.id)

        for room in G.room_list:
            G.link_victims_in_room(room, room.victim_list)

        for portal_pair in G.portal_list:
            G.connect_portal_to_rooms(portal_pair)

        for portal_pair in G.portal_list:
            G.connected_portals_to_portals(portal_pair)

        G.add_victim_to_ordered_node_list()

        return G


    @classmethod
    def generate_random_graph(cls, num_of_rooms, edge_weight_range, green_range, yellow_range,
                              portal_state="random", open_ratio=0.5, light_state="random"):
        assert isinstance(edge_weight_range, tuple) and len(edge_weight_range) == 2 and \
               all(isinstance(n, int) for n in edge_weight_range) and edge_weight_range[0] >= 1
        assert isinstance(green_range, tuple) and len(green_range) == 2 and \
               all(isinstance(n, int) for n in green_range) and green_range[0] >= 0
        assert isinstance(yellow_range, tuple) and len(yellow_range) == 2 and \
               all(isinstance(n, int) for n in yellow_range) and yellow_range[0] >= 0
        assert portal_state in ["random", "open", "close"]
        assert light_state in ["random", "open", "close"]
        assert open_ratio is None or isinstance(open_ratio, float) and 0 < open_ratio < 1
        G = Graph()
        guide = nx.connected_watts_strogatz_graph(num_of_rooms, 3, 0.5)
        guide_dict = dict()

        # add rooms according to guide, add victims in, fully connect them
        for node in guide.nodes:
            temp_victim_list = []
            for g_v in range(random.randint(green_range[0], green_range[1])):
                victim = G.add_victim(VictimType.Green)
                temp_victim_list.append(victim.id)
            for g_v in range(random.randint(yellow_range[0], yellow_range[1])):
                victim = G.add_victim(VictimType.Yellow)
                temp_victim_list.append(victim.id)
            room = G.add_room(victims=temp_victim_list)
            guide_dict[node] = room
            G.link_victims_in_room(room, temp_victim_list, edge_weight_range)

        # add portals in between edges
        for edge in guide.edges:
            room_1 = guide_dict[edge[0]]
            room_2 = guide_dict[edge[1]]
            if portal_state == "open":
                portal_tuple = G.add_portal((room_1.id, room_2.id), is_open=True)
                G.connect_portal_to_rooms(portal_tuple, edge_weight_range)
            elif portal_state == "close":
                portal_tuple = G.add_portal((room_1.id, room_2.id), is_open=False)
                G.connect_portal_to_rooms(portal_tuple, edge_weight_range)
            else:
                if random.random() < open_ratio:
                    portal_tuple = G.add_portal((room_1.id, room_2.id), is_open=True)
                    G.connect_portal_to_rooms(portal_tuple, edge_weight_range)
                else:
                    portal_tuple = G.add_portal((room_1.id, room_2.id), is_open=False)
                    G.connect_portal_to_rooms(portal_tuple, edge_weight_range)

        # Connect portal with portals
        for portal_tuple in G.portal_list:
            G.connected_portals_to_portals(portal_tuple, edge_weight_range)

        # add the Start Node
        G.add_room(id="Start")
        start_portal_tuple = G.add_portal(("Start", "R0"), True)
        G.connect_portal_to_rooms(start_portal_tuple, edge_weight_range)
        G.connected_portals_to_portals(start_portal_tuple, edge_weight_range)

        return G