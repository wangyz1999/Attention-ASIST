import asist.graph as graph

import pandas as pd
import numpy as np

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
    def parse_json_map_data(cls, data):
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