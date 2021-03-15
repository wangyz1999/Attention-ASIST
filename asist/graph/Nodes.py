from enum import Enum

class NodeType(Enum):
    Portal = 0
    Victim = 1
    Room = 2

class VictimType(Enum):
    Green = 0
    Yellow = 1
    Safe = 2
    Dead = 3

class Node:
    """
        Super class Node in Graph, three subtypes of Node
        PortalNode, VictimNode, and RoomNode
    """
    def __init__(self, id, name):
        self.id = id
        self.name = name
        self.visited_count = 0

    def __str__(self):
        return self.id if self.name is None else self.id + "_" + self.name


class PortalNode(Node):
    """
        portal Node id has format PID-RID (Portal ID - Room ID)
        Portal also comes in pairs, linked_portal_id is the portal it is connecting
    """
    def __init__(self, id, name, location, is_open):
        super().__init__(id, name)
        self.type = NodeType.Portal
        self.loc = location
        self.is_open = is_open

    def link_portal(self, other):
        assert isinstance(other, PortalNode)
        self.linked_portal = other

    def get_connected_room_id(self):
        return self.id.split("|")[1]

    def is_same_portal(self, other):
        # return True if the two portals are linked
        assert isinstance(other, PortalNode)
        return other.id.split("|")[0] == self.id.split("|")[0]

    def open_portal(self):
        self.is_open = True
        self.linked_portal.is_open = True

    def close_portal(self):
        self.is_open = False
        self.linked_portal.is_open = False

class VictimNode(Node):
    def __init__(self, id, name, victim_type, location):
        assert isinstance(victim_type, VictimType)
        super().__init__(id, name)
        self.type = NodeType.Victim
        self.victim_type = victim_type
        self.victim_type_original = victim_type
        self.loc = location

    def yellow_death(self):
        # Simulate the death of yellow victims, turns them to red
        assert self.victim_type == VictimType.Yellow, \
            "Only Yellow Victim could die, please check the victim type"
        self.victim_type = VictimType.Dead

    def triage(self):
        """ Simulate the victim triage process, can only triage Yellow and Green
        :return: A tuple of time cost and score reward (Time Cost, Reward)
        """
        if self.victim_type == VictimType.Safe or self.victim_type == VictimType.Dead:
            return 0, 0
        # TODO: Need to discuss cost and reward here
        elif self.victim_type == VictimType.Green:
            self.victim_type = VictimType.Safe
            return 7, 10
        else: # A Yellow Victim
            self.victim_type = VictimType.Safe
            return 15, 30

    def get_type_str(self):
        if self.victim_type == VictimType.Green:
            return "Green"
        if self.victim_type == VictimType.Yellow:
            return "Yellow"
        if self.victim_type == VictimType.Safe:
            return "Safe"
        if self.victim_type == VictimType.Dead:
            return "Dead"

class RoomNode(Node):
    def __init__(self, id, name, location, victims):
        super().__init__(id, name)
        self.type = NodeType.Room
        self.loc = location
        self.victim_list = victims if victims is not None else []
        self.light_on = False

    def add_victim(self, victim_id):
        assert isinstance(victim_id, str) or isinstance(victim_id, list) and \
               all(isinstance(v, str) for v in victim_id)
        self.victim_list.append(victim_id)

    def turn_light_on(self):
        self.light_on = True

    def turn_light_off(self):
        self.light_on = False

