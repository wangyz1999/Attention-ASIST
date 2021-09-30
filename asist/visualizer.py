from mapparser import MapParser
import matplotlib.pyplot as plt
import matplotlib.animation
import pandas as pd
import networkx as nx
from graph import RandomGraphGenerator
from graph.Nodes import NodeType
import json
import graph


def plot_graph(graph, save=None, hide_portal_label=False, pop_out=False):
    """ Plot the semantic graph in png form.
    :param graph: the input semantic graph
    :param save: the filename to save
    :param hide_portal_label: if True, the plotted graph will hide portal label
    :param pop_out: whether to use the matplotlib TkAgg display style
    :return: the actual node
    """
    if pop_out:
        matplotlib.use("TkAgg")
    # Get position
    pos, fix = graph.better_layout(expand_iteration=3)
    pos = graph.flip_z(pos)
    pos = graph.clockwise90(pos)
    pos = graph.clockwise90(pos)
    weight_labels = nx.get_edge_attributes(graph,'weight')
    # plt.figure(figsize=(18,27))
    plt.figure(figsize=(16,10), dpi=282)
    plt.axis("off")
    plt.tight_layout()
    color_map = graph.better_color()

    if hide_portal_label:
        label_dict = {}
        for nodes in graph.nodes():
            if nodes.id[:2] == 'c_':
                label_dict[nodes] = ""
            else:
                label_dict[nodes] = nodes

        nx.draw_networkx(graph, pos, labels=label_dict, with_labels=True, node_color=color_map, node_size=40,
                         font_size=4, width=0.5)
    else:
        nx.draw_networkx(graph, pos, with_labels=True, node_color=color_map, node_size=40,
                        font_size=4, width=0.5)
    # nx.draw_networkx_edge_labels(graph, pos, edge_labels=weight_labels)

    if save is None:
        plt.show()
    else:
        plt.savefig(save + ".png", bbox_inches='tight', pad_inches=0)


def plot_random_graph():
    """
    (Deprecated) Testing plot random graph
    """
    plt.figure(figsize=(8,8))
    graph = RandomGraphGenerator.generate_random_graph(5, (2,8), (0,2), (0,1))
    pos = nx.kamada_kawai_layout(graph)
    weight_labels = nx.get_edge_attributes(graph,'weight')
    color_map = graph.better_color()
    nx.draw(graph, pos, with_labels=True, node_color=color_map, edge_labels=weight_labels)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=weight_labels)
    plt.show()


def animate_graph_training(animation_sequence, portal_data, room_data, victim_data, with_save=False):
    """
        (Deprecated) Animate Route with animation sequence for Sparky map
    """
    # matplotlib.use("Qt5Agg")
    matplotlib.use("TkAgg")
    graph = MapParser.parse_map_data(portal_data, room_data, victim_data)
    # Get position
    pos, fix = graph.better_layout()
    pos = graph.flip_z(pos)
    pos = graph.clockwise90(pos)
    # weight_labels = nx.get_edge_attributes(graph,'weight')

    # Animation update function
    def update(animation_frame):
        # if animation_frame[0] == len(animation_sequence) - 1:
        #     plt.close(fig)

        ax.clear()
        curr_node = animation_frame[0]
        if graph[curr_node].type == NodeType.Victim:
            cost, reward = graph.triage(graph[curr_node])

        color_map = graph.better_color(curr_node)
        nx.draw(graph, pos, with_labels=False, node_color=color_map, node_size=50,
                font_size=7, width=0.5)
        # nx.draw_networkx_edge_labels(graph, pos, edge_labels=weight_labels, font_size=7)
        ax.set_title(f"Time: {animation_frame[1]:.2f}  Score: {animation_frame[2]}")
        # ax.set_title("Steps: " + str(animation_frame[0]))

    fig, ax = plt.subplots(figsize=(6,6))
    fig.subplots_adjust(left=0, bottom=0, right=1, top=0.9, wspace=None, hspace=None)
    ani = matplotlib.animation.FuncAnimation(fig, update, frames=animation_sequence, interval=200, repeat=False)
    if not with_save:
        plt.show()
    else:
        ani.save('evaluation_20.mp4')
    # del ani
    # del fig

def animate_graph_training_json(animation_sequence, json_data, with_save=None):
    """
        (Deprecated) Animate Route with animation sequence for Saturn map
    """
    # matplotlib.use("Qt5Agg")
    matplotlib.use("TkAgg")
    graph = MapParser.parse_json_map_data_new_format(json_data)
    # Get position
    pos, fix = graph.better_layout()
    pos = graph.flip_z(pos)
    pos = graph.clockwise90(pos)
    # weight_labels = nx.get_edge_attributes(graph,'weight')

    # Animation update function
    def update(animation_frame):
        # if animation_frame[0] == len(animation_sequence) - 1:
        #     plt.close(fig)

        ax.clear()
        curr_node = animation_frame
        if graph[curr_node].type == NodeType.Victim:
            cost, reward = graph.triage(graph[curr_node])

        color_map = graph.better_color(curr_node)
        nx.draw(graph, pos, with_labels=False, node_color=color_map, node_size=50,
                font_size=7, width=0.5)
        # nx.draw_networkx_edge_labels(graph, pos, edge_labels=weight_labels, font_size=7)
        # ax.set_title(f"Time: {animation_frame[1]:.2f}  Score: {animation_frame[2]}")
        # ax.set_title("Steps: " + str(animation_frame[0]))

    fig, ax = plt.subplots(figsize=(6,6))
    fig.subplots_adjust(left=0, bottom=0, right=1, top=0.9, wspace=None, hspace=None)
    ani = matplotlib.animation.FuncAnimation(fig, update, frames=animation_sequence, interval=200, repeat=False)
    if with_save is None:
        plt.show()
    else:
        ani.save(f'{with_save}.mp4')
    # del ani
    # del fig

def animate_MIP_graph(animation_sequence, json_data, with_save=None):
    """
        (Deprecated) Animate Route discovered by Mixed Integer Programming
    """
    # matplotlib.use("Qt5Agg")
    matplotlib.use("TkAgg")
    graph = MapParser.parse_json_map_data_new_format(json_data)
    # Get position
    pos, fix = graph.better_layout()
    pos = graph.flip_z(pos)
    pos = graph.clockwise90(pos)
    pos = graph.clockwise90(pos)
    # weight_labels = nx.get_edge_attributes(graph,'weight')
    # Animation update function
    def update(animation_frame):
        # if animation_frame[0] == len(animation_sequence) - 1:
        #     plt.close(fig)

        ax.clear()
        do_triage = False
        curr_node = animation_frame[0]
        if '@' in curr_node:
            do_triage = True
            curr_node = curr_node[1:]
        if do_triage:
            cost, reward = graph.triage(graph[curr_node])

        color_map = graph.better_color(curr_node)
        nx.draw(graph, pos, with_labels=False, node_color=color_map, node_size=50,
                font_size=7, width=0.5)
        # nx.draw_networkx_edge_labels(graph, pos, edge_labels=weight_labels, font_size=7)
        print(animation_frame)
        ax.set_title(f"Time: {animation_frame[1]:.2f}  Score: {animation_frame[2]}")
        # ax.set_title("Steps: " + str(animation_frame[0]))

    fig, ax = plt.subplots(figsize=(6,6))
    fig.subplots_adjust(left=0, bottom=0, right=1, top=0.9, wspace=None, hspace=None)
    ani = matplotlib.animation.FuncAnimation(fig, update, frames=animation_sequence, interval=200, repeat=False)
    if with_save is None:
        plt.show()
    else:
        ani.save(f'{with_save}.mp4')

def simulate_run(g, path):
    """
        (Deprecated) Generate Animation Sequence data from path ids
    """
    points = [0]
    times = [0]
    prev = 'ew_1'
    # agent_speed = 4.3 # full walk
    # agent_speed = 5.0 # walk + sprint
    # agent_speed = 5.6  # full sprint
    agent_speed = 5.6  # tmp
    # agent_speed = 7.3 # full sprint + jump
    for curr in path:
        point, time = points[-1], times[-1]
        do_triage = False
        if '@' in curr:
            do_triage = True
            curr = curr[1:]
        if do_triage:
            if g[curr].victim_type == graph.VictimType.Yellow:
                point += 30
                time += 15
            elif g[curr].victim_type == graph.VictimType.Green:
                point += 10
                time += 7
        points.append(point)
        print(prev, curr)
        time += g.get_edge_cost(g[prev], g[curr]) / agent_speed
        times.append(time)
        prev = curr
        assert isinstance(time, float)
    return list(zip(['ew_1']+path, times, points))


if __name__ == '__main__':
    with open('data/json/Saturn/Saturn_trial_416.json') as f:
        data = json.load(f)

    graph = MapParser.parse_saturn_map(data)

    max_count = -1
    for n in graph.nodes_list:
        count = 0
        for i in graph.get_neighbors(n):
            if i.type == NodeType.Room:
                count += 1
        max_count = max(max_count, count)
    print(len(graph.nodes_list))

    plot_graph(graph, save="Semantic_Graph_png/Saturn_trial_416", hide_portal_label=True)

    # animate_graph()
    # plot_random_graph()

