from visualizer import *
import matplotlib.pyplot as plt
import matplotlib.image as mpllimg
from asist.get_attention_problem import *
from problems import PCVRP
from ipynb.plot_pcvrp import plot_vehicle_routes_paper
import torch
from matplotlib.lines import Line2D

if __name__ == '__main__':
    with open('data/json/Saturn/Saturn_trial_416_paper.json') as f:
        data = json.load(f)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15), dpi=130)
    # axs[0, 0].plot(x, y)
    # axs[1, 1].scatter(x, y)

    img = mpllimg.imread('ab-lv-1.png')
    img_plot = ax1.imshow(img)
    ax1.axis("off")

    ax1.set_title('a) Level 1: The Original Saturn Map', fontsize=16)

    # ax1.axes.xaxis.set_ticks([])
    # ax1.axes.yaxis.set_ticks([])

    # env = AsistEnvGym(portal_data, room_data, victim_data, "as", random_victim=False)

    graph = MapParser.parse_saturn_map(data)

    max_count = -1
    for n in graph.nodes_list:
        count = 0
        for i in graph.get_neighbors(n):
            if i.type == NodeType.Room:
                count += 1
        max_count = max(max_count, count)
    print(len(graph.nodes_list))

    plot_paper_graph(graph, ax2, hide_portal_label=True)
    ax2.set_title('b) Level 2: The Semantic Map', fontsize=16)
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Normal Victims', markerfacecolor='limegreen', markersize=15),
                       Line2D([0], [0], marker='o', color='w', label='High-Value Victims', markerfacecolor='yellow', markersize=15),
                       Line2D([0], [0], marker='o', color='w', label='Rooms & Hallways', markerfacecolor='lightskyblue', markersize=15),
                       Line2D([0], [0], marker='o', color='w', label='Portals/Connections', markerfacecolor='violet', markersize=15)]
    ax2.legend(handles=legend_elements, loc='best')

    node_list = [graph['ew_1']] + graph.victim_list
    D = get_distance_matrix_original(graph, node_list)


    pos3 = ax3.matshow(D, cmap='YlGnBu')
    fig.colorbar(pos3, ax=ax3)
    ax3.set_xticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55])
    ax3.set_xticklabels(['ew_1', 'vg5', 'vg10', 'vg15', 'vg20', 'vg25', 'vg30', 'vg35', 'vg40', 'vg45', 'vg50', 'vy5',])
    ax3.set_yticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55])
    ax3.set_yticklabels(
        ['ew_1', 'vg5', 'vg10', 'vg15', 'vg20', 'vg25', 'vg30', 'vg35', 'vg40', 'vg45', 'vg50', 'vy5', ])
    ax3.set_title('c) Level 3: Objective Nodes Distance Matrix', fontsize=16)

    medic_data = PCVRP.make_dataset(size=55, filename='trial416.pkl')[0]
    medic_tour = torch.tensor([31, 24, 23, 49, 25, 37, 15, 32, 46, 16, 14, 11, 39, 41, 34, 19, 35, 3,
     22, 0, 55, 10, 43, 29, 4, 30, 5, 26, 21, 53, 36, 17, 9, 42, 40, 8,
     28, 38, 27, 0, 54, 7, 50, 1, 2, 20, 6, 33, 51, 45, 44, 12, 13, 52,
     18, 47, 48])
    medic_routes, medic_cost, medic_path_length = plot_vehicle_routes_paper(medic_data, node_list, medic_tour, ax4,
                                                                      visualize_demands=False, demand_scale=50,
                                                                      round_demand=True, return_routes=True)
    ax4.set_title('d) Level 4: The 2D[0-1] Layout After mMDS', fontsize=16)

    plt.subplots_adjust(left=0.02, bottom=0.02, right=0.98, top=0.98, wspace=0.075, hspace=0.075)
    plt.savefig("4ab-lv-mMDS.pdf")
    plt.show()
