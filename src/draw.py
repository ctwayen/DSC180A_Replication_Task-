import matplotlib.pyplot as plt
import networkx as nx
def vis_graph(graph, image_path, mode='circular'):
    plt.figure(num=None, figsize=(20, 20), dpi=80)
    plt.axis('off')
    fig = plt.figure(1)
    if mode == 'spectral':
        nx.draw_spectral(graph)
    if mode == 'spring':
        nx.draw_spring(graph)
    if mode == 'circular':
        nx.draw_circular(graph)
    #plt.show()
    plt.savefig(image_path)