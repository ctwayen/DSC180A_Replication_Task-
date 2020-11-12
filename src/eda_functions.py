import os
import math
import networkx as nx
import matplotlib.pyplot as plt

def load_cora(path_con, path_city):
    all_data = []
    all_edges = []
    with open(path_con) as f:
        all_data.extend(f.read().splitlines())
    with open(path_city) as f:
        all_edges.extend(f.read().splitlines())
    nodes = []
    for i , data in enumerate(all_data):
        elements = data.split('\t')
        nodes.append(elements[0])
    edge_list=[]
    for edge in all_edges:
        e = edge.split('\t')
        edge_list.append((e[0],e[1]))
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edge_list)
    return G

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def degree_hist(graph, sig=True, filepath):
    degrees = dict(G.degree()).values()
    if not sig:
        plt.hist(degrees)
    else:
        degrees_sigmoid = [sigmoid(x) for x in degrees]
        plt.hist(degrees_sigmoid)
    plt.savefig(filepath)
    
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

def graph_info(graph):
    print(nx.info(G))

def load_cora(path_con, path_city):
    all_data = []
    all_edges = []
    with open(path_con) as f:
        all_data.extend(f.read().splitlines())
    with open(path_city) as f:
        all_edges.extend(f.read().splitlines())
    nodes = []
    for i , data in enumerate(all_data):
        elements = data.split('\t')
        nodes.append(elements[0])
    edge_list=[]
    for edge in all_edges:
        e = edge.split('\t')
        edge_list.append((e[0],e[1]))
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edge_list)
    return G