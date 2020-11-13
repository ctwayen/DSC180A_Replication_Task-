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

def centrality_hist(G, centrality = "closeness_centrality"):
    if centrality == "closeness_centrality":
        centrality = nx.closeness_centrality(G, u=None, distance=None, wf_improved=True)
    elif centrality == "betweenness_centrality":
        centrality = nx.betweenness_centrality(G, u=None, distance=None, wf_improved=True)
    elif centrality == "degree_centrality":
        centrality = nx.degree_centrality(G, u=None, distance=None, wf_improved=True)
    values = centrality.values()
    plt.hist(values)
    sorted_cent = sorted(centrality, key=centrality.__getitem__, reverse=True)[:10]
    print(sorted_cent)