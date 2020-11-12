import pandas as pd
import numpy as np
import networkx as nx
import random

from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from src.eda_functions import vis_graph

def encode_label(y):
    '''

    '''
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(y)
    labels = to_categorical(labels)
    return labels

#https://towardsdatascience.com/graph-convolutional-networks-on-node-classification-2b6bbec1d042
class cora_loader():
    def __init__(self, path_con, path_city, image_path):
        '''
            @ params：
                path_con: path for the file cora.content
                path_city: path for the file cora.cities
                ratio: a list represent [ratio of training, ratio of test]
        '''
        all_data = []
        all_edges = []
        with open(path_con) as f:
            all_data.extend(f.read().splitlines())
        with open(path_city) as f:
            all_edges.extend(f.read().splitlines())
        random.seed(4)
        random.shuffle(all_data)
        labels = []
        nodes = []
        X = []

        for i , data in enumerate(all_data):
            elements = data.split('\t')
            labels.append(elements[-1])
            X.append(elements[1:-1])
            nodes.append(elements[0])

        self.X = np.array(X,dtype=int)
        self.y = labels
        #parse the edge
        edge_list=[]
        for edge in all_edges:
            e = edge.split('\t')
            edge_list.append((e[0],e[1]))
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edge_list)
        if image_path != None:
            vis_graph(G, image_path)
        #obtain the adjacency matrix (A)
        self.A = nx.adjacency_matrix(G)
        
    def get_train(self):
        return self.X, self.y, self.A