import pandas as pd
import networkx as nx
import torch
import numpy as np

class arxiv_loader():
    def __init__(self, edges_path='data/edges.csv', labels_path='data/labels.csv', 
                 features_path='data/features.csv', seed=40, size=0.05):
        np.random.seed(seed)
        ind = np.random.choice(169343, round(169343*size), replace=False)
        start = pd.read_csv("data/edges.csv")['start']
        end = pd.read_csv("data/edges.csv")['end']
        G = nx.Graph()
        for i in ind:
            G.add_node(i)
        for i in range(start.shape[0]):
            if start[i] in ind and end[i] in ind:
                G.add_edge(start[i], end[i])
        self.y = pd.read_csv('data/labels.csv').to_numpy()[ind].ravel()
        A = nx.adjacency_matrix(G)
        A = A.todense()
        self.A = np.asarray(A)
        features = pd.read_csv('data/features.csv')
        X = []
        for f in features['feature'][ind]:
            X.append([float(x) for x in np.array(f[1:-1].split(','))])
        self.X = np.array(X)
        
    def get_train(self):
        return self.A, self.X, self.y