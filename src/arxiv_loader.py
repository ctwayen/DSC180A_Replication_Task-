import pandas as pd
import networkx as nx
import torch
import numpy as np
import scipy

class arxiv_loader():
    def __init__(self, seed=40, size=0.05, edegs_path="data/edges.csv", labels_path="data/labels.csv", f1_path='data/f_1.npz', f2_path='data/f_2.npz', f3_path='data/f_3.npz', f4_path='data/f_4.npz', f5_path='data/f_5.npz', f6_path='data/f_6.npz'):
        np.random.seed(seed)
        ind = np.random.choice(169343, round(169343*size), replace=False)
        start = pd.read_csv(edegs_path)['start']
        end = pd.read_csv(edegs_path)['end']
        G = nx.Graph()
        for i in ind:
            G.add_node(i)
        for i in range(start.shape[0]):
            if start[i] in ind and end[i] in ind:
                G.add_edge(start[i], end[i])
        self.y = pd.read_csv(labels_path).to_numpy()[ind].ravel()
        A = nx.adjacency_matrix(G)
        A = A.todense()
        self.A = np.asarray(A)
        f_1 = scipy.sparse.load_npz(f1_path).todense()
        f_2 = scipy.sparse.load_npz(f2_path).todense()
        f_3 = scipy.sparse.load_npz(f3_path).todense()
        f_4 = scipy.sparse.load_npz(f4_path).todense()
        f_5 = scipy.sparse.load_npz(f5_path).todense()
        f_6 = scipy.sparse.load_npz(f6_path).todense()
        self.X = np.concatenate([f_1, f_2, f_3, f_4, f_5, f_6], axis=0)[ind]
    def get_train(self):
        return self.A, self.X, self.y