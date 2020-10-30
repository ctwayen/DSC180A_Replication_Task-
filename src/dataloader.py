import pandas as pd
import numpy as np
import networkx as nx

def readF(file_path, delimiter='\t', header=None, y_index=-1, ignore_columns=None):
    '''
    @ Params
        file_path: a string indicate the path of the file
        delimiter: the delimiter for delimite the file, default is '\t'
        header: default is none
        y_index: the index of labels, default is the last one
        ignore_columns: a list of indexs representing correponding columns that need to be ignored

    @ Output
        X and y in terms of numpy array
    '''
    features = pd.read_csv(file_path, header=header, delimiter=delimiter).to_numpy()
    if ignore_columns != None:
        features = features[:, [x for x in range(features.shape[1]) if x not in ignore_columns]]
    if y_index == -1:
        y_index = features.shape[1] - 1
    X = features[:, [x for x in range(features.shape[1]) if x != y_index]]
    y = features[:, y_index]
    return X.astype('int64'), y

def readedges(file_path, adj_matrix=False):
    '''
    @ params
        file_path: a string indicate the path of the file
        adj_matrix: wether return the graph as adj matrix, default False
    '''
    G = nx.read_edgelist(file_path)
    if adj_matrix:
        return nx.adjacency_matrix(G).todense()
    return G