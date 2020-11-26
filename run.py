import json
import argparse
from src.coraloader import cora_loader
from src.two_layer_gnn import GNN
from src.coraloader import encode_label
from src.LPA_GCN import LPA_GCN
import pandas as pd
import numpy as np
import networkx as nx
import random

from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, Callback
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from spektral.layers import GraphConv
import pathlib

def main():
    # Training settings
    local_path = str(pathlib.Path().parent.absolute())
    local_data = local_path + '/data'
    local_output = local_path + '/config/model-output.json'
    parser = argparse.ArgumentParser(description='Running model')
    parser.add_argument('--model', type=str, default='graph', choices=['graph', 'LPA_GCN'],
                        help='model to use for training (default: 2layerGNN)')
    parser.add_argument('--image_path', type=int, default=None,
                        help='draw the graph to the path')
    parser.add_argument('--dataset', type=str, default='cora', choices=['cora'],
                        help='data set type (default cora and only support cora now)')
    parser.add_argument('--cora_path', type=str, default=local_data,
                        help='path for the cora dataset')
    parser.add_argument('--output_path', type=str, default=local_output,
                        help='path for the output json file')
    
    parser.add_argument('--len_walk', type=int, default=3,
                        help='the length of random walk; only used when model is LPA_GCN')
    parser.add_argument('--hidden_neurons', type=int, default=200,
                        help='hidden neurons in hidden layer (GNN) (default: 200)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for trianing the model (dafault: cuda)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate (default: 1e-3)')
    parser.add_argument(--test', action = 'store_true', help='running test')
                        
    args = parser.parse_args()
                        
    if args.test:
        with open('test/testdata/test.json') as f:
             data = json.load(f)
        X, y, A = np.array(data['X']), np.array(data['y']), np.array(data['A'])
        model = GNN()
        model.fit(X, y, A)
        hist = model.train_epoch(F=2, class_number=2)
    with open(args.output_path, 'w') as f:
            json.dump(hist, f)
        
if __name__ == '__main__':
    main()
    # Examples:
    # python run.py --model graph --dataset cora
