import json
import argparse
from src.coraloader import cora_loader
from src.nlayer_gnn import gnn_nlayer
from src.coraloader import encode_label

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
    parser.add_argument('--model', type=str, default='graph', choices=['graph'],
                        help='model to use for training (default: nlayerGNN)')
    parser.add_argument('--layer_number', type=int, default=1,
                        help='input layer number for nlayerGNN')
    parser.add_argument('--image_path', type=int, default=None,
                        help='draw the graph to the path')
    
    parser.add_argument('--dataset', type=str, default='cora', choices=['cora'],
                        help='data set type (default cora and only support cora now)')
    parser.add_argument('--cora_path', type=str, default=local_data,
                        help='path for the cora dataset')
    parser.add_argument('--output_path', type=str, default=local_output,
                        help='path for the output json file')
    
    parser.add_argument('--channels', type=int, default=16,
                        help='channels output of each layer (GNN) (default: 16)')
    parser.add_argument('--dropout', type=int, default=0.1,
                        help='dropout ratio in dropout layer (default: 0.1)')
    parser.add_argument('--l2_reg', type=int, default=5e-4,
                        help='l2 regularzaytion (default: 5e-4)')
    
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate (default: 1e-3)')
    
    args = parser.parse_args()
    cora = cora_loader(args.cora_path + '/cora.content', args.cora_path + '/cora.cites', args.image_path)
    model = gnn_nlayer(channels = args.channels, dropout = args.dropout, l2_reg = args.l2_reg)
    X, y, A = cora.get_train()
    model.fit(X, y, A)
    model.model_(args.layer_number)
    model.mcompile(optimizer='adam', loss='cr', learning_rate=args.lr)
    hist = model.train(args.epochs)
    with open(args.output_path, 'w') as f:
        json.dump(hist.history, f)
    print('success write model history into file')
        
if __name__ == '__main__':
    main()
    # Examples:
    # python run.py --model graph --dataset cora