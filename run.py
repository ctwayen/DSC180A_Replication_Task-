import json
import argparse
from src.coraloader import cora_loader
from src.two_layer_gnn import GNN
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
    parser.add_argument('--image_path', type=int, default=None,
                        help='draw the graph to the path')
    parser.add_argument('--dataset', type=str, default='cora', choices=['cora'],
                        help='data set type (default cora and only support cora now)')
    parser.add_argument('--cora_path', type=str, default=local_data,
                        help='path for the cora dataset')
    parser.add_argument('--output_path', type=str, default=local_output,
                        help='path for the output json file')
    
    parser.add_argument('--hidden_neurons', type=int, default=200,
                        help='hidden neurons in hidden layer (GNN) (default: 200)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for trianing the model (dafault: cuda)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate (default: 1e-3)')
    
    args = parser.parse_args()
    cora = cora_loader(args.cora_path + '/cora.content', args.cora_path + '/cora.cites', args.image_path)
    model = GNN(hidden_neurons=args.hidden_neurons, learning_rate=args.lr, epoch=args.epochs, device=args.device)
    X, y, A = cora.get_train()
    model.fit(X, y, A)
    hist = model.train_epoch()
    with open(args.output_path, 'w') as f:
        json.dump(hist, f)
    print('success write model history into file')
        
if __name__ == '__main__':
    main()
    # Examples:
    # python run.py --model graph --dataset cora