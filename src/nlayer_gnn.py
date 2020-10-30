import pandas as pd
import numpy as np
import networkx as nx

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

class gnn_nlayer():
    def __init__ (self, channels = 16, dropout=0.1, l2_reg = 5e-4):
        '''
            channels: output channels of each layer
            dropout: dropout ratio of dropout layer
            l2_reg: if -1, then no reularization
        '''
        self.channels = channels
        self.drop_out = dropout
        self.l2_reg = l2_reg
        
    def encode_label(self, y):
        '''

        '''
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(y)
        labels = to_categorical(labels)
        return labels
    
    def fit(self, X, y, A, num_classes = 7, train_part = 0.5):
        '''
            X: features
            A: adj matrix
        '''
        self.A = GraphConv.preprocess(A).astype('f4')
        self.F = X.shape[1]
        self.N = X.shape[0]
        self.X = X
        self.y = y
        self.num_classes = num_classes
        
        #https://towardsdatascience.com/graph-convolutional-networks-on-node-classification-2b6bbec1d042
        train_idx = np.random.choice(len(self.y), int(np.ceil(train_part * len(self.y))), replace=False)

        #get the indices that do not go to traning data
        val_idx = [x for x in range(len(self.y)) if x not in train_idx]
        train_mask = np.zeros((self.N,),dtype=bool)
        train_mask[train_idx] = True
        self.train_mask = train_mask
        val_mask = np.zeros((self.N,),dtype=bool)
        val_mask[val_idx] = True
        self.val_mask = val_mask
        self.y = self.encode_label(self.y)
        
    def model_(self, n):
        input_layer = Input(shape=(self.F, ))
        filter_in = Input((self.N, ), sparse=True)
        dropout = Dropout(self.drop_out)(input_layer)
        graph_conv_ = GraphConv(self.channels,
                             activation='relu',
                             use_bias=True,
                             kernel_regularizer=l2(self.l2_reg))([dropout, filter_in])
        for i in range(1, n):
            dropout = Dropout(self.drop_out)(graph_conv_)
            graph_conv_ = GraphConv(
                                     self.channels,
                                     kernel_regularizer=l2(self.l2_reg),
                                     use_bias=True,
                                     activation='relu')([dropout, filter_in]
                                    )

        dropout_ = Dropout(self.drop_out)(graph_conv_)
        graph_conv_ = GraphConv(self.num_classes,
                                 use_bias=True,
                                 activation='softmax')([dropout_, filter_in])
        model = Model(inputs=[input_layer, filter_in], outputs=graph_conv_)
        self.model = model
        return model
    
    def mcompile(self, optimizer='adam', loss='cr', learning_rate=1e-3):
        if optimizer == 'adam':
            optimizer = Adam(lr=learning_rate)
        if loss == 'cr':
            loss = 'categorical_crossentropy'
        self.model.compile(optimizer=optimizer,
                  loss=loss,
                  weighted_metrics=['acc'])
        return self.model
    
    def train(self, epochs=50):
        validation_data = ([self.X, self.A], self.y, self.val_mask)
        hist = self.model.fit([self.X, self.A],
                  self.y,
                  sample_weight=self.train_mask,
                  epochs=epochs,
                  batch_size=self.N,
                  validation_data=validation_data,
                  shuffle=False)
        return hist