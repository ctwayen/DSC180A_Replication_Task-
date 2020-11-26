# import section
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
import torch
import pandas as pd
import networkx as nx
class two_layer_GraphNet(nn.Module):
    def __init__(self, A, device, F = 1433, class_number=7, hidden_neurons = 200):
        super(two_layer_GraphNet, self).__init__()
        # precompute adjacency matrix before training
        if A[0][0] == 0:
            A = A * 0.1 + np.identity(A.shape[0])
        self.A = torch.from_numpy(A).float()
        self.class_number = class_number
        self.fc1 = nn.Linear(F, hidden_neurons, bias=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_neurons, self.class_number, bias=True)
        self.device = device

    def forward(self, x):
        # training on full x, not batch
        x = x.float()
        # average all neighboors
        #print(x.shape)
        #A = self.A.float()
        A = self.A.to(self.device)
        #print(A.shape)
        #print(self.X.shape)
        #print(A.dtype, self.X.dtype, x.dtype)
        x = torch.matmul(A, x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        return x
    
class GNN():
    def __init__(self, hidden_neurons=200, learning_rate=1e-3, epoch=50, device='cuda'):
        self.hidden_neurons = hidden_neurons
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.device = torch.device(device)
        self.kwargs = {'num_workers': 1, 'pin_memory': True}
    
    def train(self, model, device, train_loader, optimizer):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = model(data)
            #print(output, target)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            print('training loss {:.4f}'.format(loss.item()))

    def test(self, model, device, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print('Average loss: {:.4f}, Accuracy: {:.4f}%'.format(test_loss, 100. * correct / len(test_loader.dataset)))
        return 100. * correct / len(test_loader.dataset)
    
    def encode_label(self, y):
        '''

        '''
        label_encoder = preprocessing.LabelEncoder()
        label_encoder.fit(y)
        y = label_encoder.transform(y)
        return y
    
    def fit(self, X, y, A):
        X = torch.from_numpy(X).type(torch.long)
        self.A = A
        y = torch.from_numpy(self.encode_label(y))
        self.dataset = TensorDataset(X, y)
        self.dataloader = DataLoader(self.dataset, batch_size=2708, shuffle=True, **self.kwargs)
        
    def train_epoch(self, F = 1433, class_number=7):
        model = two_layer_GraphNet(self.A, self.device, F=F, class_number=class_number)
        model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        accs = {'acc': []}
        for epoch in range(self.epoch):
            self.train(model, self.device, self.dataloader , optimizer)
            acc = self.test(model, self.device, self.dataloader)
            accs['acc'].append(acc)
        return accs
