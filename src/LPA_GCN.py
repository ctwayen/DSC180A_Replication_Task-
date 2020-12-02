import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.parameter import Parameter
import numpy as np
import torch
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import math

class LPA_GCN_layer(nn.Module):
    def __init__(self, A, F, O, len_walk, bias=True):
        super(LPA_GCN_layer, self).__init__()
        self.F = F
        self.O = O
        self.len_walk = len_walk
        self.weight = Parameter(torch.FloatTensor(F, O))
        if bias:
            self.bias = Parameter(torch.FloatTensor(O))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.A_mask = Parameter(A.clone())

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, X, A, y):
        #print(X.type(), self.weight.type())
        X = torch.mm(X, self.weight)
        A = A * self.A_mask
        A = F.normalize(A, p=1, dim=1)
        X = torch.mm(A, X)
        A = torch.matrix_power(A, self.len_walk)
        #print(A.shape, y.shape)
        y_hat = torch.mm(A, y)
        if self.bias is not None:
            return X + self.bias, y_hat
        else:
            return X, y_hat
        
class LPA_GCN_model(nn.Module):
    def __init__(self, A, len_walk, F, class_number, hid):
        super(LPA_GCN_model, self).__init__()
        self.lg1 = LPA_GCN_layer(F = F, O = hid, A=A, len_walk=len_walk)
        self.lg2 = LPA_GCN_layer(F = hid, O = class_number, A=A, len_walk=len_walk)

    def forward(self, X, A, y):
        X, y_hat = self.lg1(X, A, y)
        X = F.relu(X)
        X, y_hat = self.lg2(X, A, y_hat)
        return F.log_softmax(X, dim=1), F.log_softmax(y_hat,dim=1)
    
class LPA_GCN():
    def __init__(self, A, X, y, lamb, device='cuda', len_walk=2, F=1433, class_number=7, hid=200, val = 0.3):
        if device == 'cuda':
            self.device= torch.device('cuda')
        else:
            assert('only support cuda')
        le = preprocessing.LabelEncoder()
        one = preprocessing.OneHotEncoder(sparse=False)
        y_ = np.reshape(y, (-1, 1))
        one.fit(y_)
        labels = np.array(one.transform(y_))
        labels = torch.from_numpy(labels).type(torch.float)
        le.fit(y)
        y = le.transform(y)
        
        X = torch.tensor(X)
        X = X.type(torch.float)
        y = torch.tensor(y)
        y = y.type(torch.long)
        y = y.to(self.device)
        A = torch.from_numpy(A).float()
        
        self.X = X.to(self.device)
        self.A = A.to(self.device)
        self.y = y.to(self.device)
        self.Lambda = lamb
        self.labels = labels.to(self.device)
        
        train_idx = np.random.choice(self.X.shape[0], round(self.X.shape[0]*(1-val)), replace=False)
        val_idx = np.array([x for x in range(X.shape[0]) if x not in train_idx])
        print("Train length :{a}, Validation length :{b}".format(a=len(train_idx), b=len(val_idx)))
        
        self.idx_train = torch.LongTensor(train_idx)
        self.idx_val = torch.LongTensor(val_idx)
        
        self.lpa_gcn = LPA_GCN_model(A = self.A, len_walk=len_walk, F=F, class_number = class_number, hid = hid)
        self.lpa_gcn.to(self.device)
    
    def train(self, optimizer, epoch):
        self.lpa_gcn.train()
        optimizer.zero_grad()
        output, y_hat = self.lpa_gcn(self.X, self.A, self.labels)
        loss_gcn = F.cross_entropy(output[self.idx_train], self.y[self.idx_train])
        loss_lpa = F.nll_loss(y_hat, self.y)
        loss_train = loss_gcn + self.Lambda * loss_lpa
        loss_train.backward(retain_graph=True)
        optimizer.step()
        print('Epoch: {x}'.format(x=epoch))
        print('training loss {:.4f}'.format(loss_train.item()))
            
    def test(self):
        self.lpa_gcn.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            output, y_hat = self.lpa_gcn(self.X, self.A, self.labels)
            #print(self.idx_val)
            test_loss = F.cross_entropy(output[self.idx_val], self.y[self.idx_val], reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)[self.idx_val]
            correct += pred.eq(self.y[self.idx_val].view_as(pred)).sum().item()

        test_loss /= len(self.idx_val)
        print('Validtion: Average loss: {:.4f}, Accuracy: {:.4f}%'.format(test_loss, 100. * correct / len(self.idx_val)))
        return 100. * correct / len(self.idx_val)
    
    def train_model(self, epochs=50, lr=1e-3):
        lpa_gcn_test_acc = []
        optimizer = optim.Adam(self.lpa_gcn.parameters(), lr=lr)#, weight_decay=1e-1)

        for epoch in range(epochs):
            self.train(optimizer, epoch)
            accs = self.test()
            lpa_gcn_test_acc.append(accs)
        accs = {'acc': lpa_gcn_test_acc}
        return accs