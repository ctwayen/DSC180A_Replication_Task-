import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

class LPA(object):
    def __init__(self, A, len_walk, labels, device):
        self.A = A.to(device)
        self.len_walk = len_walk
        self.labels = labels.to(device)
        
    def forward(self, x):
        l = x.shape[0]
        labels = self.labels[:l]
        A = torch.div(self.A, torch.sum(self.A, axis=1))
        A = torch.matrix_power(A, self.len_walk)
        m = torch.mm(A, labels)
        #print(m.shape)
        return labels
    
class LPA_GCN_model(nn.Module):
    def __init__(self, A, len_walk, labels, device, F = 1433, class_number=7):
        super(LPA_GCN_model, self).__init__()
        # precompute adjacency matrix before training
        self.lpa = LPA(torch.from_numpy(A).float(), len_walk, labels, device)
        self.device = device
        if A[0][0] == 0:
            A = A * 0.1 + np.identity(A.shape[0])
        self.A = torch.from_numpy(A).float()
        self.class_number = class_number
        self.fc1 = nn.Linear(F, 200, bias=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(200, self.class_number, bias=True)
        self.fc3 = nn.Linear(2*self.class_number, self.class_number, bias=True)

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
        lpa = self.lpa.forward(x)
        #print(x.shape, lpa.shape)
        x = torch.cat((x, lpa), axis=1)
        x = self.fc3(x)
        return x

class LPA_GCN():
    def __init__(self, A, X, y, device='cuda', len_walk=3, F=1433, class_number=7):
        self.A = A
        if device == 'cuda':
            self.device = device = torch.device('cuda')
        else:
            assert('only support cuda')
        kwargs = {'num_workers': 1, 'pin_memory': True}
        #display(y.shape)
        le = preprocessing.LabelEncoder()
        one = preprocessing.OneHotEncoder(sparse=False)
        y_ = np.reshape(y, (-1, 1))
        #display(y.shape)
        one.fit(y_)
        #display(y.shape)
        labels = np.array(one.transform(y_))
        #display(y.shape)
        labels = torch.from_numpy(labels).type(torch.float)
        #display(y.shape)
        le.fit(y)
        #display(y.shape)
        y = le.transform(y)
        X = torch.tensor(X)
        X = X.type(torch.long)
        y = torch.tensor(y)
        y = y.type(torch.long)
        
        dataset = TensorDataset(X, y)
        self.dataloader = DataLoader(dataset, batch_size=2708, shuffle=True, **kwargs)
        
        self.lpa_gcn = LPA_GCN_model(A, len_walk, labels, self.device)
        self.lpa_gcn.to(self.device)
    
    def train(self, model, device, train_loader, optimizer):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
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
    
    def train_model(self, epochs=50, lr=1e-3):
        lpa_gcn_test_acc = []
        optimizer = optim.Adam(self.lpa_gcn.parameters(), lr=lr)#, weight_decay=1e-1)

        for epoch in range(epochs):
            self.train(self.lpa_gcn, self.device, self.dataloader, optimizer)
            accs = self.test(self.lpa_gcn, self.device, self.dataloader)
            lpa_gcn_test_acc.append(accs)
        accs = {'acc': lpa_gcn_test_acc}
        return accs