import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from sklearn import preprocessing
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class GCN_N_layer(nn.Module):
    def __init__(self, A, N=0, F = 1433, class_number=7, hidden_neurons=200):
        super(GCN_N_layer, self).__init__()
        self.A = A
        self.class_number = class_number
        self.N = N
        self.hidden = nn.Linear(hidden_neurons, hidden_neurons, bias=True)
        self.fc1 = nn.Linear(F, hidden_neurons, bias=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(200, self.class_number, bias=True)

    def forward(self, x):
        # training on full x, not batch
        x = x.float()
        # average all neighboors
        #print(x.shape)
        #A = self.A.float()
        #print(A.shape)
        #print(self.X.shape)
        #print(A.dtype, self.X.dtype, x.dtype)
        x = torch.matmul(self.A, x)
        x = self.fc1(x)
        x = self.relu(x)
        for i in range(self.N):
            x = self.hidden(x)
            x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        return x

class n_hidden_GCN():
    def __init__(self, A, X, y, device="cuda", N=0, F=1433, class_number = 7, hidden_neurons=200, self_weight=10, val_size=0.3):
        self.device = torch.device(device)
        le = preprocessing.LabelEncoder()
        le.fit(y)
        y = le.transform(y)
        
        X = torch.tensor(X)
        X = X.type(torch.float)
        y = torch.tensor(y)
        y = y.type(torch.long)
        y = y.to(self.device)
        
        if A[0][0] == 0:
            A = A * 1./self_weight + np.identity(A.shape[0])
        A = torch.from_numpy(A).float()
        
        self.X = X.to(self.device)
        self.A = A.to(self.device)
        self.y = y.to(self.device)
    
        self.model = GCN_N_layer(A=self.A, N=N, F=F, class_number=class_number, hidden_neurons=hidden_neurons)
        
        train_idx = np.random.choice(self.X.shape[0], round(self.X.shape[0]*(1-val_size)), replace=False)
        val_idx = np.array([x for x in range(X.shape[0]) if x not in train_idx])
        print("Train length :{a}, Validation length :{b}".format(a=len(train_idx), b=len(val_idx)))
        
        self.idx_train = torch.LongTensor(train_idx)
        self.idx_val = torch.LongTensor(val_idx)
        self.model.to(self.device)
        
    def train(self, optimizer, epoch):
        self.model.train()
        optimizer.zero_grad()
        output = self.model(self.X)
        loss = F.cross_entropy(output[self.idx_train], self.y[self.idx_train])
        loss.backward(retain_graph=True)
        optimizer.step()
        print('Epoch: {x}'.format(x=epoch))
        print('training loss {:.4f}'.format(loss.item()))
            
    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            output = self.model(self.X)
            loss = F.cross_entropy(output[self.idx_val], self.y[self.idx_val], reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)[self.idx_val]
            correct += pred.eq(self.y[self.idx_val].view_as(pred)).sum().item()

        test_loss /= len(self.idx_val)
        print('Validtion: Average loss: {:.4f}, Accuracy: {:.4f}%'.format(test_loss, 100. * correct / len(self.idx_val)))
        return 100. * correct / len(self.idx_val)
    
    def train_epoch(self, epochs=100, lr=1e-3):
        acc = []
        optimizer = optim.Adam(self.model.parameters(), lr=lr)#, weight_decay=1e-1)

        for epoch in range(epochs):
            self.train(optimizer, epoch)
            accs = self.test()
            acc.append(accs)
        accs = {'acc': acc}
        return accs
    
    def draw_cora(self):
        self.model.eval()
        output = self.model(self.X)
        pred = output.argmax(dim=1, keepdim=True)
        A = self.A.cpu().detach().numpy()
        Y = pred.cpu().detach().numpy()
        G = nx.from_numpy_matrix(A, nx.DiGraph())
        plt.figure(figsize=(10,10))
        pos = nx.spring_layout(G, seed=675)
        nx.draw(G, pos=pos, node_size=10, edge_size=1,node_color=Y, cmap = 'tab10')
        plt.show()