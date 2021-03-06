import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt

class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        """
        To add dropout or other idk
        input_size: df.shape[1]
        output_size: nb_clust
        """
        self.fc1 = nn.Linear(in_features = input_size, out_features = 100)

        self.fc2 = nn.Linear(in_features = 100, out_features=40)
        self.fc3 = nn.Linear(in_features = 40, out_features = 1)
        
        self.bn1 = nn.BatchNorm1d(num_features=100)
        self.bn2 = nn.BatchNorm1d(num_features=40)
        self.dropout = nn.Dropout(0.25)
        self.relu = nn.ReLU()
        
    def forward(self, x, n_clusters):
        x = self.fc1(x.float())
        x = self.bn1(x)
        x = self.relu(x)
        
        
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        if n_clusters == 1:
            # binary classification
            # returns a probability scalar
            x = self.fc3(x)
            
            return x
            
        else:
            # clustering
            # return a vector of size number of clusters
            x = self.fc2(x)
            #x = self.fc3(x)
            return x
