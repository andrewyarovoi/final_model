import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualPointBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualPointBlock, self).__init__()
        self.dim = dim
        self.lin1 = nn.Conv1d(dim, dim, 1)
        self.bn1 = nn.BatchNorm1d(dim)
        self.lin2 = nn.Conv1d(dim, dim, 1)
        self.bn2 = nn.BatchNorm1d(dim)

    def forward(self, x):
        out = self.lin1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.lin2(out)
        out = self.bn2(out)
        out = x + out
        out = F.relu(out)
        return out

class MLP(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels=None):
        super(MLP, self).__init__()
        self.out_channels = in_channels if out_channels is None else out_channels
        
        self.lin1 = nn.Conv1d(in_channels, self.out_channels, 1)
        self.bn1 = nn.BatchNorm1d(self.out_channels)
        self.lin2 = nn.Conv1d(self.out_channels, self.out_channels, 1)
        self.bn2 = nn.BatchNorm1d(self.out_channels)
        self.lin3 = nn.Conv1d(self.out_channels, self.out_channels, 1)
        
    def forward(self, x):
        y = F.relu(self.bn1(self.lin1(x)))
        y = F.relu(self.bn2(self.lin2(y)))
        y = self.lin3(y)
        return y

