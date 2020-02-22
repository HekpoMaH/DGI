import torch.nn as nn
import torch_geometric
import torch_geometric.nn as nng

class GAEEnc(nn.Module):
    def __init__(self, in_features, out_features):
        super(GAEEnc, self).__init__()
        self.gcn1 = nng.GCNConv(in_features, out_features)
        self.gcn1.reset_parameters()
        self.relu = nn.ReLU()
        self.gcn2 = nng.GCNConv(2*out_features, out_features)
        self.gcn2.reset_parameters()
    def forward(self, x, edge_index):
        return self.gcn1(x, edge_index)

class VGAEEnc(nn.Module):
    def __init__(self, in_features, out_features):
        super(VGAEEnc, self).__init__()
        self.gcn1 = nng.GCNConv(in_features, 2*out_features)
        self.gcn1.reset_parameters()
        self.relu = nn.ReLU()
        self.gcn_mu = nng.GCNConv(2*out_features, out_features)
        self.gcn_mu.reset_parameters()
        self.gcn_sigma = nng.GCNConv(2*out_features, out_features)
        self.gcn_sigma.reset_parameters()
    def forward(self, x, edge_index):
        hidden = self.relu(self.gcn1(x, edge_index))
        mu = self.gcn_mu(hidden, edge_index)
        sigma = self.gcn_sigma(hidden, edge_index)
        return mu, sigma
