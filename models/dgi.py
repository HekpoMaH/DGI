import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from layers import GCN, AvgReadout, Discriminator, GraphSkip, SGCInductive, GATSum

class GNNPlusAct(nn.Module):
    def __init__(self, n_in, n_h, activation, gnn_type='GCNConv', K=None):
        super(GNNPlusAct, self).__init__()
        self.act = nn.PReLU() if activation == "prelu" else activation
        if "SGC" in gnn_type:
            self.act = lambda x: x
        if gnn_type == "GCNConv":
            self.gnn = torch_geometric.nn.GCNConv(n_in, n_h)
        elif gnn_type == "GATConv":
            self.gnn = torch_geometric.nn.GATConv(n_in, n_h)
        elif gnn_type == "GATConvMean":
            self.gnn = torch_geometric.nn.GATConv(n_in, n_h, heads=K, concat=False)
        elif gnn_type == "GATConvSum":
            self.gnn = GATSum(n_in, n_h, heads=K, concat=False)
        elif gnn_type == "SGConv":
            self.gnn = torch_geometric.nn.SGConv(n_in, n_h, K=K)
        elif gnn_type == "SGCInductive":
            self.gnn = SGCInductive.SGCInductive(n_in, n_h, K=K)
        else:
            print("UNKNOWN ARCHITECTURE")
            exit(0)
        self.gnn.reset_parameters()
    def forward(self, x, edge_index):
        return self.act(self.gnn(x, edge_index))


class DGI(nn.Module):
    def __init__(self, n_in, n_h, activation, update_rule="GCNConv", batch_size=1, K=None):
        super(DGI, self).__init__()

        if "GraphSkip" in update_rule:
            self.gnn = GraphSkip.GraphSkip(n_in, n_h, activation, convolution=update_rule, K=K)
            # has reset parameters and activation in constructor
        else:
            self.gnn = GNNPlusAct(n_in, n_h, activation, update_rule, K=K)
            # has reset parameters and activation in constructor

        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h, batch_size)
        # has reset parameters

    def forward(self, seq1, seq2, edge_index, batch=None, edge_index_alt=None, embed_gae=False):
        if embed_gae:
            return self.embed(seq1, edge_index, embed_gae=embed_gae)

        h_1 = self.gnn(seq1, edge_index)
        # h_1 = self.act(h_1)

        s = self.read(h_1, batch)
        s = self.sigm(s)

        h_2 = self.gnn(seq2, edge_index if edge_index_alt is None else edge_index_alt)
        # h_2 = self.act(h_2)
        # print(h_1.shape, h_2.shape)

        ret = self.disc(s, h_1, h_2, batch=batch)
        return ret

    # Detach the return variables
    def embed(self, seq, edge_index, batch=None, embed_gae=False, standardise=False):
        def standardise_data(data):
            means = data.mean(dim=1, keepdim=True)
            stds = data.std(dim=1, keepdim=True)
            data = (data - means)/stds
            return data
        h_1 = self.gnn(seq, edge_index)

        # h_1 = self.act(h_1)
        s = self.read(h_1, batch)
        s = self.sigm(s)

        if standardise:
            h_1 = standardise_data(h_1)
            s = standardise_data(s)
        if embed_gae:
            return h_1.detach()
        return h_1.detach(), s.detach()

if __name__ == "__main__":
    dataset_train = PPI(
        "./geometric_datasets/"+dataset,
        split="train",
        transform=torch_geometric.transforms.NormalizeFeatures()
    )
