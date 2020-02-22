import numpy as np
import torch
import torch.nn as nn
import torch_geometric
from layers import GCN, AvgReadout, Discriminator, GraphSkip

class GNNPlusAct(nn.Module):
    def __init__(self, n_in, n_h, activation, gnn_type='GCNConv'):
        super(GNNPlusAct, self).__init__()
        self.act = nn.PReLU() if activation == "prelu" else activation
        if gnn_type == "GCNConv":
            self.gnn = torch_geometric.nn.GCNConv(n_in, n_h)
        elif gnn_type == "GATConv":
            self.gnn = torch_geometric.nn.GATConv(n_in, n_h)
        elif gnn_type == "SGConv":
            self.gnn = torch_geometric.nn.SGConv(n_in, n_h, K=3)
        else:
            print("UNKNOWN ARCHITECTURE")
            exit(0)
        self.gnn.reset_parameters()
    def forward(self, x, edge_index):
        return self.act(self.gnn(x, edge_index))


class DGI(nn.Module):
    def __init__(self, n_in, n_h, activation, update_rule="GCNConv", batch_size=1):
        super(DGI, self).__init__()

        if update_rule=="MeanPool":
            self.gnn = GraphSkip.GraphSkip(n_in, n_h, activation)
            # has reset parameters and activation in constructor
        else:
            self.gnn = GNNPlusAct(n_in, n_h, activation, update_rule)
            # has reset parameters and activation in constructor

        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h, batch_size)
        # has reset parameters

    def forward(self, seq1, seq2, edge_index, batch=None, samp_bias1=None, samp_bias2=None, edge_index_alt=None):
        h_1 = self.gnn(seq1, edge_index)
        # h_1 = self.act(h_1)

        s = self.read(h_1, batch)
        s = self.sigm(s)

        h_2 = self.gnn(seq2, edge_index if edge_index_alt is None else edge_index_alt)
        # h_2 = self.act(h_2)
        # print(h_1.shape, h_2.shape)

        ret = self.disc(s, h_1, h_2, samp_bias1, samp_bias2, batch=batch)
        return ret

    # Detach the return variables
    def embed(self, seq, edge_index, batch):
        h_1 = self.gnn(seq, edge_index)
        # h_1 = self.act(h_1)
        s = self.read(h_1, batch)
        s = self.sigm(s)

        return h_1.detach(), s.detach()

if __name__ == "__main__":
    dataset_train = PPI(
        "./geometric_datasets/"+dataset,
        split="train",
        transform=torch_geometric.transforms.NormalizeFeatures()
    )
