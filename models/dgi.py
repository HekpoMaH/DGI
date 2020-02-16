import numpy as np
import torch
import torch.nn as nn
import torch_geometric
from layers import GCN, AvgReadout, Discriminator, GraphSkip

class DGI(nn.Module):
    def __init__(self, n_in, n_h, activation, update_rule="GCNConv", batch_size=1):
        super(DGI, self).__init__()

        if update_rule=="GCNConv":
            self.gcn = torch_geometric.nn.GCNConv(n_in, n_h)
            self.gcn.reset_parameters()
        if update_rule=="MeanPool":
            self.gcn = GraphSkip.GraphSkip(n_in, n_h, activation)

        self.act = nn.PReLU() if activation == "prelu" else activation
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h, batch_size)

    def forward(self, seq1, seq2, edge_index, batch=None, samp_bias1=None, samp_bias2=None, edge_index_alt=None):
        h_1 = self.gcn(seq1, edge_index)
        h_1 = self.act(h_1)

        s = self.read(h_1, batch)
        s = self.sigm(s)

        h_2 = self.gcn(seq2, edge_index if edge_index_alt is None else edge_index_alt)
        h_2 = self.act(h_2)
        print(h_1.shape, h_2.shape)

        ret = self.disc(s, h_1, h_2, samp_bias1, samp_bias2, batch=batch)
        return ret

    # Detach the return variables
    def embed(self, seq, edge_index, batch):
        h_1 = self.gcn(seq, edge_index)
        s = self.read(h_1, batch)

        return h_1.detach(), s.detach()

if __name__ == "__main__":
    dataset_train = PPI(
        "./geometric_datasets/"+dataset,
        split="train",
        transform=torch_geometric.transforms.NormalizeFeatures()
    )
