import torch
import torch.nn as nn
import torch_geometric.nn as nng
import torch_geometric.utils as utilsg

class SGCInductive(nng.MessagePassing):
    def __init__(self, in_channels, out_channels, K=1, cached=False, bias=True):
        super(SGCInductive, self).__init__(aggr='mean')  # "Add" aggregation.
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.cached = cached
        self.cached_result = None

        self.lin = nn.Linear(in_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.cached_result = None
        self.cached_num_edges = None

    def forward(self, x, edge_index):
        edge_index, _ = utilsg.add_remaining_self_loops(edge_index, num_nodes=x.size(0))

        x = self.lin(x)
        for k in range(self.K):
            x = self.propagate(edge_index, x=x)

        return x

    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        return aggr_out

