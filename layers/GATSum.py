import torch_geometric.nn as nng

class GATSum(nng.GATConv):
    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.sum(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out
