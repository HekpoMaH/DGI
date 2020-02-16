import torch
import torch.nn as nn
import torch_geometric.nn as nng

# Applies an average on seq, of shape (batch, nodes, features)
# While taking into account the masking of msk
class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, x, batch=None):
        if batch is None:
            batch = torch.zeros(
                x.shape[0],
                dtype=torch.long,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
        return nng.global_mean_pool(x, batch)

