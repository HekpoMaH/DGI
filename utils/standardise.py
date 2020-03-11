import torch
import torch.nn as nn

class Standardise(object):
    def __call__(self, data):
        if hasattr(data, "train_mask"):
            mask = data.train_mask
        else:
            mask = torch.ones_like(data.x, dtype=torch.bool, device='cuda')

        means = data.x[mask].mean(dim=1, keepdim=True)
        stds = data.x[mask].std(dim=1, keepdim=True)
        data.x[mask] = (data.x[mask] - means)/stds
        return data
