import torch
import torch.nn as nn
import torch_geometric

class GraphSkip(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GraphSkip, self).__init__()
        self.fc_skip = nn.Linear(in_ft, out_ft, bias=False)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)
        # That's equal to mean pooling -- check the docs
        self.mp1 = torch_geometric.nn.SAGEConv(in_ft, out_ft)
        self.mp1.reset_parameters()
        self.mp2 = torch_geometric.nn.SAGEConv(out_ft, out_ft)
        self.mp2.reset_parameters()
        self.mp3 = torch_geometric.nn.SAGEConv(out_ft, out_ft)
        self.mp3.reset_parameters()

        self.act = nn.PReLU() if act == 'prelu' else act
        

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x, edge_index):
        h_1 = self.act(self.mp1(x, edge_index))
        h_2 = self.act(self.mp2(self.fc_skip(x) + h_1, edge_index))
        h_3 = self.act(self.mp3(self.fc_skip(x) + h_1 + h_2, edge_index))
        return h_3
