import torch
import torch.nn as nn
import torch_geometric
from layers import SGCInductive, GATSum

class GraphSkip(nn.Module):
    def build_convolutions(self, in_ft, out_ft, convolution, extra_param=None):
        self.convolution = convolution
        if convolution == "GraphSkip":
            # That's equal to mean pooling -- check the docs
            conv_class = torch_geometric.nn.SAGEConv
            self.conv1 = conv_class(in_ft, out_ft)
            self.conv2 = conv_class(out_ft, out_ft)
            self.conv3 = conv_class(out_ft, out_ft)
        elif "GATConvMean" in convolution:
            self.conv1 = torch_geometric.nn.GATConv(in_ft, out_ft, heads=extra_param, concat=False)
            self.conv2 = torch_geometric.nn.GATConv(out_ft, out_ft, heads=extra_param, concat=False)
        elif "GATConvSum" in convolution:
            self.conv1 = GATSum(in_ft, out_ft, heads=extra_param, concat=False)
            self.conv2 = GATSum(out_ft, out_ft, heads=extra_param, concat=False)
        elif "GATConv" in convolution:
            conv_class = torch_geometric.nn.GATConv
            self.conv1 = conv_class(in_ft, out_ft)
            self.conv2 = conv_class(out_ft, out_ft)
            self.conv3 = conv_class(out_ft, out_ft)
        elif "SGCInductive" in convolution:
            self.conv1 = SGCInductive.SGCInductive(in_ft, out_ft, K=extra_param)
            self.conv2 = SGCInductive.SGCInductive(out_ft, out_ft, K=extra_param)
            self.conv3 = SGCInductive.SGCInductive(out_ft, out_ft, K=extra_param)

        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        if "GATConvSum" not in convolution and "GATConvMean" not in convolution:
            self.conv3.reset_parameters()

    def __init__(self, in_ft, out_ft, act, bias=True, convolution="MeanPool", K=None):
        super(GraphSkip, self).__init__()
        self.fc_skip = nn.Linear(in_ft, out_ft, bias=bias)
        self.build_convolutions(in_ft, out_ft, convolution, extra_param=K)

        self.act = nn.PReLU() if act == 'prelu' else act

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x, edge_index):
        h_1 = self.act(self.conv1(x, edge_index))
        h_2 = self.act(self.conv2(self.fc_skip(x) + h_1, edge_index))
        if "GATConvSum" not in self.convolution and "GATConvMean" not in self.convolution:
            h_3 = self.act(self.conv3(self.fc_skip(x) + h_1 + h_2, edge_index))
        else:
            h_3 = h_2
        return h_3
