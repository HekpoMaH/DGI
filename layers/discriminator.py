import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, n_h, batch_size):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, batch_size)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, s, h_pl, h_mi, s_bias1=None, s_bias2=None, batch=None):
        if batch is None:
            batch = torch.zeros(h_pl.shape[0], dtype=torch.long)

        print(s.shape)
        s_x = s[batch]
        print(h_pl.shape, h_mi.shape, s_x.shape)
        exit(0)
        sc_1 = torch.squeeze(self.f_k(h_pl, s_x), 1)
        sc_2 = torch.squeeze(self.f_k(h_mi, s_x), 1)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 0)
        return logits
