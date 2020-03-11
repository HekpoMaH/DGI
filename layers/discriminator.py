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

    def forward(self, s, h_pl, h_mi, batch=None):
        if batch is None:
            batch = torch.zeros(h_pl.shape[0], dtype=torch.long)

        s_x = s.expand_as(h_pl) # expands [1, f_dim] to [N_pl, f_dim]
        sc_1 = torch.squeeze(self.f_k(h_pl, s_x), 1) # [N_pl, 1] squeezed [N_pl]
        s_x = s.expand_as(h_mi) # expands [1, f_dim] to [N_mi, f_dim] 
        sc_2 = torch.squeeze(self.f_k(h_mi, s_x), 1) # [N_mi, 1] squeezed [N_mi]

        logits = torch.cat((sc_1, sc_2), 0)
        return logits
