import torch
import torch.nn as nn
import torch.nn.functional as F

import wandb
wandb.init()

class SLU(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.layer = nn.Linear(in_dim, out_dim)
        self.act_fn = lambda x: x * torch.sigmoid(100.0 * x)

    def forward(self, x):
        x = self.act_fn(self.layer(x))
        x = torch.max(x, dim=1)[0]
        return x


class OneHotConvNet(nn.Module):
    def __init__(
            self,
            n_tokens: int = 23,
            kernel_size: int = 5,
            input_size: int = 256,
            dropout: float = 0.0,
            make_one_hot=True):
        super(OneHotConvNet, self).__init__()
        self.encoder = nn.Conv1d(n_tokens, input_size, kernel_size=kernel_size)
        self.embedding = SLU(
            in_dim=input_size,
            out_dim=input_size * 2
        )
        self.decoder = nn.Linear(input_size * 2, 1)
        self.n_tokens = n_tokens
        self.dropout = nn.Dropout(dropout)  # TODO: actually add this to model
        self.input_size = input_size
        self._make_one_hot = make_one_hot

    def forward(self, x):
        if self._make_one_hot:
            x = F.one_hot(x, num_classes=self.n_tokens)
        x = x.permute(0, 2, 1).float()
        x = self.encoder(x).permute(0, 2, 1)
        x = self.dropout(x)
        x = self.embedding(x)
        output = self.decoder(x).squeeze(1)
        return output