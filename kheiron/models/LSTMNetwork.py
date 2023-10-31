import torch
import torch.nn.functional as F
from torch import nn

class BiLSTMAtt(nn.Module):
    def __init__(self, embed_dim=128,
                 rnn_dim=256, n_layers_rnn=2, dropout=0.3,
                 n_linear=2, n_classes=1, vocab_size=23):
        super().__init__()
        self.rnn_dim = rnn_dim
        self.embed_dim = embed_dim
        self.embedding_layer = nn.Embedding(vocab_size, self.embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(self.embed_dim, self.rnn_dim,
                            num_layers=n_layers_rnn,
                            bidirectional=True,
                            batch_first=True)
        self.lstm.dropout = dropout
        self.att_w = nn.Parameter(torch.randn(1, rnn_dim, 1))
        self.fc_layers = nn.ModuleList([nn.Linear(rnn_dim, rnn_dim) for _ in range(n_linear - 1)])
        self.linear_out = nn.Linear(rnn_dim, n_classes)

    def attention(self, rnn_output, bs, slen):
        rnn_output = rnn_output.view(bs, slen, 2,
                                       self.rnn_dim).sum(dim=2)
        att = torch.bmm(torch.tanh(rnn_output),
                        self.att_w.repeat(bs, 1, 1))
        att_weights = F.softmax(att, dim=1)
        out_att = torch.tanh(torch.bmm(rnn_output.transpose(1, 2), att_weights).squeeze(2))
        return out_att, att_weights

    def forward(self, input):
        inp = self.dropout(self.embedding_layer(input))
        rnn_output, (_, _) = self.lstm(inp)
        rnn_output = self.dropout(rnn_output)
        out_att, weighted = self.attention(rnn_output, input.shape[0], input.shape[1])
        output = self.dropout(out_att)
        for layer in self.fc_layers:
            output = layer(output)
            output = self.dropout(output)
            output = F.relu(output)
        return self.linear_out(output), weighted
