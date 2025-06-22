import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter, Module


class GCNLayer(Module):
    def __init__(self, input_dim, output_dim, bias=True):
        super(GCNLayer, self).__init__()
        self.weight = Parameter(torch.FloatTensor(input_dim, output_dim))
        self.bias = Parameter(torch.FloatTensor(output_dim)) if bias else None

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, X, A):
        support = torch.spmm(X, self.weight)
        output = torch.spmm(A, support)
        return output + self.bias if self.bias is not None else output


class GCN(Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super(GCN, self).__init__()
        self.gc1 = GCNLayer(input_dim, hidden_dim)
        self.dropout = dropout

    def forward(self, X, adj):
        X = self.gc1(X, adj)
        X = F.relu(X)
        X = F.dropout(X, self.dropout, training=self.training)
        return X


class AttentionHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AttentionHead, self).__init__()
        self.q = nn.Linear(input_dim, output_dim)
        self.k = nn.Linear(input_dim, output_dim)
        self.v = nn.Linear(input_dim, output_dim)

    def forward(self, x, attention_mask=None):
        Q, K, V = self.q(x), self.k(x), self.v(x)
        scale = Q.size(1) ** 0.5
        scores = torch.bmm(Q, K.transpose(1, 2)) / scale
        scores = scores.masked_fill(attention_mask, -1e9)
        attn = F.softmax(scores, dim=-1)
        return torch.bmm(attn, V)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, input_dim, output_dim):
        super(MultiHeadAttention, self).__init__()
        self.heads = nn.ModuleList([
            AttentionHead(input_dim, output_dim) for _ in range(num_heads)
        ])
        self.linear = nn.Linear(output_dim * num_heads, input_dim)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x, attention_mask):
        head_outputs = [head(x, attention_mask) for head in self.heads]
        concatenated = torch.cat(head_outputs, dim=-1)
        return self.norm(self.linear(concatenated))


class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=4, dropout=0.1):
        super(Encoder, self).__init__()
        self.attention = MultiHeadAttention(num_heads, input_dim, output_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(output_dim, input_dim),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x, attention_mask):
        context = self.attention(x, attention_mask)
        output = self.feed_forward(context)
        return self.norm(output)


class BertSimpleModel(nn.Module):
    def __init__(self, vocab_size, input_dim, output_dim, attention_heads=4, user_size=1, mode=0):
        super(BertSimpleModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, input_dim)
        self.gcn = GCN(input_dim, input_dim, dropout=0.5)
        self.encoder = Encoder(input_dim, output_dim, attention_heads)
        self.token_pred_layer = nn.Linear(input_dim, vocab_size)
        self.user_pred_layer = nn.Linear(input_dim, user_size)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.mode = mode

    def forward(self, input_tensor, attention_mask, X_vocab, adj):
        if self.mode == 0:
            # Token Prediction
            embedded_vocab = self.embedding(X_vocab)
            gcn_output = self.gcn(embedded_vocab, adj)
            input_embedded = gcn_output[input_tensor]
            encoded = self.encoder(input_embedded, attention_mask)
            return self.softmax(self.token_pred_layer(encoded))
        else:
            # User Classification
            input_embedded = self.embedding(input_tensor)
            encoded = self.encoder(input_embedded, attention_mask)
            return self.user_pred_layer(encoded[:, 0])

    def change_mode(self, new_mode):
        if new_mode not in [0, 1]:
            raise ValueError("Mode must be 0 (MLM) or 1 (Classification)")
        self.mode = new_mode
