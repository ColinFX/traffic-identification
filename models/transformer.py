# https://www.kaggle.com/code/arunmohan003/transformer-from-scratch-using-pytorch

import math

import numpy as np
import torch
import torch.nn as nn


# TODO URGENT add positional encoding to TransformerEncoder
class PositionalEmbedding(torch.nn.Module):
    def __init__(self, seq_len: int, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.pos_embedding = torch.zeros(seq_len, self.embed_dim)
        for pos in range(seq_len):
            for i in range(0, self.embed_dim, 2):
                self.pos_embedding[pos, i] = math.sin(pos / (10000 ** (2 * i / self.embed_dim)))
                self.pos_embedding[pos, i + 1] = math.cos(pos / (10000 ** (2 * (i + 1) / self.embed_dim)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # in, out: batch_size * seq_len * embed_dim
        return x * math.sqrt(self.embed_dim) + self.pos_embedding


# class MultiHeadAttention(torch.nn.Module):
#     def __init__(self, embed_dim: int, n_heads: int = 5):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.n_heads = n_heads
#         self.single_head_dim = int(self.embed_dim / self.n_heads)
#         self.query_matrix = torch.nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
#         self.key_matrix = torch.nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
#         self.value_matrix = torch.nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
#         self.out = torch.nn.Linear(self.n_heads * self.single_head_dim, self.embed_dim)
#
#     def forward(self, key: torch.Tensor, query: torch.Tensor, value: torch.Tensor):
#         # key, query, value: batch_size * seq_len * embed_dim
#         batch_size = key.shape[0]
#         seq_len = key.shape[1]
#         key = key.view(batch_size, seq_len, self.n_heads, self.single_head_dim)
#         query = query.view(batch_size, seq_len, self.n_heads, self.single_head_dim)
#         value = value.view(batch_size, seq_len, self.n_heads, self.single_head_dim)
#         # key, query, value: batch_size * seq_len * n_heads * (embed_dim/n_heads)
#         k = self.key_matrix(key)
#         q = self.query_matrix(query)
#         v = self.value_matrix(value)
#         # k, q, v: batch_size * seq_len * n_heads * (embed_dim/n_heads)
#         k = k.transpose(1, 2)
#         q = q.transpose(1, 2)
#         v = v.transpose(1, 2)
#         # k, q, v: batch_size * n_heads * seq_len * (embed_dim/n_heads)
#         k_adjusted = k.transpose(-1, -2)  # batch_size * n_heads * (embed_dim/n_heads) * seq_len
#         product = torch.matmul(q, k_adjusted)  # batch_size * n_heads * seq_len * seq_len
#         product = product / math.sqrt(self.single_head_dim)
#         scores = torch.nn.functional.softmax(product, dim=-1)  # batch_size * n_heads * seq_len * seq_len
#         scores = torch.matmul(scores, v)  # batch_size * n_heads * seq_len * (embed_dim/n_heads)
#         concat = scores.transpose(1, 2).contiguous().view(
#             batch_size,
#             seq_len,
#             self.single_head_dim*self.n_heads
#         )  # batch_size * n_heads * embed_dim
#         return self.out(concat)
#
#
# class TransformerBlock(torch.nn.Module):
#     def __init__(self, embed_dim: int, expansion_factor: int = 4, n_heads: int = 5):
#         super().__init__()
#         self.attention = MultiHeadAttention(embed_dim, n_heads)
#         self.norm1 = torch.nn.LayerNorm(embed_dim)
#         self.norm2 = torch.nn.LayerNorm(embed_dim)
#         self.feed_forward = torch.nn.Sequential(
#             torch.nn.Linear(embed_dim, expansion_factor*embed_dim),
#             torch.nn.ReLU(),
#             torch.nn.Linear(expansion_factor*embed_dim, embed_dim)
#         )
#         self.dropout1 = torch.nn.Dropout(0.2)
#         self.dropout2 = torch.nn.Dropout(0.2)
#
#     def forward(self, key: torch.Tensor, query: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
#         # key, query, value: batch_size * seq_len * embed_dim
#         attention_out = self.attention(key, query, value)  # batch_size * seq_len * embed_dim
#         attention_residual_out = attention_out + value  # batch_size * seq_len * embed_dim
#         norm1_out = self.dropout1(self.norm1(attention_residual_out))  # batch_size * seq_len * embed_dim
#         feed_forward_out = self.feed_forward(norm1_out)  # batch_size * seq_len * embed_dim
#         feed_forward_residual_out = feed_forward_out + norm1_out  # batch_size * seq_len * embed_dim
#         return self.dropout2(self.norm2(feed_forward_residual_out))  # batch_size * seq_len * embed_dim
#
#
# class TransformerEncoder(torch.nn.Module):
#     def __init__(
#             self,
#             seq_len: int,
#             embed_dim: int,
#             num_layers: int = 2,
#             expansion_factor: int = 4,
#             n_heads: int = 5
#     ):
#         super().__init__()
#         self.positional_encoder = PositionalEmbedding(seq_len, embed_dim)
#         self.layers = torch.nn.ModuleList([
#             TransformerBlock(embed_dim, expansion_factor, n_heads) for _ in range(num_layers)
#         ])
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.positional_encoder(x)  # x: batch_size * seq_len * embed_dim
#         for layer in self.layers:
#             x = layer(x, x, x)
#         return x  # batch_size * seq_len * embed_dim
#
#
# class MLPClassifier(torch.nn.Module):
#     def __init__(self, seq_len: int, embed_dim: int, num_classes: int):
#         super().__init__()
#         self.seq_len = seq_len
#         self.embed_dim = embed_dim
#         self.layer1 = [torch.nn.Linear(embed_dim, 1) for _ in range(seq_len)]
#         self.relu = torch.nn.ReLU()
#         self.layer2 = torch.nn.Linear(seq_len, num_classes)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # batch_size * seq_len * embed_dim
#         x = x.view(-1, self.embed_dim)  # (batch_size*seq_len) * embed_dim
#         x = torch.cat([self.layer1[idx % self.seq_len](x[idx]) for idx in range(x.shape[0])])  # (batch_size*seq_len)
#         x = x.view(-1, self.seq_len)  # batch_size * seq_len
#         x = self.layer2(x)  # batch_size * num_classes
#         return torch.softmax(x, dim=0)
#
#
# class TransformerClassifier(torch.nn.Module):
#     def __init__(
#             self,
#             params: utils.HyperParams,
#             embed_dim: int,
#             num_classes: int,
#     ):
#         # TODO: use params
#         super().__init__()
#         self.encoder = TransformerEncoder(
#             seq_len=params.window_size*20,
#             embed_dim=embed_dim,
#             num_layers=params.num_layers,
#             expansion_factor=params.expansion_factor,
#             n_heads=params.n_heads)
#         self.mlp = MLPClassifier(
#             seq_len=params.window_size*20,
#             embed_dim=embed_dim,
#             num_classes=num_classes)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # x: batch_size * seq_len * embed_dim
#         x = self.encoder(x)
#         return self.mlp(x)


class TransformerEncoderClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO URGENT get all these parameters from call
        self.encoder = torch.nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=54, nhead=6),
            num_layers=6
        )
        self.linear = nn.Linear(in_features=540, out_features=9)

    def forward(self, data_batch: torch.Tensor):
        # batch_size * sequence_length * embedding_length
        data_batch = data_batch.permute(1, 0, 2)  # sequence_length * batch_size * embedding_length
        data_batch = self.encoder(data_batch)
        data_batch = data_batch.permute(1, 0, 2)  # batch_size * sequence_length * embedding_length
        data_batch = data_batch.reshape(data_batch.shape[0], -1)  # batch_size * sequence_length*embedding_length
        data_batch = self.linear(data_batch)  # batch_size * out_features
        return data_batch


class TransformerClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = torch.nn.Transformer(d_model=54, nhead=6, batch_first=True)
        self.linear = torch.nn.Linear(in_features=540, out_features=9)

    def forward(self, data_batch: torch.Tensor):
        data_batch = self.transformer(data_batch, torch.zeros_like(data_batch))
        data_batch = data_batch.reshape(data_batch.shape[0], -1)
        data_batch = self.linear(data_batch)
        return data_batch


def loss_fn(outputs: torch.Tensor, labels: torch.Tensor) -> torch.FloatTensor:
    loss = nn.CrossEntropyLoss()
    return loss(outputs, labels)


def accuracy(outputs: np.ndarray[np.float32], labels: np.ndarray[np.int64]) -> np.float64:
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs == labels) / float(labels.size)


metrics = {"accuracy": accuracy}
