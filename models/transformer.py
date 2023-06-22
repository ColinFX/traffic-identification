# https://www.kaggle.com/code/arunmohan003/transformer-from-scratch-using-pytorch

import math

import torch

import utils


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
        return x * math.sqrt(self.embed_dim) + self.pos_embedding


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, embed_dim: int, n_heads: int = 5):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.single_head_dim = int(self.embed_dim / self.n_heads)
        self.query_matrix = torch.nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        self.key_matrix = torch.nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        self.value_matrix = torch.nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        self.out = torch.nn.Linear(self.n_heads * self.single_head_dim, self.embed_dim)

    def forward(self, key: torch.Tensor, query: torch.Tensor, value: torch.Tensor):
        batch_size = key.shape[0]
        seq_len = key.shape[1]
        seq_len_query = query.shape[1]
        key = key.view(batch_size, seq_len, self.n_heads, self.single_head_dim)
        query = query.view(batch_size, seq_len_query, self.n_heads, self.single_head_dim)
        value = value.view(batch_size, seq_len, self.n_heads, self.single_head_dim)
        k = self.key_matrix(key)
        q = self.query_matrix(query)
        v = self.value_matrix(value)
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        k_adjusted = k.transpose(-1, -2)
        product = torch.matmul(q, k_adjusted)
        product = product / math.sqrt(self.single_head_dim)
        scores = torch.nn.functional.softmax(product, dim=-1)
        scores = torch.matmul(scores, v)
        concat = scores.transpose(1, 2).contiguous().view(
            batch_size,
            seq_len_query,
            self.single_head_dim*self.n_heads
        )
        return self.out(concat)


class TransformerBlock(torch.nn.Module):
    def __init__(self, embed_dim: int, expansion_factor: int = 4, n_heads: int = 5):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, n_heads)
        self.norm1 = torch.nn.LayerNorm(embed_dim)
        self.norm2 = torch.nn.LayerNorm(embed_dim)
        self.feed_forward = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, expansion_factor*embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(expansion_factor*embed_dim, embed_dim)
        )
        self.dropout1 = torch.nn.Dropout(0.2)
        self.dropout2 = torch.nn.Dropout(0.2)

    def forward(self, key: torch.Tensor, query: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        attention_out = self.attention(key, query, value)
        attention_residual_out = attention_out + value
        norm1_out = self.dropout1(self.norm1(attention_residual_out))
        feed_forward_out = self.feed_forward(norm1_out)
        feed_forward_residual_out = feed_forward_out + norm1_out
        return self.dropout2(self.norm2(feed_forward_residual_out))


class TransformerEncoder(torch.nn.Module):
    def __init__(self, seq_len: int, embed_dim: int, num_layers: int = 2, expansion_factor: int = 4, n_heads: int = 5):
        super().__init__()
        self.positional_encoder = PositionalEmbedding(seq_len, embed_dim)
        self.layers = torch.nn.ModuleList([
            TransformerBlock(embed_dim, expansion_factor, n_heads) for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.positional_encoder(x)
        for layer in self.layers:
            x = layer(x, x, x)
        return x


class MLPClassifier(torch.nn.Module):
    def __init__(self, seq_len: int, embed_dim: int, num_classes: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.layer1 = [torch.nn.Linear(embed_dim, 1) for _ in range(seq_len)]
        self.relu = torch.nn.ReLU()
        self.layer2 = torch.nn.Linear(seq_len, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.layer1[idx](x[idx]) for idx in range(self.embed_dim)])
        x = self.relu(x)
        x = self.layer2(x)
        return torch.softmax(x, dim=0)


class TransformerClassifier(torch.nn.Module):
    def __init__(
            self,
            params: utils.HyperParams,
            embed_dim: int,
            num_classes: int,
    ):
        # TODO: use params
        super().__init__()
        self.encoder = TransformerEncoder(
            seq_len=params.window_size*20,
            embed_dim=embed_dim,
            num_layers=params.num_layers,
            expansion_factor=params.expansion_factor,
            n_heads=params.n_heads)
        self.mlp = MLPClassifier(
            seq_len=params.window_size*20,
            embed_dim=embed_dim,
            num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        return self.mlp(x)


def loss_fn(y_pred_proba: torch.Tensor, y_true: torch.Tensor) -> torch.FloatTensor:
    return torch.nn.CrossEntropyLoss()(y_pred_proba, y_true)


def accuracy(y_pred_proba: torch.Tensor, y_true: torch.Tensor) -> torch.FloatTensor:
    y_pred = torch.argmax(y_pred_proba, dim=1)
    return torch.sum(y_pred == y_true) / y_true.shape[0]


metrics = {"accuracy": accuracy}
