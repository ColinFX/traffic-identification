import math

import torch
import torch.nn as nn

from models.lstm import LSTMClassifier


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, seq_len: int, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.pos_embedding = torch.zeros(seq_len, self.embed_dim)
        for pos in range(seq_len):
            for i in range(0, self.embed_dim, 2):
                self.pos_embedding[pos, i] = math.sin(pos / (100 ** (2 * (i + 1) / self.embed_dim)))
                if i + 1 < self.embed_dim:
                    self.pos_embedding[pos, i + 1] = math.cos(pos / (100 ** (2 * (i + 2) / self.embed_dim)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # in, out: batch_size * seq_len * embed_dim
        return x * math.sqrt(self.embed_dim) + self.pos_embedding.to(x.device)


class TransformerEncoderClassifier(nn.Module):
    def __init__(
            self,
            raw_embedding_len: int,
            sequence_length: int,
            num_classes: int,
            target_embedding_len: int = 64,
            transformer_num_head: int = 8,
            transformer_num_layers: int = 6,
            lstm_hidden_size: int = 256,
            lstm_num_layers: int = 5,
            upstream_model: str = "linear",
            downstream_model: str = "linear"
    ):
        """
        upstream_model: str, one of ("padding", "linear")
        downstream_model: str, one of ("linear", "lstm")
        """
        super().__init__()
        self.raw_embedding_len = raw_embedding_len
        self.target_embedding_len = target_embedding_len
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.upstream_model = upstream_model
        self.downstream_model = downstream_model

        self.positional_embedding = PositionalEmbedding(seq_len=sequence_length, embed_dim=raw_embedding_len)

        if upstream_model == "linear":
            self.linear1 = torch.nn.Linear(in_features=raw_embedding_len, out_features=target_embedding_len)
        elif upstream_model != "padding":
            raise ValueError("upstream model has to be one of ('padding', 'linear')")

        self.encoder = torch.nn.TransformerEncoder(
            encoder_layer=torch.nn.TransformerEncoderLayer(
                d_model=target_embedding_len,
                nhead=transformer_num_head,
                dropout=0.2,
            ),
            num_layers=transformer_num_layers
        )

        if downstream_model == "linear":
            self.linear2 = torch.nn.Linear(in_features=target_embedding_len*sequence_length, out_features=num_classes)
        elif downstream_model == "lstm":
            self.lstm_classifier = LSTMClassifier(
                embedding_len=target_embedding_len,
                num_classes=num_classes,
                hidden_size=lstm_hidden_size,
                num_layers=lstm_num_layers
            )
        else:
            raise ValueError("downstream model has to be one of ('linear', 'lstm')")

    def forward(self, data_batch: torch.Tensor):
        # batch_size * sequence_length * raw_embedding_length
        data_batch = self.positional_embedding(data_batch)

        if self.upstream_model == "linear":
            data_batch = self.linear1(data_batch)
        elif self.upstream_model == "padding":
            data_batch = torch.nn.functional.pad(data_batch, (0, self.target_embedding_len-self.raw_embedding_len))
        # batch_size * sequence_length * target_embedding_length

        data_batch = data_batch.permute(1, 0, 2)
        # sequence_length * batch_size * target_embedding_length
        data_batch = self.encoder(data_batch)
        data_batch = data_batch.permute(1, 0, 2)
        # batch_size * sequence_length * target_embedding_length

        if self.downstream_model == "linear":
            data_batch = data_batch.reshape(data_batch.shape[0], -1)
            # batch_size * sequence_length*target_embedding_length
            data_batch = self.linear2(data_batch)
        elif self.downstream_model == "lstm":
            data_batch = self.lstm_classifier(data_batch)
        # batch_size * num_classes

        return data_batch
