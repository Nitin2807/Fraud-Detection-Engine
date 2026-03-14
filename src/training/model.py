from __future__ import annotations

import torch
from torch import nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, output_neurons_layer: int, dropout_rate: float, num_layers: int):
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        encoder_layers = []
        current_dim = input_dim
        next_dim = output_neurons_layer
        self.layer_sizes = []

        for _ in range(num_layers):
            encoder_layers.append(nn.Linear(current_dim, next_dim))
            encoder_layers.append(nn.BatchNorm1d(next_dim))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(dropout_rate))
            self.layer_sizes.append(next_dim)
            current_dim = next_dim
            next_dim = max(5, next_dim // 2)

        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        reverse_sizes = self.layer_sizes[::-1]
        current_dim = self.layer_sizes[-1]
        target_sizes = reverse_sizes[1:] + [input_dim]

        for target_dim in target_sizes:
            decoder_layers.append(nn.Linear(current_dim, target_dim))
            if target_dim != input_dim:
                decoder_layers.append(nn.BatchNorm1d(target_dim))
                decoder_layers.append(nn.ReLU())
                decoder_layers.append(nn.Dropout(dropout_rate))
            current_dim = target_dim

        self.decoder = nn.Sequential(*decoder_layers)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))
