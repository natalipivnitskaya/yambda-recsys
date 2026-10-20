"""Neural rankers: concatenated feature embeddings -> (optional cross network) -> MLP -> task heads."""
from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn

from recsys.ranking.features import PiecewiseLinearEncoding, UnifiedEmbedding


class FeatureEncoder(nn.Module):
    """Concatenates encoded numeric, categorical and multivalent features into one vector ``x0``."""

    def __init__(
        self,
        numeric: PiecewiseLinearEncoding,
        categorical: UnifiedEmbedding,
        sparse_columns: Sequence[str],
        multivalent_columns: Sequence[str],
    ) -> None:
        super().__init__()
        self.numeric = numeric
        self.categorical = categorical
        self.sparse_columns = list(sparse_columns)
        self.multivalent_columns = list(multivalent_columns)
        self.out_dim = numeric.out_dim + sum(categorical.dims[c] for c in self.sparse_columns + self.multivalent_columns)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        parts = [self.numeric(batch["dense"])]
        parts += [self.categorical.lookup(c, batch["sparse"][:, i]) for i, c in enumerate(self.sparse_columns)]
        parts += [
            self.categorical.lookup_bag(c, batch[f"{c}_ids"], batch[f"{c}_lengths"])
            for c in self.multivalent_columns
        ]
        return torch.cat(parts, dim=1)


def mlp(in_dim: int, hidden: Sequence[int], dropout: float) -> nn.Sequential:
    layers = []
    for width in hidden:
        layers += [nn.Linear(in_dim, width), nn.ReLU(), nn.Dropout(dropout)]
        in_dim = width
    return nn.Sequential(*layers)


class CrossNetwork(nn.Module):
    """DCN-v2 cross network with low-rank experts (Wang et al., "DCN V2", 2021).

    Each layer computes ``x_{l+1} = x_l + sum_k g_k(x_l) * x0 ⊙ (U_k V_kᵀ x_l + b_k)``,
    i.e. explicit feature crosses of growing order. ``num_experts=1`` is the plain
    low-rank DCN-v2 layer; more experts give the "mixture of low-rank experts" variant
    with a softmax gate per layer.
    """

    def __init__(self, dim: int, num_layers: int, rank: int, num_experts: int = 1) -> None:
        super().__init__()
        self.V = nn.ParameterList([nn.Parameter(torch.randn(num_experts, dim, rank) / dim ** 0.5) for _ in range(num_layers)])
        self.U = nn.ParameterList([nn.Parameter(torch.randn(num_experts, rank, dim) / rank ** 0.5) for _ in range(num_layers)])
        self.b = nn.ParameterList([nn.Parameter(torch.zeros(num_experts, dim)) for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(dim, num_experts) for _ in range(num_layers)]) if num_experts > 1 else None

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        x = x0
        for layer, (V, U, b) in enumerate(zip(self.V, self.U, self.b)):
            low_rank = torch.einsum("bd,kdr->bkr", x, V)
            experts = x0.unsqueeze(1) * (torch.einsum("bkr,krd->bkd", low_rank, U) + b)  # (batch, experts, dim)
            if self.gates is None:
                mixed = experts[:, 0]
            else:
                weights = torch.softmax(self.gates[layer](x), dim=-1).unsqueeze(-1)
                mixed = (weights * experts).sum(dim=1)
            x = x + mixed
        return x


class Ranker(nn.Module):
    """Shared bottom (encoder [+ cross network] + MLP) with one logit per task."""

    def __init__(
        self,
        encoder: FeatureEncoder,
        tasks: Sequence[str],
        hidden: Sequence[int] = (512, 256),
        cross_layers: int = 0,
        cross_rank: int = 64,
        cross_experts: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.tasks = list(tasks)
        self.encoder = encoder
        dim = encoder.out_dim
        self.cross = CrossNetwork(dim, cross_layers, cross_rank, cross_experts) if cross_layers else None
        self.deep = mlp(dim, hidden, dropout)
        self.heads = nn.Linear(hidden[-1], len(self.tasks))

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.encoder(batch)
        if self.cross is not None:
            x = self.cross(x)
        return self.heads(self.deep(x))
