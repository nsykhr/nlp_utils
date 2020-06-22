import numpy as np

import torch
from torch import nn
from torch.nn import functional as F


class SoftmaxLoss(nn.Module):
    """
    Given some embeddings and some targets, applies a linear layer
    to create logits over possible words and then returns the
    negative log likelihood.
    """
    def __init__(self,
                 num_words: int,
                 embedding_dim: int,
                 ignore_index: int = -100) -> None:
        super().__init__()

        self.softmax_w = nn.Parameter(
                torch.randn(embedding_dim, num_words) / np.sqrt(embedding_dim)
        )
        self.softmax_b = nn.Parameter(torch.zeros(num_words))
        self.ignore_index = ignore_index

    def forward(self, embeddings: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # pylint: disable=arguments-differ
        # embeddings is size (n, embedding_dim)
        # targets is (batch_size,) with the correct class id
        probs = F.log_softmax(
                torch.matmul(embeddings, self.softmax_w) + self.softmax_b,
                dim=-1
        )

        return F.nll_loss(probs, targets.long(), ignore_index=self.ignore_index, reduction='sum')
