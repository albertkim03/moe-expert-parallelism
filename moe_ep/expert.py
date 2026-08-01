"""A single expert.

An "expert" sounds exotic. It is not: it is an ordinary two-layer feed-forward
network, exactly the FFN block you would find inside any transformer layer. The
only thing that makes it an *expert* is that a router decides which tokens go
into it.

    x  ->  Linear(d_model, d_ff)  ->  activation  ->  Linear(d_ff, d_model)  ->  y

STEP 4 of the build plan.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """One expert FFN.

    Shapes
    ------
    input   (n_tokens, d_model)
    output  (n_tokens, d_model)

    `n_tokens` is whatever arrives — it changes every step under expert
    parallelism, because it depends on how the router happened to route.
    """

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=True)
        self.fc2 = nn.Linear(d_ff, d_model, bias=True)
        self.act = nn.ReLU() # TRADEOFF

        for p in self.parameters():
            p.is_expert = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ (n_tokens, d_model) -> (n_tokens, d_model) """
        x                              # input layer   — shape (d_model,)
        h = self.act(self.fc1(x))      # hidden layer  — shape (d_ff,)
        y = self.fc2(h)                # output layer  — shape (d_model,)

        return y
