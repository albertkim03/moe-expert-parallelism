"""
The router is a single small Linear layer that scores every expert for every
token, and picks the best k. It is the ONLY part of an MoE layer that is
replicated on every rank, which is precisely why its gradients need an
all-reduce and the experts' do not.
"""

import torch
import torch.nn as nn


class Router(nn.Module):
    """Scores experts for tokens and picks the top-k.

    Shapes
    ------
    input    x       (T, d_model)
    outputs  topk_idx    (T, k)   int64, which experts were chosen
             topk_gate   (T, k)   float, the probability of each chosen expert
             probs       (T, E)   float, the full softmax — needed for the aux loss
    """

    def __init__(self, d_model: int, num_experts: int, top_k: int = 1) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # No bias: a constant offset would bias every token toward the same experts regardless of content
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # Score experts
        logits = self.gate(x)                                   # (T, E)
        probs = torch.softmax(logits, dim=-1)                   # (T, E) | normalize each token's scores into a distribution over experts, summing to 1
        topk_gate, topk_idx = probs.topk(self.top_k, dim=-1)    # (T, k)

        # if k > 1, renormalise topk_gate so each row sums to 1 again
        if self.top_k > 1:
            topk_sum = topk_gate.sum(dim=-1, keepdim=True)      # (T, 1): one sum per token
            topk_gate = topk_gate / topk_sum

        return (topk_idx, topk_gate, probs)
