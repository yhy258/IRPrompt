import torch
import torch.nn as nn


class SSLAdapter(nn.Module):
    """Project SSL global embedding into a sequence of tokens for cross-attention.

    - Inputs: (B, Din)
    - Outputs: (B, T, Dout)
    """

    def __init__(self, in_dim: int, out_dim: int = 1024, num_tokens: int = 8) -> None:
        super().__init__()
        assert num_tokens > 0, "num_tokens must be > 0"
        self.num_tokens = num_tokens
        self.out_dim = out_dim

        self.mapper = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim * num_tokens),
        )

        # Optional learned bias tokens (initial prompts)
        self.token_bias = nn.Parameter(torch.zeros(num_tokens, out_dim))
        self.project = nn.Linear(out_dim, out_dim)
        nn.init.trunc_normal_(self.token_bias, std=0.02)

    def forward(self, global_emb: torch.Tensor) -> torch.Tensor:
        bsz = global_emb.shape[0]
        projected = self.mapper(global_emb)  # (B, out_dim*T)
        tokens = projected.view(bsz, self.num_tokens, self.out_dim)
        tokens = tokens * self.token_bias.unsqueeze(0)
        tokens = self.project(tokens)
        return tokens



















