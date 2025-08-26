import math
import torch
import torch.nn as nn

class SelfAttentionBlock(nn.Module):
            def __init__(self, dim: int, heads: int = 8, mlp_ratio: float = 2.0, dropout: float = 0.1):
                super().__init__()
                self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
                self.norm1 = nn.LayerNorm(dim)
                self.ffn = nn.Sequential(
                    nn.Linear(dim, int(dim * mlp_ratio)),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(int(dim * mlp_ratio), dim),
                )
                self.norm2 = nn.LayerNorm(dim)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                attn_out, _ = self.attn(x, x, x)
                x = self.norm1(x + attn_out)
                f = self.ffn(x)
                x = self.norm2(x + f)
                return x

class PromptGenModule(nn.Module):
    def __init__(self, prompt_dim=128, num_prompts=5):
        super().__init__()
        self.prompt_dim = prompt_dim
        self.num_prompts = num_prompts

        # lightweight extractor to reduce overhead
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        self.prompt_generator = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, num_prompts * prompt_dim),
        )

        self.prompt_embed = nn.Parameter(torch.randn(num_prompts, prompt_dim))
        nn.init.xavier_uniform_(self.prompt_embed)

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        prompt_weights = self.prompt_generator(features)
        prompt_weights = prompt_weights.view(-1, self.num_prompts, self.prompt_dim)
        prompts = prompt_weights + self.prompt_embed.unsqueeze(0)
        return prompts


class PromptInteractionModule(nn.Module):
    def __init__(self, feature_dim, prompt_dim=128, num_prompts=5):
        super().__init__()
        self.feature_dim = feature_dim
        self.prompt_dim = prompt_dim
        self.num_prompts = num_prompts

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=8, dropout=0.1, batch_first=True
        )
        self.prompt_proj = nn.Linear(prompt_dim, feature_dim)
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim),
        )

    def forward(self, features, prompts):
        projected_prompts = self.prompt_proj(prompts)
        attended_features, _ = self.cross_attention(
            query=features, key=projected_prompts, value=projected_prompts
        )
        features = self.norm1(features + attended_features)
        ffn_output = self.ffn(features)
        enhanced_features = self.norm2(features + ffn_output)
        return enhanced_features


class PromptAwareEncoder(nn.Module):
    def __init__(self, base_encoder, prompt_dim=128, num_prompts=5):
        super().__init__()
        self.base_encoder = base_encoder

        self.pgm = PromptGenModule(prompt_dim, num_prompts)

        if hasattr(base_encoder, 'config'):
            self.feature_dim = base_encoder.config.latent_channels
        else:
            self.feature_dim = 512

        self.pim = PromptInteractionModule(self.feature_dim, prompt_dim, num_prompts)
        self.intermediate_features = None
        self._register_hooks()

    def _register_hooks(self):
        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor) and output.dim() == 4:
                b, c, h, w = output.shape
                if c == self.feature_dim:
                    self.intermediate_features = output.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)

        if hasattr(self.base_encoder, 'conv_out'):
            self.base_encoder.conv_out.register_forward_hook(hook_fn)
        elif hasattr(self.base_encoder, 'mid_block') and hasattr(self.base_encoder.mid_block, 'resnets'):
            self.base_encoder.mid_block.resnets[-1].register_forward_hook(hook_fn)

    def forward(self, x):
        prompts = self.pgm(x)
        encoded = self.base_encoder(x)
        if self.intermediate_features is not None:
            enhanced_features = self.pim(self.intermediate_features, prompts)
            b, hw, c = enhanced_features.shape
            h = w = int(math.sqrt(hw))
            _ = enhanced_features.view(b, h, w, c).permute(0, 3, 1, 2)
        return encoded


class LatentPromptAdapter(nn.Module):
    """
    Adapter F that operates in latent space using PromptIR blocks.

    - Generates prompts from the degraded RGB image with `PromptGenModule`.
    - Interacts prompts with a learned latent feature stack via `PromptInteractionModule`.
    - Predicts a clean latent ``z_clean_hat`` from degraded latent ``z_degraded``.

    This module is intentionally lightweight; it can be trained while keeping
    the Stable Diffusion components (VAE encoder/decoder and UNet) frozen.
    """

    def __init__(
        self,
        latent_channels: int = 4,
        feature_dim: int = 256,
        prompt_dim: int = 128,
        num_prompts: int = 5,
        num_heads: int = 8,
        mode: str = "direct",  # "direct" predicts z_clean; "residual" predicts delta
    ) -> None:
        super().__init__()
        assert mode in ("direct", "residual"), "mode must be 'direct' or 'residual'"
        self.mode = mode

        # Prompt generator consumes degraded RGB image in [-1, 1] range
        self.pgm = PromptGenModule(prompt_dim=prompt_dim, num_prompts=num_prompts)

        # Shallow latent feature tower
        self.encoder = nn.Sequential(
            nn.Conv2d(latent_channels, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, feature_dim, 3, padding=1),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(feature_dim, 128, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, latent_channels, 3, padding=1),
        )

        class SelfAttentionBlock(nn.Module):
            def __init__(self, dim: int, heads: int = 8, mlp_ratio: float = 2.0, dropout: float = 0.1):
                super().__init__()
                self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
                self.norm1 = nn.LayerNorm(dim)
                self.ffn = nn.Sequential(
                    nn.Linear(dim, int(dim * mlp_ratio)),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(int(dim * mlp_ratio), dim),
                )
                self.norm2 = nn.LayerNorm(dim)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                attn_out, _ = self.attn(x, x, x)
                x = self.norm1(x + attn_out)
                f = self.ffn(x)
                x = self.norm2(x + f)
                return x

        # Two-stage [self-attn -> prompt-interaction] blocks
        self.sa1 = SelfAttentionBlock(feature_dim, heads=num_heads)
        self.pim1 = PromptInteractionModule(feature_dim=feature_dim, prompt_dim=prompt_dim, num_prompts=num_prompts)
        self.sa2 = SelfAttentionBlock(feature_dim, heads=num_heads)
        self.pim2 = PromptInteractionModule(feature_dim=feature_dim, prompt_dim=prompt_dim, num_prompts=num_prompts)

    def forward(self, z_degraded: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_degraded: latent tensor shaped [B, C=latent_channels, H, W]
            x_degraded: RGB degraded image in [-1, 1], shaped [B, 3, 512, 512] (or compatible)
        Returns:
            z_pred: predicted clean latent ("direct") or residual to add ("residual")
        """
        prompts = self.pgm(z_degraded)

        feats = self.encoder(z_degraded)
        b, c, h, w = feats.shape
        seq = feats.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)

        # Block 1: attention -> prompt block
        seq = self.sa1(seq)
        seq = self.pim1(seq, prompts)

        # Block 2: attention -> prompt block
        seq = self.sa2(seq)
        seq = self.pim2(seq, prompts)

        feats = seq.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        z_out = self.decoder(feats)

        if self.mode == "residual":
            return z_degraded + z_out
        return z_out
