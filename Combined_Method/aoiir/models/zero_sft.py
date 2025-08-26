import torch
import torch.nn as nn
import torch.nn.functional as F


class ZeroSFTBlock(nn.Module):
    """
    Zero-initialized FiLM-style modulator that produces per-channel gamma/beta.
    Conditioning comes from concatenated pooled text embedding and pooled control features.
    """
    def __init__(self, channel_dim: int, text_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.channel_dim = channel_dim
        self.text_dim = text_dim

        self.mlp = nn.Sequential(
            nn.Linear(text_dim + channel_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * channel_dim),
        )
        # Zero-init last layer to start as identity transformation
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, feature_map: torch.Tensor, pooled_text: torch.Tensor) -> torch.Tensor:
        b, c, h, w = feature_map.shape
        assert c == self.channel_dim, f"Channel mismatch: {c} vs {self.channel_dim}"
        # Global average pool control features
        ctrl_vec = F.adaptive_avg_pool2d(feature_map, output_size=1).flatten(1)  # [B, C]
        # Fuse with pooled text
        fused = torch.cat([pooled_text, ctrl_vec], dim=1)  # [B, text_dim + C]
        gamma_beta = self.mlp(fused)  # [B, 2C]
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = gamma.view(b, c, 1, 1)
        beta = beta.view(b, c, 1, 1)
        return feature_map * (1.0 + gamma) + beta


class ZeroSFTModulator(nn.Module):
    """
    Holds multiple ZeroSFTBlocks keyed by channel size.
    Applies modulation to lists of residual feature maps produced by ControlNet per UNet block.
    """
    def __init__(self, text_hidden_dim: int = 1024, known_channel_dims=None, hidden_dim: int = 512):
        super().__init__()
        if known_channel_dims is None:
            # Typical SD v2 base UNet channels
            known_channel_dims = [320, 640, 1280]
        self.text_hidden_dim = text_hidden_dim
        self.hidden_dim = hidden_dim

        self.blocks = nn.ModuleDict({
            str(c): ZeroSFTBlock(channel_dim=c, text_dim=text_hidden_dim, hidden_dim=hidden_dim)
            for c in known_channel_dims
        })

    def _get_block(self, channel_dim: int) -> ZeroSFTBlock:
        key = str(channel_dim)
        if key not in self.blocks:
            # Lazily create a new block for unseen channel sizes
            self.blocks[key] = ZeroSFTBlock(channel_dim=channel_dim, text_dim=self.text_hidden_dim, hidden_dim=self.hidden_dim)
        return self.blocks[key]

    def forward(self, residuals_list, pooled_text: torch.Tensor):
        """
        residuals_list: list[Tensor] or tuple[Tensor], each [B, C, H, W]
        pooled_text: [B, text_hidden_dim]
        Returns list with the same structure, modulated.
        """
        if residuals_list is None:
            return None
        modulated = []
        for feat in residuals_list:
            if feat is None:
                modulated.append(None)
                continue
            c = feat.shape[1]
            block = self._get_block(c)
            modulated.append(block(feat, pooled_text))
        return modulated















