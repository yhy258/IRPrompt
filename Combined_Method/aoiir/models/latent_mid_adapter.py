import torch
import torch.nn as nn


class ZeroConv2d(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1):
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        nn.init.zeros_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class LatentMidAdapter(nn.Module):
    """
    Trimmed Control adaptor that maps latent z_y (B,4,H,W) to a mid-block residual (B,C_mid,H,W).
    Uses zero-initialized conv so base UNet behavior is preserved at start.
    """
    def __init__(self, in_channels: int, mid_channels: int, hidden_channels: int = 128, copy_from_unet: nn.Module | None = None):
        super().__init__()
        self.proj_in = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.body = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.proj_out = ZeroConv2d(hidden_channels, mid_channels, kernel_size=1)

        # Optional: copy-init from UNet conv_in if shapes match (SD v2: 4->320 conv)
        if copy_from_unet is not None and hasattr(copy_from_unet, "conv_in"):
            src = copy_from_unet.conv_in
            if isinstance(src, nn.Conv2d) and src.weight.shape == self.proj_in.weight.shape:
                with torch.no_grad():
                    self.proj_in.weight.copy_(src.weight)
                    if self.proj_in.bias is not None and src.bias is not None:
                        self.proj_in.bias.copy_(src.bias)

    def forward(self, z_y: torch.Tensor) -> torch.Tensor:
        h = self.proj_in(z_y)
        h = self.body(h)
        h = self.proj_out(h)
        return h


