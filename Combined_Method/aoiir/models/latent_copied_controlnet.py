import copy
import torch
import torch.nn as nn


class ZeroConv2d(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(in_channels, out_channels, kernel_size=1, padding=0)
        nn.init.zeros_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class LatentCopiedControlNet(nn.Module):
    """
    ControlNet-like adaptor that accepts latent (B,4,H,W) and is copy-initialized from a base UNet encoder.
    It deep-copies the UNet and uses forward hooks to capture down/mid features when run on z_cond.
    Zero-initialized 1x1 convs map captured features into residuals for the base UNet.
    """
    def __init__(self, base_unet: nn.Module):
        super().__init__()
        in_dim = base_unet.in_channels
        self.z_residual_zero = ZeroConv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1)
        self.z_t_conv = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1)
        # Deep-copy encoder + time/text paths by cloning full UNet for simplicity
        self.control_unet = copy.deepcopy(base_unet)
        # Build zero-conv heads per down block and one for mid
        self.down_zero = nn.ModuleList()
        for db in self.control_unet.down_blocks:
            # assume output channels equals db.out_channels via hidden states' C after the block
            # we will infer at runtime and create lazily if needed
            self.down_zero.append(None)
        self.mid_zero = None

        # Buffers to store hooks
        self._captured = {}
        # Capture down and mid features via hooks (ordered)
        self._mid_feat = None
        self._down_feats_seq = []

        # Pre-initialize zero-conv heads for all down resnet outputs (copy-time), and for mid
        self.down_zero = nn.ModuleList()
        for db in self.control_unet.down_blocks:
            if hasattr(db, 'resnets'):
                for res in db.resnets:
                    # infer channel dimension from conv2
                    if hasattr(res, 'conv2') and isinstance(res.conv2, nn.Conv2d):
                        ch = res.conv2.out_channels
                    else:
                        # fallback to first conv weight if structure differs
                        ch = next((m.out_channels for m in res.modules() if isinstance(m, nn.Conv2d)), None)
                        if ch is None:
                            raise RuntimeError("Unable to infer channels for down resnet")
                    self.down_zero.append(ZeroConv2d(ch, ch))
                    # register hook to collect outputs in the same order
                    res.register_forward_hook(self._append_down_feat)

        # Mid channels from last resnet of mid_block
        if hasattr(self.control_unet.mid_block, 'resnets') and len(self.control_unet.mid_block.resnets) > 0:
            res_m = self.control_unet.mid_block.resnets[-1]
            if hasattr(res_m, 'conv2') and isinstance(res_m.conv2, nn.Conv2d):
                ch_m = res_m.conv2.out_channels
            else:
                ch_m = next((m.out_channels for m in res_m.modules() if isinstance(m, nn.Conv2d)), None)
                if ch_m is None:
                    raise RuntimeError("Unable to infer channels for mid resnet")
            self.mid_zero = ZeroConv2d(ch_m, ch_m)
        else:
            # fallback: run-time infer will raise if not present
            self.mid_zero = None

        def _store_mid(module, args, output):
            self._mid_feat = output
        self.control_unet.mid_block.register_forward_hook(_store_mid)

    def _make_store_hook(self, key: str):
        def hook(module, args, output):
            # output is a hidden state tensor for blocks
            self._captured[key] = output
        return hook

    def forward(self, z_t: torch.Tensor, sample: torch.Tensor, timestep: torch.Tensor, encoder_hidden_states: torch.Tensor):
        # Run copied UNet; hook will capture mid feature
        sample = self.z_t_conv(z_t) + self.z_residual_zero(sample)
        self._down_feats_seq = []
        _ = self.control_unet(sample=sample, timestep=timestep, encoder_hidden_states=encoder_hidden_states)
        mid_feat = self._mid_feat
        if mid_feat is None:
            raise RuntimeError("mid feature not captured")
        if self.mid_zero is None:
            # create once if not set (should not happen if mid has resnets)
            self.mid_zero = ZeroConv2d(mid_feat.shape[1], mid_feat.shape[1]).to(mid_feat.device)
        mid_residual = self.mid_zero.to(mid_feat.device)(mid_feat)
        # Assemble down residuals in call order matching UNet resnet sequence
        down_residuals = []
        if len(self.down_zero) != len(self._down_feats_seq):
            raise RuntimeError(f"down heads ({len(self.down_zero)}) != captured features ({len(self._down_feats_seq)})")
        for i, feat in enumerate(self._down_feats_seq):
            down_residuals.append(self.down_zero[i].to(feat.device)(feat))
        return tuple(down_residuals), mid_residual

    def _append_down_feat(self, module, args, output):
        # Called in forward order; keep sequence
        self._down_feats_seq.append(output)


