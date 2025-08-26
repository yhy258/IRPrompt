import warnings
from typing import Dict

import torch
from torchmetrics.functional import structural_similarity_index_measure as ssim


def _to01(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)


@torch.no_grad()
def psnr_ssim(x: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
    x01 = _to01(x)
    y01 = _to01(y)
    mse = torch.mean((x01 - y01) ** 2)
    eps = 1e-8
    psnr_val = 10.0 * torch.log10(1.0 / torch.clamp(mse, min=eps))
    ssim_val = ssim(x01, y01, data_range=1.0)
    return {"psnr": psnr_val, "ssim": ssim_val}


@torch.no_grad()
def lpips(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    try:
        from torch_fidelity.sample_similarity_lpips import SampleSimilarityLPIPS
        # Create once per call to keep module self-contained. Run on CPU to avoid dtype/device mismatch.
        metric = SampleSimilarityLPIPS("lpips-vgg16", sample_similarity_resize=256, sample_similarity_dtype="float32").eval()
        x01 = _to01(x).float().cpu()
        y01 = _to01(y).float().cpu()
        return metric(x01, y01).mean().to(x.device)
    except Exception as e:
        warnings.warn(f"LPIPS metric unavailable: {e}")
        return torch.tensor(float('nan'), device=x.device)


@torch.no_grad()
def dists(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    try:
        import piq
        x01 = _to01(x)
        y01 = _to01(y)
        return piq.DISTS(reduction='mean').to(x.device)(x01, y01)
    except Exception as e:
        warnings.warn(f"DISTS metric unavailable: {e}")
        return torch.tensor(float('nan'), device=x.device)


# Removed CLIPIQA per user request


