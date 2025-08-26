import argparse
import os
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image

from aoiir.datasets.multi_degradation import MultiDegradationDataset
from aoiir.metrics.metrics import psnr_ssim
from aoiir.pipelines.dps_promptir import DPSPromptIRPipeline


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to01(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)


def build_subset(dataset: MultiDegradationDataset, subset_ratio: float, seed: int) -> Subset:
    total = len(dataset)
    n = max(1, int(total * subset_ratio))
    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total, generator=g)[:n].tolist()
    return Subset(dataset, indices)


def try_load_adapter_from_ckpt(adapter: torch.nn.Module, ckpt_path: str) -> None:
    """
    Load adapter weights from either:
    - Full Lightning checkpoint containing 'state_dict' with keys like 'pipeline.adapter.*' or 'adapter.*'
    - Raw adapter state_dict saved directly
    """
    ckpt_path_folder = '/home/joon/ImageRestoration-AllInOne/Combined_Method/runs/dps_promptir/lightning_logs/version_16/checkpoints'
    ckpt_files = os.listdir(ckpt_path_folder)
    ckpt_path = os.path.join(ckpt_path_folder, ckpt_files[-1])
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # If Lightning checkpoint
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    # Detect prefix
    prefixes = [
        "pipeline.adapter.",
        "adapter.",
        "model.pipeline.adapter.",
    ]

    filtered: Dict[str, torch.Tensor] = {}
    found_prefix = None
    for p in prefixes:
        keys = [k for k in state_dict.keys() if k.startswith(p)]
        if len(keys) > 0:
            found_prefix = p
            for k in keys:
                filtered[k[len(p):]] = state_dict[k]
            break

    if not filtered:
        # Assume it's already adapter-only
        if isinstance(state_dict, dict):
            filtered = state_dict
        else:
            raise RuntimeError("Unsupported checkpoint format for adapter loading.")

    missing, unexpected = adapter.load_state_dict(filtered, strict=False)
    if missing:
        print(f"[warn] Missing adapter keys: {len(missing)} (showing first 10): {missing[:10]}")
    if unexpected:
        print(f"[warn] Unexpected adapter keys: {len(unexpected)} (showing first 10): {unexpected[:10]}")
    print(f"Loaded adapter weights from '{ckpt_path}' with prefix='{found_prefix}'.")


@torch.no_grad()
def run_batch(args, pipeline: DPSPromptIRPipeline, x_D: torch.Tensor, x_G: torch.Tensor, steps: int, cfg: float, do_classifier_free_guidance: bool, num_forwards: int = 3, manifold_eq: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:

    # Compute posterior latents
    z_D, z_G, _, _, _ = pipeline(x_D, x_G)
    z_D = pipeline.several_forwards(x_D, num_forwards=num_forwards) # N, B, C, H, W
    x_direct = pipeline.decode(z_G)

    # DPS sampling (internally enables grad for measurement step)
    if args.sampling_method == 'dps':
        z_rec = pipeline.dps_sample(z_D, num_steps=steps, dps_scale=args.dps_scale, cfg=cfg, do_classifier_free_guidance=do_classifier_free_guidance, manifold_eq=manifold_eq)
    elif args.sampling_method == 'pixel_space_dps':
        z_rec = pipeline.pixel_space_dps_sample(z_D, num_steps=steps, dps_scale=args.pixel_dps_scale, cfg=cfg, do_classifier_free_guidance=do_classifier_free_guidance, manifold_eq=manifold_eq)
    elif args.sampling_method == 'beyond_first_tweedie':
        z_rec = pipeline.beyond_first_tweedie(z_D, num_steps=steps, stoc_avg_steps=args.stoc_avg_steps, rand_num=args.rand_num, pixel_dps_scale=args.pixel_dps_scale, second_order_scale=args.second_order_scale)
    elif args.sampling_method == 'combined_dps':
        z_rec = pipeline.combined_dps_sample(z_D, num_steps=steps, latent_dps_scale=args.dps_scale, pixel_dps_scale=args.pixel_dps_scale, cfg=cfg, do_classifier_free_guidance=do_classifier_free_guidance, manifold_eq=manifold_eq)
    else:
        raise ValueError(f"Invalid sampling method: {args.sampling_method}")
    x_rec = pipeline.decode(z_rec)
    return x_direct, x_rec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/home/joon/ImageRestoration-AllInOne/Combined_Method/aoiir/datasets/dataset')
    parser.add_argument('--sd_model', type=str, default='stabilityai/stable-diffusion-2-1-base')
    parser.add_argument('--adapter_ckpt', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--num_forwards', type=int, default=3)
    parser.add_argument('--stoc_avg_steps', type=int, default=2)
    parser.add_argument('--rand_num', type=int, default=3)
    parser.add_argument('--dps_scale', type=float, default=0)
    parser.add_argument('--pixel_dps_scale', type=float, default=0)
    parser.add_argument('--second_order_scale', type=float, default=0)
    parser.add_argument('--sampling_method', type=str, default='dps')
    parser.add_argument('--cfg', type=float, default=0)
    parser.add_argument('--do_classifier_free_guidance', type=bool, default=False)
    parser.add_argument('--scheduler_type', type=str, default='ddim')
    parser.add_argument('--subset_ratio', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out_dir', type=str, default='inference_dps_promptir')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--manifold_eq', type=bool, default=False)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    # Dataset and loader (subset 1%)
    full_ds = MultiDegradationDataset(dataset_root=args.data_root, patch_size=256)
    sub_ds = build_subset(full_ds, args.subset_ratio, args.seed)
    loader = DataLoader(sub_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Build pipeline and load adapter weights
    device = torch.device(args.device)
    pipeline = DPSPromptIRPipeline(sd_model_name=args.sd_model, adapter=None, scheduler_type=args.scheduler_type, device=str(device))
    pipeline.adapter.eval()
    try_load_adapter_from_ckpt(pipeline.adapter, args.adapter_ckpt)
    pipeline.to(device)

    # Metrics accumulators
    sum_psnr_direct = 0.0
    sum_ssim_direct = 0.0
    sum_psnr_dps = 0.0
    sum_ssim_dps = 0.0
    count_images = 0

    for batch_idx, (x_D, x_G) in enumerate(loader):
        x_D = x_D.to(device)
        x_G = x_G.to(device)

        x_direct, x_rec = run_batch(args, pipeline, x_D, x_G, steps=args.steps, cfg=args.cfg, do_classifier_free_guidance=args.do_classifier_free_guidance, num_forwards=args.num_forwards, manifold_eq=args.manifold_eq)

        # Metrics (per-batch average)
        m1 = psnr_ssim(x_direct, x_G)
        m2 = psnr_ssim(x_rec, x_G)
        bsz = x_D.size(0)
        sum_psnr_direct += float(m1['psnr']) * bsz
        sum_ssim_direct += float(m1['ssim']) * bsz
        sum_psnr_dps += float(m2['psnr']) * bsz
        sum_ssim_dps += float(m2['ssim']) * bsz
        count_images += bsz

        # Save grid: degraded | clean | posterior-decode | dps-decode
        grid = []
        n = min(4, x_D.size(0))
        for i in range(n):
            grid.extend([to01(x_D[i].detach().cpu()), to01(x_G[i].detach().cpu()), to01(x_direct[i].detach().cpu()), to01(x_rec[i].detach().cpu())])
        save_path = os.path.join(args.out_dir, f"batch_{batch_idx:04d}.png")
        save_image(grid, save_path, nrow=4)
        print(f"Saved {save_path}")

    # Report averages
    if count_images > 0:
        avg_psnr_direct = sum_psnr_direct / count_images
        avg_ssim_direct = sum_ssim_direct / count_images
        avg_psnr_dps = sum_psnr_dps / count_images
        avg_ssim_dps = sum_ssim_dps / count_images
        print(f"[Direct] PSNR={avg_psnr_direct:.3f} SSIM={avg_ssim_direct:.4f}")
        print(f"[DPS]    PSNR={avg_psnr_dps:.3f} SSIM={avg_ssim_dps:.4f}")
    else:
        print("No samples in subset. Nothing to run.")


if __name__ == '__main__':
    main()



