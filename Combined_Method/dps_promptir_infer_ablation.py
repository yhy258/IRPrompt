import argparse
import csv
import os
from itertools import product
from typing import Iterable, List, Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image

from aoiir.datasets.multi_degradation import MultiDegradationDataset
from aoiir.metrics.metrics import psnr_ssim, lpips
from aoiir.pipelines.dps_promptir import DPSPromptIRPipeline

@torch.no_grad()
def run_batch(args, pipeline: DPSPromptIRPipeline, x_D: torch.Tensor, x_G: torch.Tensor, steps: int, cfg: float, do_classifier_free_guidance: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    # Compute posterior latents
    z_D, z_G, _, _, _ = pipeline(x_D, x_G)
    x_direct = pipeline.decode(z_G)
    
    if args.sampling_method == 'dps':
        z_rec = pipeline.dps_sample(z_D, num_steps=steps, dps_scale=args.dps_scale, cfg=cfg, do_classifier_free_guidance=do_classifier_free_guidance)
    elif args.sampling_method == 'pixel_space_dps':
        z_rec = pipeline.pixel_space_dps_sample(z_D, num_steps=steps, dps_scale=args.pixel_dps_scale, cfg=cfg, do_classifier_free_guidance=do_classifier_free_guidance)
    elif args.sampling_method == 'beyond_first_tweedie':
        z_rec = pipeline.beyond_first_tweedie(z_D, num_steps=steps, stoc_avg_steps=args.stoc_avg_steps, rand_num=args.rand_num, pixel_dps_scale=args.pixel_dps_scale, second_order_scale=args.second_order_scale)
    elif args.sampling_method == 'combined_dps':
        z_rec = pipeline.combined_dps_sample(z_D, num_steps=steps, latent_dps_scale=args.dps_scale, pixel_dps_scale=args.pixel_dps_scale, cfg=cfg, do_classifier_free_guidance=do_classifier_free_guidance)
    else:
        raise ValueError(f"Invalid sampling method: {args.sampling_method}")
    x_rec = pipeline.decode(z_rec)
    return x_direct, x_rec

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


def parse_list(arg: str, typ=float) -> List:
    parts = [p.strip() for p in arg.split(",") if p.strip() != ""]
    if typ is bool:
        return [p.lower() in ("1", "true", "yes", "y") for p in parts]
    return [typ(p) for p in parts]


@torch.no_grad()
def run_case(
    args,
    pipeline: DPSPromptIRPipeline,
    loader: DataLoader,
    steps: int,
    cfg: float,
    do_cfg: bool,
    device: torch.device,
    out_dir: str,
    case_tag: str,
) -> Tuple[float, float, float]:
    case_dir = os.path.join(out_dir, case_tag)
    os.makedirs(case_dir, exist_ok=True)

    sum_psnr = 0.0
    sum_ssim = 0.0
    sum_lpips = 0.0
    count_images = 0

    for batch_idx, (x_D, x_G) in enumerate(loader):
        x_D = x_D.to(device)
        x_G = x_G.to(device)

        x_direct, x_rec = run_batch(args, pipeline, x_D, x_G, steps, cfg, do_cfg)

        # Metrics (per-batch average)
        m = psnr_ssim(x_rec, x_G)
        l = lpips(x_rec, x_G)
        bsz = x_D.size(0)
        sum_psnr += float(m["psnr"]) * bsz
        sum_ssim += float(m["ssim"]) * bsz
        sum_lpips += float(l) * bsz
        count_images += bsz

        # Save grid: degraded | clean | dps-decode
        grid = []
        n = min(4, x_D.size(0))
        for i in range(n):
            grid.extend([to01(x_D[i].detach().cpu()), to01(x_G[i].detach().cpu()), to01(x_rec[i].detach().cpu())])
        save_path = os.path.join(case_dir, f"batch_{batch_idx:04d}.png")
        save_image(grid, save_path, nrow=3)

    if count_images == 0:
        return 0.0, 0.0, 0.0

    return sum_psnr / count_images, sum_ssim / count_images, sum_lpips / count_images


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/home/joon/ImageRestoration-AllInOne/Combined_Method/aoiir/datasets/dataset')
    parser.add_argument('--sd_model', type=str, default='stabilityai/stable-diffusion-2-1-base')
    parser.add_argument('--adapter_ckpt', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--steps_list', type=str, default='')
    parser.add_argument('--subset_ratio', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out_dir', type=str, default='inference_dps_promptir_ablation')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    # Lists for ablation
    parser.add_argument('--sampling_method', type=str, default='dps')
    parser.add_argument('--stoc_avg_steps', type=str, default='0,1,2,3')
    parser.add_argument('--rand_nums', type=str, default='0,1,2,3')
    parser.add_argument('--dps_scales', type=str, default='0,0.5,1.0')
    parser.add_argument('--pixel_dps_scales', type=str, default='0,0.5,1.0')
    parser.add_argument('--second_order_scales', type=str, default='0,0.01,0.03,0.05')
    parser.add_argument('--cfgs', type=str, default='0,2,4,7.5')
    parser.add_argument('--scheduler_types', type=str, default='ddim,dpm')
    parser.add_argument('--do_cfgs', type=str, default='false,true')
    parser.add_argument('--csv_path', type=str, default='ablation_results.csv')

    args = parser.parse_args()

    args.out_dir = args.out_dir + '-' + args.sampling_method if args.sampling_method != 'dps' else args.out_dir

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    # Dataset and loader (subset)
    full_ds = MultiDegradationDataset(dataset_root=args.data_root, patch_size=256)
    sub_ds = build_subset(full_ds, args.subset_ratio, args.seed)
    loader = DataLoader(sub_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = torch.device(args.device)

    # Parse grids
    stoc_avg_steps = parse_list(args.stoc_avg_steps, int)
    rand_nums = parse_list(args.rand_nums, int)
    dps_scales = parse_list(args.dps_scales, float)
    pixel_dps_scales = parse_list(args.pixel_dps_scales, float)
    second_order_scales = parse_list(args.second_order_scales, float)
    cfgs = parse_list(args.cfgs, float)
    scheduler_types = [s.strip() for s in args.scheduler_types.split(',') if s.strip()]
    do_cfgs = parse_list(args.do_cfgs, bool)
    steps_list = parse_list(args.steps_list, int) if args.steps_list.strip() else [int(args.steps)]
    
    ### sampling method마다 다른 조합 설정
    if args.sampling_method == 'dps':
        args_name_list = ['dps_scale', 'cfg', 'scheduler_type', 'steps', 'do_classifier_free_guidance']
    elif args.sampling_method == 'pixel_space_dps':
        args_name_list = ['pixel_dps_scale', 'cfg', 'scheduler_type', 'steps', 'do_classifier_free_guidance']
    elif args.sampling_method == 'beyond_first_tweedie':
        args_name_list = ['stoc_avg_steps', 'rand_num', 'pixel_dps_scale', 'second_order_scale', 'scheduler_type', 'steps']
    elif args.sampling_method == 'combined_dps':
        args_name_list = ['dps_scale', 'pixel_dps_scale', 'cfg', 'scheduler_type', 'steps', 'do_classifier_free_guidance']
    else:
        ValueError(f"Invalid sampling method: {args.sampling_method}")
    args.dps_scale = 0
    args.pixel_dps_scale = 0
    args.stoc_avg_steps = 0
    args.rand_num = 0
    args.second_order_scale = 0
    # args.cfg = 0
    args.do_classifier_free_guidance = False
    do_cfg = False
    cfg = 0

    # Prepare CSV
    args.csv_path = args.sampling_method + "-" + args.csv_path if args.sampling_method != 'dps' else args.csv_path
    csv_file = os.path.join(args.out_dir, args.csv_path)
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(args_name_list + ['PSNR', 'SSIM', 'LPIPS'])

        # Iterate grid
        for scheduler_type in scheduler_types:
            # Recreate pipeline per scheduler to ensure clean state
            pipeline = DPSPromptIRPipeline(sd_model_name=args.sd_model, adapter=None, device=str(device), scheduler_type=scheduler_type)
            pipeline.adapter.eval()
            # Load adapter weights
            # Reuse helper from single-run script to support flexible ckpts
            from dps_promptir_infer import try_load_adapter_from_ckpt
            try_load_adapter_from_ckpt(pipeline.adapter, args.adapter_ckpt)
            pipeline.to(device)

            if args.sampling_method == 'dps':
                for steps_val, dps_scale in product(steps_list, dps_scales):
                    print('Step:', steps_val)
                    args.dps_scale = dps_scale
                    for do_cfg in do_cfgs:
                        for cfg in cfgs:
                            case_tag = f"sched-{scheduler_type}_steps-{steps_val}_dps-{dps_scale}_cfg-{cfg}_doCFG-{int(do_cfg)}"
                            print(f"Running case: {case_tag}")
                            psnr, ssim_val, lp = run_case(
                                args=args,
                                pipeline=pipeline,
                                loader=loader,
                                steps=steps_val,
                                cfg=cfg,
                                do_cfg=do_cfg,
                                device=device,
                                out_dir=args.out_dir,
                                case_tag=case_tag,
                            )
                            writer.writerow([dps_scale, cfg, scheduler_type, steps_val, bool(do_cfg), psnr, ssim_val, lp])
                            f.flush()
                            print(f"Case done: steps={steps_val} PSNR={psnr:.3f}, SSIM={ssim_val:.4f}, LPIPS={lp:.4f}")
                            if do_cfg == False:
                                break
            elif args.sampling_method == 'pixel_space_dps':
                for steps_val, pixel_dps_scale in product(steps_list, pixel_dps_scales):
                    args.pixel_dps_scale = pixel_dps_scale
                    for do_cfg in do_cfgs:
                        for cfg in cfgs:
                            case_tag = f"sched-{scheduler_type}_steps-{steps_val}_pixel_dps-{pixel_dps_scale}_cfg-{cfg}_doCFG-{int(do_cfg)}"
                            print(f"Running case: {case_tag}")
                            psnr, ssim_val, lp = run_case(
                                args=args,
                                pipeline=pipeline,
                                loader=loader,
                                steps=steps_val,
                                cfg=cfg,
                                do_cfg=do_cfg,
                                device=device,
                                out_dir=args.out_dir,
                                case_tag=case_tag,
                            )
                            writer.writerow([pixel_dps_scale, cfg, scheduler_type, steps_val, bool(do_cfg), psnr, ssim_val, lp])
                            f.flush()
                            print(f"Case done: steps={steps_val} PSNR={psnr:.3f}, SSIM={ssim_val:.4f}, LPIPS={lp:.4f}")
                            if do_cfg == False:
                                break
            elif args.sampling_method == 'combined_dps':
                for steps_val, dps_scale, pixel_dps_scale in product(steps_list, dps_scales, pixel_dps_scales):
                    args.dps_scale = dps_scale
                    args.pixel_dps_scale = pixel_dps_scale
                    for do_cfg in do_cfgs:
                        for cfg in cfgs:
                            case_tag = f"sched-{scheduler_type}_steps-{steps_val}_dps-{dps_scale}_pixel_dps-{pixel_dps_scale}_cfg-{cfg}_doCFG-{int(do_cfg)}"  
                            print(f"Running case: {case_tag}")
                            psnr, ssim_val, lp = run_case(
                                args=args,
                                pipeline=pipeline,
                                loader=loader,
                                steps=steps_val,
                                cfg=cfg,
                                do_cfg=do_cfg,
                                device=device,
                                out_dir=args.out_dir,
                                case_tag=case_tag,
                            )
                            writer.writerow([dps_scale, pixel_dps_scale, cfg, scheduler_type, steps_val, bool(do_cfg), psnr, ssim_val, lp])
                            f.flush()
                            print(f"Case done: steps={steps_val} PSNR={psnr:.3f}, SSIM={ssim_val:.4f}, LPIPS={lp:.4f}")
                            if do_cfg == False:
                                break
            elif args.sampling_method == 'beyond_first_tweedie':
                for steps_val, stoc_avg_steps, rand_num, pixel_dps_scale, second_order_scale in product(steps_list, stoc_avg_steps, rand_nums, pixel_dps_scales, second_order_scales):
                    args.stoc_avg_steps = stoc_avg_steps
                    args.rand_num = rand_num
                    args.pixel_dps_scale = pixel_dps_scale
                    args.second_order_scale = second_order_scale

                    case_tag = f"sched-{scheduler_type}_steps-{steps_val}_stoc_avg-{stoc_avg_steps}_rand-{rand_num}_pixel_dps-{pixel_dps_scale}_second_order-{second_order_scale}_cfg-{cfg}_doCFG-{int(do_cfg)}"
                    print(f"Running case: {case_tag}")
                    psnr, ssim_val, lp = run_case(
                        args=args,
                        pipeline=pipeline,
                        loader=loader,
                        steps=steps_val,
                        cfg=cfg,
                        do_cfg=do_cfg,
                        device=device,
                        out_dir=args.out_dir,
                        case_tag=case_tag,
                    )
                    writer.writerow([stoc_avg_steps, rand_num, pixel_dps_scale, second_order_scale, scheduler_type, steps_val, psnr, ssim_val, lp])
                    f.flush()
                    print(f"Case done: steps={steps_val} PSNR={psnr:.3f}, SSIM={ssim_val:.4f}, LPIPS={lp:.4f}")


if __name__ == '__main__':
    main()


