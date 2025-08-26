import argparse
from aoiir.pipelines.sd_restoration import StableDiffusionRestorationPipeline
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Stable Diffusion Image Restoration Pipeline")
    parser.add_argument("--encoder_degraded_path", type=str, default="none",
                        help="Path to trained E_D checkpoint (or 'none' to use E_G)")
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to input image or directory")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Path to output image or directory")
    parser.add_argument("--sd_model_name", type=str, default="stabilityai/stable-diffusion-2-1-base",
                        help="Stable Diffusion model name")
    parser.add_argument("--num_diffusion_steps", type=int, default=50,
                        help="Number of diffusion steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                        help="Guidance scale for diffusion")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")

    args = parser.parse_args()
    pipeline = StableDiffusionRestorationPipeline(
        encoder_degraded_path=args.encoder_degraded_path,
        sd_model_name=args.sd_model_name,
        device=args.device,
    )
    input_path = Path(args.input_path)
    if input_path.is_file():
        pipeline.restore_image(str(input_path), args.output_path, args.num_diffusion_steps, args.guidance_scale)
    elif input_path.is_dir():
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(input_path.glob(f"*{ext}"))
            image_paths.extend(input_path.glob(f"*{ext.upper()}"))
        if not image_paths:
            print(f"No images found in {input_path}")
            return
        output_dir = args.output_path or str(input_path / "restored")
        pipeline.restore_batch([str(p) for p in image_paths], output_dir, args.num_diffusion_steps, args.guidance_scale)
    else:
        print(f"Input path {input_path} does not exist")


if __name__ == "__main__":
    main()
