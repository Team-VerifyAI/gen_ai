#!/usr/bin/env python3
"""
FLUX schnell 512x512 generation
Fast and stable for testing
"""
import os
import sys
import time
from pathlib import Path
import torch
import gc

def main():
    output_dir = Path("outputs")  # Relative to gen_image directory
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt = "A serene landscape with mountains and a lake at sunset, highly detailed, photorealistic"

    print(f"Starting FLUX schnell 512x512 generation...")
    print(f"Prompt: {prompt}")

    try:
        from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack
        from flux.util import configs, load_ae, load_clip, load_flow_model, load_t5
        from einops import rearrange
        from PIL import Image

        model_name = "flux-schnell"
        device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"\nLoading {model_name} model...")
        start_time = time.time()

        # Load models with CPU offloading
        t5 = load_t5("cpu", max_length=256)
        clip = load_clip("cpu")
        model = load_flow_model(model_name, device=device)
        ae = load_ae(model_name, device="cpu")

        print(f"Models loaded in {time.time() - start_time:.2f}s")

        # Generate 512x512
        width = 512
        height = 512
        num_steps = 4  # schnell optimal
        guidance = 3.5
        seed = 42

        print(f"\nGenerating {width}x{height} image...")
        gen_start = time.time()

        x = get_noise(1, height, width, device=device, dtype=torch.bfloat16, seed=seed)
        inp = prepare(t5, clip, x, prompt=prompt)

        for key in inp:
            if isinstance(inp[key], torch.Tensor) and inp[key].device.type == "cpu":
                inp[key] = inp[key].to(device)

        config = configs[model_name]
        timesteps = get_schedule(num_steps, inp["img"].shape[1], shift=False)

        with torch.inference_mode():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                x = denoise(model, **inp, timesteps=timesteps, guidance=guidance)

        # Decode on CPU
        x = unpack(x.float(), height, width)
        x_cpu = x.cpu()
        torch.cuda.empty_cache()

        with torch.inference_mode():
            x = ae.decode(x_cpu)

        gen_time = time.time() - gen_start
        print(f"Generated in {gen_time:.2f}s")

        # Save
        x = x.clamp(-1, 1)
        x = rearrange(x[0], "c h w -> h w c")
        img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"schnell_{timestamp}_{width}x{height}.png"
        img.save(output_path)

        print(f"\n✓ Success! Saved to: {output_path}")
        print(f"Total time: {time.time() - start_time:.2f}s")

        return 0

    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
