#!/usr/bin/env python3
"""
Simple FLUX image generation test script
Generates a single test image using FLUX.1 [schnell] model
"""
import os
import sys
import time
from pathlib import Path

def main():
    # Setup output directory
    output_dir = Path("outputs")  # Relative to gen_image directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Test prompt
    prompt = "A serene landscape with mountains and a lake at sunset, highly detailed, photorealistic"

    print(f"Starting FLUX image generation test...")
    print(f"Prompt: {prompt}")
    print(f"Output directory: {output_dir}")

    try:
        # Import FLUX modules
        print("\nImporting FLUX modules...")
        from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack
        from flux.util import configs, load_ae, load_clip, load_flow_model, load_t5
        import torch
        from einops import rearrange
        from PIL import Image

        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

        # Use FLUX.1 [schnell] - the fastest Apache-2.0 licensed model
        model_name = "flux-schnell"
        device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"\nLoading {model_name} model on {device}...")
        start_time = time.time()

        # Load models with memory optimization
        # Load T5 on CPU to save GPU memory, CLIP on GPU
        print("Loading text encoders (T5 on CPU, CLIP on GPU for memory efficiency)...")
        t5 = load_t5("cpu", max_length=256)  # T5-XXL is huge, keep on CPU
        clip = load_clip(device)

        print("Loading main flow model...")
        model = load_flow_model(model_name, device=device)

        print("Loading autoencoder...")
        ae = load_ae(model_name, device=device)

        load_time = time.time() - start_time
        print(f"Models loaded in {load_time:.2f} seconds")

        # Generate image
        print(f"\nGenerating image...")
        gen_start = time.time()

        # Get config for schnell model
        config = configs[model_name]

        # Set generation parameters (reduced resolution for 24GB GPU)
        width = 512  # Reduced from 1024 to save memory
        height = 512  # Reduced from 1024 to save memory
        num_steps = 4  # schnell is optimized for 4 steps
        guidance = 3.5
        seed = 42

        print(f"Parameters: {width}x{height}, {num_steps} steps, guidance={guidance}, seed={seed}")

        # Prepare input
        rng = torch.Generator(device=device).manual_seed(seed)
        x = get_noise(1, height, width, device=device, dtype=torch.bfloat16, seed=seed)

        # Encode prompt
        inp = prepare(t5, clip, x, prompt=prompt)

        # Get timesteps
        timesteps = get_schedule(num_steps, inp["img"].shape[1], shift=(model_name != "flux-schnell"))

        # Denoise
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            x = denoise(model, **inp, timesteps=timesteps, guidance=guidance)

        # Decode
        x = unpack(x.float(), height, width)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            x = ae.decode(x)

        gen_time = time.time() - gen_start
        print(f"Image generated in {gen_time:.2f} seconds")

        # Convert to PIL Image
        x = x.clamp(-1, 1)
        x = rearrange(x[0], "c h w -> h w c")
        img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())

        # Save image
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"test_image_{timestamp}.png"
        img.save(output_path)

        print(f"\nSuccess! Image saved to: {output_path}")
        print(f"Total time: {time.time() - start_time:.2f} seconds")

        return 0

    except Exception as e:
        print(f"\nError during generation: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
