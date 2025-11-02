#!/usr/bin/env python3
"""
HunyuanImage-2.1 Single Image Test
24GB GPU with FP8 quantization
"""
import os
import sys
import time
from pathlib import Path

# IMPORTANT: Update these paths to match your environment
HUNYUAN_PATH = '/data/YOUR_USERNAME/repos/HunyuanImage-2.1'

# Change to HunyuanImage directory (required for relative paths)
os.chdir(HUNYUAN_PATH)

# Environment setup
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
sys.path.insert(0, HUNYUAN_PATH)
from hyimage.diffusion.pipelines.hunyuanimage_pipeline import HunyuanImagePipeline

# Test parameters - Reprompt-level detailed prompt
PROMPT = """A professional headshot portrait captures a single young adult Korean woman positioned centrally in the frame, viewed from the shoulders up. The subject has fair skin with an oval face and soft features, and displays a neutral expression with a calm, professional demeanor, her gaze directed straight at the camera with composed confidence. Her medium length straight black hair frames her face naturally, falling just past her shoulders with a natural sheen. She is wearing minimal accessories, maintaining a clean professional appearance. She wears business attire - a well-tailored navy blue blazer over a white collared shirt, which is rendered with realistic fabric textures showing natural draping and subtle wrinkles. The subject is positioned against a plain gray background with a smooth, matte finish, creating a clean, professional aesthetic typical of corporate photography. The scene is illuminated by soft, diffused studio lighting from the front, creating subtle highlights on her facial features and gentle shadows that define her facial structure. Photorealistic rendering with highly detailed facial features including natural skin pores, realistic skin texture, individual hair strands with natural highlights, and accurate fabric details in the clothing. Sharp focus on the eyes and face with crisp detail throughout, maintaining professional depth. Single person only in frame, professional headshot composition with the subject perfectly centered and level with the camera."""

WIDTH = 2048  # Full resolution without refiner (refiner causes OOM)
HEIGHT = 2048
NUM_STEPS = 50
GUIDANCE = 3.5
SHIFT = 5
SEED = 42

def main():
    output_dir = Path("outputs/hunyuan_test")  # Relative path from gen_image directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("HunyuanImage-2.1 Test - FP8 Quantization")
    print("=" * 60)
    print(f"Resolution: {WIDTH}x{HEIGHT}")
    print(f"Steps: {NUM_STEPS}")
    print(f"Guidance: {GUIDANCE}")
    print(f"Shift: {SHIFT}")

    try:
        # GPU info
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

        start_time = time.time()

        # Load pipeline with FP8 quantization
        print("\nLoading HunyuanImage-2.1 with FP8 quantization...")
        print("(This may take 5-10 minutes on first load)")

        model_name = "hunyuanimage-v2.1"
        pipe = HunyuanImagePipeline.from_pretrained(
            model_name=model_name,
            use_fp8=True,  # Enable FP8 for 24GB GPU
            reprompt_model=None,  # Disable reprompt - we use detailed prompts instead
            # Ensure all offloading options are enabled (should be default True)
            enable_stage1_offloading=True,
            enable_refiner_offloading=True,
            enable_text_encoder_offloading=True,
            enable_full_dit_offloading=True,
            enable_vae_offloading=True,
            enable_byt5_offloading=True
        )
        pipe = pipe.to("cuda")

        load_time = time.time() - start_time
        print(f"\nPipeline loaded in {load_time:.1f}s")
        print(f"GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

        # Generate
        print(f"\nGenerating {WIDTH}x{HEIGHT} image...")
        print(f"Prompt: {PROMPT[:100]}...")

        gen_start = time.time()

        image = pipe(
            prompt=PROMPT,
            width=WIDTH,
            height=HEIGHT,
            use_reprompt=False,  # Disabled - we use extremely detailed prompts instead
            use_refiner=False,   # Disabled - requires 200GB system RAM (GitHub issue #22)
            num_inference_steps=NUM_STEPS,
            guidance_scale=GUIDANCE,
            shift=SHIFT,
            seed=SEED
        )

        gen_time = time.time() - gen_start

        print(f"\nGeneration complete in {gen_time:.1f}s")
        print(f"Peak GPU: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

        # Save
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"hunyuan_{WIDTH}x{HEIGHT}_{timestamp}.png"
        image.save(str(output_path))

        print(f"\n✓ Image saved: {output_path}")
        print(f"Total time: {time.time() - start_time:.1f}s")

        return 0

    except torch.cuda.OutOfMemoryError as e:
        print(f"\n✗ GPU OOM Error!", file=sys.stderr)
        print(f"Try: use_fp8=True and reduce resolution", file=sys.stderr)
        if torch.cuda.is_available():
            print(f"Peak GPU: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB", file=sys.stderr)
        return 1

    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
