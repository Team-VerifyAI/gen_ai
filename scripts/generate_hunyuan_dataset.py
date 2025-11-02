#!/usr/bin/env python3
"""
HunyuanImage-2.1 Batch Generation for Deepfake Dataset
FP8 quantization for 24GB GPU
"""
import os
import sys
import time
import json
from pathlib import Path

# IMPORTANT: Update these paths to match your environment
HUNYUAN_PATH = '/data/YOUR_USERNAME/repos/HunyuanImage-2.1'
GEN_IMAGE_PATH = '/data/YOUR_USERNAME/repos/gen_image'

# Add paths first
sys.path.insert(0, HUNYUAN_PATH)
sys.path.insert(0, GEN_IMAGE_PATH)

# Change to HunyuanImage directory (required for relative paths in model loading)
os.chdir(HUNYUAN_PATH)

# Environment setup
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
from hyimage.diffusion.pipelines.hunyuanimage_pipeline import HunyuanImagePipeline
from prompts_deepfake import generate_prompt_batch

# Configuration
WIDTH = 896  # Reduced from 1024 to fit 24GB VRAM with refiner
HEIGHT = 896
NUM_STEPS = 50
GUIDANCE = 3.5
SHIFT = 5

# Dataset configuration - 무제한 생성 (시간 제한까지 계속)
NUM_PROMPTS = 10000  # Very large number
IMAGES_PER_PROMPT = 5
START_INDEX = 0

def generate_single_image(pipe, prompt, seed, output_path):
    """Generate a single image"""
    try:
        torch.cuda.empty_cache()

        image = pipe(
            prompt=prompt,
            width=WIDTH,
            height=HEIGHT,
            use_reprompt=False,  # Disabled - we use extremely detailed prompts instead
            use_refiner=False,  # Disabled - requires 200GB system RAM (GitHub issue #22)
            num_inference_steps=NUM_STEPS,
            guidance_scale=GUIDANCE,
            shift=SHIFT,
            seed=seed
        )

        image.save(str(output_path))
        return True

    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    output_dir = Path(GEN_IMAGE_PATH) / "outputs/hunyuan_dataset"
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_file = output_dir / "metadata.jsonl"

    print("=" * 60)
    print("HunyuanImage-2.1 Dataset Generation")
    print("=" * 60)
    print(f"Resolution: {WIDTH}x{HEIGHT}")
    print(f"Steps: {NUM_STEPS}")
    print(f"Output: {output_dir}")
    print(f"Will run until SLURM time limit")

    # GPU info
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # Load pipeline
    print("\nLoading HunyuanImage-2.1 with FP8...")
    start = time.time()

    model_name = "hunyuanimage-v2.1"
    pipe = HunyuanImagePipeline.from_pretrained(
        model_name=model_name,
        use_fp8=True,
        reprompt_model=None,  # Disable reprompt - we use detailed prompts instead
        enable_stage1_offloading=True,
        enable_refiner_offloading=True,
        enable_text_encoder_offloading=True,
        enable_full_dit_offloading=True,
        enable_vae_offloading=True,
        enable_byt5_offloading=True
    )
    pipe = pipe.to("cuda")

    print(f"Pipeline loaded in {time.time() - start:.1f}s")

    # Generate prompts
    print(f"\nGenerating {NUM_PROMPTS} diverse prompts...")
    prompts = generate_prompt_batch(NUM_PROMPTS, balanced=True)

    # Generation loop
    total_images = NUM_PROMPTS * IMAGES_PER_PROMPT
    generated_count = 0
    failed_count = 0
    total_start = time.time()

    metadata_fp = open(metadata_file, 'a')

    try:
        for prompt_idx, prompt in enumerate(prompts):
            if prompt_idx < START_INDEX:
                continue

            print(f"\n[{prompt_idx + 1}/{NUM_PROMPTS}] Prompt: {prompt[:80]}...")

            for seed_idx in range(IMAGES_PER_PROMPT):
                seed = prompt_idx * 1000 + seed_idx
                filename = f"fake_{prompt_idx:05d}_{seed_idx:02d}.png"
                output_path = output_dir / filename

                # Skip if exists
                if output_path.exists():
                    print(f"  Skip {filename} (exists)")
                    generated_count += 1
                    continue

                gen_start = time.time()
                success = generate_single_image(pipe, prompt, seed, output_path)
                gen_time = time.time() - gen_start

                if success:
                    generated_count += 1
                    print(f"  ✓ {filename} ({gen_time:.1f}s)")

                    # Metadata
                    metadata = {
                        "filename": filename,
                        "prompt": prompt,
                        "seed": seed,
                        "label": "fake",
                        "resolution": f"{WIDTH}x{HEIGHT}",
                        "steps": NUM_STEPS,
                        "guidance": GUIDANCE,
                        "shift": SHIFT,
                        "model": "hunyuanimage-v2.1",
                        "generation_time": gen_time
                    }
                    metadata_fp.write(json.dumps(metadata) + '\n')
                    metadata_fp.flush()
                else:
                    failed_count += 1
                    print(f"  ✗ {filename} (failed)")

                # Progress
                elapsed = time.time() - total_start
                avg_time = elapsed / generated_count if generated_count > 0 else 100
                remaining = (total_images - generated_count) * avg_time
                print(f"  Progress: {generated_count}/{total_images} "
                      f"({generated_count/total_images*100:.1f}%) "
                      f"ETA: {remaining/3600:.1f}h")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user!")
        print(f"Resume with START_INDEX={prompt_idx}")
    finally:
        metadata_fp.close()

    # Summary
    total_time = time.time() - total_start
    print(f"\n{'=' * 60}")
    print("Generation Complete")
    print(f"{'=' * 60}")
    print(f"Generated: {generated_count}/{total_images}")
    print(f"Failed: {failed_count}")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Avg time: {total_time/generated_count:.1f}s per image")
    print(f"Metadata: {metadata_file}")

if __name__ == "__main__":
    main()
