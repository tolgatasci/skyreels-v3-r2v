"""
SkyReels-V3 R2V - RunPod Serverless Handler
Reference-to-Video: 1-4 reference images + prompt → 5s video
Portrait 9:16 natively supported (auto aspect ratio from ref image)
"""

import os
import sys
import io
import base64
import time
import random
import tempfile

# Add SkyReels-V3 to path
sys.path.insert(0, "/app/SkyReels-V3")

import runpod
import torch

# ─── Global State ───────────────────────────────────────────
pipeline = None
MODEL_LOADED = False


def load_model():
    """Load SkyReels-V3 R2V pipeline (lazy, first request)"""
    global pipeline, MODEL_LOADED

    if MODEL_LOADED and pipeline is not None:
        return pipeline

    print("=" * 60)
    print("SkyReels-V3 R2V - Loading Model")
    print("=" * 60)

    # HuggingFace login
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)
        print("HuggingFace login OK")

    start = time.time()

    from skyreels_v3.pipelines import ReferenceToVideoPipeline
    from skyreels_v3.modules import download_model

    # Download/cache model to Network Volume
    vol_path = "/runpod-volume/models/SkyReels-V3-R2V"
    if os.path.isdir(vol_path) and os.listdir(vol_path):
        model_path = vol_path
        print(f"Model cached: {model_path}")
    else:
        print("Downloading Skywork/SkyReels-V3-Reference2Video ...")
        model_path = download_model("Skywork/SkyReels-V3-Reference2Video")
        # Copy to volume for next cold start
        if os.path.exists("/runpod-volume"):
            import shutil
            try:
                print(f"Caching to {vol_path} ...")
                shutil.copytree(model_path, vol_path, dirs_exist_ok=True)
                model_path = vol_path
                print("Cached OK")
            except Exception as e:
                print(f"Cache failed (using HF cache): {e}")

    # Pipeline options
    offload = os.environ.get("OFFLOAD", "true").lower() == "true"
    low_vram = os.environ.get("LOW_VRAM", "false").lower() == "true"

    print(f"Creating pipeline (offload={offload}, low_vram={low_vram}) ...")
    pipeline = ReferenceToVideoPipeline(
        model_path=model_path,
        offload=offload,
        low_vram=low_vram,
    )

    elapsed = time.time() - start
    print(f"Model loaded in {elapsed:.1f}s")
    print("=" * 60)

    MODEL_LOADED = True
    return pipeline


def decode_ref_images(ref_images_input):
    """Decode reference images from base64 or URL to PIL"""
    from PIL import Image
    from diffusers.utils import load_image

    images = []
    for item in ref_images_input:
        if item.startswith("http://") or item.startswith("https://"):
            images.append(load_image(item))
        else:
            img_bytes = base64.b64decode(item)
            images.append(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
    return images


def handler(job):
    """
    RunPod handler for SkyReels-V3 Reference-to-Video.

    Input:
        prompt (str): Text description of the video
        ref_images (list[str]): 1-4 reference images (base64 or URL)
        duration (int): Video duration in seconds (default: 5)
        seed (int): Random seed, -1 for random (default: -1)
        resolution (str): "480P", "540P", or "720P" (default: "720P")

    Output:
        video (str): Base64 encoded MP4 video (24fps)
        seed (int), duration (int), resolution (str), fps (int),
        width (int), height (int), inference_time (float)

    Notes:
        - Aspect ratio auto-detected from first reference image
        - Portrait 9:16 supported (pass a portrait reference image)
        - Only 8 denoising steps (very fast!)
    """
    job_input = job.get("input", {})

    prompt = job_input.get("prompt", "")
    ref_images_input = job_input.get("ref_images", [])
    duration = job_input.get("duration", 5)
    seed = job_input.get("seed", -1)
    resolution = job_input.get("resolution", "720P")

    # ─── Validate ───────────────────────────────────────────
    if not prompt:
        return {"error": "prompt is required"}

    if not ref_images_input:
        return {"error": "ref_images is required (list of 1-4 base64 or URL strings)"}

    if len(ref_images_input) > 4:
        return {"error": "Maximum 4 reference images supported"}

    if resolution not in ("480P", "540P", "720P"):
        return {"error": f"Invalid resolution: {resolution}. Use 480P, 540P, or 720P"}

    print(f"\n{'='*60}")
    print(f"SkyReels-V3 R2V Generation")
    print(f"Duration: {duration}s | Resolution: {resolution}")
    print(f"Refs: {len(ref_images_input)} image(s)")
    print(f"Prompt: {prompt[:120]}...")
    print(f"{'='*60}")

    # ─── Load model ─────────────────────────────────────────
    try:
        pipe = load_model()
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": f"Model load failed: {str(e)}"}

    # ─── Decode reference images ────────────────────────────
    try:
        ref_imgs = decode_ref_images(ref_images_input)
        print(f"Decoded {len(ref_imgs)} reference image(s)")
        print(f"First ref size: {ref_imgs[0].size}")
    except Exception as e:
        return {"error": f"Failed to decode ref_images: {str(e)}"}

    # ─── Seed ───────────────────────────────────────────────
    if seed < 0:
        seed = random.randint(0, 2**32 - 1)
    print(f"Seed: {seed}")

    # ─── Generate ───────────────────────────────────────────
    try:
        gen_start = time.time()

        video_out = pipe.generate_video(
            ref_imgs=ref_imgs,
            prompt=prompt,
            duration=duration,
            seed=seed,
            resolution=resolution,
        )

        gen_time = time.time() - gen_start
        print(f"Generated in {gen_time:.1f}s ({len(video_out)} frames)")

        # ─── Encode output video ────────────────────────────
        import imageio
        tmp_path = tempfile.mktemp(suffix=".mp4")
        imageio.mimwrite(tmp_path, video_out, fps=24, quality=8)

        with open(tmp_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode("utf-8")

        file_size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
        os.unlink(tmp_path)

        # Get dimensions from first frame
        h, w = video_out[0].shape[:2]

        print(f"Output: {file_size_mb:.1f}MB, {w}x{h}, {len(video_out)} frames")
        print(f"{'='*60}\n")

        return {
            "video": video_b64,
            "seed": seed,
            "duration": duration,
            "resolution": resolution,
            "fps": 24,
            "width": w,
            "height": h,
            "num_frames": len(video_out),
            "inference_time": round(gen_time, 2),
        }

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {
            "error": "Out of GPU memory. Try: lower resolution (540P/480P), enable LOW_VRAM=true, or shorter duration"
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


# ─── Entry Point ────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SkyReels-V3 R2V - RunPod Serverless Worker")
    print("=" * 60)

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_mem / 1024**3
        print(f"GPU: {gpu_name}")
        print(f"VRAM: {vram_gb:.1f}GB")
    else:
        print("WARNING: CUDA not available!")

    print(f"HF_TOKEN: {'Set' if os.environ.get('HF_TOKEN') else 'NOT SET!'}")
    print(f"OFFLOAD: {os.environ.get('OFFLOAD', 'true')}")
    print(f"LOW_VRAM: {os.environ.get('LOW_VRAM', 'false')}")
    print("=" * 60 + "\n")

    print("Starting worker...")
    runpod.serverless.start({"handler": handler})
