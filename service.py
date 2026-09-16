"""service.py
Core generation service wrapping Hugging Face Diffusers pipelines for:
- text2img
- img2img
- inpainting
- upscaling (Real-ESRGAN)

Provides scheduler selection, seed management, xformers/CPU offload best-effort optimizations,
and result saving/versioning (outputs/<timestamp>_...)
"""
import os
import io
import json
import time
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from dotenv import load_dotenv
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
)
from diffusers import (
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
)

# Optional upscaler
try:
    from realesrgan import RealESRGAN
    REAL_ESRGAN_AVAILABLE = True
except Exception:
    REAL_ESRGAN_AVAILABLE = False

load_dotenv()

MODEL_ID = os.environ.get("MODEL_ID", "runwayml/stable-diffusion-v1-5")
HF_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Lazy-loaded pipelines
_pipes = {
    "txt2img": None,
    "img2img": None,
    "inpaint": None,
}


def _get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _enable_optimizations(pipe):
    # Try enabling xformers memory efficient attention
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    # Try enabling attention slicing to reduce peak memory
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass
    # Model offload if accelerate provides it
    try:
        pipe.enable_model_cpu_offload()
    except Exception:
        pass


def _set_scheduler(pipe, name: Optional[str]):
    if not name:
        return pipe
    name = name.lower()
    scheduler = None
    if "dpm" in name or "dpmsolver" in name:
        scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    elif "lms" in name:
        scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
    elif "euler-ancestral" in name or "euler_ancestral" in name:
        scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    elif "euler" in name:
        scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    else:
        # Unknown scheduler, fallback to current
        return pipe
    pipe.scheduler = scheduler
    return pipe


def _load_pipelines():
    device = _get_device()
    global _pipes
    if _pipes["txt2img"] is None:
        kwargs = {}
        if device == "cuda":
            kwargs["torch_dtype"] = torch.float16
        # Load main text2img pipeline
        _pipes["txt2img"] = StableDiffusionPipeline.from_pretrained(MODEL_ID, use_auth_token=HF_TOKEN, **kwargs)
        _pipes["txt2img"] = _pipes["txt2img"].to(device)
        _enable_optimizations(_pipes["txt2img"])

    if _pipes["img2img"] is None:
        kwargs = {}
        if device == "cuda":
            kwargs["torch_dtype"] = torch.float16
        _pipes["img2img"] = StableDiffusionImg2ImgPipeline.from_pretrained(MODEL_ID, use_auth_token=HF_TOKEN, **kwargs)
        _pipes["img2img"] = _pipes["img2img"].to(device)
        _enable_optimizations(_pipes["img2img"])

    if _pipes["inpaint"] is None:
        kwargs = {}
        if device == "cuda":
            kwargs["torch_dtype"] = torch.float16
        # Inpainting pipeline may require a different repo id if using specialized inpaint model
        try:
            _pipes["inpaint"] = StableDiffusionInpaintPipeline.from_pretrained(MODEL_ID, use_auth_token=HF_TOKEN, **kwargs)
        except Exception:
            # Fallback: try loading same model into inpaint pipeline
            _pipes["inpaint"] = StableDiffusionInpaintPipeline.from_pretrained(MODEL_ID, use_auth_token=HF_TOKEN, **kwargs)
        _pipes["inpaint"] = _pipes["inpaint"].to(device)
        _enable_optimizations(_pipes["inpaint"])


def _seed_generator(seed: Optional[int]):
    device = _get_device()
    if seed is None:
        gen = None
    else:
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))
    return gen


def _save_image_and_meta(image: Image.Image, meta: dict) -> str:
    ts = int(time.time())
    seed = meta.get("seed", "-")
    fname = f"{ts}_s{seed}_{meta.get('type','img')}.png"
    outpath = OUTPUT_DIR / fname
    image.save(outpath)
    # Save metadata json next to image
    meta_path = outpath.with_suffix(".json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return str(outpath)


def generate_text2img(prompt: str, num_images: int = 1, steps: int = 20, guidance_scale: float = 7.5,
                      width: int = 512, height: int = 512, seed: Optional[int] = None, scheduler: Optional[str] = None):
    _load_pipelines()
    pipe = _pipes["txt2img"]
    _set_scheduler(pipe, scheduler)
    gen = _seed_generator(seed)
    results = []
    for i in range(num_images):
        out = pipe(prompt, num_inference_steps=int(steps), guidance_scale=float(guidance_scale), height=height, width=width, generator=gen)
        image = out.images[0]
        meta = dict(type="txt2img", prompt=prompt, steps=steps, guidance=guidance_scale, width=width, height=height,
                    seed=seed, scheduler=scheduler)
        path = _save_image_and_meta(image, meta)
        results.append({"path": path, "meta": meta})
    return results


def generate_img2img(init_image: Image.Image, prompt: str, strength: float = 0.8, num_images: int = 1,
                     steps: int = 20, guidance_scale: float = 7.5, seed: Optional[int] = None, scheduler: Optional[str] = None):
    _load_pipelines()
    pipe = _pipes["img2img"]
    _set_scheduler(pipe, scheduler)
    gen = _seed_generator(seed)
    results = []
    init_image = init_image.convert("RGB")
    for i in range(num_images):
        out = pipe(prompt=prompt, image=init_image, strength=float(strength), num_inference_steps=int(steps), guidance_scale=float(guidance_scale), generator=gen)
        image = out.images[0]
        meta = dict(type="img2img", prompt=prompt, strength=strength, steps=steps, guidance=guidance_scale,
                    seed=seed, scheduler=scheduler)
        path = _save_image_and_meta(image, meta)
        results.append({"path": path, "meta": meta})
    return results


def inpaint(init_image: Image.Image, mask_image: Image.Image, prompt: str, strength: float = 0.8, num_images: int = 1,
            steps: int = 20, guidance_scale: float = 7.5, seed: Optional[int] = None, scheduler: Optional[str] = None):
    _load_pipelines()
    pipe = _pipes["inpaint"]
    _set_scheduler(pipe, scheduler)
    gen = _seed_generator(seed)
    results = []
    # Ensure correct mode
    init_image = init_image.convert("RGB")
    mask_image = mask_image.convert("RGB")
    for i in range(num_images):
        out = pipe(prompt=prompt, image=init_image, mask_image=mask_image, strength=float(strength), num_inference_steps=int(steps), guidance_scale=float(guidance_scale), generator=gen)
        image = out.images[0]
        meta = dict(type="inpaint", prompt=prompt, strength=strength, steps=steps, guidance=guidance_scale,
                    seed=seed, scheduler=scheduler)
        path = _save_image_and_meta(image, meta)
        results.append({"path": path, "meta": meta})
    return results


def upscale(image: Image.Image, scale: int = 2, out_tile: int = 0) -> dict:
    """Upscale using Real-ESRGAN if available. Returns dict with path and meta.
    Note: Real-ESRGAN python package may require additional installation steps (CUDA, build) on some systems.
    """
    image = image.convert("RGB")
    meta = dict(type="upscale", scale=scale)
    if not REAL_ESRGAN_AVAILABLE:
        # fallback: simple PIL resize (nearest/ lanczos)
        w, h = image.size
        new = image.resize((w * scale, h * scale), resample=Image.LANCZOS)
        path = _save_image_and_meta(new, meta)
        return {"path": path, "meta": meta, "note": "realesrgan not available; used PIL upscale"}

    device = _get_device()
    # RealESRGAN expects numpy arrays; the library RealESRGAN wraps model loading
    try:
        model = RealESRGAN(device, scale=scale)
        model.load_weights("RealESRGAN_x2")
        output = model.predict(image)
        path = _save_image_and_meta(output, meta)
        return {"path": path, "meta": meta}
    except Exception as e:
        # fallback
        w, h = image.size
        new = image.resize((w * scale, h * scale), resample=Image.LANCZOS)
        path = _save_image_and_meta(new, meta)
        return {"path": path, "meta": meta, "note": f"realesrgan failed: {e}; used PIL upscale"}
