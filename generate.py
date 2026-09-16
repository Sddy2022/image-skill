#!/usr/bin/env python3
"""
简单 CLI 接口使用 Diffusers Pipeline 生成图像。
示例:
  python generate.py --prompt "A cyberpunk cityscape" --outdir outputs --num-images 2
"""
import os
import argparse
from pathlib import Path
from dotenv import load_dotenv
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

load_dotenv()

def load_pipeline(model_dir_or_id, hf_token=None, device="cuda"):
    # 优先尝试从本地目录加载；否则从huggingface hub加载
    kwargs = {}
    if torch.cuda.is_available():
        torch_dtype = torch.float16
        kwargs.update({"torch_dtype": torch_dtype})
    else:
        device = "cpu"
    if os.path.isdir(model_dir_or_id):
        pipe = StableDiffusionPipeline.from_pretrained(model_dir_or_id, **kwargs)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(model_dir_or_id, use_auth_token=hf_token, **kwargs)
    if device == "cuda":
        pipe = pipe.to("cuda")
    else:
        pipe = pipe.to("cpu")
    return pipe

def generate(prompt, outdir="outputs", model_id=None, hf_token=None, num_images=1, steps=20, guidance_scale=7.5, width=512, height=512):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    pipe = load_pipeline(model_id, hf_token)
    for i in range(num_images):
        image = pipe(prompt, num_inference_steps=steps, guidance_scale=guidance_scale, height=height, width=width).images[0]
        out_path = Path(outdir) / f"gen_{i+1}.png"
        image.save(out_path)
        print("Saved:", out_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="outputs")
    parser.add_argument("--model-id", type=str, default=os.environ.get("MODEL_ID", "runwayml/stable-diffusion-v1-5"))
    parser.add_argument("--num-images", type=int, default=1)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    args = parser.parse_args()

    hf_token = os.environ.get("HUGGINGFACE_TOKEN")
    generate(args.prompt, outdir=args.outdir, model_id=args.model_id, hf_token=hf_token,
             num_images=args.num_images, steps=args.steps, guidance_scale=args.guidance_scale,
             width=args.width, height=args.height)

if __name__ == "__main__":
    main()
