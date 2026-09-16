#!/usr/bin/env python3
"""
简易 Gradio Web UI，用于交互式生成图像。
运行:
  python app.py
打开: http://localhost:7860
"""
import os
from dotenv import load_dotenv
load_dotenv()

import gradio as gr
import torch
from diffusers import StableDiffusionPipeline

MODEL_ID = os.environ.get("MODEL_ID", "runwayml/stable-diffusion-v1-5")
HF_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")

def get_pipe():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kwargs = {}
    if device == "cuda":
        kwargs["torch_dtype"] = torch.float16
    # 尝试从本地目录优先加载
    if os.path.isdir(MODEL_ID):
        pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, **kwargs)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, use_auth_token=HF_TOKEN, **kwargs)
    return pipe.to(device)

pipe = None

def generate(prompt, steps=20, guidance=7.5, num_images=1, width=512, height=512):
    global pipe
    if pipe is None:
        pipe = get_pipe()
    images = []
    for _ in range(num_images):
        out = pipe(prompt, num_inference_steps=int(steps), guidance_scale=float(guidance),
                   height=int(height), width=int(width))
        images.append(out.images[0])
    return images

with gr.Blocks() as demo:
    gr.Markdown("# Stable Diffusion - Gradio 示例")
    with gr.Row():
        prompt = gr.Textbox(label="Prompt", value="A cute robot painting a landscape")
    with gr.Row():
        steps = gr.Slider(minimum=1, maximum=50, step=1, value=20, label="Steps")
        guidance = gr.Slider(minimum=1.0, maximum=20.0, step=0.5, value=7.5, label="Guidance scale")
        num_images = gr.Slider(minimum=1, maximum=4, step=1, value=1, label="Num images")
    with gr.Row():
        width = gr.Dropdown(choices=[256, 512, 768], value=512, label="Width")
        height = gr.Dropdown(choices=[256, 512, 768], value=512, label="Height")
    btn = gr.Button("Generate")
    gallery = gr.Gallery(label="Results").style(grid=[2], height="auto")
    btn.click(fn=generate, inputs=[prompt, steps, guidance, num_images, width, height], outputs=gallery)

if __name__ == "__main__":
    demo.launch()
