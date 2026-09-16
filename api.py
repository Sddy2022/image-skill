"""api.py
FastAPI server exposing endpoints for:
- POST /generate (text->image)
- POST /img2img (image + prompt)
- POST /inpaint (image + mask + prompt)
- POST /upscale (image)
- POST /batch (batch jobs)

Run: uvicorn api:app --host 0.0.0.0 --port 8000
"""
import io
import os
import uuid
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Optional, List
from PIL import Image

import service

app = FastAPI(title="image-skill API")


class GenParams(BaseModel):
    prompt: str
    num_images: int = 1
    steps: int = 20
    guidance: float = 7.5
    width: int = 512
    height: int = 512
    seed: Optional[int] = None
    scheduler: Optional[str] = None


@app.post("/generate")
async def generate_text(params: GenParams):
    results = service.generate_text2img(params.prompt, num_images=params.num_images, steps=params.steps,
                                        guidance_scale=params.guidance, width=params.width, height=params.height,
                                        seed=params.seed, scheduler=params.scheduler)
    return JSONResponse(content={"results": results})


@app.post("/img2img")
async def img2img(prompt: str = Form(...), strength: float = Form(0.8), steps: int = Form(20), guidance: float = Form(7.5),
                  num_images: int = Form(1), seed: Optional[int] = Form(None), scheduler: Optional[str] = Form(None),
                  image: UploadFile = File(...)):
    img_bytes = await image.read()
    init_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    results = service.generate_img2img(init_image, prompt=prompt, strength=strength, num_images=num_images,
                                       steps=steps, guidance_scale=guidance, seed=seed, scheduler=scheduler)
    return JSONResponse(content={"results": results})


@app.post("/inpaint")
async def inpaint(prompt: str = Form(...), strength: float = Form(0.8), steps: int = Form(20), guidance: float = Form(7.5),
                  num_images: int = Form(1), seed: Optional[int] = Form(None), scheduler: Optional[str] = Form(None),
                  image: UploadFile = File(...), mask: UploadFile = File(...)):
    img_bytes = await image.read()
    mask_bytes = await mask.read()
    init_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    mask_image = Image.open(io.BytesIO(mask_bytes)).convert("RGB")
    results = service.inpaint(init_image, mask_image, prompt=prompt, strength=strength, num_images=num_images,
                               steps=steps, guidance_scale=guidance, seed=seed, scheduler=scheduler)
    return JSONResponse(content={"results": results})


@app.post("/upscale")
async def upscale(scale: int = Form(2), image: UploadFile = File(...)):
    img_bytes = await image.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    res = service.upscale(img, scale=scale)
    return JSONResponse(content=res)


@app.post("/batch")
async def batch(prompts: List[str] = Form(...), num_images: int = Form(1), steps: int = Form(20), guidance: float = Form(7.5),
                width: int = Form(512), height: int = Form(512), seed: Optional[int] = Form(None), scheduler: Optional[str] = Form(None)):
    job_id = str(uuid.uuid4())
    results = []
    for p in prompts:
        res = service.generate_text2img(p, num_images=num_images, steps=steps, guidance_scale=guidance, width=width, height=height, seed=seed, scheduler=scheduler)
        results.append({"prompt": p, "results": res})
    return JSONResponse(content={"job_id": job_id, "results": results})
