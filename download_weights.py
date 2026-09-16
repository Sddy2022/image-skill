#!/usr/bin/env python3
"""
下载模型快照到本地（使用 huggingface_hub.snapshot_download）。
用法:
  python download_weights.py --model-id runwayml/stable-diffusion-v1-5 --output-dir models/sd-v1-5
"""
import argparse
import os
from huggingface_hub import snapshot_download
from dotenv import load_dotenv

load_dotenv()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, required=True, help="模型 id（Hugging Face）")
    parser.add_argument("--output-dir", type=str, default="models", help="权重保存目录")
    parser.add_argument("--token-env", type=str, default="HUGGINGFACE_TOKEN", help="环境变量名，包含 HF token")
    args = parser.parse_args()

    token = os.environ.get(args.token_env)
    if not token:
        print(f"请设置环境变量 {args.token_env}，例如在 .env 中或导出到 shell。")
        return

    out = snapshot_download(repo_id=args.model_id, cache_dir=args.output_dir, token=token)
    print("模型已下载至：", out)

if __name__ == "__main__":
    main()
