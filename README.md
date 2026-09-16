# image-skill (Stable Diffusion + Diffusers) 示例

说明
- 这是一个最小可用的图像生成 skill，使用 Hugging Face Diffusers（Stable Diffusion）。
- 提供 CLI（generate.py）和 Gradio Web UI（app.py）。
- 我没有把模型权重包含在仓库（建议不要把大权重放仓库），而是提供下载脚本 download_weights.py。

前提
- 一台有 NVIDIA GPU（推荐）或 CPU（性能差很多）。
- Python 3.9+，CUDA 驱动（如果用 GPU）。
- 一个 Hugging Face 访问令牌（HUGGINGFACE_TOKEN），用于下载受限模型（如 `runwayml/stable-diffusion-v1-5` 或其他）。

安装依赖
1. 建议使用虚拟环境：
   python -m venv .venv
   source .venv/bin/activate  # Linux / macOS
   .venv\Scripts\activate     # Windows

2. 安装：
   pip install -r requirements.txt

配置环境变量
- 在项目根目录复制 .env.example 为 .env 并填写 HUGGINGFACE_TOKEN：
  cp .env.example .env
  # 编辑 .env 填写 HUGGINGFACE_TOKEN

下载模型权重（推荐在有 GPU 的机器）
- 可使���脚本：
  python download_weights.py --model-id runwayml/stable-diffusion-v1-5 --output-dir models/stable-diffusion-v1-5

运行 CLI 生成图像
- 示例：
  python generate.py --prompt "A cute robot painting a landscape" --outdir outputs --num-images 2

运行 Web UI（Gradio）
- 启动：
  python app.py
- 在浏览器打开 http://localhost:7860

打包为 zip（在项目根目录执行）
- UNIX:
  zip -r image-skill.zip README.md requirements.txt .env.example download_weights.py generate.py app.py
- Windows PowerShell:
  Compress-Archive -Path * -DestinationPath image-skill.zip

注意与建议
- 模型权重通常很大（GB 级），不要把它们放入仓库；在“豆包”里放 zip 时，请仅上传代码与说明并在第一次运行时下载权重，或把模型放在共享存储。
- 若无 GPU，可考虑使用 smaller checkpoint / ONNX / 4-bit quantized runtimes（需要额外步骤）。
