#!/usr/bin/env bash
# make_zip.sh - 打包项目为 zip
# 用法：
#   ./make_zip.sh            # 生成 image-skill.zip，默认排除 models/.venv/outputs/.env 等
#   ./make_zip.sh -o out.zip # 指定输出文件名
#   ./make_zip.sh -m         # 包含 models 目录（风险：文件可能非常大）
set -euo pipefail

OUTFILE="image-skill.zip"
INCLUDE_MODELS=0

show_help() {
  cat <<EOF
Usage: $0 [-m] [-o outfile.zip] [-h]
  -m    Include models/ directory and large checkpoint files in the zip (dangerous: may be very large)
  -o    Output zip filename (default: image-skill.zip)
  -h    Show this help
EOF
}

while getopts "mo:h" opt; do
  case "$opt" in
    m) INCLUDE_MODELS=1 ;;
    o) OUTFILE="$OPTARG" ;;
    h) show_help; exit 0 ;;
    *) show_help; exit 1 ;;
  esac
done

# 默认排除模式（zip -x supports patterns）
EXCLUDES=(
  ".venv/*"
  "venv/*"
  "outputs/*"
  "models/*"
  "__pycache__/*"
  "*.pyc"
  ".DS_Store"
  ".env"               # 不要把 .env 上传到公开位置
  "*.ckpt"
  "*.safetensors"
  "*.pt"
  "*.bin"
  "*.ckpt.index"
)

if [ "$INCLUDE_MODELS" -eq 1 ]; then
  echo "注意：你选择了包含 models 和大文件，确认继续？(y/N)"
  read -r CONFIRM
  if [[ "$CONFIRM" != "y" && "$CONFIRM" != "Y" ]]; then
    echo "取消：未包含 models。"
    INCLUDE_MODELS=0
  else
    # 如果包含模型，则从排除列表移除 models/*
    NEW_EX=()
    for p in "${EXCLUDES[@]}"; do
      if [ "$p" != "models/*" ]; then
        NEW_EX+=("$p")
      fi
    done
    EXCLUDES=("${NEW_EX[@]}")
    echo "将包含 models/ 目录（请确保有足够磁盘与上传带宽）。"
  fi
fi

# 构造 zip 排除参数
EXCLUDE_PARAMS=()
for p in "${EXCLUDES[@]}"; do
  EXCLUDE_PARAMS+=("-x" "$p")
done

# 使用 zip 打包当前目录（排除脚本本身的临时文件）
echo "Creating $OUTFILE ..."
# Remove existing zip if exists
rm -f "$OUTFILE"

# Use zip if available
if command -v zip >/dev/null 2>&1; then
  # zip -r <outfile> . -x patterns
  zip -r "$OUTFILE" . "${EXCLUDE_PARAMS[@]}" >/dev/null
else
  # fallback to tar.gz if zip isn't installed
  echo "zip command not found; fallback to tar.gz (output will be named .tar.gz)"
  if [ "${OUTFILE##*.}" != "gz" ]; then
    OUTFILE="${OUTFILE%.*}.tar.gz"
  fi
  tar --exclude='.venv' --exclude='venv' --exclude='outputs' --exclude='models' --exclude='.env' -czf "$OUTFILE" .
fi

echo "Zip created: $OUTFILE"

# 生成文件列表（便于上传到豆包时确认）
if command -v unzip >/dev/null 2>&1 && [[ "$OUTFILE" == *.zip ]]; then
  unzip -l "$OUTFILE" | sed -n '4,$p' | sed '$d' > "${OUTFILE}.manifest.txt"
  echo "Manifest saved to ${OUTFILE}.manifest.txt"
elif [[ "$OUTFILE" == *.tar.gz ]]; then
  tar -tzf "$OUTFILE" > "${OUTFILE}.manifest.txt"
  echo "Manifest saved to ${OUTFILE}.manifest.txt"
fi

echo "Done."
