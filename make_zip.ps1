<#
  make_zip.ps1 - Windows 打包脚本（PowerShell）
  用法:
    .\make_zip.ps1            # 生成 image-skill.zip（不包含 models/.env/.venv/outputs）
    .\make_zip.ps1 -Out "out.zip" -IncludeModels
#>
param(
  [string]$Out = "image-skill.zip",
  [switch]$IncludeModels
)

$excludes = @(".venv","venv","outputs",".env","*.ckpt","*.safetensors","*.pt","models")
if ($IncludeModels) {
  Write-Host "IncludeModels set: models 将会包含（请注意大小）"
  $excludes = $excludes | Where-Object { $_ -ne "models" }
} else {
  Write-Host "默认不包含 models 目录。"
}

# Build file list
$files = Get-ChildItem -Recurse -File | Where-Object {
  $p = $_.FullName
  foreach ($e in $excludes) {
    if ($e -like "*.*" -and $_.Name -like $e) { return $false }
    if ($p -like "*\$e*") { return $false }
  }
  return $true
}

# Use a temp folder to copy files (so Compress-Archive works nicely)
$tmp = Join-Path $env:TEMP ("packtmp_" + [guid]::NewGuid().ToString())
New-Item -ItemType Directory -Path $tmp | Out-Null
foreach ($f in $files) {
  $rel = $f.FullName.Substring((Get-Location).Path.Length).TrimStart('\')
  $dest = Join-Path $tmp $rel
  New-Item -ItemType Directory -Path (Split-Path $dest) -Force | Out-Null
  Copy-Item $f.FullName -Destination $dest -Force
}

if (Test-Path $Out) { Remove-Item $Out -Force }
Compress-Archive -Path (Join-Path $tmp '*') -DestinationPath $Out -Force
Write-Host "Created $Out"
Remove-Item -Recurse -Force $tmp
