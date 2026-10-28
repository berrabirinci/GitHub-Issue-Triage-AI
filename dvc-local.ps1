$ErrorActionPreference = 'Stop'
$env:DVC_GLOBAL_CONFIG_DIR = Join-Path $PSScriptRoot '.dvc-user-config'
$env:DVC_SYSTEM_CONFIG_DIR = Join-Path $PSScriptRoot '.dvc-system-config'
$env:DVC_SITE_CACHE_DIR = Join-Path $PSScriptRoot '.dvc-site-cache'
& (Join-Path $PSScriptRoot '.dvc-venv\Scripts\dvc.exe') @args
exit $LASTEXITCODE
