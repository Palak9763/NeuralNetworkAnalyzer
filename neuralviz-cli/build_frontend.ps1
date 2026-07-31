# build_frontend.ps1
# Run this from the project root (NeuralNetworkAnalyzer-main/) to:
#   1. Build the frontend with Vite
#   2. Copy the output into the neuralviz CLI package
#
# Requires Node.js and npm to be on PATH.
# After running this, do: pip install -e neuralviz-cli/ to pick up the new build.

$ErrorActionPreference = "Stop"
# neuralviz-cli/ is a subdirectory of the project root, so go up one level
$ROOT = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Definition)

Write-Host "[1/3] Building frontend..." -ForegroundColor Cyan
Push-Location (Join-Path $ROOT "frontend")
try {
    npm install
    npm run build
} finally {
    Pop-Location
}

$DIST_SRC  = Join-Path (Join-Path $ROOT "frontend") "dist"
$DIST_DEST = Join-Path (Join-Path $ROOT "neuralviz-cli") (Join-Path "neuralviz" "frontend_dist")

Write-Host "[2/3] Copying dist/ -> neuralviz-cli/neuralviz/frontend_dist/ ..." -ForegroundColor Cyan
if (Test-Path $DIST_DEST) {
    # Remove existing dist but keep the .gitkeep
    Get-ChildItem -Path $DIST_DEST -Exclude ".gitkeep" | Remove-Item -Recurse -Force
}

Copy-Item -Recurse -Force -Path (Join-Path $DIST_SRC "*") -Destination $DIST_DEST

Write-Host "[3/3] Done." -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:"
Write-Host "  pip install -e neuralviz-cli/     # install/refresh local editable install"
Write-Host "  neuralviz path/to/model.py         # test it!"
