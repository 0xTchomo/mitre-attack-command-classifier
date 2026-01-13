param(
  [string]$RepoPath = (Get-Location).Path,
  [switch]$Force
)

function Write-Info($msg) { Write-Host "[INFO] $msg" -ForegroundColor Cyan }
function Write-Warn($msg) { Write-Host "[WARN] $msg" -ForegroundColor Yellow }

function Ensure-Dir([string]$Path) {
  if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
    New-Item -ItemType Directory -Path $Path | Out-Null
    Write-Info "Created directory: $Path"
  } else {
    Write-Info "Directory exists: $Path"
  }
}

function Ensure-File([string]$Path, [string]$Content = "") {
  if (Test-Path -LiteralPath $Path -PathType Leaf) {
    if ($Force) {
      Set-Content -LiteralPath $Path -Value $Content -Encoding UTF8
      Write-Warn "Overwritten file (Force): $Path"
    } else {
      Write-Info "File exists (kept): $Path"
    }
  } else {
    Set-Content -LiteralPath $Path -Value $Content -Encoding UTF8
    Write-Info "Created file: $Path"
  }
}

# --- Validate repo root (optional warning) ---
$gitDir = Join-Path $RepoPath ".git"
if (-not (Test-Path -LiteralPath $gitDir -PathType Container)) {
  Write-Warn "No .git folder found in: $RepoPath"
  Write-Warn "Run this script from the ROOT of your cloned repo (where .git exists), or pass -RepoPath."
}

Write-Info "RepoPath: $RepoPath"

# --- Directories to create ---
$dirs = @(
  "command_classifier",
  "notebooks",
  "data\raw",
  "data\sample",
  "models",
  "reports\figures",
  ".github\workflows"
)

foreach ($d in $dirs) {
  Ensure-Dir (Join-Path $RepoPath $d)
}

# --- .gitkeep for empty folders (so Git can track them) ---
$gitkeepPaths = @(
  "notebooks\.gitkeep",
  "data\sample\.gitkeep",
  "reports\figures\.gitkeep",
  ".github\workflows\.gitkeep"
)
foreach ($p in $gitkeepPaths) {
  Ensure-File (Join-Path $RepoPath $p) "# keep"
}

# --- .gitignore (safe defaults for python + data/models) ---
$gitignore = @"
# Python
__pycache__/
*.py[cod]
*.pyo
*.pyd
*.so
*.egg-info/
dist/
build/
.cache/
.pytest_cache/
.mypy_cache/
.ruff_cache/

# Jupyter
.ipynb_checkpoints/

# Environments
.venv/
venv/
env/

# OS / IDE
.DS_Store
Thumbs.db
.vscode/
.idea/

# Data & Models (don't commit large/private artifacts)
data/raw/
models/
"@

Ensure-File (Join-Path $RepoPath ".gitignore") $gitignore

# --- Minimal requirements files (adjust later) ---
$reqML = @"
numpy
pandas
scikit-learn
matplotlib
jupyter
"@
Ensure-File (Join-Path $RepoPath "requirements.txt") $reqML

$reqDL = @"
tensorflow
numpy
pandas
scikit-learn
matplotlib
jupyter
"@
Ensure-File (Join-Path $RepoPath "requirements-dl.txt") $reqDL

# --- Placeholder python entry points (non-destructive) ---
$trainLR = @"
\"\"\"
Train a baseline ML model (TF-IDF + Logistic Regression) for MITRE ATT&CK command classification.
Fill in the dataset path and columns based on your CSV.
\"\"\"

def main():
    print(\"TODO: implement train_lr pipeline\")

if __name__ == \"__main__\":
    main()
"@
Ensure-File (Join-Path $RepoPath "command_classifier\train_lr.py") $trainLR

$predict = @"
\"\"\"
Simple inference demo.
Load a trained model and predict technique_grouped from a proctitle string.
\"\"\"

def main():
    cmd = input(\"Enter a command/proctitle: \").strip()
    print(f\"TODO: predict label for: {cmd}\")

if __name__ == \"__main__\":
    main()
"@
Ensure-File (Join-Path $RepoPath "command_classifier\predict.py") $predict

Write-Info "Done. Next steps:"
Write-Host "  tree /f" -ForegroundColor Green
Write-Host "  git status" -ForegroundColor Green
Write-Host "  git add ." -ForegroundColor Green
Write-Host "  git commit -m `"Initialize professional project structure`"" -ForegroundColor Green
Write-Host "  git push" -ForegroundColor Green
