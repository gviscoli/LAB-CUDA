# setup.ps1 — Setup LAB-CUDA su Windows 11
# Esegui da PowerShell come amministratore:
#   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
#   .\setup.ps1

Write-Host "╔══════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  LAB-CUDA Setup — RTX 4080 + i9 + 96GB  ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════╝" -ForegroundColor Cyan

# 1. Crea venv
Write-Host "`n[1/5] Creando ambiente virtuale..." -ForegroundColor Yellow
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Aggiorna pip
Write-Host "[2/5] Aggiornando pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# 3. PyTorch CUDA 12.4
Write-Host "[3/5] Installando PyTorch CUDA 12.4..." -ForegroundColor Yellow
pip install torch==2.6.0+cu124 torchvision torchaudio `
    --index-url https://download.pytorch.org/whl/cu124

# 4. CuPy
Write-Host "[4/5] Installando CuPy CUDA 12.x..." -ForegroundColor Yellow
pip install cupy-cuda12x

# 5. Resto dipendenze
Write-Host "[5/5] Installando dipendenze lab..." -ForegroundColor Yellow
pip install -r requirements.txt

# Verifica
Write-Host "`nVerifica ambiente..." -ForegroundColor Green
python shared/utils/gpu_info.py

Write-Host "`n✅ Setup completato!" -ForegroundColor Green
Write-Host "Avvia Jupyter con: jupyter lab" -ForegroundColor Cyan
