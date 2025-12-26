# Simple check for numpy 1.x availability on Python 3.14

Write-Host "Checking numpy 1.x precompiled packages for Python 3.14..." -ForegroundColor Cyan
Write-Host ""

python --version
Write-Host ""

$versions = @("1.26.4", "1.26.3", "1.26.2", "1.25.2", "1.24.4")
$found = $false

foreach ($version in $versions) {
    Write-Host -NoNewline "Checking numpy $version... "
    
    $result = pip download --dry-run --no-deps --prefer-binary --only-binary :all: "numpy==$version" 2>&1
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "OK (precompiled package available)" -ForegroundColor Green
        Write-Host ""
        Write-Host "You can install with:" -ForegroundColor Yellow
        Write-Host "pip install numpy==$version --prefer-binary" -ForegroundColor Cyan
        Write-Host "pip install opencv-python==4.8.1.78" -ForegroundColor Cyan
        $found = $true
        break
    } else {
        Write-Host "FAILED (no precompiled package)" -ForegroundColor Red
    }
}

Write-Host ""
if (-not $found) {
    Write-Host "RESULT: No numpy 1.x precompiled packages found for Python 3.14" -ForegroundColor Red
    Write-Host ""
    Write-Host "Solutions:" -ForegroundColor Yellow
    Write-Host "1. Use Python 3.11 or 3.12 (recommended)" -ForegroundColor White
    Write-Host "2. Install Visual Studio Build Tools" -ForegroundColor White
    Write-Host "3. Use conda instead of pip" -ForegroundColor White
}

