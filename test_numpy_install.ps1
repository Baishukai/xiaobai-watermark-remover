# Test numpy 1.x version installation (check mode)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Testing numpy 1.x version availability" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

python --version
Write-Host ""

# Test several versions, actually try to download (but don't install)
$versions = @("1.26.4", "1.26.3", "1.26.2", "1.25.2", "1.24.4")
$available = @()

foreach ($version in $versions) {
    Write-Host "Testing numpy $version..." -ForegroundColor Yellow
    
    # Use pip download --dry-run to check if there's a precompiled package
    $output = pip download --dry-run --no-deps --prefer-binary --only-binary :all: "numpy==$version" 2>&1
    
    if ($LASTEXITCODE -eq 0 -or $output -match "\.whl") {
        Write-Host "  OK: numpy $version has precompiled package available" -ForegroundColor Green
        $available += $version
    } else {
        Write-Host "  FAILED: numpy $version has no precompiled package" -ForegroundColor Red
        # Show detailed error
        if ($output -match "ERROR" -or $output -match "No matching distribution") {
            Write-Host "    Reason: No matching precompiled package found" -ForegroundColor Red
        }
    }
    Write-Host ""
}

Write-Host "========================================" -ForegroundColor Cyan
if ($available.Count -gt 0) {
    Write-Host "Found $($available.Count) available versions:" -ForegroundColor Green
    foreach ($v in $available) {
        Write-Host "  - numpy==$v" -ForegroundColor Green
    }
    Write-Host ""
    Write-Host "You can install with:" -ForegroundColor Yellow
    Write-Host "pip install numpy==$($available[0]) --prefer-binary" -ForegroundColor Cyan
} else {
    Write-Host "FAILED: No numpy 1.x precompiled packages found" -ForegroundColor Red
    Write-Host ""
    Write-Host "This means Python 3.14 is too new, no precompiled numpy 1.x packages available" -ForegroundColor Yellow
    Write-Host "Suggested solutions:" -ForegroundColor Yellow
    Write-Host "1. Use Python 3.11 or 3.12 (Recommended)" -ForegroundColor White
    Write-Host "2. Install Visual Studio Build Tools to compile numpy" -ForegroundColor White
    Write-Host "3. Wait for numpy official updates" -ForegroundColor White
}
