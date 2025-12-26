# 检查 numpy 1.x 版本是否有预编译包（Python 3.14）

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "检查 numpy 1.x 预编译包可用性" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 显示 Python 版本
python --version
Write-Host ""

# 测试几个常见的 numpy 1.x 版本
$versions = @("1.26.4", "1.26.3", "1.26.2", "1.26.1", "1.26.0", "1.25.2", "1.25.1", "1.24.4")
$available = @()

foreach ($version in $versions) {
    Write-Host "检查 numpy $version..." -ForegroundColor Yellow -NoNewline
    # 使用 pip index versions 或直接尝试下载（dry-run）
    $result = pip index versions numpy 2>&1 | Select-String $version
    if ($result) {
        Write-Host "  ✓ 版本存在" -ForegroundColor Green
        
        # 尝试检查是否有 wheel 文件
        Write-Host "  检查预编译包..." -ForegroundColor Yellow -NoNewline
        $wheelCheck = pip download --dry-run --no-deps --prefer-binary "numpy==$version" 2>&1
        if ($wheelCheck -match "\.whl" -or $wheelCheck -match "Found existing installation") {
            Write-Host "  ✓ 可能有预编译包" -ForegroundColor Green
            $available += $version
        } else {
            Write-Host "  ✗ 没有预编译包" -ForegroundColor Red
        }
    } else {
        Write-Host "  ✗ 版本不存在" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
if ($available.Count -gt 0) {
    Write-Host "找到可用的版本:" -ForegroundColor Green
    foreach ($v in $available) {
        Write-Host "  - numpy $v" -ForegroundColor Green
    }
    Write-Host ""
    Write-Host "建议使用: numpy==$($available[0])" -ForegroundColor Yellow
} else {
    Write-Host "❌ 没有找到任何 numpy 1.x 的预编译包" -ForegroundColor Red
    Write-Host ""
    Write-Host "建议:" -ForegroundColor Yellow
    Write-Host "1. 使用 Python 3.11 或 3.12" -ForegroundColor Yellow
    Write-Host "2. 安装 Visual Studio Build Tools 来编译 numpy" -ForegroundColor Yellow
    Write-Host "3. 等待相关包的更新" -ForegroundColor Yellow
}

