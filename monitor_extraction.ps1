param(
    [int]$IntervalSeconds = 10
)

$featurePath = "C:\Users\rishi\CATALOG\features\Features_serengeti\standard_features"
$expectedFiles = 6  # train/test/val x 16/32

Write-Host "=== FEATURE EXTRACTION PROGRESS ===" -ForegroundColor Cyan
Write-Host "Expected: $expectedFiles .pt files (train/test/val x 16/32 clip modes)" -ForegroundColor Gray
Write-Host ""

$startTime = Get-Date
$lastCount = 0

while ($true) {
    $fileCount = (Get-ChildItem "$featurePath\*.pt" -ErrorAction SilentlyContinue).Count
    $elapsed = (Get-Date) - $startTime
    $pythonRunning = (Get-Process python -ErrorAction SilentlyContinue).Count -gt 0
    
    # Calculate progress bar
    $progressPercent = [int]($fileCount / $expectedFiles * 100)
    $barLength = 30
    $filledLength = [int]($barLength * $fileCount / $expectedFiles)
    $bar = "#" * $filledLength + "-" * ($barLength - $filledLength)
    
    # Calculate ETA (rough estimate)
    if ($fileCount -gt 0 -and $elapsed.TotalSeconds -gt 0) {
        $timePerFile = $elapsed.TotalSeconds / $fileCount
        $remainingFiles = $expectedFiles - $fileCount
        $etaSeconds = $timePerFile * $remainingFiles
        $etaTime = (Get-Date).AddSeconds($etaSeconds)
        $etaString = $etaTime.ToString("HH:mm:ss")
    } else {
        $etaString = "calculating..."
    }
    
    # Status
    $status = if ($pythonRunning) { "[RUNNING]" } else { "[STOPPED]" }
    $elapsedString = "{0:hh\:mm\:ss}" -f $elapsed
    
    Clear-Host
    Write-Host "=== FEATURE EXTRACTION PROGRESS ===" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Status:     $status" -ForegroundColor $(if ($pythonRunning) { 'Green' } else { 'Red' })
    Write-Host "Files Done: $fileCount / $expectedFiles"
    Write-Host "Progress:   [$bar] $progressPercent%"
    Write-Host ""
    Write-Host "Elapsed:    $elapsedString"
    Write-Host "ETA Done:   $etaString"
    Write-Host ""
    Write-Host "Generated files:" -ForegroundColor Gray
    Get-ChildItem "$featurePath\*.pt" -ErrorAction SilentlyContinue | ForEach-Object {
        $size = "{0:N1}" -f ($_.Length / 1MB)
        Write-Host "  * $($_.Name) ($size MB)"
    }
    
    if ($fileCount -eq $expectedFiles) {
        Write-Host ""
        Write-Host "COMPLETE!" -ForegroundColor Green
        break
    }
    
    if (-not $pythonRunning -and $fileCount -lt $expectedFiles) {
        Write-Host ""
        Write-Host "WARNING: Python process stopped but extraction incomplete!" -ForegroundColor Yellow
    }
    
    Start-Sleep -Seconds $IntervalSeconds
}
