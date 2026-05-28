# run_overnight_v2.ps1
# 第二輪夜跑：MuZero v2 + PPO-Lag v2 + CPO v2
#
# 修正說明：
#   MuZero      - Dirichlet 探索噪音 + support [0,500] + td_steps 5 + 5000 ep (~6h)
#   PPO-Lag     - cost_limit 25→5，強迫 λ 啟動展示安全機制 (~2h)
#   CPO         - cost_limit 25→5，與 PPO-Lag 對齊方便教學對比 (~1h)
#
# 執行方式：.\run_overnight_v2.ps1

$PYTHON = "C:\Users\666\Desktop\RL\venv\Scripts\python.exe"
$BASE   = "C:\Users\666\Desktop\RL"
$LOG    = "$BASE\overnight_log_v2.txt"

if (Test-Path $LOG) { Remove-Item $LOG }

function Log-Write {
    param([string]$Msg, [string]$Color = "White")
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Msg"
    Write-Host $line -ForegroundColor $Color
    Add-Content -Path $LOG -Value $line
}

function Run-Job {
    param(
        [string]$Name,
        [string]$Dir,
        [string[]]$ClearDirs = @()
    )

    Log-Write "" "White"
    Log-Write "========================================"  "Cyan"
    Log-Write "  開始: $Name" "Cyan"
    Log-Write "========================================"  "Cyan"

    foreach ($d in $ClearDirs) {
        $p = Join-Path $Dir $d
        if (Test-Path $p) {
            Remove-Item -Recurse -Force $p
            Log-Write "  [清除] $p" "Yellow"
        }
    }

    $t0 = Get-Date
    Push-Location $Dir
    & $PYTHON -u train.py > "train_output.txt" 2> "train_err.txt"
    $exitCode = $LASTEXITCODE
    Pop-Location

    $elapsed = (Get-Date) - $t0
    $h = [int]$elapsed.TotalHours; $m = $elapsed.Minutes; $s = $elapsed.Seconds
    $timeStr = "${h}h ${m}m ${s}s"

    if ($exitCode -eq 0) {
        Log-Write "  完成: $Name | 耗時 $timeStr" "Green"
    } else {
        Log-Write "  失敗: $Name | exit=$exitCode | 耗時 $timeStr" "Red"
        Log-Write "  錯誤詳情: $Dir\train_err.txt" "Red"
    }
}

# ── 開始 ──────────────────────────────────────
$start = Get-Date
Log-Write "第二輪夜跑開始: $start" "Yellow"
Log-Write "預計: MuZero v2(~6h) → PPO-Lag v2(~2h) → CPO v2(~1h)" "Yellow"

# 1. MuZero v2（清除舊 buggy 結果，全新訓練）
Run-Job `
    "MuZero v2 (5000ep, Dirichlet+support[0,500]+td_steps=5)" `
    "$BASE\05_Model_Based\2019_MuZero" `
    @("checkpoints", "runs", "best_checkpoints")

# 2. PPO-Lagrangian v2（cost_limit=5，強迫 λ 啟動）
Run-Job `
    "PPO-Lagrangian v2 (cost_limit=5, ent_coef=0.01)" `
    "$BASE\10_Safe_RL\2019_PPO_Lagrangian" `
    @("checkpoints", "runs")

# 3. CPO v2（cost_limit=5）
Run-Job `
    "CPO v2 (cost_limit=5)" `
    "$BASE\10_Safe_RL\2017_CPO" `
    @("checkpoints", "runs")

# ── 結束 ──────────────────────────────────────
$total = (Get-Date) - $start
$th = [int]$total.TotalHours; $tm = $total.Minutes
Log-Write "" "White"
Log-Write "========================================"  "Yellow"
Log-Write "  全部完成！總耗時 ${th}h ${tm}m" "Yellow"
Log-Write "  Master log: $LOG" "Yellow"
Log-Write "========================================"  "Yellow"
