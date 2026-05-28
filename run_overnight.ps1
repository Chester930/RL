# run_overnight.ps1
# 夜跑自動訓練腳本
# 執行方式：在 PowerShell 終端輸入 .\run_overnight.ps1 後離開即可
# 預計順序：MuZero(~4h) → DDPG(~2h) → TD3(~2h) → PPO-Lag(~4h) → CPO(~5h)
# 12 小時內確定完成前 4 個；CPO 視速度而定

$PYTHON = "C:\Users\666\Desktop\RL\venv\Scripts\python.exe"
$BASE   = "C:\Users\666\Desktop\RL"
$LOG    = "$BASE\overnight_log.txt"

# 清除舊的 master log
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

    # 清除指定目錄（舊 checkpoint / logs）
    foreach ($d in $ClearDirs) {
        $p = Join-Path $Dir $d
        if (Test-Path $p) {
            Remove-Item -Recurse -Force $p
            Log-Write "  [清除] $p" "Yellow"
        }
    }

    $t0 = Get-Date
    Push-Location $Dir

    # 執行訓練，stdout → train_output.txt，stderr → train_err.txt（均覆蓋）
    & $PYTHON -u train.py > "train_output.txt" 2> "train_err.txt"
    $exitCode = $LASTEXITCODE

    Pop-Location

    $elapsed = (Get-Date) - $t0
    $h = [int]$elapsed.TotalHours
    $m = $elapsed.Minutes
    $s = $elapsed.Seconds
    $timeStr = "${h}h ${m}m ${s}s"

    if ($exitCode -eq 0) {
        Log-Write "  完成: $Name | 耗時 $timeStr" "Green"
    } else {
        Log-Write "  失敗: $Name | exit=$exitCode | 耗時 $timeStr" "Red"
        Log-Write "  錯誤詳情請看: $Dir\train_err.txt" "Red"
    }
}

# ──────────────────────────────────────────────
#  夜跑開始
# ──────────────────────────────────────────────
$globalStart = Get-Date
Log-Write "夜跑開始: $globalStart" "Yellow"
Log-Write "預計任務清單：" "Yellow"
Log-Write "  1. MuZero      D-3  (~4h) - 修正 policy target，重新訓練 3000 ep" "Yellow"
Log-Write "  2. DDPG        D-1  (~2h) - 補存 checkpoint（200K steps Pendulum）" "Yellow"
Log-Write "  3. TD3         D-2  (~2h) - 補存 checkpoint（200K steps Pendulum）" "Yellow"
Log-Write "  4. PPO-Lag     M-1  (~4h) - 首次完整訓練（300K steps SafePendulum）" "Yellow"
Log-Write "  5. CPO         M-2  (~5h) - 首次完整訓練（300K steps SafePendulum）" "Yellow"
Log-Write "" "White"

# ──────────────────────────────────────────────
#  1. MuZero (清除舊 buggy 結果，全新訓練)
# ──────────────────────────────────────────────
Run-Job `
    "MuZero D-3 (3000ep, CartPole-v1)" `
    "$BASE\05_Model_Based\2019_MuZero" `
    @("checkpoints", "runs", "best_checkpoints")

# ──────────────────────────────────────────────
#  2. DDPG (從頭訓練 200K steps)
# ──────────────────────────────────────────────
Run-Job `
    "DDPG D-1 (200K steps, Pendulum-v1)" `
    "$BASE\04_Actor_Critic_Continuous\2015_DDPG"

# ──────────────────────────────────────────────
#  3. TD3 (從頭訓練 200K steps)
# ──────────────────────────────────────────────
Run-Job `
    "TD3 D-2 (200K steps, Pendulum-v1)" `
    "$BASE\04_Actor_Critic_Continuous\2018_TD3"

# ──────────────────────────────────────────────
#  4. PPO-Lagrangian (300K steps SafePendulum)
# ──────────────────────────────────────────────
Run-Job `
    "PPO-Lagrangian M-1 (300K steps, SafePendulum)" `
    "$BASE\10_Safe_RL\2019_PPO_Lagrangian" `
    @("runs")

# ──────────────────────────────────────────────
#  5. CPO (300K steps SafePendulum, TRPO 較慢)
# ──────────────────────────────────────────────
Run-Job `
    "CPO M-2 (300K steps, SafePendulum)" `
    "$BASE\10_Safe_RL\2017_CPO" `
    @("runs")

# ──────────────────────────────────────────────
#  全部完成
# ──────────────────────────────────────────────
$totalElapsed = (Get-Date) - $globalStart
$th = [int]$totalElapsed.TotalHours
$tm = $totalElapsed.Minutes
Log-Write "" "White"
Log-Write "========================================"  "Yellow"
Log-Write "  全部完成！總耗時 ${th}h ${tm}m" "Yellow"
Log-Write "  Master log: $LOG" "Yellow"
Log-Write "========================================"  "Yellow"
