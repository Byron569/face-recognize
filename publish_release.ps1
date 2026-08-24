# AI Monitor — 分发脚本 (GitHub Release 资产发布 / PowerShell)
#
# 用途: 把「不随仓库分发」的模型/素材统一打到 GitHub Release, 供
#       install_python.ps1 从 Release 下载, 无需再依赖第三方官方源。
#
# 当前分发范围(本机资产 → Release):
#   pose_plugin/models/yolov8n-pose.pt  (6.5MB, 姿态检测模型)
#
#   buffalo_l(人脸识别, 约 325MB) 与 queda.mp4(私有测试视频) 不进入 Release,
#   前者仍走 insightface 官方包(见 README), 后者仅本地验证用。
#
# 前置要求:
#   - 已安装 GitHub CLI (winget install GitHub.cli) 且已登录: gh auth login
#   - PowerShell 执行策略: Set-ExecutionPolicy -Scope Process Bypass
#
# 用法:
#   .\publish_release.ps1                       # 打 release v0.1.0 + 上传模型
#   .\publish_release.ps1 -Tag v1.2.0           # 指定 tag
#   .\publish_release.ps1 -Tag v1.2.0 -Repo Byron569/face-recognize
#   .\publish_release.ps1 -OnlyUpload           # 只上传资产到已存在的 release, 不新建
#   .\publish_release.ps1 -DryRun               # 只打印将要执行的命令, 不改动远端

[CmdletBinding()]
param(
    [string]$Repo = '',
    [string]$Tag = 'V2.1.0',
    [string]$Title = 'AI Monitor 分发资产',
    [switch]$OnlyUpload,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$ConfirmPreference = 'None'

# 定位 git(file 在 PATH; 常见安装路径; GitHub Desktop 内置)
function Resolve-Git {
    foreach ($c in @('git', "$env:ProgramFiles\Git\cmd\git.exe", "${env:ProgramFiles(x86)}\Git\cmd\git.exe",
            "$env:LOCALAPPDATA\GitHubDesktop\*\resources\app\git\cmd\git.exe")) {
        $cmd = Get-Command $c -ErrorAction SilentlyContinue
        if ($cmd) { return $cmd.Source }
        $expanded = @(Get-Item $c -ErrorAction SilentlyContinue)
        if ($expanded -and (Test-Path $expanded[0].FullName)) { return $expanded[0].FullName }
    }
    return $null
}
$Git = Resolve-Git
if (-not $Git) { Write-Warning "未找到 git, 请用 -Repo Byron569/<name> 显式指定仓库" }

# 定位 gh 可执行文件(winget 安装后 shell PATH 可能未刷新)
function Resolve-Gh {
    foreach ($c in @('gh', "$env:ProgramFiles\GitHub CLI\gh.exe", "${env:ProgramFiles(x86)}\GitHub CLI\gh.exe")) {
        $cmd = Get-Command $c -ErrorAction SilentlyContinue
        if ($cmd) { return $cmd.Source }
        if ($c -and (Test-Path $c)) { return $c }
    }
    throw "未找到 gh。请安装: winget install GitHub.cli 并运行 gh auth login"
}

$Gh = Resolve-Gh

if (-not $Repo) {
    # 从远端 remote 推导, 形如 Byron569/face-recognize
    if ($Git) {
        $origin = (& $Git remote get-url origin 2>$null | Select-Object -First 1)
        if ($origin -and $origin -match 'github\.com[:/]([^/]+)/([^/]+?)(\.git)?/?$') {
            $Repo = "$($Matches[1])/$($Matches[2])"
        }
    }
    if (-not $Repo) {
        throw "无法推导仓库, 请用 -Repo Byron569/<name> 指定"
    }
}

# 待上传资产(本机路径)
$Assets = @(
    (Join-Path $PSScriptRoot 'pose_plugin\models\yolov8n-pose.pt')
)

foreach ($a in $Assets) {
    if (-not (Test-Path $a)) { throw "待上传资产不存在: $a" }
}

Write-Host "Repo  : $Repo"
Write-Host "Tag   : $Tag"
Write-Host "Assets:"
foreach ($a in $Assets) { Write-Host ("  - {0}  [{1:N1} MB]" -f $a, ((Get-Item $a).Length / 1MB)) }

# ── 1. 确保 Release 存在 ────────────────────────────────────
# 注意: gh release view 对不存在的 release 会往 stderr 输出并返回非零,
# 这里需在 $ErrorActionPreference='Stop' 下容忍该情况。
$exists = $false
$origPreference = $ErrorActionPreference
$ErrorActionPreference = 'SilentlyContinue'
try {
    & $Gh @('release', 'view', $Tag, '--repo', $Repo) 2>$null | Out-Null
    $exists = ($LASTEXITCODE -eq 0)
} finally {
    $ErrorActionPreference = $origPreference
}

if (-not $OnlyUpload) {
    # Release 说明(含与 V1.1.0 的更新对比) —— 用数组 join 避免 here-string 结束符缩进陷阱
    $NotesLines = @(
        "## V2.1.0 相比 V1.1.0 的更新"
        ""
        "### 新功能"
        "- **摔倒检测（YOLOv8-Pose 融合）** — 新增独立的 GPU Worker 进程（PyTorch CUDA + FP16），多摄像头共享单一推理引擎，基于时间的状态机（NORMAL/POTENTIAL/FALLEN），输出 fall_potential / fall_detected / fall_recovered 事件。"
        "- **事件可靠投递** — Worker Journal(WAL SQLite) → 父进程 ACK 确认 + 退避重试 → PostgreSQL + WebSocket；失败事件转到 EventSpool，不丢失。"
        "- **实时骨架叠加** — 前端 Canvas 绘制 17 关键点 + 目标框，探测层 TTL 失效，200ms 内实时更新。"
        "- **健康接口** — GET /system/fall-runtime 返回脱敏运行快照（worker/gpu/model/delivery/cameras）。"
        "- **摄像头实时注册** — 动作引导五步多角度采集（前->左->右->上->下），自动捕捉、单方向重采、视频源切换（本机+系统监控摄像头）。"
        "- **一键安装/便携部署** — 新增 install_python.ps1 自动创建后端与 Worker 虚拟环境、安装依赖、下载并校验模型；配置路径改为按仓库根动态解析，clone 即跑。"
        ""
        "### 改进"
        "- 摄像头注册姓名改用 state 保存，修复提交丢失与黑屏/无反应。"
        "- 上下/左右方向判定修复，预览镜像，Steps 展示修正。"
        "- README 重写为工程化风格，便于他人克隆快速上手。"
        ""
        "### 架构"
        "- 姿态项目源码并入 pose_plugin/ 子目录；后端与姿态 Worker 双虚拟环境隔离，CUDA 硬性要求（无 CPU 回退）。"
        ""
        "## 资产说明"
        "- 本 Release 包含姿态检测模型 **yolov8n-pose.pt**（install_python.ps1 会优先从这里下载）。"
        "- buffalo_l 人脸识别器超大文件（约 325MB）不随仓库分发，请按 README 从官方包下载。"
    )
    $Notes = $NotesLines -join "`n"
    if (-not $exists) {
        $releaseCmd = "gh release create '$Tag' --repo '$Repo' --title '$Title' --notes <见脚本生成内容>"
        if ($DryRun) { Write-Host "DRYRUN: $releaseCmd" } else {
            Write-Host "`n[创建 Release $Tag]"
            & $Gh @('release', 'create', $Tag, '--repo', $Repo, '--title', $Title, '--notes', $Notes)
            if ($LASTEXITCODE -ne 0) { throw "创建 Release 失败" }
        }
    } else {
        Write-Host "`nRelease $Tag 已存在, 跳过创建"
    }
}

# ── 2. 上传资产 ─────────────────────────────────────────────
foreach ($a in $Assets) {
    $mb = [math]::Round((Get-Item $a).Length / 1MB, 1)
    Write-Host "`n[上传] $a (${mb} MB)"
    if ($DryRun) {
        Write-Host "DRYRUN: gh release upload '$Tag' '--repo' '$Repo' '--clobber' '--file' '$a'"
    } else {
        & $Gh @('release', 'upload', $Tag, '--repo', $Repo, '--clobber', '--file', $a)
        if ($LASTEXITCODE -ne 0) { throw "上传失败: $a" }
    }
}

Write-Host @"

==========================================================
  分发完成!
  Release: https://github.com/$Repo/releases/tag/$Tag
  install_python.ps1 会自动从该 Release 下载模型(优先)
  ✔ 上传前请确保已 gh auth login(脚本当前 token 对 repo 有写权限)
==========================================================
"@