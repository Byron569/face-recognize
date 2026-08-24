# AI Monitor — 一键安装脚本 (Windows / PowerShell)
#
# 用途: 仓库 clone 下来之后, 一条命令完成环境准备:
#   1. 后端 Python 虚拟环境  .venv                (FastAPI + InsightFace + onnxruntime-gpu)
#   2. 姿态 GPU Worker 虚拟环境 pose_plugin/.venv-worker (PyTorch CUDA + YOLOv8-Pose)
#   3. 下载/校验推理模型   yolov8n-pose.pt + buffalo_l
#   4. (可选) 前端 npm install + build
#
# 用法:
#   .\install_python.ps1                 # 后端 + 姿态Worker + 模型
#   .\install_python.ps1 -SkipFrontend   # 跳过前端
#   .\install_python.ps1 -SkipModels     # 跳过模型下载
#   .\install_python.ps1 -OnlyBackend    # 只装后端
#   .\install_python.ps1 -OnlyWorker     # 只装姿态 Worker
#   .\install_python.ps1 -Mirror         # 后端 pip 走清华镜像(国内加速)
#
# 前置要求: 已安装 Python 3.12(姿态 Worker 使用 cp312 wheel), Node.js 16+。
#           启用 PowerShell 脚本执行时的执行策略限制:
#           Set-ExecutionPolicy -Scope Process Bypass

[CmdletBinding()]
param(
    [switch]$SkipFrontend,
    [switch]$SkipModels,
    [switch]$OnlyBackend,
    [switch]$OnlyWorker,
    [switch]$Mirror
)

$ErrorActionPreference = 'Stop'
$ConfirmPreference = 'None'

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not (Test-Path (Join-Path $Root 'configs\default.yaml'))) {
    $Root = $PSScriptRoot   # 脚本直接在仓库根目录时
}

$BackendPy  = Join-Path $Root '.venv\Scripts\python.exe'
$WorkerPy   = Join-Path $Root 'pose_plugin\.venv-worker\Scripts\python.exe'

function Write-Step($title) { Write-Host "[STEP] $title" -ForegroundColor Cyan }

function Resolve-Python {
    param([string]$Hint)
    $p = $Hint
    if (Test-Path $p) { return $p }
    # 用户最初在此机练出的 Python 3.12: 优先加进 PATH 再探测
    foreach ($cand in @('python', 'py')) {
        $cmd = Get-Command $cand -ErrorAction SilentlyContinue
        if ($cmd) {
            if ($cand -eq 'py') {
                $v = & py -3.12 -c "import sys;print(sys.version.split()[0])" 2>$null
                if ($v) {
                    $exe = & py -3.12 -c "import sys;print(sys.executable)" 2>$null
                    if ($exe) { return $exe }
                }
            } else {
                return $cmd.Source
            }
        }
    }
    throw "未找到 Python。请安装 Python 3.12 并加入 PATH，然后重试。"
}

function Invoke-Fallback {
    # 统一命令执行: 失败即抛错退出
    param([string]$cmd, [string[]]$argsList, [string]$desc)
    Write-Host "`n>> $desc"
    & $cmd @argsList
    if ($LASTEXITCODE -ne 0) {
        throw "$desc 失败(exit=$LASTEXITCODE)"
    }
}

# ──────────────────────────────────────────────────────────────
Write-Step "根目录: $Root"

# ── 1. 后端环境 ──────────────────────────────────────────────
if (-not $OnlyWorker) {
    Write-Step "1/5  后端 Python 虚拟环境 (.venv)"
    $SystemPy = Resolve-Python -Hint (Join-Path $Root '.venv\Scripts\python.exe')
    if (Test-Path $BackendPy) {
        Write-Host "    已存在 .venv，跳过创建"
    } else {
        & $SystemPy -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw "创建 .venv 失败" }
    }

    Write-Host "`n    安装 backend/requirements.txt ..."
    if ($Mirror) {
        & $BackendPy -m pip install -r (Join-Path $Root 'backend\requirements.txt') `
            -i https://pypi.tuna.tsinghua.edu.cn/simple
    } else {
        & $BackendPy -m pip install -r (Join-Path $Root 'backend\requirements.txt')
    }
    if ($LASTEXITCODE -ne 0) { throw "后端依赖安装失败" }

    # insightface(人脸识别内核, 由 requirements.txt 安装; 校验可导入)
    Write-Host "`n    insightface 导入校验 ..."
    & $BackendPy -c "import insightface; print('insightface', getattr(insightface,'__version__','?'))"
    if ($LASTEXITCODE -ne 0) { Write-Host "    ⚠ insightface 导入失败, 可重跑或用 -Mirror 换镜像重装" -ForegroundColor Yellow }

    # GPU 检查(onnxruntime CUDAExecutionProvider)
    Write-Host "`n    onnxruntime 可用 Provider 检查(CUDAExecutionProvider=CUDA 就绪)"
    & $BackendPy -c "import onnxruntime; prov=onnxruntime.get_available_providers(); print(prov); assert any('CUDAExecutionProvider'==p for p in prov) or any('TensorrtExecutionProvider'==p for p in prov)"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "    ⚠ 未检测到 CUDAExecutionProvider: 人脸推理将走 CPU(onnxruntime-gpu 需与 CUDA 运行库匹配)。" -ForegroundColor Yellow
    }
}

# ── 2. 姿态 GPU Worker 环境 ─────────────────────────────────
if (-not $OnlyBackend) {
    Write-Step "2/5  姿态 GPU Worker 虚拟环境 (pose_plugin/.venv-worker)"
    if (-not (Test-Path $WorkerPy)) {
        $WorkerBase = Resolve-Python -Hint $WorkerPy
        & $WorkerBase -m venv (Join-Path $Root 'pose_plugin\.venv-worker')
        if ($LASTEXITCODE -ne 0) { throw "创建 .venv-worker 失败" }
    } else {
        Write-Host "    已存在 .venv-worker，跳过创建"
    }

    # PyTorch CUDA (cu126) 从 PyTorch 官方源安装(可移植, 不依赖本机 wheel 路径)
    Write-Host "`n    安装 PyTorch 2.7.0+cu126 / torchvision 0.22.0+cu126 ..."
    & $WorkerPy -m pip install --index-url https://download.pytorch.org/whl/cu126 `
        torch==2.7.0 torchvision==0.22.0
    if ($LASTEXITCODE -ne 0) {
        Write-Host "    torch cu126 安装失败, 尝试国内 PyTorch 镜像 ..." -ForegroundColor Yellow
        & $WorkerPy -m pip install --index-url https://mirrors.aliyun.com/pytorch-wheels/cu126 `
            torch==2.7.0 torchvision==0.22.0
        if ($LASTEXITCODE -ne 0) { throw "PyTorch cu126 安装失败" }
    }

    # Worker 其余依赖(YOLOv8-Pose)
    Write-Host "`n    安装 YOLOv8-Pose 依赖 ..."
    if ($Mirror) {
        & $WorkerPy -m pip install "ultralytics==8.4.126" PyYAML opencv-python scipy nvidia-ml-py numpy `
            -i https://pypi.tuna.tsinghua.edu.cn/simple
    } else {
        & $WorkerPy -m pip install "ultralytics==8.4.126" PyYAML opencv-python scipy nvidia-ml-py numpy
    }
    if ($LASTEXITCODE -ne 0) { throw "Worker 依赖安装失败" }

    # GPU 真实校验
    Write-Host "`n    CUDA 可用性校验(真实加载) ..."
    & $WorkerPy -c "import torch; ok=torch.cuda.is_available(); print('cuda.available=',ok); assert ok; print('gpu=',torch.cuda.get_device_name(0))"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "    ⚠ 当前环境无可用 NVIDIA GPU/驱动, 姿态检测无法运行(CUDA 硬性要求)。" -ForegroundColor Yellow
    }
}

# ── 3. 模型下载/校验 ────────────────────────────────────────
if (-not $SkipModels -and -not $OnlyWorker -and -not $OnlyBackend) {
    Write-Step "3/5  推理模型"
    $PoseModelDir = Join-Path $Root 'pose_plugin\models'
    New-Item -ItemType Directory -Force -Path $PoseModelDir | Out-Null

    # yolov8n-pose.pt (来自 Ultralytics 官方 release; 用仓库内置 .sha256 校验)
    $PosePt  = Join-Path $PoseModelDir 'yolov8n-pose.pt'
    $ShaFile = Join-Path $PoseModelDir 'yolov8n-pose.pt.sha256'
    if (-not (Test-Path $PosePt)) {
        Write-Host "`n    下载 yolov8n-pose.pt(约 8MB, Ultralytics 官方源)..."
        $url = 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-pose.pt'
        try {
            Invoke-WebRequest -Uri $url -OutFile $PosePt -UseBasicParsing
        } catch {
            Write-Host "    ⚠ 直连 pytorch assets 失败, 尝试镜像 assets ..." -ForegroundColor Yellow
            $url = 'https://modelscope.cn/models/Ultralytics/YOLOv8-Pose/resolve/master/yolov8n-pose.pt'
            Invoke-WebRequest -Uri $url -OutFile $PosePt -UseBasicParsing
        }
    } else {
        Write-Host "    已存在 yolov8n-pose.pt，跳过下载"
    }

    # SHA-256 校验(与仓库 .sha256 一致)
    if (Test-Path $ShaFile) {
        $want = (Get-Content $ShaFile).Trim()
        $got  = (Get-FileHash $PosePt -Algorithm SHA256).Hash.ToLower()
        if ($got -ne $want -and $got -ne $want.ToLower()) {
            throw "yolov8n-pose.pt SHA-256 校验失败! 期望=$want 实际=$got"
        }
        Write-Host "    ✔ yolov8n-pose.pt SHA-256 校验通过"
    }

    # buffalo_l 人脸识别器(高精度, 超 GitHub 单文件限制, 需官方包)
    $BuffaloL = Join-Path $Root 'models\buffalo_l'
    if (-not (Test-Path (Join-Path $BuffaloL 'w600k_r50.onnx'))) {
        Write-Host "`n    buffalo_l 人脸识别器未检测到 w600k_r50.onnx"
        Write-Host "    官方下载: https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
        Write-Host "    下载后解压到: $BuffaloL"
        Write-Host "    (轻量级 buffalo_s 已随仓库提供; 高精度请按 README『Models』章节操作)" -ForegroundColor Yellow
    } else {
        Write-Host "    ✔ buffalo_l 已就绪"
    }
}

# ── 4. 前端依赖 ─────────────────────────────────────────────
$DoFrontend = (-not $SkipFrontend) -and (-not $OnlyBackend) -and (-not $OnlyWorker)
if ($DoFrontend) {
    Write-Step "4/5  前端依赖 (npm install + build)"
    if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
        Write-Host "    ⚠ 未找到 Node/npm, 跳过前端(可之后手动: cd frontend; npm install)" -ForegroundColor Yellow
    } else {
        Push-Location (Join-Path $Root 'frontend')
        try {
            if (Test-Path 'node_modules') {
                Write-Host "    node_modules 已存在, 跳过 npm install"
            } else {
                npm install --no-audit --no-fund
                if ($LASTEXITCODE -ne 0) { Write-Host "    ⚠ npm install 失败" -ForegroundColor Yellow }
            }
            Write-Host "`n    构建前端 ..."
            npm run build 2>$null
            if ($LASTEXITCODE -ne 0) { Write-Host "    ⚠ 前端 build 失败(可忽略; 开发模式用 npm run dev)" -ForegroundColor Yellow }
        } finally {
            Pop-Location
        }
    }
}

# ── 5. 收尾提示 ─────────────────────────────────────────────
Write-Step "5/5  完成"

Write-Host @"

==========================================================
 环境安装完成! 接下来启动服务:
==========================================================
 1) 数据库(PostgreSQL 16, 一次性):
    docker run -d --name ai-monitor-db -p 5432:5432 ^
      -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=123456 -e POSTGRES_DB=ai_monitor postgres:16-alpine

 2) 后端(项目根):
    .\.venv\Scripts\python.exe -m alembic -c backend\alembic.ini upgrade head  # 或:
    cd backend; ..\.venv\Scripts\python.exe -m alembic upgrade head
    cd backend; ..\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000

 3) 前端(另开终端):
    cd frontend; npm run dev    # http://localhost:3000

 4) 启用姿态检测: 编辑 configs/default.yaml,
    tasks.fall_detection.enabled -> true
    (worker.python 应指向本机: $Root\pose_plugin\.venv-worker\Scripts\python.exe)

 5) 故障排查:
    - GPU/Worker 错误: set  env AI_MONITOR_POSE_WORKER_STDERR=<logfile> 观察 stderr
    - CUDA 校验:  .\pose_plugin\.venv-worker\Scripts\python.exe -m scripts.gpu_smoke
==========================================================
"@