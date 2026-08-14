FROM python:3.12-slim

WORKDIR /app

# 系统依赖(OpenCV / 运行时)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 后端依赖(含 InsightFace 推理内核;GPU 镜像请把 onnxruntime 换成 onnxruntime-gpu 并挂载 NVIDIA runtime)
COPY backend/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# 项目源码
COPY configs/ /app/configs/
COPY vision/ /app/vision/
COPY backend/ /app/backend/
COPY models/ /app/models/

WORKDIR /app/backend

# 数据库连接(容器内覆盖 .env;密码等敏感项请用环境变量注入)
ENV AIM_PROJECT_ROOT=/app
ENV AIM_DATABASE_URL=postgresql+asyncpg://postgres:123456@db:5432/ai_monitor

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
