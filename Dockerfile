FROM python:3.10

WORKDIR /workspace

# 复制项目代码进容器
COPY . /workspace

# 安装系统依赖
RUN apt-get update && \
    apt-get install -y build-essential libgmp-dev imagemagick && \
    rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# 进入 bash，方便交互
CMD ["bash"]
