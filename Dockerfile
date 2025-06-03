# ===== Dockerfile  =====
# 基础镜像：Python 3.6  （满足 neupy + TensorFlow 1.x 依赖）
FROM python:3.6

WORKDIR /workspace
COPY . /workspace

# 1️⃣ 先升级 pip
RUN pip install --upgrade pip

# 2️⃣ 先装 numpy（必须 >=1.15 以兼容 gudhi，且 <=1.16.* 才不和 TF1.13 冲突）
RUN pip install numpy==1.16.6

# 3️⃣ 其余科学计算包（scipy / matplotlib 选兼容 1.16.6 的最高版本）
RUN pip install scipy==1.2.3 \
                matplotlib==2.2.5

# 4️⃣ 其余项目依赖一次装齐
RUN pip install \
        networkx==2.0 \
        imageio==2.2.0 \
        future==0.16.0 \
        bresenham \
        gudhi==3.8.0 \
        neupy==0.8.2

# 默认进入 bash
CMD ["bash"]
# =======================
