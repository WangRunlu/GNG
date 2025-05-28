# 选用老版本 Python 3.6，兼容旧依赖
FROM python:3.8

WORKDIR /workspace
COPY . /workspace

# 先升 pip，再分两步装依赖
RUN pip install --upgrade pip
RUN pip install numpy==1.13.1
RUN pip install networkx==2.0 imageio==2.2.0 matplotlib==2.0.2 future==0.16.0

CMD ["bash"]
