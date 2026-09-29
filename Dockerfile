# 多阶段构建：先构建 wheel，再以非 root 用户运行
FROM python:3.11-slim AS builder
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --user build && python -m build --wheel -o /wheels

FROM python:3.11-slim
RUN useradd -m sfw
USER sfw
WORKDIR /app

# 复制构建产物与依赖
COPY --from=builder /wheels /wheels
COPY --from=builder /root/.local /home/sfw/.local
ENV PATH=/home/sfw/.local/bin:$PATH
RUN pip install --user /wheels/*.whl

# 挂载点：数据与配置从宿主机/卷注入，不进镜像（生产配置管理）
COPY config ./config
COPY data ./data

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD sfw version >/dev/null || exit 1

ENTRYPOINT ["sfw"]
CMD ["analyze", "data/pingan_000001.yaml"]
