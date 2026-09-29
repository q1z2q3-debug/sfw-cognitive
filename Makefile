# SFW-10.0 常用开发命令
.PHONY: install dev test lint typecheck build run fetch clean

PYTHON ?= python

install:            ## 常规安装
	$(PYTHON) -m pip install -e .

dev:                ## 开发安装（含测试/实时行情依赖）
	$(PYTHON) -m pip install -e ".[dev,data]"

test:               ## 运行单元测试
	$(PYTHON) -m pytest tests/

lint:               ## 代码检查
	ruff check src/ tests/

typecheck:          ## 类型检查（提示性）
	mypy src/sfw --ignore-missing-imports || true

build:              ## 构建 wheel + sdist
	$(PYTHON) -m build

run:                ## 运行平安银行示例分析
	sfw analyze data/pingan_000001.yaml --out report.md --json result.json

fetch:              ## 拉取实时行情并生成数据文件（需 [data] 依赖）
	sfw fetch sz000001 --name 平安银行 --out data/auto_sz000001.yaml

clean:              ## 清理构建产物
	rm -rf build dist *.egg-info .pytest_cache __pycache__ src/*/__pycache__

help:
	@grep -E '^[a-zA-Z_-]+:.*##' Makefile | awk 'BEGIN{FS=":.*## "}{printf "  %-12s %s\n", $$1, $$2}'
