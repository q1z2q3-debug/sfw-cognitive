# SFW-10.0 · 股票分析认知固件（生产级实现）

`SFW-10.0 · 回测闭环 + 概率校准` 的**生产化实现**：把原有的单文件原型（`sfw7_core.py`
等）重构为**可安装、可配置、可测试、数据驱动、带 CLI** 的 Python 包。

> 本包输出含启发式表达层（九维概率云 / 卦象 / 沙盘概率），**不构成投资建议，
> 不预测涨跌，不承诺收益**。具体决策与风险由个人承担。

**发布与存档**
- PyPI：`pip install sfw-cognitive` → <https://pypi.org/project/sfw-cognitive/>
- 源码（GitHub，开源）：<https://github.com/q1z2q3-debug/sfw-cognitive>
- 存档（Zenodo · DOI 10.5281/zenodo.23040435）：<https://zenodo.org/record/23040435>
- 认知固件（即用型提示词，全英文 · SFW-12.0，SFW-11.5 的认知升级版）：
  [`firmware/SFW-12.0-cognition-augmented-en.md`](firmware/SFW-12.0-cognition-augmented-en.md)
  · Zenodo 独立存档 DOI 10.5281/zenodo.23040599 · https://zenodo.org/record/23040599

---

## 1. 安装

要求 Python ≥ 3.10，依赖 `numpy` 与 `PyYAML`。

```bash
cd sfw-10.0-prod
pip install -e .            # 常规安装（含 CLI `sfw`）
pip install -e ".[dev]"     # 开发安装（含 pytest）
```

## 2. 用法

### 分析一个标的

```bash
sfw analyze data/pingan_000001.yaml                # 打印 Markdown 报告
sfw analyze data/pingan_000001.yaml --out report.md --json result.json
sfw analyze data/pingan_000001.yaml --config config/my.yaml --market-vol 0.25
```

> `data/pingan_000001.yaml` 含 `valuation.bank` 段，`analyze` 会额外输出
> **银行专项三表模型（银行口径）与专项估值**。

### 拉取实时行情（可选依赖）

```bash
pip install "sfw-cognitive[data]"     # 安装 pandas + akshare
sfw fetch sz000001 --name 平安银行 --out data/auto_sz000001.yaml
```

自动拉取行情与九维技术面指标；PE/PB/股息率/三表等需人工补充后再分析。
无 akshare 时该命令会明确报错，不影响其余功能。

### 自检（单元测试）

```bash
sfw selfcheck          # 等价于：python -m pytest tests/
```

### 版本

```bash
sfw version
```

## 3. 结构

```
sfw-10.0-prod/
├── pyproject.toml          # 打包元数据 + CLI 入口（sfw）
├── config/default.yaml     # 引擎/评分/估值/回测 算法参数（唯一可调参数库）
├── .github/workflows/      # CI (ci.yml) + 发布 (release.yml)
├── Makefile                # install/test/lint/build/run/fetch
├── Dockerfile              # 多阶段构建、非 root、HEALTHCHECK
├── docker-compose.yml      # 编排（数据/配置只读注入）
├── src/sfw/
│   ├── config.py           # 配置加载与校验（深度合并 + 白名单）
│   ├── core.py             # 认知引擎（概率云/分形/裂变/拓扑/涌现/存在）
│   ├── scoring.py          # 九维评分卡 + 银行专项评分
│   ├── valuation.py        # 通用三表预测 + 多方法估值 + 敏感性
│   ├── bank_model.py       # 银行专用三表模板 + 银行敏感性 + PB-ROE/DDM 专项估值
│   ├── market_data.py      # 实时行情接入（akshare 适配，可选依赖、优雅降级）
│   ├── backtest.py         # 回测指标 + 组合相关性
│   ├── data.py             # 输入数据模式、加载与校验
│   ├── report.py           # SFW-10.0 报告渲染与流水线编排
│   └── cli.py              # 命令行入口（analyze / fetch / selfcheck / version）
├── data/pingan_000001.yaml # 示例标的输入数据（含银行三表输入）
└── tests/                  # 单元测试（核心/评分/估值/银行模型/行情/数据/流水线）
```

## 4. 生产化要点

- **数据驱动**：业务数值全部来自 `data/*.yaml`，引擎不做任何硬编码标的；
  新增标的只需新增数据文件。
- **配置集中**：阈值/权重/温度等算法参数集中在 `config/default.yaml`，
  可通过 `--config` 覆盖，运行时做类型与范围校验。
- **银行专用三表**：`bank_model.py` 提供银行口径三表（利息净收入、手续费、
  其他非息、信用减值/拨备）与专项估值（PB-ROE、历史PB分位、DDM）及
  敏感性（LPR/不良率/存款成本/信贷增速），由 `valuation.bank` 输入驱动。
- **实时行情接入**：`market_data.py` 封装 akshare 取数与九维指标计算，
  缺依赖/网络时明确降级为"数据不可得"，不编造数值。
- **缺失数据显式处理**：缺失指标按中性(0)处理，并在报告中标注"数据不可得"，
  不静默失败、不编造数值。
- **可测试**：`tests/` 覆盖核心、评分、估值、银行模型、行情、数据校验与
  完整流水线（当前 49 个用例）。
- **CI/CD**：GitHub Actions（多 Python 版本 lint + 测试 + 构建）、Makefile、
  Dockerfile（多阶段、非 root、HEALTHCHECK）、docker-compose。
- **合规**：报告固定输出免责声明；回测默认使用确定性随机模拟数据并明确标注。

## 5. 新增一个标的

1. 复制 `data/pingan_000001.yaml` 为 `data/<标的>.yaml`；
2. 填写 `meta`（名称/代码/日期/反方/证伪）、`market`（价格/估值）、
   `scoring`（九维指标，缺失项省略即可）、`valuation`（三表与估值输入）；
3. （可选）银行股：在 `valuation.bank` 填入生息资产/净息差/不良率/拨备/
   资本/股息等输入，`analyze` 会额外输出银行专项三表与专项估值；
4. 运行 `sfw analyze data/<标的>.yaml --out <标的>.md`。

## 6. 说明与局限

- 九维概率云、卦象、沙盘情境概率均为**启发式表达层**，非外部统计或期权隐含
  数据，报告已显式标注主观性。
- 回测为**确定性模拟数据**，仅用于演示与自检；接入真实行情后方可作决策参考。
- 组合相关性返回的是行业经验值估算，非实测。
