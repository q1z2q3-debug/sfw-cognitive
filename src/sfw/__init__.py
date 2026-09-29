"""SFW-10.0 · 股票分析认知固件（生产级实现）。

模块划分：
- config.py     配置加载与校验
- core.py       认知引擎（概率云 / 分形 / 裂变 / 拓扑 / 涌现 / 存在）
- scoring.py    九维评分卡 + 银行专项评分
- valuation.py  三表预测 / 多方法估值 / 敏感性
- backtest.py   回测指标 / 组合相关性
- data.py       输入数据模式、加载与校验
- report.py     SFW-10.0 报告渲染
- cli.py        命令行入口

使用方式：安装后执行 ``sfw analyze data/pingan_000001.yaml``，
详见 README.md。
"""

__version__ = "10.1.0"

from .config import SFWConfig, load_config
from .core import (  # noqa: F401
    ProbabilityCloud,
    SFWEngine,
    dynamic_temperature,
    softmax_cloud,
)

__all__ = ["SFWConfig", "__version__", "load_config"]
