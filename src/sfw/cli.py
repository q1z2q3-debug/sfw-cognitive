"""命令行入口：``sfw``。

子命令：
    analyze      对指定数据文件运行完整分析并输出报告（可 --internalize 内化判断）
    fetch        通过 akshare 拉取实时行情并生成数据文件（可选依赖）
    internalize  大模型内化：System 1 模式库 状态/学习/反馈
    selfcheck    运行确定性自检（单元测试 + 引擎演示）
    version      打印版本
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from . import __version__
from .config import load_config
from .data import load_stock_data
from .internalize import DualSystemInternalizer, analysis_action_signature, state_to_index, to_trits
from .report import generate_report, run_analysis

logger = logging.getLogger("sfw")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _cmd_fetch(args: argparse.Namespace) -> int:
    """拉取实时行情并生成数据文件。"""
    try:
        import yaml

        from .market_data import fetch_to_data_dict, require_deps
    except ImportError as exc:
        print(f"错误：缺少依赖（{exc}）。请执行 pip install 'sfw-cognitive[data]'", file=sys.stderr)
        return 2
    try:
        require_deps()
    except RuntimeError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2
    try:
        data = fetch_to_data_dict(symbol=args.symbol, name=args.name, days=args.days)
    except Exception as exc:  # noqa: BLE001
        print(f"错误：取数失败（{exc}）。请确认网络与 akshare 可用。", file=sys.stderr)
        return 1

    out = Path(args.out or f"data/{args.symbol}.yaml")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"数据文件已生成：{out.resolve()}")
    print("提示：自动拉取的为行情/技术面字段，PE/PB/股息率/三表需人工补充后再分析。")
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    try:
        cfg = load_config(args.config)
        data = load_stock_data(args.data)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("输入错误：%s", exc)
        print(f"错误：{exc}", file=sys.stderr)
        return 2

    res = run_analysis(data, cfg, market_vol=args.market_vol)
    report_md = generate_report(res)

    # 大模型内化：把本次固件判断写入 System 1 模式库
    if args.internalize:
        wtp = (res.bank["weighted_target_price"] if res.bank else res.valuation["weighted_target_price"])
        action = analysis_action_signature(res.engine.get_state_vector(), res.engine_report, wtp)
        inter = DualSystemInternalizer(library_path=args.library, min_confidence=0.6)
        inter.load()
        inter.library.learn(state_to_index(res.engine.get_state_vector()),
                            to_trits(res.engine.get_state_vector()),
                            data.meta.get("stock", "") or str(data.meta.get("code", "")), action)
        path = inter.save()
        print(f"已内化本次判断到模式库：{path}")
        print(f"  {action}")

    if args.out:
        out = Path(args.out)
        out.write_text(report_md, encoding="utf-8")
        print(f"报告已写入：{out.resolve()}")
    else:
        print(report_md)

    if args.json:
        # 结构化结果（供下游消费）
        payload = {
            "meta": data.meta,
            "market": data.market,
            "scores": {d: res.scoring.results[d].raw_score for d in res.scoring.results},
            "engine": res.engine_report,
            "valuation": {
                "weighted_target_price": res.valuation["weighted_target_price"],
                "breakdown": res.valuation["breakdown"],
            },
            "backtest": {
                "annual_return": res.backtest.annual_return,
                "sharpe": res.backtest.sharpe_ratio,
                "max_drawdown": res.backtest.max_drawdown,
            },
        }
        if res.bank:
            payload["bank"] = {
                "weighted_target_price": res.bank["weighted_target_price"],
                "breakdown": res.bank["breakdown"],
                "net_profit_2026e": res.bank["income"].net_profit[0],
                "eps_2026e": res.bank["income"].eps[0],
            }
        json_path = args.json if args.json != "" else (args.out or "sfw_result.json")
        Path(json_path).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(f"结构化结果已写入：{Path(json_path).resolve()}")

    return 0


def _cmd_selfcheck(args: argparse.Namespace) -> int:
    import unittest
    suite = unittest.defaultTestLoader.discover("tests")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


def _cmd_internalize(args: argparse.Namespace) -> int:
    """大模型内化：S1 模式库的状态 / 学习 / 反馈。"""
    inter = DualSystemInternalizer(library_path=args.library)
    inter.load()
    op = args.subcommand

    if op == "status":
        g = inter.graduation()
        print(f"阶段：{g['stage']}  (school → internalize → graduate)")
        print(f"  模式数：{g['patterns']}")
        print(f"  覆盖状态：{g['distinct_states']} / {19683}（{g['state_coverage']:.4%}）")
        print(f"  累计使用：{g['total_uses']}  平均置信：{g['avg_confidence']:.3f}")
        print(f"  LLM 依赖度：{g['llm_dependency']}")
        return 0

    if op == "learn":
        state9 = [float(x) for x in args.state.split(",")]
        learned = inter.library.learn(state_to_index(state9), to_trits(state9), args.input, args.action)
        inter.save()
        print(f"已内化：idx={learned.state_index} hash={learned.input_hash} → {args.action}")
        return 0

    if op == "feedback":
        rec = inter.feedback(args.hash, success=args.success)
        if rec is None:
            print(f"未找到 hash={args.hash} 的模式", file=sys.stderr)
            return 1
        inter.save()
        print(f"反馈已回灌：hash={args.hash} success={args.success} → 成功率 {rec.success_rate:.3f}")
        return 0

    if op == "reset":
        inter.library.records = []
        inter.save()
        print(f"模式库已清空：{args.library}")
        return 0

    print(f"未知子命令：{op}", file=sys.stderr)
    return 2


def _cmd_version(_args: argparse.Namespace) -> int:
    print(f"SFW-10.0 cognitive engine v{__version__}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sfw",
        description="股票分析认知固件 SFW-10.0（生产级实现）",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="输出调试日志")
    sub = parser.add_subparsers(dest="command", required=True)

    pa = sub.add_parser("analyze", help="对数据文件运行完整分析")
    pa.add_argument("data", help="数据文件路径（YAML）")
    pa.add_argument("--config", default=None, help="配置文件路径（可选）")
    pa.add_argument("--out", default=None, help="Markdown 报告输出路径")
    pa.add_argument("--json", nargs="?", const="", default=None, help="同时输出结构化 JSON（可指定路径）")
    pa.add_argument("--market-vol", type=float, default=0.15, help="年化波动率（用于温度选择）")
    pa.add_argument("--internalize", action="store_true",
                    help="分析后将本次固件判断内化写入 System 1 模式库")
    pa.add_argument("--library", default="output/sfw_patterns.json",
                    help="内化模式库路径（默认 output/sfw_patterns.json）")
    pa.set_defaults(func=_cmd_analyze)

    pf = sub.add_parser("fetch", help="通过 akshare 拉取实时行情并生成数据文件")
    pf.add_argument("symbol", help="证券代码，如 sz000001")
    pf.add_argument("--name", default="", help="标的名称（写入 meta.stock）")
    pf.add_argument("--days", type=int, default=750, help="行情天数")
    pf.add_argument("--out", default=None, help="数据文件输出路径（默认 data/<symbol>.yaml）")
    pf.set_defaults(func=_cmd_fetch)

    pi = sub.add_parser("internalize", help="大模型内化：System 1 模式库状态/学习/反馈")
    pi.add_argument("--library", default="output/sfw_patterns.json", help="模式库路径")
    pis = pi.add_subparsers(dest="subcommand", required=True)
    pis.add_parser("status", help="查看内化阶段与覆盖率").set_defaults(subcommand="status")
    pl = pis.add_parser("learn", help="手动内化一条判断")
    pl.add_argument("--state", required=True, help="九维状态，逗号分隔，如 1,0,-1,...")
    pl.add_argument("--input", required=True, help="触发输入文本")
    pl.add_argument("--action", required=True, help="判断签名/动作")
    pl.set_defaults(subcommand="learn")
    pb = pis.add_parser("feedback", help="回灌成功/失败以强化模式")
    pb.add_argument("--hash", required=True, help="模式 input_hash")
    pb.add_argument("--success", type=bool, required=True, help="本次是否成功")
    pb.set_defaults(subcommand="feedback")
    pis.add_parser("reset", help="清空模式库").set_defaults(subcommand="reset")
    pi.set_defaults(func=_cmd_internalize)

    ps = sub.add_parser("selfcheck", help="运行单元测试自检")
    ps.set_defaults(func=_cmd_selfcheck)

    pv = sub.add_parser("version", help="打印版本")
    pv.set_defaults(func=_cmd_version)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
