"""Replay fixed targets under explicitly hypothetical past-volume budgets."""

import json
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .capacity import trailing_capacity
from .execution import settle_portfolio
from .inference import by_adjust
from .market_benchmark import (
    BOOKS,
    attach_exposures,
    build_orders,
    digest,
    scenario_marks,
    summarize,
)


def capacity_plan(config: Path) -> tuple[dict, dict, Path]:
    plan = tomllib.loads(config.read_text(encoding="utf-8"))
    root = config.resolve().parent.parent
    paths = {k: (root / v).resolve() for k, v in plan["paths"].items()}
    if any(not path.is_relative_to(root) for path in paths.values()) or set(paths) != set(
        plan["input_hashes"]
    ) | {"output", "methodology"}:
        raise ValueError("Explicit workspace paths and frozen data inputs required")
    for name, expected in plan["input_hashes"].items():
        if digest(paths[name]) != expected:
            raise ValueError(f"Capacity input changed: {name}")
        if paths["output"].is_relative_to(paths[name].parent):
            raise ValueError("Keep output outside source directories")
    study = plan["study"]
    if (
        study["capacity_policy"] != "staged"
        or not study["keep_original_signals_contracts_targets"]
        or any(
            study[k]
            for k in (
                "independent_new_sample",
                "formal_p_value_computed",
                "actual_execution_certified",
            )
        )
        or not 0 < study["max_margin_fraction"] <= 1
        or study["initial_cash"] <= 0
        or study["cost_bps"] < 0
        or not 0 < study["margin_rate"] <= 1
        or study["volume_window_sessions"] < 2
        or not study["past_volume_fractions"]
        or len(set(study["past_volume_fractions"])) != len(study["past_volume_fractions"])
        or any(not 0 < f <= 1 for f in study["past_volume_fractions"])
    ):
        raise ValueError("Invalid capacity plan or claim boundaries")
    return plan, paths, root / "reports/protocols" / config.stem / "receipt.json"


def freeze_capacity(config: Path) -> Path:
    plan, paths, receipt = capacity_plan(config)
    payload = {
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": digest(config),
        "methodology_sha256": digest(paths["methodology"]),
        "inputs": plan["input_hashes"],
        "historical_results_previously_viewed": True,
        "independent_preregistration": False,
    }
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with receipt.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    return receipt


def compare_original(result, source: Path):
    for name in ("daily", "trades", "holdings", "rejected"):
        old = pd.read_csv(source / f"{name}.csv", parse_dates=["date"])
        actual = getattr(result, name)
        if name == "daily":
            actual = actual.reset_index()
        pd.testing.assert_frame_equal(
            actual.reset_index(drop=True),
            old.reset_index(drop=True),
            check_dtype=False,
            atol=1e-7,
            rtol=1e-12,
        )


def run_capacity_study(config: Path) -> Path:
    plan, paths, receipt = capacity_plan(config)
    frozen = json.loads(receipt.read_text(encoding="utf-8"))
    if frozen["config_sha256"] != digest(config) or frozen["methodology_sha256"] != digest(
        paths["methodology"]
    ):
        raise ValueError("Capacity protocol changed after its freeze")
    started = datetime.now(timezone.utc)
    if started <= datetime.fromisoformat(frozen["frozen_utc"]):
        raise ValueError("Freeze must precede computation")
    source = paths["source_run"].parent
    original = json.loads(paths["source_run"].read_text(encoding="utf-8"))
    for filename, expected in original["output_sha256"].items():
        if digest(source / filename.replace("\\", "/")) != expected:
            raise ValueError(f"Saved original output changed: {filename}")
    curves_path = Path(original["inputs"]["curves"]["path"])
    if digest(curves_path) != original["inputs"]["curves"]["sha256"]:
        raise ValueError("Original feature input changed")
    bars = pd.read_csv(paths["bars"], parse_dates=["date"])
    calendar = pd.read_csv(paths["calendar"], parse_dates=["date"])
    curves = pd.read_csv(curves_path, parse_dates=["date"])
    study = plan["study"]
    prior_study = original["protocol"]["study"]
    prior_execution = original["protocol"]["execution"]
    if (
        study["initial_cash"] != prior_study["initial_cash"]
        or study["cost_bps"] != prior_execution["primary_cost_bps"]
        or study["margin_rate"] != prior_execution["primary_margin_rate"]
        or study["max_margin_fraction"] != prior_execution["max_margin_fraction"]
        or any(
            plan["input_hashes"][name] != original["inputs"][name]["sha256"]
            for name in ("bars", "calendar")
        )
    ):
        raise ValueError("Keep original capital, costs, margin and paired data inputs unchanged")
    orders, targets, clocks = build_orders(
        curves, calendar, prior_study, original["protocol"]["multipliers"]
    )
    for book in BOOKS:
        old = pd.read_csv(source / f"orders_{book}.csv", parse_dates=["date", "decision_at"])
        pd.testing.assert_frame_equal(orders[book], old, check_dtype=False)
    marks = pd.read_csv(source / "market_marks.csv", parse_dates=["date"])
    output = paths["output"] / started.strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    targets.to_csv(output / "target_audit.csv", index=False, encoding="utf-8-sig")
    clocks.to_csv(output / "assumed_calendar.csv", index=False, encoding="utf-8-sig")
    for book, frame in orders.items():
        frame.to_csv(output / f"orders_{book}.csv", index=False, encoding="utf-8-sig")
    scenarios = [
        ("unlimited", None),
        *[(f"past_volume_{f:g}", f) for f in study["past_volume_fractions"]],
    ]
    rows = []
    for scenario, fraction in scenarios:
        inputs = scenario_marks(marks, study["cost_bps"], study["margin_rate"], "assumed_full_open")
        if fraction is not None:
            budgets = trailing_capacity(
                bars.loc[bars.contract.isin(marks.contract.unique())],
                calendar,
                fraction,
                study["volume_window_sessions"],
            )
            needed = marks[["date", "contract"]].merge(
                budgets, on=["date", "contract"], validate="one_to_one"
            )
            needed.to_csv(output / f"{scenario}_budgets.csv", index=False, encoding="utf-8-sig")
            inputs = inputs.merge(
                needed[["date", "contract", "max_trade_lots", "capacity_available_at"]],
                on=["date", "contract"],
                validate="one_to_one",
            )
        for book in BOOKS:
            print(f"Capacity replay: {scenario}/{book}", flush=True)
            folder = output / scenario / book
            folder.mkdir(parents=True)
            try:
                result = settle_portfolio(
                    inputs,
                    clocks,
                    orders[book],
                    study["initial_cash"],
                    study["max_margin_fraction"],
                    capacity_policy="atomic" if fraction is None else "staged",
                )
                attach_exposures(result, original["protocol"]["multipliers"])
                if fraction is None:
                    compare_original(result, source / "cost5_margin0.2_assumed_full_open" / book)
                for name in ("daily", "trades", "holdings", "rejected", "capacity_events"):
                    getattr(result, name).to_csv(
                        folder / f"{name}.csv", index=name == "daily", encoding="utf-8-sig"
                    )
                events = result.capacity_events
                incomplete = events.remaining_lots.ne(0) if len(events) else pd.Series(dtype=bool)
                rows.append(
                    {
                        "scenario": scenario,
                        "book": book,
                        "status": "completed",
                        **summarize(result, study["initial_cash"]),
                        "capacity_attempts": len(events),
                        "incomplete_attempts": int(incomplete.sum()),
                        "days_with_incomplete_attempts": int(
                            events.loc[incomplete, "date"].nunique()
                        )
                        if len(events)
                        else 0,
                        "terminal_held_contracts": int(result.daily.contracts_held.iloc[-1]),
                    }
                )
            except ValueError as exc:
                if fraction is None:
                    raise
                (folder / "failure.json").write_text(
                    json.dumps({"error": str(exc)}, ensure_ascii=False), encoding="utf-8"
                )
                rows.append(
                    {"scenario": scenario, "book": book, "status": "failed", "error": str(exc)}
                )
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "summary.csv", index=False, encoding="utf-8-sig")
    previous = pd.read_csv(source / "registered_exploration.csv")
    if len(previous) != 16:
        raise ValueError("Expected the complete 16-item exploration record")
    registry = pd.concat(
        [
            previous,
            pd.DataFrame(
                [
                    {
                        "analysis": "past_volume_execution_capacity",
                        "signal": "unchanged_saved_targets",
                        "hac_p": np.nan,
                        "claim_eligible": False,
                        "reason": "Descriptive capacity/deferral study; no calibrated p-value",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    registry["family_17_by_q"] = by_adjust(registry.hac_p.to_numpy())
    registry["family_17_size"] = 17
    registry.to_csv(output / "registered_exploration.csv", index=False, encoding="utf-8-sig")
    (output / "registered_exploration_source.csv").write_bytes(
        (source / "registered_exploration.csv").read_bytes()
    )
    run = {
        "version": __version__,
        "created_utc": started.isoformat(),
        "protocol": plan,
        "freeze_receipt": frozen,
        "protocol_sha256": digest(config),
        "source_run_sha256": digest(paths["source_run"]),
        "original_targets_unchanged": True,
        "original_primary_ledgers_reproduced": True,
        "known_exploration_family": 17,
        "independent_new_sample": False,
        "actual_execution_certified": False,
        "formal_p_value_computed": False,
        "source_sha256": {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")},
        "output_sha256": {
            p.relative_to(output).as_posix(): digest(p) for p in output.rglob("*.csv")
        },
    }
    (output / "run.json").write_text(
        json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>容量与延迟执行</title><style>body{font:16px/1.8 system-ui,'Microsoft YaHei';max-width:1300px;margin:40px auto;padding:24px;background:#f3f6f9;color:#183247}section{background:white;padding:24px;border-radius:12px;margin:20px 0}table{border-collapse:collapse;font-size:13px;white-space:nowrap}th,td{padding:9px;border-bottom:1px solid #ddd}.scroll{overflow:auto}.notice{background:#fff0d9;padding:20px}</style><h1>过去成交量约束下的执行</h1><p class="notice">前20交易日量预算，1%与5%两种固定压力参数。未完成目标分日重试，旧约平完才开新约。仍假设预算内可按日线开盘价成交；结果不是实际成交认证。终端残留时回报包含未清头寸盯市。</p>"""
    columns = [
        c
        for c in [
            "scenario",
            "book",
            "status",
            "total_return",
            "max_drawdown",
            "fees",
            "trade_legs",
            "incomplete_attempts",
            "days_with_incomplete_attempts",
            "terminal_held_contracts",
            "terminal_unliquidated",
            "error",
        ]
        if c in summary
    ]
    page += f'<section><h2>完整固定情景</h2><div class="scroll">{summary[columns].to_html(index=False, escape=True, float_format=lambda x: f"{x:.6f}")}</div></section>'
    page += '<section><p>unlimited为原完整成交假设；past_volume_0.01/0.05为过去量1%/5%预算。total_return与max_drawdown是小数；incomplete_attempts为未达目标的尝试次数，包含重复重试。</p><a href="summary.csv">完整汇总</a> · <a href="run.json">输入和方法记录</a><p>原目标和三个原主情景资金账已复现。新问题记为第17项探索；历史已看过，未计算正式显著性或确认新因子。</p></section></html>'
    (output / "index.html").write_text(page, encoding="utf-8")
    return output
