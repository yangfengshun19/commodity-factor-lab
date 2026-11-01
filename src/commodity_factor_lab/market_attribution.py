"""Accounting explanations of saved market books; creates no new strategy or test."""

import hashlib
import json
import tomllib
from datetime import datetime, timezone
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, expected, label, atol=1e-6):
    if not np.allclose(actual, expected, atol=atol, rtol=1e-11, equal_nan=False):
        raise ValueError(f"Attribution reconciliation failed: {label}")


def lifecycle(before: set, after: set, reasons: set) -> str:
    if not reasons:
        return "no_trade"
    if "margin_liquidation" in reasons:
        return "margin_liquidation"
    if "terminal_liquidation" in reasons:
        return "terminal_liquidation"
    if before and after and before != after:
        return "roll_including_any_resize"
    if not before and after:
        return "entry"
    if before and not after:
        return "exit"
    return "same_contract_resize_or_reverse"


def attribute_book(
    bars: pd.DataFrame,
    daily: pd.DataFrame,
    trades: pd.DataFrame,
    holdings: pd.DataFrame,
    codes: list[str],
    multipliers: dict,
    cost_bps: float,
    book: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Previous and current contract positions are valued separately on roll days."""
    marks, days, fills, positions = (f.copy() for f in (bars, daily, trades, holdings))
    for frame in (marks, days, fills, positions):
        frame["date"] = pd.to_datetime(frame.date).dt.normalize()
    if (
        marks.duplicated(["date", "contract"]).any()
        or not days.date.is_unique
        or not days.date.is_monotonic_increasing
        or positions.duplicated(["date", "contract"]).any()
        or fills.duplicated(["date", "contract"]).any()
    ):
        raise ValueError("Ambiguous dated accounting keys")
    if not fills.date.isin(days.date).all() or not positions.date.isin(days.date).all():
        raise ValueError("Saved transactions/positions outside accounting calendar")
    quotes = {(r.date, r.contract): r for r in marks.itertuples()}
    by_day = {d: g for d, g in positions.groupby("date")}
    by_fill = {d: g for d, g in fills.groupby("date")}
    before, last = {}, {}
    rows, checks = [], []
    for record in days.itertuples():
        day = record.date
        current_frame = by_day.get(day, positions.iloc[:0])
        current = dict(zip(current_frame.contract, current_frame.lots, strict=True))
        if current_frame.code.duplicated().any():
            raise ValueError("Each product must have at most one saved closing contract")
        today = by_fill.get(day, fills.iloc[:0])
        implied = before.copy()
        for t in today.itertuples():
            if implied.get(t.contract, 0) != t.old_lots or t.delta_lots != t.new_lots - t.old_lots:
                raise ValueError("Trade quantities disagree with previous holdings")
            if t.new_lots:
                implied[t.contract] = t.new_lots
            else:
                implied.pop(t.contract, None)
        if implied != current:
            raise ValueError("Trades fail to explain saved closing holdings")
        overnight, intraday = dict.fromkeys(codes, 0.0), dict.fromkeys(codes, 0.0)
        before_names, after_names = {c: set() for c in codes}, {c: set() for c in codes}
        notionals = dict.fromkeys(codes, 0.0)
        for contract, lots in before.items():
            raw = quotes[(day, contract)]
            if not np.isfinite([raw.open, raw.settle]).all():
                raise ValueError("Missing required held-contract price")
            overnight[raw.code] += lots * multipliers[raw.code] * (raw.open - last[contract])
            before_names[raw.code].add(contract)
        for contract, lots in current.items():
            raw = quotes[(day, contract)]
            if not np.isfinite([raw.open, raw.settle]).all():
                raise ValueError("Missing required current-contract price")
            saved = current_frame.loc[current_frame.contract.eq(contract)].iloc[0]
            close(saved.settle, raw.settle, "saved settlement")
            close(saved.multiplier, multipliers[raw.code], "saved multiplier")
            intraday[raw.code] += lots * multipliers[raw.code] * (raw.settle - raw.open)
            notionals[raw.code] = lots * raw.settle * multipliers[raw.code]
            last[contract] = raw.settle
            after_names[raw.code].add(contract)
        for code in codes:
            events = today.loc[today.code.eq(code)]
            opened_fee = closed_fee = 0.0
            for t in events.itertuples():
                raw = quotes[(day, t.contract)]
                close(t.fill_price, raw.open, "saved opening fill")
                unit = abs(raw.open) * multipliers[code] * cost_bps / 10000
                close(t.fee, (t.opened_lots + t.closed_lots) * unit, "split fee")
                opened_fee += t.opened_lots * unit
                closed_fee += t.closed_lots * unit
            fees = float(events.fee.sum())
            gross = overnight[code] + intraday[code]
            rows.append(
                {
                    "book": book,
                    "date": day,
                    "code": code,
                    "overnight_pnl": overnight[code],
                    "intraday_pnl": intraday[code],
                    "gross_pnl": gross,
                    "fees": fees,
                    "opened_fee": opened_fee,
                    "closed_fee": closed_fee,
                    "net_pnl": gross - fees,
                    "signed_notional": notionals[code],
                    "gross_notional": abs(notionals[code]),
                    "trade_legs": len(events),
                    "traded_notional": float(events.notional.sum()),
                    "event": lifecycle(before_names[code], after_names[code], set(events.reason)),
                }
            )
        parts = rows[-len(codes) :]
        residuals = {
            "overnight_difference": sum(r["overnight_pnl"] for r in parts) - record.overnight_pnl,
            "intraday_difference": sum(r["intraday_pnl"] for r in parts) - record.intraday_pnl,
            "fee_difference": sum(r["fees"] for r in parts) - record.fees,
            "net_cash_difference": sum(r["net_pnl"] for r in parts)
            - (record.equity - record.previous_equity),
            "gross_exposure_difference": sum(r["gross_notional"] for r in parts)
            - record.gross_notional,
            "net_exposure_difference": sum(r["signed_notional"] for r in parts)
            - record.net_notional,
        }
        for label, difference in residuals.items():
            close(difference, 0.0, label)
        checks.append({"book": book, "date": day, **residuals})
        before = current
    return pd.DataFrame(rows), pd.DataFrame(checks)


def risk_decomposition(parts: pd.DataFrame, cash: float, annualization: int = 252):
    matrix = parts.pivot(index="date", columns="code", values="net_pnl") / cash
    if matrix.isna().any().any() or len(matrix) < 2:
        raise ValueError("Risk accounting requires the complete dated grid")
    values = matrix.to_numpy()
    centered = values - values.mean(axis=0)
    total = values.sum(axis=1)
    total_centered = total - total.mean()
    contributions = centered.T @ total_centered / (len(matrix) - 1)
    variance = float(total.var(ddof=1))
    close(contributions.sum(), variance, "variance contributions", atol=1e-16)
    table = pd.DataFrame(
        {
            "code": matrix.columns,
            "variance_contribution": contributions,
            "variance_share": contributions / variance if variance > 0 else np.nan,
        }
    )
    gross_daily = parts.groupby("date").gross_pnl.sum() / cash
    exposure = parts.groupby("date")[["signed_notional", "gross_notional"]].sum()
    summary = {
        "cash_normalized_net_volatility": float(np.sqrt(variance * annualization)),
        "cash_normalized_gross_volatility": float(gross_daily.std(ddof=1) * np.sqrt(annualization)),
        "average_gross_exposure_on_initial_cash": float(exposure.gross_notional.mean() / cash),
        "average_net_exposure_on_initial_cash": float(exposure.signed_notional.mean() / cash),
        "average_absolute_net_exposure_on_initial_cash": float(
            exposure.signed_notional.abs().mean() / cash
        ),
        "daily_cash_variance": variance,
    }
    return table, summary


def drawdown_attribution(parts, daily, initial_cash, initial_date):
    wealth = np.r_[initial_cash, daily.equity.to_numpy()]
    drawdown = wealth / np.maximum.accumulate(wealth) - 1
    trough = int(np.argmin(drawdown))
    peak = int(np.argmax(wealth[: trough + 1]))
    dates = [pd.Timestamp(initial_date), *pd.to_datetime(daily.date)]
    selected = parts.loc[parts.date.gt(dates[peak]) & parts.date.le(dates[trough])]
    components = parts[["code"]].drop_duplicates().set_index("code")
    components["net_pnl_in_drawdown"] = (
        selected.groupby("code").net_pnl.sum().reindex(components.index, fill_value=0)
    )
    components["drawdown_contribution"] = components.net_pnl_in_drawdown / wealth[peak]
    close(components.drawdown_contribution.sum(), drawdown[trough], "drawdown window", atol=1e-12)
    return components.reset_index(), {
        "peak_date": str(dates[peak].date()),
        "trough_date": str(dates[trough].date()),
        "peak_equity": float(wealth[peak]),
        "trough_equity": float(wealth[trough]),
        "max_drawdown": float(drawdown[trough]),
        "window_sessions": selected.date.nunique(),
    }


def target_diagnostics(targets: pd.DataFrame, curves: pd.DataFrame):
    records, direction = [], []
    features = curves.copy()
    features["date"] = pd.to_datetime(features.date)
    for day, all_books in targets.groupby("date", sort=True):
        momentum = all_books.loc[all_books.book.eq("momentum")].set_index("code").sort_index()
        blend = all_books.loc[all_books.book.eq("momentum_carry")].set_index("code").sort_index()
        if not momentum.index.equals(blend.index) or not momentum.contract.equals(blend.contract):
            raise ValueError("Target comparison must keep the same products and contracts")
        signal_day = pd.Timestamp(momentum.signal_date.iloc[0])
        f = features.loc[features.date.eq(signal_day)].set_index("code").loc[momentum.index]
        m, c = f.momentum_126d.rank(method="average"), f.carry.rank(method="average")
        m, c = m - m.mean(), c - c.mean()
        mixed_norm = float(((m + c) / 2).abs().sum())
        base_norm = float(m.abs().sum())
        for book, group in all_books.groupby("book"):
            w = group.weight.to_numpy()
            gross, squared = float(abs(w).sum()), float((w * w).sum())
            records.append(
                {
                    "date": day,
                    "book": book,
                    "max_absolute_weight": float(abs(w).max()),
                    "sum_squared_weights": squared,
                    "effective_absolute_names": gross * gross / squared if squared > 0 else np.nan,
                    "target_gross_weight": gross,
                    "momentum_score_abs_sum": base_norm,
                    "mixed_score_abs_sum": mixed_norm,
                    "score_normalization_ratio": base_norm / mixed_norm if mixed_norm else np.nan,
                }
            )
        for code in momentum.index:
            wm, wb = momentum.loc[code, "weight"], blend.loc[code, "weight"]
            if wm == 0 and wb == 0:
                state = "both_zero"
            elif wm == 0:
                state = "blend_only_nonzero"
            elif wb == 0:
                state = "momentum_only_nonzero"
            else:
                state = (
                    "same_nonzero_direction"
                    if np.sign(wm) == np.sign(wb)
                    else "opposite_nonzero_direction"
                )
            direction.append(
                {
                    "date": day,
                    "code": code,
                    "momentum_weight": wm,
                    "blend_weight": wb,
                    "momentum_rank_score": m.loc[code],
                    "carry_rank_score": c.loc[code],
                    "weight_difference": wb - wm,
                    "direction_state": state,
                }
            )
    return pd.DataFrame(records), pd.DataFrame(direction)


def run_market_attribution(config: Path) -> Path:
    plan = tomllib.loads(config.read_text(encoding="utf-8"))
    root = config.resolve().parent.parent
    paths = {k: (root / v).resolve() for k, v in plan["paths"].items()}
    if (
        any(not p.is_relative_to(root) for p in paths.values())
        or digest(paths["source_run"]) != plan["input_hashes"]["source_run"]
    ):
        raise ValueError("Attribution inputs must match the frozen local experiment")
    spec = plan["analysis"]
    if (
        spec["scenario"] != "cost5_margin0.2_assumed_full_open"
        or spec["books"] != ["equal_weight_long", "momentum", "momentum_carry"]
        or spec["known_exploration_family"] != 16
        or any(
            spec[k]
            for k in (
                "formal_p_value_computed",
                "new_strategy_generated",
                "new_prediction_test",
                "causal_mechanism_identified",
                "independent_new_sample",
            )
        )
    ):
        raise ValueError("Keep the existing experiment and descriptive scope")
    freeze = json.loads(
        (root / "reports/protocols/market_attribution_20261031/receipt.json").read_text(
            encoding="utf-8"
        )
    )
    if freeze["config_sha256"] != digest(config) or freeze["methodology_sha256"] != digest(
        root / "docs/market_attribution_methodology.md"
    ):
        raise ValueError("Attribution protocol or methodology changed")
    original = json.loads(paths["source_run"].read_text(encoding="utf-8"))
    previous = paths["source_run"].parent
    saved_hashes = {
        name.replace("\\", "/"): value for name, value in original["output_sha256"].items()
    }
    if len(saved_hashes) != len(original["output_sha256"]):
        raise ValueError("Ambiguous normalized output paths")
    selected = [
        "target_audit.csv",
        "assumed_calendar.csv",
        "summary.csv",
        "registered_exploration.csv",
    ]
    selected += [
        f"{spec['scenario']}/{b}/{n}.csv"
        for b in spec["books"]
        for n in ("daily", "trades", "holdings", "rejected")
    ]
    inputs = {
        "source_run": {"path": str(paths["source_run"]), "sha256": digest(paths["source_run"])}
    }
    for name in selected:
        expected = saved_hashes[name]
        if digest(previous / name) != expected:
            raise ValueError(f"Saved market result changed: {name}")
        inputs[name] = {"path": str(previous / name), "sha256": expected}
    for name in ("bars", "curves"):
        entry = original["inputs"][name]
        path = Path(entry["path"])
        if not path.is_relative_to(root) or digest(path) != entry["sha256"]:
            raise ValueError(f"Original source changed: {name}")
        inputs[name] = entry
    bars = pd.read_csv(inputs["bars"]["path"], parse_dates=["date"])
    curves = pd.read_csv(inputs["curves"]["path"], parse_dates=["date"])
    targets = pd.read_csv(previous / "target_audit.csv", parse_dates=["date", "signal_date"])
    calendar = pd.read_csv(
        previous / "assumed_calendar.csv", parse_dates=["date", "previous_signal_date"]
    )
    study = original["protocol"]["study"]
    cash = study["initial_cash"]
    rows, reconciliations, risks, variances, windows, window_parts = [], [], [], [], [], []
    for book in spec["books"]:
        directory = previous / spec["scenario"] / book
        daily, trades, holdings = (
            pd.read_csv(directory / f"{n}.csv", parse_dates=["date"])
            for n in ("daily", "trades", "holdings")
        )
        parts, checks = attribute_book(
            bars,
            daily,
            trades,
            holdings,
            study["codes"],
            original["protocol"]["multipliers"],
            5.0,
            book,
        )
        rows.append(parts)
        reconciliations.append(checks)
        variance, risk = risk_decomposition(parts, cash)
        variances.append(variance.assign(book=book))
        risks.append({"book": book, **risk})
        ddparts, window = drawdown_attribution(
            parts, daily, cash, calendar.previous_signal_date.iloc[0]
        )
        windows.append({"book": book, **window})
        window_parts.append(ddparts.assign(book=book))
    parts = pd.concat(rows, ignore_index=True)
    parts["year"] = parts.date.dt.year
    product = (
        parts.groupby(["book", "code"], sort=True)
        .agg(
            gross_pnl=("gross_pnl", "sum"),
            fees=("fees", "sum"),
            net_pnl=("net_pnl", "sum"),
            average_signed_notional=("signed_notional", "mean"),
            average_gross_notional=("gross_notional", "mean"),
            traded_notional=("traded_notional", "sum"),
            trade_legs=("trade_legs", "sum"),
        )
        .reset_index()
    )
    product["net_contribution_on_initial_cash"] = product.net_pnl / cash
    yearly = (
        parts.groupby(["book", "year", "code"])[["gross_pnl", "fees", "net_pnl"]]
        .sum()
        .reset_index()
    )
    yearly["net_contribution_on_initial_cash"] = yearly.net_pnl / cash

    def difference(frame, keys):
        left = frame.loc[frame.book.eq("momentum_carry")].set_index(keys)
        right = frame.loc[frame.book.eq("momentum")].set_index(keys)
        result = (
            left[["gross_pnl", "fees", "net_pnl"]] - right[["gross_pnl", "fees", "net_pnl"]]
        ).reset_index()
        result["net_difference_on_initial_cash"] = result.net_pnl / cash
        return result

    by_year = yearly.groupby(["book", "year"])[["gross_pnl", "fees", "net_pnl"]].sum().reset_index()
    fees = (
        parts.groupby(["book", "event"])
        .agg(
            fees=("fees", "sum"),
            opened_fee=("opened_fee", "sum"),
            closed_fee=("closed_fee", "sum"),
            product_days=("date", "size"),
            trade_legs=("trade_legs", "sum"),
        )
        .reset_index()
    )
    structure, directions = target_diagnostics(targets, curves)
    tables = {
        "daily_product_attribution": parts,
        "product_attribution": product,
        "product_year_attribution": yearly,
        "annual_attribution": by_year,
        "paired_product_difference": difference(product, ["code"]),
        "paired_annual_difference": difference(by_year, ["year"]),
        "fees_by_event": fees,
        "risk_summary": pd.DataFrame(risks),
        "variance_contributions": pd.concat(variances, ignore_index=True),
        "drawdown_windows": pd.DataFrame(windows),
        "drawdown_contributions": pd.concat(window_parts, ignore_index=True),
        "target_structure": structure,
        "target_directions": directions,
        "daily_reconciliation": pd.concat(reconciliations, ignore_index=True),
    }
    started = datetime.now(timezone.utc)
    if started <= datetime.fromisoformat(freeze["frozen_utc"]):
        raise ValueError("Freeze must precede this drill-down")
    output = paths["output"] / started.strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    for name, frame in tables.items():
        frame.to_csv(output / f"{name}.csv", index=False, encoding="utf-8-sig")
    # Exact bytes, including older correction columns and family metadata.
    (output / "registered_exploration.csv").write_bytes(
        (previous / "registered_exploration.csv").read_bytes()
    )
    run = {
        "version": __version__,
        "created_utc": started.isoformat(),
        "kind": "accounting_drilldown_existing_market_experiment",
        "protocol": plan,
        "protocol_sha256": digest(config),
        "freeze_receipt": freeze,
        "inputs": inputs,
        "original_run": str(paths["source_run"]),
        "initial_cash": cash,
        "known_exploration_family": 16,
        "new_strategies": 0,
        "new_prediction_tests": 0,
        "formal_p_value_computed": False,
        "independent_new_sample": False,
        "causal_mechanism_identified": False,
        "actual_execution_certified": False,
        "source_sha256": {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")},
        "output_sha256": {p.name: digest(p) for p in output.glob("*.csv")},
    }
    (output / "run.json").write_text(
        json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(output, tables, run)
    return output


def write_report(output, tables, run):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paired = tables["paired_product_difference"].sort_values("net_pnl")
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    ax.barh(
        paired.code,
        paired.net_difference_on_initial_cash * 100,
        color=["#cf7458" if v < 0 else "#238b8a" for v in paired.net_pnl],
    )
    ax.axvline(0, color="#718096", lw=0.8)
    ax.set_xlabel("Blend minus momentum: cumulative contribution, percentage points")
    ax.set_title("All products / saved primary scenario / descriptive accounting")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output / "product_difference.png", dpi=150)
    plt.close(fig)
    names = {
        "book": "组合",
        "code": "品种",
        "year": "年份",
        "gross_pnl": "毛盈亏差/元",
        "fees": "费用或费用差/元",
        "net_pnl": "净盈亏差/元",
        "net_difference_on_initial_cash": "累计净差/初始资金",
        "max_drawdown": "最大回撤",
        "cash_normalized_net_volatility": "年化净盈亏波动/初始资金",
        "average_gross_exposure_on_initial_cash": "平均总绝对敞口/初始资金",
        "average_net_exposure_on_initial_cash": "平均净敞口/初始资金",
        "peak_date": "峰值日期",
        "trough_date": "谷值日期",
    }

    def table(frame):
        return (
            '<div class="scroll">'
            + frame.rename(columns=names).to_html(
                index=False, border=0, float_format=lambda x: f"{x:,.5f}"
            )
            + "</div>"
        )

    structure = (
        tables["target_structure"]
        .groupby("book")[["max_absolute_weight", "sum_squared_weights", "effective_absolute_names"]]
        .mean()
        .reset_index()
    )
    total = paired[["gross_pnl", "fees", "net_pnl"]].sum()
    links = " · ".join(f'<a href="{escape(name)}.csv">{escape(name)}</a>' for name in tables)
    html = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>既有组合的盈亏与风险拆解</title>
<style>body{{font:15px/1.8 "Segoe UI","Microsoft YaHei",sans-serif;background:#f3f6f8;color:#173149;max-width:1200px;margin:32px auto;padding:24px}}section{{background:white;border:1px solid #dce5ec;border-radius:12px;padding:22px;margin:22px 0}}.notice{{background:#fff3df;padding:20px}}.scroll{{overflow:auto}}table{{border-collapse:collapse;white-space:nowrap;font-size:12px}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #dce5ec}}img{{width:100%}}a{{color:#147d92}}</style>
<h1>混合组合为什么落后：从保存的资金账拆解</h1><p class="notice">沿用原5bp/20%保证金/假设开盘全量成交三本账，不生成新策略，不改变权重、窗口或品种。所有结论是已看历史上的会计解释，实际成交、来源初发和独立样本均未认证。</p>
<section><h2>毛盈亏与交易摩擦分别贡献多少</h2><p>混合减动量：毛盈亏差 {total.gross_pnl:,.2f} 元，额外费用 {total.fees:,.2f} 元，净差 {total.net_pnl:,.2f} 元。净差等于毛差减费用差。各品种/年度都公开，不根据贡献删品种。</p><img src="product_difference.png" alt="原十品种的混合减动量累计贡献">{table(tables["paired_product_difference"])}{table(tables["paired_annual_difference"])}</section>
<section><h2>实际敞口与风险</h2><p>波动率来自每日现金盈亏除初始资金，按252日、样本标准差描述性折算；不是风险匹配后的Alpha。品种方差份额可为负，完整文件提供相加恒等式。</p>{table(tables["risk_summary"])}</section>
<section><h2>分数混合后的目标结构</h2><p>表内是每次调仓指标的时间平均。“两组秩分数各50%后归一”不等于两套组合各半资金；相互抵消后再归一可能让剩余品种权重集中。只诊断原订单，不替换成事后更优策略。</p>{table(structure)}</section>
<section><h2>最大回撤发生在哪段时间</h2><p>按各自资金曲线找峰谷，不保证是同一市场区间。品种窗口净贡献除峰权益应还原其最大回撤，不能把跨区间差当因果效果。</p>{table(tables["drawdown_windows"])}</section>
<section><h2>完整可复核资料</h2><p>表内收益贡献用小数，0.01表示1%。费用/毛盈亏的差都为“混合减动量”。没有新增p值或统计确认；16项原登记逐字节保留。</p><p>{links} · <a href="registered_exploration.csv">原16项登记</a> · <a href="run.json">输入/源码/输出哈希</a></p></section></html>"""
    (output / "index.html").write_text(html, encoding="utf-8")
