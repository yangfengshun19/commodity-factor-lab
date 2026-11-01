"""Frozen, explicitly assumed market simulations; never certified account returns."""

import hashlib
import json
import tomllib
from datetime import datetime, timezone
from html import escape
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .curve_factors import utc_column
from .execution import ExecutionResult, settle_portfolio
from .inference import by_adjust

BOOKS = ("equal_weight_long", "momentum", "momentum_carry")
FILLS = ("assumed_full_open", "block_flat_ohlc")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_plan(config: Path) -> tuple[dict, dict]:
    plan = tomllib.loads(config.read_text(encoding="utf-8"))
    root = config.resolve().parent.parent
    paths = {k: (root / v).resolve() for k, v in plan["paths"].items()}
    if set(paths) != set(plan["input_hashes"]) | {"output"} or any(
        not p.is_relative_to(root) for p in paths.values()
    ):
        raise ValueError("Every input needs a frozen hash and workspace-contained path")
    for name, expected in plan["input_hashes"].items():
        if paths["output"].is_relative_to(paths[name].parent):
            raise ValueError("Output must be outside inputs")
        if digest(paths[name]) != expected:
            raise ValueError(f"Frozen market input changed: {name}")
    s, e, v = plan["study"], plan["execution"], plan["evaluation"]
    if (
        s["books"] != list(BOOKS)
        or len(set(s["codes"])) != len(s["codes"])
        or s["min_assets"] < 8
        or s["rebalance_sessions"] != 5
        or s["initial_cash"] != 10_000_000
        or s["gross_notional_on_initial_cash"] != 1
        or s["blend_momentum_weight"] != 0.5
        or s["blend_carry_weight"] != 0.5
        or e["one_way_all_in_cost_bps"] != [0, 5, 10, 20]
        or e["margin_rates"] != [0.2, 0.4]
        or e["fill_scenarios"] != list(FILLS)
        or e["max_margin_fraction"] != 0.5
        or e["clock_kind"] != "synthetic_event_order_not_actual_session_times"
        or v["registered_experiments_before"] != 15
        or v["registered_experiments_after"] != 16
        or not v["new_descriptive_experiment_counts_as_p_one"]
        or any(
            v[k]
            for k in (
                "formal_p_value_computed",
                "independent_new_sample",
                "historical_first_vintage_certified",
                "actual_execution_certified",
            )
        )
    ):
        raise ValueError("Keep the fixed design and claim boundaries")
    if set(plan["multipliers"]) != set(s["codes"]) or any(
        not np.isfinite(m) or m <= 0 for m in plan["multipliers"].values()
    ):
        raise ValueError("Each product needs an explicit positive assumed multiplier")
    manifest = json.loads(paths["bars_manifest"].read_text(encoding="utf-8"))
    if manifest["normalized_sha256"] != plan["input_hashes"]["bars"]:
        raise ValueError("Market bars disagree with their source manifest")
    return plan, paths


def weights(group: pd.DataFrame, book: str) -> pd.Series:
    """Only contemporaneous ranks, common universe and fixed gross exposure."""
    if book not in BOOKS or group.empty:
        raise ValueError("Unknown book or empty universe")
    if book == "equal_weight_long":
        return pd.Series(1 / len(group), index=group.index)
    momentum = group.momentum_126d.rank(method="average")
    score = momentum - momentum.mean()
    if book == "momentum_carry":
        carry = group.carry.rank(method="average")
        score = 0.5 * score + 0.5 * (carry - carry.mean())
    gross = score.abs().sum()
    return score / gross if gross > 0 else score * 0


def build_orders(
    curves: pd.DataFrame, calendar: pd.DataFrame, study: dict, multipliers: dict
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """Generate every target without reading future prices, labels or outcomes."""
    features, sessions = curves.copy(), calendar.copy()
    for frame in (features, sessions):
        frame["date"] = pd.to_datetime(frame.date).dt.normalize()
    sessions["signal_cutoff"] = utc_column(sessions.signal_cutoff)
    if (
        not sessions.date.is_unique
        or not sessions.date.is_monotonic_increasing
        or features.duplicated(["date", "code"]).any()
        or not features.date.isin(sessions.date).all()
    ):
        raise ValueError("Ambiguous feature/session keys")
    evaluation = sessions.loc[sessions.date.between(study["start"], study["end"])].copy()
    if len(evaluation) < 2:
        raise ValueError("Need at least two evaluation sessions")
    positions = {d: i for i, d in enumerate(sessions.date)}
    if positions[evaluation.date.iloc[0]] == 0:
        raise ValueError("First execution needs a previous-session signal")
    grouped = {d: g for d, g in features.groupby("date")}
    orders = {b: [] for b in BOOKS}
    audit, clocks = [], []
    for number, day in enumerate(evaluation.date):
        previous = sessions.iloc[positions[day] - 1]
        cutoff = sessions.loc[sessions.date.eq(day), "signal_cutoff"].iloc[0]
        # These deliberately invented clocks encode sequence only. They cannot
        # establish the actual opening time of the source daily open price.
        clocks.append(
            {
                "date": day,
                "open_at": previous.signal_cutoff + pd.Timedelta(1, unit="m"),
                "previous_signal_date": previous.date,
                "settlement_at": cutoff - pd.Timedelta(1, unit="m"),
                "clock_kind": "synthetic_event_order_not_actual_session_times",
                "actual_session_certified": False,
            }
        )
        if number % study["rebalance_sessions"] or number == len(evaluation) - 1:
            continue
        group = grouped.get(previous.date, features.iloc[:0]).copy()
        group = group.loc[group.code.isin(study["codes"])].copy()
        good = (
            np.isfinite(group[["momentum_126d", "carry", "reference_settle"]]).all(axis=1)
            & group.reference_settle.gt(0)
            & group.front_contract.notna()
            & group.front_expiry.notna()
        )
        common = group.loc[good].sort_values("code")
        if common.code.duplicated().any() or common.front_contract.duplicated().any():
            raise ValueError("Ambiguous common universe")
        enough = len(common) >= study["min_assets"]
        # Mention every excluded product even when the remaining cross-section
        # meets the minimum. The ledger closes whichever old contract is held
        # for that product, including an earlier rejected roll.
        excluded = set(study["codes"]) - set(common.code) if enough else set(study["codes"])
        if excluded:
            history = (
                features.loc[features.date.le(previous.date)]
                .dropna(subset=["front_contract"])
                .sort_values("date")
                .groupby("code", sort=True)
                .tail(1)
            )
            for book in BOOKS:
                for row in history.loc[history.code.isin(excluded)].itertuples():
                    orders[book].append(
                        {
                            "date": day,
                            "contract": row.front_contract,
                            "target_lots": 0,
                            "decision_at": previous.signal_cutoff,
                        }
                    )
                    if enough:
                        audit.append(
                            {
                                "date": day,
                                "signal_date": previous.date,
                                "code": row.code,
                                "book": book,
                                "status": "excluded_product_zero_target",
                                "common_assets": len(common),
                                "contract": row.front_contract,
                                "weight": 0.0,
                                "target_lots": 0,
                            }
                        )
        if not enough:
            audit.append(
                {
                    "date": day,
                    "signal_date": previous.date,
                    "code": "ALL",
                    "book": "ALL",
                    "status": "insufficient_common_universe",
                    "common_assets": len(common),
                }
            )
            continue
        for book in BOOKS:
            allocation = weights(common, book)
            for index, row in common.iterrows():
                amount = (
                    allocation.loc[index]
                    * study["initial_cash"]
                    * study["gross_notional_on_initial_cash"]
                )
                lots = int(np.trunc(amount / (row.reference_settle * multipliers[row.code])))
                orders[book].append(
                    {
                        "date": day,
                        "contract": row.front_contract,
                        "target_lots": lots,
                        "decision_at": previous.signal_cutoff,
                    }
                )
                audit.append(
                    {
                        "date": day,
                        "signal_date": previous.date,
                        "decision_at": previous.signal_cutoff,
                        "code": row.code,
                        "book": book,
                        "status": "ok",
                        "common_assets": len(common),
                        "contract": row.front_contract,
                        "weight": allocation.loc[index],
                        "reference_settle": row.reference_settle,
                        "multiplier": multipliers[row.code],
                        "target_lots": lots,
                        "front_expiry": row.front_expiry,
                    }
                )
    columns = ["date", "contract", "target_lots", "decision_at"]
    return (
        {b: pd.DataFrame(rows, columns=columns) for b, rows in orders.items()},
        pd.DataFrame(audit),
        pd.DataFrame(clocks),
    )


def prepare_marks(
    bars: pd.DataFrame,
    curves: pd.DataFrame,
    clocks: pd.DataFrame,
    orders: dict[str, pd.DataFrame],
    multipliers: dict,
) -> pd.DataFrame:
    """Use original prices; fail on missing marks instead of selecting around them."""
    contracts = set().union(*(set(o.contract) for o in orders.values()))
    expiry = curves.loc[
        curves.front_contract.isin(contracts), ["date", "front_contract", "front_expiry"]
    ].copy()
    expiry["date"] = pd.to_datetime(expiry.date)
    if expiry.duplicated(["date", "front_contract"]).any():
        raise ValueError("Ambiguous dated expiry versions")
    marks = bars.copy()
    marks["date"] = pd.to_datetime(marks.date)
    marks = marks.loc[marks.date.isin(clocks.date) & marks.contract.isin(contracts)].copy()
    if marks.duplicated(["date", "contract"]).any():
        raise ValueError("Duplicate market marks")
    if not np.isfinite(marks.settle).all() or marks[["open", "settle"]].le(0).any().any():
        raise ValueError("Missing/nonpositive required price; do not fill or drop")
    marks["previous_signal_date"] = marks.date.map(clocks.set_index("date").previous_signal_date)
    enriched = []
    for contract, group in marks.groupby("contract", sort=True):
        versions = expiry.loc[expiry.front_contract.eq(contract)].rename(
            columns={"date": "expiry_signal_date", "front_expiry": "expiry"}
        )
        joined = pd.merge_asof(
            group.sort_values("previous_signal_date"),
            versions[["expiry_signal_date", "expiry"]].sort_values("expiry_signal_date"),
            left_on="previous_signal_date",
            right_on="expiry_signal_date",
            direction="backward",
        )
        # Not-yet-known contracts have no target or position. A missing mark
        # for any actual held/ordered contract still fails in the ledger.
        enriched.append(joined.loc[joined.expiry.notna()])
    marks = pd.concat(enriched, ignore_index=True)
    marks["multiplier"] = marks.code.map(multipliers)
    known = clocks.set_index("date").open_at - pd.Timedelta(1, unit="m")
    marks["spec_available_at"] = marks.date.map(known)
    marks["spec_clock_is_assumed"] = True
    marks["historical_first_vintage_certified"] = False
    marks["actual_execution_certified"] = False
    return marks


def scenario_marks(marks: pd.DataFrame, cost: float, margin: float, fill: str) -> pd.DataFrame:
    if fill not in FILLS:
        raise ValueError("Unknown fill scenario")
    # Do not invent an opening price for dormant, zero-volume contracts. All
    # such rows remain in market_marks.csv. If any is actually held/ordered,
    # the ledger fails on that missing required mark, rather than dropping an
    # instrument from the portfolio or substituting its settlement price.
    valid = np.isfinite(marks[["open", "high", "low", "close", "settle"]]).all(axis=1)
    result = marks.loc[valid].copy()
    result["fee_rate"] = result["close_fee_rate"] = cost / 10000
    result["fee_per_lot"] = result["close_fee_per_lot"] = 0.0
    result["margin_rate"] = margin
    result["can_buy"] = result["can_sell"] = True
    if fill == "block_flat_ohlc":
        flat = result[["open", "high", "low", "close"]].nunique(axis=1).eq(1)
        result.loc[flat, ["can_buy", "can_sell"]] = False
    return result


def summarize(result: ExecutionResult, initial_cash: float, annualization: int = 252) -> dict:
    daily = result.daily
    wealth = np.r_[initial_cash, daily.equity.to_numpy()]
    drawdown = wealth / np.maximum.accumulate(wealth) - 1
    trades = result.trades
    turnover = (
        trades.delta_lots.abs()
        * trades.fill_price.abs()
        * trades.contract.map(
            result.holdings.drop_duplicates("contract").set_index("contract").multiplier.to_dict()
        )
    )
    # A contract opened and closed within a session may not appear in EOD
    # holdings; callers also attach monetary turnover directly to each trade.
    if "notional" in trades:
        turnover = trades.notional
    sigma = daily.net_return.std(ddof=1)
    return {
        "sessions": len(daily),
        "final_equity": float(daily.equity.iloc[-1]),
        "net_pnl": float(daily.equity.iloc[-1] - initial_cash),
        "total_return": float(daily.equity.iloc[-1] / initial_cash - 1),
        "annualized_return": float(
            (daily.equity.iloc[-1] / initial_cash) ** (annualization / len(daily)) - 1
        ),
        "max_drawdown": float(drawdown.min()),
        "descriptive_sharpe": float(daily.net_return.mean() / sigma * np.sqrt(annualization))
        if sigma > 0
        else np.nan,
        "fees": float(daily.fees.sum()),
        "trade_legs": len(trades),
        "traded_notional": float(turnover.sum()),
        "rejected_product_orders": len(result.rejected),
        "margin_call_days": int(daily.margin_call.sum()),
        "terminal_unliquidated": bool(daily.terminal_unliquidated.iloc[-1]),
        "max_cash_reconciliation_error": float(daily.reconciliation_error.abs().max()),
        "average_gross_notional": float(daily.gross_notional.mean()),
        "average_net_notional": float(daily.net_notional.mean()),
        "maximum_margin_to_equity": float((daily.reserved_margin / daily.equity).max()),
    }


def attach_exposures(result: ExecutionResult, multipliers: dict) -> None:
    h = result.holdings
    h["signed_notional"] = h.lots * h.settle * h.multiplier
    sums = h.groupby("date").signed_notional.agg(
        net_notional="sum", gross_notional=lambda v: v.abs().sum()
    )
    for column in ("gross_notional", "net_notional"):
        result.daily[column] = sums[column].reindex(result.daily.index, fill_value=0)
    result.trades["notional"] = (
        result.trades.delta_lots.abs()
        * result.trades.fill_price.abs()
        * result.trades.code.map(multipliers)
    )


def append_exploration(prior: pd.DataFrame) -> pd.DataFrame:
    if len(prior) != 15:
        raise ValueError("Do not discard earlier explorations")
    registry = prior.copy()
    row = {name: np.nan for name in registry.columns}
    row.update(
        {
            "analysis": "fixed_market_benchmark",
            "signal": "momentum_plus_carry",
            "controls": "momentum;equal_weight_long;zero_interest_cash",
            "hac_p": np.nan,
            "claim_eligible": False,
            "reason": "descriptive assumed execution; no formal p or new sample",
        }
    )
    # Preserve earlier family sizes and corrections exactly. The new family
    # gets new columns, rather than rewriting older experiment metadata.
    registry.loc[len(registry)] = row
    registry["family_16_by_q"] = by_adjust(registry.hac_p.to_numpy())
    registry["family_16_size"] = 16
    return registry


def run_market_benchmark(config: Path) -> Path:
    plan, paths = load_plan(config)
    root = config.resolve().parent.parent
    freeze = root / "reports/protocols/market_benchmark_20261031/receipt.json"
    frozen = json.loads(freeze.read_text(encoding="utf-8"))
    if frozen["config_sha256"] != digest(config):
        raise ValueError("Protocol differs from frozen receipt")
    started = datetime.now(timezone.utc)
    if started <= datetime.fromisoformat(frozen["frozen_utc"]):
        raise ValueError("Protocol freeze must precede computation")
    curves = pd.read_csv(paths["curves"], parse_dates=["date"])
    bars = pd.read_csv(paths["bars"], parse_dates=["date"])
    calendar = pd.read_csv(paths["calendar"], parse_dates=["date"])
    orders, targets, clocks = build_orders(curves, calendar, plan["study"], plan["multipliers"])
    marks = prepare_marks(bars, curves, clocks, orders, plan["multipliers"])
    stamp = started.strftime("%Y%m%dT%H%M%S%fZ")
    output = paths["output"] / stamp
    output.mkdir(parents=True, exist_ok=False)
    for name, frame in (
        ("target_audit", targets),
        ("assumed_calendar", clocks),
        ("market_marks", marks),
    ):
        frame.to_csv(output / f"{name}.csv", index=False, encoding="utf-8-sig")
    for book, frame in orders.items():
        frame.to_csv(output / f"orders_{book}.csv", index=False, encoding="utf-8-sig")
    summary, annual = [], []
    e, s = plan["execution"], plan["study"]
    for cost, margin, fill in product(e["one_way_all_in_cost_bps"], e["margin_rates"], FILLS):
        scenario = f"cost{cost:g}_margin{margin:g}_{fill}"
        inputs = scenario_marks(marks, cost, margin, fill)
        for book in BOOKS:
            print(f"Assumed market ledger: {scenario}/{book}", flush=True)
            result = settle_portfolio(
                inputs, clocks, orders[book], s["initial_cash"], e["max_margin_fraction"]
            )
            attach_exposures(result, plan["multipliers"])
            destination = output / scenario / book
            destination.mkdir(parents=True)
            for name in ("daily", "trades", "holdings", "rejected"):
                getattr(result, name).to_csv(
                    destination / f"{name}.csv", index=name == "daily", encoding="utf-8-sig"
                )
            dimensions = {
                "scenario": scenario,
                "book": book,
                "cost_bps": cost,
                "margin_rate": margin,
                "fill_scenario": fill,
            }
            summary.append(
                {**dimensions, **summarize(result, s["initial_cash"], s["annualization_sessions"])}
            )
            for year, group in result.daily.groupby(result.daily.index.year):
                annual.append(
                    {
                        **dimensions,
                        "year": int(year),
                        "sessions": len(group),
                        "pnl_on_initial_cash": float(
                            (group.equity.iloc[-1] - group.previous_equity.iloc[0])
                            / s["initial_cash"]
                        ),
                        "within_year_return": float(
                            group.equity.iloc[-1] / group.previous_equity.iloc[0] - 1
                        ),
                        "fees": float(group.fees.sum()),
                    }
                )
    summary = pd.DataFrame(summary)
    comparisons = []
    for scenario, group in summary.groupby("scenario", sort=False):
        by_book = group.set_index("book")
        comparisons.append(
            {
                "scenario": scenario,
                "cost_bps": group.cost_bps.iloc[0],
                "margin_rate": group.margin_rate.iloc[0],
                "fill_scenario": group.fill_scenario.iloc[0],
                "blend_minus_momentum_return_on_initial_cash": float(
                    by_book.loc["momentum_carry", "total_return"]
                    - by_book.loc["momentum", "total_return"]
                ),
                "blend_minus_momentum_fee": float(
                    by_book.loc["momentum_carry", "fees"] - by_book.loc["momentum", "fees"]
                ),
                "blend_minus_momentum_drawdown": float(
                    by_book.loc["momentum_carry", "max_drawdown"]
                    - by_book.loc["momentum", "max_drawdown"]
                ),
            }
        )
    comparisons = pd.DataFrame(comparisons)
    registry = append_exploration(pd.read_csv(paths["prior_exploration"]))
    for name, frame in (
        ("summary", summary),
        ("annual", pd.DataFrame(annual)),
        ("comparisons", comparisons),
        ("registered_exploration", registry),
    ):
        frame.to_csv(output / f"{name}.csv", index=False, encoding="utf-8-sig")
    run = {
        "version": __version__,
        "started_utc": started.isoformat(),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "kind": "retrospective_market_simulation_with_explicit_assumptions",
        "protocol": plan,
        "protocol_sha256": digest(config),
        "freeze_receipt": frozen,
        "inputs": {
            k: {"path": str(paths[k]), "sha256": h} for k, h in plan["input_hashes"].items()
        },
        "market_price_rows_used": len(marks),
        "books": len(summary),
        "independent_new_sample": False,
        "actual_execution_certified": False,
        "historical_first_vintage_certified": False,
        "formal_p_value_computed": False,
        "confirmed_investment_findings": 0,
        "known_exploration_family": 16,
        "source_sha256": {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")},
        "limitations": [
            "年包是修订后的历史；时点、乘数全历史、真实手续费/保证金尚未完整认证。",
            "时钟仅为事件顺序占位；日线开盘完整成交及flat-OHLC压力假设均未经实盘认证。",
            "0/5/10/20bp是单边总摩擦假设，20%/40%是固定保证金压力参数。",
            "历史已看过；固定50/50组合、共同品种、目标金额及所有情景在新结果前冻结。",
            "描述性表现不能证明独立Alpha、因果关系、统计显著或未来投资收益。",
        ],
    }
    run["output_sha256"] = {str(p.relative_to(output)): digest(p) for p in output.rglob("*.csv")}
    (output / "run.json").write_text(
        json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(output, summary, comparisons, pd.DataFrame(annual), run)
    return output


def write_report(
    output: Path, summary: pd.DataFrame, comparisons: pd.DataFrame, annual: pd.DataFrame, run: dict
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    e = run["protocol"]["execution"]
    primary = summary.loc[
        summary.cost_bps.eq(e["primary_cost_bps"])
        & summary.margin_rate.eq(e["primary_margin_rate"])
        & summary.fill_scenario.eq(e["primary_fill_scenario"])
    ]
    scenario = primary.scenario.iloc[0]
    fig, ax = plt.subplots(figsize=(10, 4.3))
    for book in BOOKS:
        daily = pd.read_csv(output / scenario / book / "daily.csv", parse_dates=["date"])
        # Explicitly include the starting-cash baseline before first trading PnL.
        initial = run["protocol"]["study"]["initial_cash"]
        dates = [daily.date.iloc[0] - pd.Timedelta(1, unit="D"), *daily.date]
        ax.plot(dates, np.r_[1, daily.equity.to_numpy() / initial], label=book)
    ax.set_title("Assumed opening fills / 5 bps one-way / 20% margin; historical simulation")
    ax.set_ylabel("Equity / initial cash")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output / "equity.png", dpi=150)
    plt.close(fig)
    labels = {
        "book": "组合",
        "total_return": "累计收益（假设）",
        "annualized_return": "年化收益（假设）",
        "max_drawdown": "最大回撤",
        "fees": "总摩擦金额",
        "cost_bps": "单边总摩擦bp",
        "margin_rate": "假设保证金",
        "fill_scenario": "成交假设",
        "year": "年份",
        "pnl_on_initial_cash": "年盈亏/初始资金",
        "within_year_return": "年内收益",
        "blend_minus_momentum_return_on_initial_cash": "混合减动量累计收益差",
        "rejected_product_orders": "拒单",
        "margin_call_days": "结算超预算日",
    }

    def table(frame):
        return (
            '<div class="scroll">'
            + frame.rename(columns=labels).to_html(
                index=False, border=0, float_format=lambda v: f"{v:,.5f}"
            )
            + "</div>"
        )

    links = " · ".join(
        f'<a href="{name}.csv">{label}</a>'
        for name, label in (
            ("summary", "48本完整账摘要"),
            ("comparisons", "16组配对差"),
            ("annual", "全部年度表现"),
            ("target_audit", "信号与整数目标"),
            ("assumed_calendar", "占位时钟"),
            ("market_marks", "原价格与假设字段"),
            ("registered_exploration", "探索登记16项"),
        )
    )
    yearly = annual.loc[annual.scenario.eq(scenario)]
    page = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>固定市场基准与成本敏感性</title>
<style>body{{font:15px/1.8 "Segoe UI","Microsoft YaHei",sans-serif;background:#f3f6f8;color:#173149;max-width:1200px;margin:32px auto;padding:24px}}section{{background:white;border:1px solid #dce5ec;border-radius:12px;padding:22px;margin:22px 0}}.notice{{background:#fff3df;padding:20px}}.scroll{{overflow:auto}}table{{border-collapse:collapse;white-space:nowrap;font-size:12px}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #dce5ec}}img{{width:100%}}a{{color:#147d92}}</style>
<h1>固定市场基准：加入期限结构，是否改善动量？</h1><p class="notice">这是2023—2025真实历史价格、明确假设下的现金模拟。不是已认证真实成交业绩，不是独立新增样本。时钟只是顺序占位；费用和保证金是压力参数。所有预设情景完整保留，没有按结果改权重、窗口或品种。</p>
<section><h2>固定主情景：单边5bp、保证金20%、假设开盘全量成交</h2><p>同一十品种、同一共同资格、每5交易日调仓，初始1000万元固定名义金额；整数手数按前一日结算价计算。混合为动量秩与carry秩各50%，目标总绝对敞口100%。现金基准无利息收益为0。</p><img src="equity.png" alt="明确假设下的三组合历史资金曲线">{table(primary[["book", "total_return", "annualized_return", "max_drawdown", "fees", "rejected_product_orders", "margin_call_days"]])}</section>
<section><h2>所有成本、保证金和成交假设下的配对差</h2><p>正数表示本段历史里混合组合累计收益较高，不代表统计确认。flat-OHLC阻单是事后压力假设，不是实盘成交证据或收益下界。</p>{table(comparisons)}</section>
<section><h2>固定主情景逐年结果</h2>{table(yearly[["book", "year", "pnl_on_initial_cash", "within_year_return", "fees"]])}<p>年盈亏/初始资金可相加；年内收益按各年起点权益计算。收益表使用小数，0.01表示1%。最后交易日开盘清仓，两条换月腿各计成本。</p></section>
<section><h2>证据边界与可复核文件</h2><ul>{"".join("<li>" + escape(x) + "</li>" for x in run["limitations"])}</ul><p>{links} · <a href="run.json">冻结协议、源码和文件哈希</a></p><p>各情景/组合子目录保存daily、trades、holdings、rejected四张完整账。不计算正式p值；探索族15→16，旧负结果及旧校正列保留。48本压力账不是48个独立发现。</p></section></html>"""
    (output / "index.html").write_text(page, encoding="utf-8")
