"""Offline synthetic contracts through factors, explicit orders and cash settlement."""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .contract_report import table
from .contract_research import file_hash, run_contract_research
from .engine import quantile_groups, signal_weights
from .execution import settle_portfolio
from .official import PILOT_CODES


def fixed_budget_orders(
    curves: pd.DataFrame,
    calendar: pd.DataFrame,
    bars: pd.DataFrame,
    signal: str,
    start: str,
    initial_cash: float = 10_000_000,
    gross_budget: float = 0.5,
    rebalance_sessions: int = 20,
    min_assets: int = 6,
) -> pd.DataFrame:
    """Size at the preceding signal's known settlement, never at tomorrow's open.

    Notional budget is fixed to initial capital, not a hidden daily rebalance or
    risk parity model. Mentioned products are complete targets for the ledger.
    """
    if not 0 < gross_budget <= 1 or initial_cash <= 0 or rebalance_sessions < 1:
        raise ValueError("Invalid fixed capital budget or schedule")
    dates = pd.DatetimeIndex(pd.to_datetime(calendar.date))
    if not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError("Order calendar must be complete and chronological")
    source = bars.copy()
    source["date"] = pd.to_datetime(source.date)
    source = source.set_index(["date", "contract"])
    values = curves.pivot(index="date", columns="code", values="carry")
    common = pd.DataFrame(True, index=values.index, columns=values.columns)
    for name in ("carry", "momentum_126d", "basis_momentum"):
        common &= curves.pivot(index="date", columns="code", values=name).notna()
    common &= common.sum(axis=1).ge(min_assets).to_numpy()[:, None]
    if signal == "equal_weight_long":
        targets = common.astype(float)
        targets = targets.div(targets.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    else:
        factor = curves.pivot(index="date", columns="code", values=signal).where(common)
        targets = signal_weights(quantile_groups(factor, 3, min_assets), 3)
    records = []
    last_reference = {}
    signal_rows = curves.set_index(["date", "code"])
    first = next((i for i in range(1, len(dates) - 1) if dates[i] >= pd.Timestamp(start)), None)
    if first is None:
        raise ValueError("No execution period after warm-up")
    for i in range(first, len(dates) - 1, rebalance_sessions):
        day, execution_day = dates[i - 1], dates[i]
        for code, weight in targets.loc[day].items():
            row = signal_rows.loc[(day, code)]
            contract = row.get("front_contract")
            if not isinstance(contract, str):
                if weight:
                    raise ValueError("No known contract for an active signal")
                if code in last_reference:
                    records.append(
                        {
                            "date": execution_day,
                            "contract": last_reference[code],
                            "target_lots": 0,
                            "decision_at": row.signal_cutoff,
                            "signal_date": day,
                            "sizing_settle": np.nan,
                            "target_nominal_weight": 0.0,
                        }
                    )
                continue
            last_reference[code] = contract
            spec = source.loc[(day, contract)]
            if pd.Timestamp(spec.spec_available_at) > pd.Timestamp(row.signal_cutoff):
                raise ValueError("Sizing specification not known at the signal cutoff")
            unit_notional = abs(float(row.reference_settle)) * float(spec.multiplier)
            if not np.isfinite(unit_notional) or unit_notional <= 0:
                raise ValueError("Invalid known notional for integer sizing")
            lots = int(
                np.sign(weight)
                * np.floor(initial_cash * gross_budget * abs(weight) / unit_notional)
            )
            records.append(
                {
                    "date": execution_day,
                    "contract": contract,
                    "target_lots": lots,
                    "decision_at": row.signal_cutoff,
                    "signal_date": day,
                    "sizing_settle": row.reference_settle,
                    "target_nominal_weight": weight * gross_budget,
                }
            )
    return pd.DataFrame(records)


def generate_contract_demo(root: Path, seed: int = 20261029) -> tuple[Path, pd.DataFrame]:
    if root.exists():
        raise FileExistsError("Synthetic contract directory must be new")
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", "2024-12-31")
    expiries = pd.date_range("2020-03-31", "2026-12-31", freq="QE-DEC")
    common = rng.normal(0, 0.004, len(dates))
    rows, metadata = [], []
    for number, code in enumerate(PILOT_CODES):
        levels = (1000 + 700 * number) * np.exp(
            np.cumsum(common + rng.normal(0, 0.008, len(dates)))
        )
        previous = {}
        for date, level in zip(dates, levels):
            for expiry in expiries:
                days = (expiry - date).days
                if not 0 < days <= 540:
                    continue
                contract = f"{code}{expiry:%y%m}"
                slope = 0.03 * np.sin(date.dayofyear / 365.25 * 2 * np.pi + number) + number * 0.005
                settle = float(level * np.exp(slope * days / 365.25))
                opening = previous.get(contract, settle) * np.exp(rng.normal(0, 0.002))
                previous[contract] = settle
                rows.append(
                    {
                        "date": date,
                        "code": code,
                        "contract": contract,
                        "open": opening,
                        "high": max(opening, settle) * 1.003,
                        "low": min(opening, settle) * 0.997,
                        "close": settle,
                        "settle": settle,
                        "volume": 10000,
                        "open_interest": 50000,
                        "multiplier": 10,
                        "margin_rate": 0.12,
                        "fee_rate": 0.0002,
                        "fee_per_lot": 1.0,
                        "close_fee_rate": 0.0002,
                        "close_fee_per_lot": 1.0,
                        "can_buy": True,
                        "can_sell": True,
                        "spec_available_at": "2019-12-01T16:00:00+08:00",
                        "available_at": f"{date:%Y-%m-%d}T16:00:00+08:00",
                        "expiry": expiry,
                    }
                )
        for expiry in expiries:
            metadata.append(
                {
                    "contract": f"{code}{expiry:%y%m}",
                    "code": code,
                    "expiry": expiry,
                    "listed": "2019-01-01",
                    "metadata_available_at": "2019-12-01T16:00:00+08:00",
                    "source": "SYNTHETIC",
                }
            )
    inventory = []
    for number, code in enumerate(PILOT_CODES):
        for date in pd.date_range("2016-01-01", "2024-12-31", freq="W-FRI"):
            value = float(
                (5000 + number * 500)
                * np.exp(0.3 * np.sin(date.dayofyear / 365.25 * 2 * np.pi) + rng.normal(0, 0.12))
            )
            inventory.append(
                {
                    "code": code,
                    "period_date": date,
                    "available_at": f"{date:%Y-%m-%d}T16:00:00+08:00",
                    "value": value,
                    "unit": "synthetic_units",
                    "source": "SYNTHETIC",
                }
            )
    for folder, name, frame in (
        ("market", "bars", pd.DataFrame(rows)),
        ("metadata", "metadata", pd.DataFrame(metadata)),
        ("warrants", "warrants", pd.DataFrame(inventory)),
    ):
        directory = root / folder
        directory.mkdir(parents=True)
        path = directory / f"{name}.csv"
        frame.to_csv(path, index=False)
        (directory / "manifest.json").write_text(
            json.dumps(
                {
                    "kind": f"synthetic_contract_{name}",
                    "synthetic": True,
                    "seed": seed,
                    "normalized_sha256": file_hash(path),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    calendar = pd.DataFrame(
        {
            "date": dates,
            "open_at": [f"{d:%Y-%m-%d}T09:00:00+08:00" for d in dates],
            "signal_cutoff": [f"{d:%Y-%m-%d}T20:30:00+08:00" for d in dates],
        }
    )
    calendar.to_csv(root / "calendar.csv", index=False)
    return root / "market/bars.csv", calendar


def run_contract_demo(output: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    source = Path("data/contract_demo") / stamp
    bars_path, calendar = generate_contract_demo(source)
    config_dir = source / "configs"
    config_dir.mkdir()
    protocol = config_dir / "pilot.toml"
    protocol.write_text(
        f'''[paths]
bars = "market/bars.csv"
metadata = "metadata/metadata.csv"
warrants = "warrants/warrants.csv"
output = "{output.as_posix()}"
[study]
codes = ["cu", "al", "zn", "rb", "hc", "ru"]
start = "2020-01-01"
end = "2024-12-31"
evaluation_start = "2023-01-01"
holding_sessions = 20
entry_lag_sessions = 1
min_assets = 6
hac_lags = 40
min_inference_obs = 252
information_cutoff_local = "20:30:00"
retrospective_bar_availability = false
inventory_min_seasons = 3
inventory_min_observations = 24
[curve]
min_days = 60
max_days = 540
min_volume = 100
min_open_interest = 1000
momentum_sessions = 126
basis_momentum_sessions = 252
''',
        encoding="utf-8",
    )
    directory = run_contract_research(protocol)
    curves = pd.read_csv(directory / "curves.csv", parse_dates=["date"])
    bars = pd.read_csv(bars_path, parse_dates=["date"])
    ledger_calendar = calendar.loc[calendar.date.ge("2023-01-01")].copy()
    ledger_bars = bars.loc[bars.date.isin(ledger_calendar.date)].copy()
    ledger_dir = directory / "settlement_demo"
    ledger_dir.mkdir()
    summaries, equities = [], {}
    for signal in ("equal_weight_long", "momentum_126d", "carry", "basis_momentum"):
        print(f"Synthetic settlement ledger: {signal}", flush=True)
        orders = fixed_budget_orders(curves, calendar, bars, signal, "2023-01-01")
        result = settle_portfolio(ledger_bars, ledger_calendar, orders, initial_cash=10_000_000)
        destination = ledger_dir / signal
        destination.mkdir()
        orders.to_csv(destination / "orders.csv", index=False)
        for name in ("daily", "trades", "rejected", "holdings"):
            getattr(result, name).to_csv(destination / f"{name}.csv", index=name == "daily")
        equities[signal] = result.daily.equity / 10_000_000
        summaries.append(
            {
                "signal": signal,
                "final_equity": result.daily.iloc[-1].equity,
                "fees": result.daily.fees.sum(),
                "trade_legs": len(result.trades),
                "rejected_product_orders": len(result.rejected),
                "max_cash_reconciliation_error": result.daily.reconciliation_error.abs().max(),
                "terminal_unliquidated": bool(result.daily.iloc[-1].terminal_unliquidated),
                "claim_eligible": False,
            }
        )
    summary = pd.DataFrame(summaries)
    summary.to_csv(ledger_dir / "summary.csv", index=False)
    links = "".join(
        "<section><h2>"
        + item["signal"]
        + '</h2><div class="links">'
        + "".join(
            f'<a href="{item["signal"]}/{name}.csv">{label}</a>'
            for name, label in (
                ("orders", "信号与目标订单"),
                ("daily", "每日权益与保证金"),
                ("trades", "成交及费用"),
                ("rejected", "订单拒绝记录"),
                ("holdings", "结算持仓"),
            )
        )
        + "</div></section>"
        for item in summaries
    )
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4))
    # Native datetime plotting avoids pandas' deprecated Matplotlib Period formatter.
    for label, series in equities.items():
        axes[0].plot(series.index.to_pydatetime(), series.to_numpy(), label=label)
    for label in ("equity", "reserved_margin", "available_cash"):
        axes[1].plot(
            result.daily.index.to_pydatetime(), result.daily[label].to_numpy(), label=label
        )
    axes[0].set_title("SYNTHETIC integer-contract cash equity")
    axes[1].set_title("SYNTHETIC collateral accounting; final model")
    for axis in axes:
        axis.legend()
        axis.set_xlabel("date")
    fig.tight_layout()
    fig.savefig(ledger_dir / "ledger.png", dpi=140)
    plt.close(fig)
    page = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>合成合约现金账</title><style>body{{font:15px/1.7 "Segoe UI","Microsoft YaHei",sans-serif;color:#173149;background:#f3f6f8;max-width:1250px;margin:40px auto;padding:25px}}table{{border-collapse:collapse;font-size:12px;white-space:nowrap}}td,th{{padding:12px;border:1px solid #dce5ec}}img{{width:100%}}.notice{{background:#fff1df;padding:20px}}.scroll{{overflow:auto}}.links{{display:flex;gap:22px;flex-wrap:wrap}}a{{color:#147d92}}section{{background:#fff;padding:15px 25px;border-radius:10px;margin:20px 0}}</style><h1>合成合约现金结算演示</h1><p class="notice">所有行情、仓单、合约参数和交易时段均为合成输入，仅验证程序。图中盈亏不代表真实市场业绩；没有据此选模型。</p><p>初始资金 1,000 万，名义预算固定为初始资金的 50%，20 日调仓、整数手数；以已知信号日结算价定手数。预算不随净值自动复投，也未实现风险平价。首次开仓、双腿换月和最后开盘清仓均计费；保证金只占用资金，不重复计损。</p><img src="ledger.png" alt="合成现金账与保证金占用">{table(summary)}{links}<p><a href="../index.html">因子诊断报告</a> · <a href="summary.csv">现金账汇总</a> · <a href="manifest.json">输入来源和演示设定</a></p></html>"""
    (ledger_dir / "index.html").write_text(page, encoding="utf-8")
    report = directory / "index.html"
    html = report.read_text(encoding="utf-8").replace(
        "</main>",
        '<section><h2>合成交易账验证</h2><p>下面的合成演示只检验整数合约、现金结算、成本和保证金记账，不表示真实投资表现。</p><a href="settlement_demo/index.html">查看完整现金账、交易记录与订单拒绝</a></section></main>',
    )
    report.write_text(html, encoding="utf-8")
    (ledger_dir / "manifest.json").write_text(
        json.dumps(
            {
                "synthetic": True,
                "initial_cash": 10_000_000,
                "fixed_initial_notional_budget": 0.5,
                "rebalance_sessions": 20,
                "source_bars_sha256": file_hash(bars_path),
                "calendar_sha256": file_hash(source / "calendar.csv"),
                "signal_curve_sha256": file_hash(directory / "curves.csv"),
                "claim_eligible": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return directory
