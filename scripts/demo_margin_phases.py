"""Synthetic cash and auction submission checks; no market performance."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from commodity_factor_lab.execution import settle_portfolio


def run_demo(output: Path) -> Path:
    if output.exists():
        raise FileExistsError("Choose a new synthetic demonstration directory")
    dates = pd.bdate_range("2023-01-02", periods=3)
    calendar = pd.DataFrame(
        {
            "date": dates,
            "open_at": [f"{day:%Y-%m-%d}T09:00:00+08:00" for day in dates],
            "settlement_at": [f"{day:%Y-%m-%d}T16:00:00+08:00" for day in dates],
        }
    )
    bars = pd.DataFrame(
        {
            "date": dates,
            "code": "demo",
            "contract": "SYNTHETIC",
            "open": 100.0,
            "settle": 100.0,
            "multiplier": 10,
            "margin_rate": [0.1, 0.3, 0.3],
            "settlement_margin_rate": 0.3,
            "fee_rate": 0.0,
            "fee_per_lot": 1.0,
            "close_fee_rate": 0.0,
            "close_fee_per_lot": 1.0,
            "can_buy": True,
            "can_sell": True,
            "spec_available_at": "2022-12-01T16:00:00+08:00",
            "settlement_spec_available_at": "2023-01-02T12:00:00+08:00",
            "expiry": "2023-12-29",
        }
    )
    orders = pd.DataFrame(
        {
            "date": [dates[0]],
            "contract": ["SYNTHETIC"],
            "target_lots": [1],
            "decision_at": ["2023-01-01T17:00:00+08:00"],
        }
    )
    result = settle_portfolio(
        bars, calendar, orders, initial_cash=1000.0, max_margin_fraction=0.2, liquidate=False
    )
    assert result.rejected.empty
    assert list(result.daily.equity) == [999.0, 998.0, 998.0]
    assert list(result.daily.opening_reserved_margin) == [100.0, 0.0, 0.0]
    assert list(result.daily.reserved_margin) == [300.0, 0.0, 0.0]
    assert list(result.daily.margin_call) == [True, False, False]
    assert list(result.trades.reason) == ["rebalance", "margin_liquidation"]
    assert result.daily.reconciliation_error.eq(0).all()
    auction_calendar = calendar.copy()
    auction_calendar["order_cutoff_at"] = [f"{day:%Y-%m-%d}T08:59:00+08:00" for day in dates]
    cutoff_checks = []
    for clock, expected_rejection in (("08:58:59", False), ("08:59:00", True), ("08:59:30", True)):
        candidate = orders.copy()
        candidate["decision_at"] = f"2023-01-02T{clock}+08:00"
        try:
            settle_portfolio(
                bars,
                auction_calendar,
                candidate,
                initial_cash=1000.0,
                max_margin_fraction=0.2,
                liquidate=False,
            )
            rejected, reason = False, "accepted_before_deadline"
        except ValueError as error:
            if "order submission cutoff" not in str(error):
                raise
            rejected, reason = True, str(error)
        assert rejected == expected_rejection
        cutoff_checks.append(
            {
                "synthetic": True,
                "decision_at": candidate.iloc[0].decision_at,
                "order_cutoff_at": auction_calendar.iloc[0].order_cutoff_at,
                "open_at": auction_calendar.iloc[0].open_at,
                "rejected": rejected,
                "reason": reason,
            }
        )
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(cutoff_checks).to_csv(output / "auction_cutoff_checks.csv", index=False)
    for name, frame in {
        "bars": bars,
        "calendar": calendar,
        "orders": orders,
        "daily": result.daily,
        "trades": result.trades,
        "rejected": result.rejected,
        "holdings": result.holdings,
    }.items():
        frame.to_csv(output / f"{name}.csv", index=name == "daily")
    source = Path(__file__).resolve().parents[1] / "src/commodity_factor_lab/execution.py"
    (output / "run.json").write_text(
        json.dumps(
            {
                "synthetic": True,
                "initial_cash": 1000.0,
                "max_margin_fraction": 0.2,
                "settlement_parameter_known_after_open_but_before_settlement": True,
                "manual_expected_equity": [999.0, 998.0, 998.0],
                "manual_expected_opening_margin": [100.0, 0.0, 0.0],
                "manual_expected_settlement_margin": [300.0, 0.0, 0.0],
                "max_cash_reconciliation_error": float(
                    result.daily.reconciliation_error.abs().max()
                ),
                "execution_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "real_market_performance": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>合成保证金两阶段验证</title><style>body{max-width:900px;margin:40px auto;padding:24px;font:17px/1.85 system-ui,'Microsoft YaHei',sans-serif;background:#f4f7fa;color:#193348}a{color:#126b83}</style><h1>合成例子：开盘可开仓，结算后需减仓</h1><p>全部价格、比例、手续费、时段和订单均为人工设定，不是市场业绩。</p><p>资金1,000，一手价格100、乘数10，开仓费1。开盘保证金10%，占用100，符合20%预算；当天午间才知结算比例30%，结算占用300，权益仍是999，触发下一开盘减仓。次日平仓费1，权益998、保证金归零；价格不变，没有交易盈亏。</p><p>人工竞价时序例子另设置08:59申报截止与09:00执行边界：08:58:59决策通过，08:59及08:59:30被拒绝。此处只验证严格截止，不证明真实市场成交。</p><p><a href="auction_cutoff_checks.csv">三种决策时刻的校验</a></p><p>不能将结算比例提前用于开盘拒单，也不能把新增200保证金当作交易亏损。原有单比例接口继续支持同日比例不变的明确输入；真实账户、历史时段及成交仍须外部认证。</p><p><a href="daily.csv">逐日现金</a> · <a href="trades.csv">逐笔记录</a> · <a href="bars.csv">人工参数</a> · <a href="calendar.csv">人工时段</a> · <a href="run.json">手算值与源码指纹</a></p></html>"""
    (output / "index.html").write_text(page, encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(run_demo(parser.parse_args().output))
