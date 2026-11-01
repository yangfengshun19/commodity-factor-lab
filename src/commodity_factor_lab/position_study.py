"""Source-qualified same-contract broker positioning; no zero filling or re-selection."""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .contract_report import table
from .contract_research import partial_rank_ic
from .fundamentals import normalized_position_flow
from .inference import by_adjust, hac_mean
from .position_ingest import position_plan, sha256
from .positions import parse_position_ranks


def qualify_positions(
    ranks: pd.DataFrame, bars: pd.DataFrame, calendar: pd.DataFrame
) -> pd.DataFrame:
    if ranks.duplicated(["date", "contract"]).any() or bars.duplicated(["date", "contract"]).any():
        raise ValueError("Duplicate contract dates cannot qualify as position observations")
    data = ranks.merge(
        bars[["date", "contract", "open_interest"]],
        on=["date", "contract"],
        how="left",
        validate="one_to_one",
    )
    cutoffs = calendar.set_index("date").signal_cutoff
    data["signal_cutoff"] = pd.to_datetime(data.date.map(cutoffs), utc=True)
    data["available_at"] = pd.to_datetime(data.available_at, utc=True)
    complete = data.status.eq("complete_top20")
    data.loc[complete & (data.available_at.isna() | data.signal_cutoff.isna()), "status"] = (
        "unknown_publication_cutoff"
    )
    data.loc[complete & data.available_at.gt(data.signal_cutoff), "status"] = (
        "published_after_cutoff"
    )
    oi = pd.to_numeric(data.open_interest, errors="coerce")
    valid_oi = np.isfinite(oi) & oi.gt(0) & oi.mod(1).eq(0)
    data.loc[data.status.eq("complete_top20") & ~valid_oi, "status"] = "missing_or_invalid_oi"
    overflow = data[["long_top20", "short_top20"]].gt(oi, axis=0).any(axis=1)
    data.loc[data.status.eq("complete_top20") & overflow, "status"] = (
        "side_total_exceeds_single_sided_oi"
    )
    data["net_top20"] = (data.long_top20 - data.short_top20).where(data.status.eq("complete_top20"))
    return data


def same_contract_position_panel(
    qualified: pd.DataFrame, curves: pd.DataFrame, calendar: pd.DataFrame, sessions: int = 20
) -> pd.DataFrame:
    dates = pd.DatetimeIndex(calendar.date)
    if not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError("Position panel requires a unique sorted full session calendar")
    if curves.duplicated(["date", "code"]).any():
        raise ValueError("Duplicate selected product dates")
    net = qualified.pivot(index="date", columns="contract", values="net_top20").reindex(dates)
    oi = qualified.pivot(index="date", columns="contract", values="open_interest").reindex(
        index=dates, columns=net.columns
    )
    flow = normalized_position_flow(net, oi, sessions)
    counts = net.notna().rolling(sessions + 1, min_periods=1).sum()
    statuses = qualified.set_index(["date", "contract"]).status
    indexes = {date: i for i, date in enumerate(dates)}
    rows = []
    for item in curves.itertuples():
        if item.date not in indexes:
            raise ValueError("Selected date is not in the declared position calendar")
        index = indexes[item.date]
        contract = item.front_contract
        row = {
            "date": item.date,
            "code": item.code,
            "contract": contract,
            "normalized_position_flow": np.nan,
            "current_rank_status": "not_disclosed",
            "complete_window_sessions": 0,
            "net_today": np.nan,
            "net_lag20": np.nan,
            "open_interest": np.nan,
            "lookback_date": pd.NaT,
            "status": "not_disclosed",
        }
        if not isinstance(contract, str):
            row["status"] = "no_selected_contract"
        elif contract in net:
            state = statuses.get((item.date, contract), "not_disclosed")
            row.update(
                current_rank_status=state,
                complete_window_sessions=int(counts.at[item.date, contract]),
                net_today=net.at[item.date, contract],
                open_interest=oi.at[item.date, contract],
            )
            if index < sessions:
                row["status"] = "warmup"
            else:
                prior = dates[index - sessions]
                row.update(lookback_date=prior, net_lag20=net.at[prior, contract])
                value = flow.at[item.date, contract]
                row.update(
                    normalized_position_flow=value,
                    status="ok" if np.isfinite(value) else "incomplete_same_contract_window",
                )
        rows.append(row)
    return pd.DataFrame(rows)


def run_position_study(config: Path) -> Path:
    raw, paths, planned = position_plan(config)
    receipt_path = paths["raw"] / "manifest.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt["protocol_sha256"] != sha256(config) or receipt["planned_dates"] != planned:
        raise ValueError("Ranking responses use a different protocol")
    by_date = {item["date"]: item for item in receipt["records"]}
    if len(by_date) != len(receipt["records"]) or set(by_date) != set(planned):
        raise ValueError("All planned dates must be attempted before running the study")
    all_ranks, inputs = [], []
    for date in planned:
        item = by_date[date]
        if item["status"] != "ok":
            continue
        source = paths["raw"] / "responses" / f"{date}.json"
        if sha256(source) != item["sha256"]:
            raise ValueError("Ranking response no longer matches its receipt")
        all_ranks.append(parse_position_ranks(source.read_bytes(), date))
        inputs.append({"date": date, "sha256": item["sha256"]})
    if not all_ranks:
        raise ValueError("No valid source responses")
    calendar = pd.read_csv(paths["calendar"], parse_dates=["date"])
    calendar = calendar.loc[calendar.date.isin(pd.to_datetime(planned))].copy()
    ranks = pd.concat(all_ranks, ignore_index=True)
    codes = raw["study"]["codes"]
    ranks = ranks.loc[ranks.contract.str.extract(r"^([a-z]+)", expand=False).isin(codes)]
    bars = pd.read_csv(paths["bars"], parse_dates=["date"])
    bars = bars.loc[bars.date.isin(calendar.date) & bars.code.isin(codes)].copy()
    curves = pd.read_csv(paths["curves"], parse_dates=["date"])
    curves = curves.loc[curves.date.isin(calendar.date) & curves.code.isin(codes)].copy()
    qualified = qualify_positions(ranks, bars, calendar)
    panel = same_contract_position_panel(
        qualified, curves, calendar, raw["study"]["lookback_sessions"]
    )
    labels = pd.read_csv(paths["labels"], parse_dates=["date"])
    data = curves.merge(
        panel[["date", "code", "normalized_position_flow"]],
        on=["date", "code"],
        validate="one_to_one",
    )
    data = data.merge(
        labels[["date", "code", "target"]], on=["date", "code"], validate="one_to_one"
    )
    controls = tuple(raw["study"]["controls"])
    daily = pd.DataFrame(
        [
            {
                "date": date,
                **partial_rank_ic(
                    group, "normalized_position_flow", controls, raw["study"]["min_assets"]
                ),
            }
            for date, group in data.groupby("date")
        ]
    )
    evaluation = daily.loc[daily.date.ge(raw["study"]["evaluation_start"])].copy()
    stats = hac_mean(
        evaluation.partial_rank_ic, raw["study"]["hac_lags"], raw["study"]["min_inference_obs"]
    )
    prior = pd.read_csv(paths["prior_exploration"])
    # Preserve the earlier gated slot; activation is a separate declared exploration.
    current = pd.DataFrame(
        [
            {
                "analysis": "source_qualified_position_activation",
                "signal": "normalized_position_flow",
                "days": stats["nobs"],
                "mean_partial_rank_ic": stats["mean"],
                "hac_t": stats["t"],
                "hac_p": stats["p"],
                "inference_status": stats["status"],
                "controls": ";".join(controls),
                "claim_eligible": False,
            }
        ]
    )
    registry = pd.concat([prior, current], ignore_index=True)
    registry["registered_exploration_by_q"] = by_adjust(registry.hac_p.to_numpy())
    current["registered_exploration_by_q"] = registry.iloc[-1].registered_exploration_by_q
    coverage = (
        panel.assign(year=panel.date.dt.year)
        .groupby(["code", "year"])
        .agg(
            selected=("date", "count"),
            current_complete=("current_rank_status", lambda s: s.eq("complete_top20").sum()),
            full_window=("normalized_position_flow", "count"),
        )
        .reset_index()
    )
    annual = (
        evaluation.assign(year=evaluation.date.dt.year)
        .groupby("year")
        .agg(days=("partial_rank_ic", "count"), mean_partial_rank_ic=("partial_rank_ic", "mean"))
        .reset_index()
    )
    output = paths["output"] / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    for name, frame in (
        ("qualified_contract_ranks", qualified),
        ("selected_position_panel", panel),
        ("daily_ic", daily),
        ("summary", current),
        ("registered_exploration", registry),
        ("coverage", coverage),
        ("yearly_diagnostics", annual),
    ):
        frame.to_csv(output / f"{name}.csv", index=False, encoding="utf-8-sig")
    run = {
        "version": __version__,
        "kind": "source_qualified_retrospective_position_study",
        "protocol": raw,
        "protocol_sha256": sha256(config),
        "source_manifest_sha256": sha256(receipt_path),
        "raw_responses": inputs,
        "planned_dates": len(planned),
        "successful_dates": len(inputs),
        "known_exploration_family": len(registry),
        "full_legacy_exploration_covered": False,
        "claim_eligible": False,
        "real_investment_backtest": False,
        "source_sha256": {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
    }
    (output / "run.json").write_text(
        json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    page = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>同合约持仓研究</title><style>body{{font:16px/1.8 'Segoe UI','Microsoft YaHei',sans-serif;color:#173149;background:#f2f6f9;max-width:1250px;margin:35px auto;padding:24px}}section{{background:#fff;border-radius:12px;padding:24px;margin:24px 0}}table{{font-size:13px;border-collapse:collapse;white-space:nowrap}}td,th{{padding:10px;border-bottom:1px solid #dce5ec}}.scroll{{overflow:auto}}.notice{{background:#fff1df;padding:20px}}a{{color:#147d92}}</style><h1>持仓排名：从来源准入到固定检验</h1><p class="notice">本轮尝试 {len(planned)} 个固定日期，{len(inputs)} 个取得可解析响应。沿用原十品种和选约，严格要求同一合约连续 21 个交易日、完整双方 Top20、已发布时钟及单边持仓校验。历史已被查看，结果不构成独立发现或真实投资业绩。</p><section><h2>增量预测结果</h2>{table(current)}<p>控制动量、carry、基差动量，至少八品种；二十交易日同合约未来标签，四十日 HAC，至少 252 个有效日期才推断。数据不足时保留不可检验，不降低门槛。原先未就绪项保留，激活作为新增一次探索，完整已登记族为 {len(registry)} 项，仍未覆盖全部旧搜索。</p></section><section><h2>覆盖与年度</h2>{table(coverage)}{table(annual)}<p>前置窗口为 2022-12-05 起的二十日。缺失、晚发、排名不完整和持仓量异常不补零；先计算逐合约变化，再按信号日原选合约取值，避免把换约当成增减仓。</p></section><section><h2>解释边界</h2><p>Top20 多头合计减 Top20 空头合计是两个各自排序集合之差，成员可变化；它不是固定投资者持仓、商业套保压力或全市场净持仓。当前可下载版本及更新时间仍不能认证所有初发历史版本。年包总持仓来自官方修订导出，数据算术一致不等于交易时点与成交条件已认证。</p></section><section><h2>完整可复核产物</h2><a href="selected_position_panel.csv">所选合约与缺失原因</a> · <a href="qualified_contract_ranks.csv">合约排名与总持仓校验</a> · <a href="daily_ic.csv">逐日诊断</a> · <a href="registered_exploration.csv">全部登记探索</a> · <a href="run.json">协议与来源哈希</a></section></html>"""
    (output / "index.html").write_text(page, encoding="utf-8")
    return output
