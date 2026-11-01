"""Replay historical window records against the current signal's information cutoff."""

import hashlib
import json
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .contract_report import table
from .contract_research import partial_rank_ic
from .inference import by_adjust, hac_mean


def replay_position_availability(
    qualified: pd.DataFrame, curves: pd.DataFrame, calendar: pd.DataFrame, sessions: int = 20
) -> pd.DataFrame:
    if (
        sessions < 1
        or qualified.duplicated(["date", "contract"]).any()
        or curves.duplicated(["date", "code"]).any()
    ):
        raise ValueError("Availability replay requires unique observations and a positive window")
    dates = pd.DatetimeIndex(calendar.date)
    if not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError("Availability calendar must be complete, unique and sorted")
    cutoffs = pd.to_datetime(calendar.signal_cutoff, utc=True)
    if cutoffs.isna().any():
        raise ValueError("Current signal cutoffs cannot be unknown")
    clocks = pd.to_datetime(qualified.available_at, utc=True)
    sources = {}
    for row, clock in zip(qualified.itertuples(), clocks, strict=True):
        quantities = np.array([row.long_top20, row.short_top20, row.open_interest], dtype=float)
        valid = (
            row.status in {"complete_top20", "published_after_cutoff"}
            and np.isfinite(quantities).all()
            and (quantities >= 0).all()
            and (quantities % 1 == 0).all()
            and quantities[2] > 0
            and max(quantities[:2]) <= quantities[2]
            and not pd.isna(clock)
        )
        if valid:
            sources[(row.date, row.contract)] = (
                float(quantities[0] - quantities[1]),
                float(quantities[2]),
                clock,
            )
    indexes = {date: i for i, date in enumerate(dates)}
    rows = []
    for row in curves.itertuples():
        if row.date not in indexes:
            raise ValueError("Selected observation is outside the replay calendar")
        i = indexes[row.date]
        cutoff = cutoffs.iloc[i]
        item = {
            "date": row.date,
            "code": row.code,
            "contract": row.front_contract,
            "signal_cutoff": cutoff,
            "lookback_date": pd.NaT,
            "structural_window_points": 0,
            "known_window_points": 0,
            "structural_full_window": False,
            "latest_window_available_at": pd.NaT,
            "normalized_position_flow": np.nan,
            "status": "warmup",
        }
        if i >= sessions:
            item["lookback_date"] = dates[i - sessions]
            window = [
                sources.get((date, row.front_contract)) for date in dates[i - sessions : i + 1]
            ]
            present = [point for point in window if point is not None]
            known = [point for point in present if point[2] <= cutoff]
            item.update(
                structural_window_points=len(present),
                known_window_points=len(known),
                structural_full_window=len(present) == sessions + 1,
            )
            if len(present) != sessions + 1:
                item["status"] = "incomplete_same_contract_window"
            else:
                item["latest_window_available_at"] = max(point[2] for point in present)
                if len(known) != sessions + 1:
                    item["status"] = "window_version_not_yet_known"
                else:
                    item.update(
                        normalized_position_flow=(window[-1][0] - window[0][0]) / window[-1][1],
                        status="ok",
                    )
        rows.append(item)
    return pd.DataFrame(rows)


def run_availability_review(config: Path) -> Path:
    raw = tomllib.loads(config.read_text(encoding="utf-8"))
    root = config.resolve().parent.parent
    paths = {k: (root / v).resolve() for k, v in raw["paths"].items()}
    files = {
        "baseline_run": paths["baseline_run"] / "run.json",
        "qualified_ranks": paths["baseline_run"] / "qualified_contract_ranks.csv",
        "strict_panel": paths["baseline_run"] / "selected_position_panel.csv",
        "strict_daily": paths["baseline_run"] / "daily_ic.csv",
        "raw_manifest": paths["raw"] / "manifest.json",
        **{k: paths[k] for k in ("calendar", "curves", "labels", "prior_exploration")},
    }
    receipts = []
    for name, path in files.items():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != raw["input_hashes"][name]:
            raise ValueError(f"Frozen information-set input changed: {name}")
        receipts.append({"name": name, "sha256": digest})
    study = raw["study"]
    if (
        study["direction"] != 1
        or study["lookback_sessions"] != 20
        or not study["require_all_window_sessions"]
        or study["min_assets"] < 8
        or study["min_inference_obs"] < 252
        or study["hac_lags"] < study["holding_sessions"]
    ):
        raise ValueError("Keep the registered direction, complete window and inference gates")
    baseline = json.loads(files["baseline_run"].read_text(encoding="utf-8"))
    for field in (
        "holding_sessions",
        "entry_lag_sessions",
        "lookback_sessions",
        "direction",
        "controls",
        "min_assets",
        "hac_lags",
        "min_inference_obs",
        "evaluation_start",
    ):
        if study[field] != baseline["protocol"]["study"][field]:
            raise ValueError("Information-set comparison cannot change the frozen study design")
    strict = pd.read_csv(files["strict_panel"], parse_dates=["date"])
    calendar = pd.read_csv(paths["calendar"], parse_dates=["date"])
    calendar = calendar.loc[calendar.date.isin(strict.date)].copy()
    curves = pd.read_csv(paths["curves"], parse_dates=["date"])
    curves = curves.loc[curves.date.isin(calendar.date)].copy()
    qualified = pd.read_csv(files["qualified_ranks"], parse_dates=["date"])
    panel = replay_position_availability(qualified, curves, calendar, study["lookback_sessions"])
    paired = panel.merge(
        strict[["date", "code", "normalized_position_flow"]].rename(
            columns={"normalized_position_flow": "strict_signal"}
        ),
        on=["date", "code"],
        validate="one_to_one",
    )
    existing = paired.strict_signal.notna()
    if not np.allclose(
        paired.loc[existing, "normalized_position_flow"],
        paired.loc[existing, "strict_signal"],
        atol=1e-14,
        rtol=1e-14,
    ):
        raise ValueError("Replay must preserve every originally admissible signal")
    paired["newly_admissible_with_current_cutoff"] = (
        paired.strict_signal.isna() & paired.normalized_position_flow.notna()
    )
    labels = pd.read_csv(paths["labels"], parse_dates=["date"])
    controls = tuple(study["controls"])
    data = curves.merge(
        paired[
            ["date", "code", "normalized_position_flow", "strict_signal", "structural_full_window"]
        ],
        on=["date", "code"],
        validate="one_to_one",
    )
    data = data.merge(
        labels[["date", "code", "target"]], on=["date", "code"], validate="one_to_one"
    )
    daily = pd.DataFrame(
        [
            {
                "date": date,
                **partial_rank_ic(group, "normalized_position_flow", controls, study["min_assets"]),
            }
            for date, group in data.groupby("date")
        ]
    )
    evaluation = daily.loc[daily.date.ge(study["evaluation_start"])].copy()
    stats = hac_mean(evaluation.partial_rank_ic, study["hac_lags"], study["min_inference_obs"])
    current = pd.DataFrame(
        [
            {
                "analysis": "current_cutoff_position_information_set",
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
    prior = pd.read_csv(paths["prior_exploration"])
    registry = pd.concat([prior, current], ignore_index=True)
    registry["registered_exploration_by_q"] = by_adjust(registry.hac_p.to_numpy())
    current["registered_exploration_by_q"] = registry.iloc[-1].registered_exploration_by_q
    complete_controls = np.isfinite(data[["target", *controls]]).all(axis=1)
    coverage = []
    for name, usable in (
        ("strict_own_day_cutoff", data.strict_signal.notna()),
        ("known_by_current_cutoff", data.normalized_position_flow.notna()),
        ("clock_unverified_upper_bound_only", data.structural_full_window),
    ):
        subset = data.loc[data.date.ge(study["evaluation_start"])].copy()
        subset["source_available"] = usable.loc[subset.index]
        subset["complete_predictor_inputs"] = (
            usable.loc[subset.index] & complete_controls.loc[subset.index]
        )
        for year, group in subset.groupby(subset.date.dt.year):
            counts = group.groupby("date").complete_predictor_inputs.sum()
            coverage.append(
                {
                    "information_set": name,
                    "year": year,
                    "product_dates": int(group.source_available.sum()),
                    "days_with_eight_complete_inputs": int(counts.ge(study["min_assets"]).sum()),
                    "predictive_statistics_computed": name != "clock_unverified_upper_bound_only",
                }
            )
    coverage = pd.DataFrame(coverage)
    annual = (
        evaluation.assign(year=evaluation.date.dt.year)
        .groupby("year")
        .agg(days=("partial_rank_ic", "count"), mean_partial_rank_ic=("partial_rank_ic", "mean"))
        .reset_index()
    )
    late = (
        qualified.loc[
            qualified.status.eq("published_after_cutoff"), ["date", "available_at", "signal_cutoff"]
        ]
        .drop_duplicates()
        .sort_values("date")
    )
    baseline_summary = registry.loc[
        registry.analysis.eq("source_qualified_position_activation")
    ].copy()
    output = paths["output"] / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    for name, frame in (
        ("selected_panel", paired),
        ("daily_ic", daily),
        ("summary", current),
        ("coverage", coverage),
        ("yearly_diagnostics", annual),
        ("registered_exploration", registry),
        ("late_file_dates", late),
    ):
        frame.to_csv(output / f"{name}.csv", index=False, encoding="utf-8-sig")
    run = {
        "version": __version__,
        "kind": "current_signal_information_set_replay",
        "protocol": raw,
        "protocol_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "inputs": receipts,
        "earlier_admissible_signals_preserved": int(existing.sum()),
        "newly_admissible_with_current_cutoff": int(
            paired.newly_admissible_with_current_cutoff.sum()
        ),
        "historical_pre_cutoff_versions_recovered": 0,
        "future_file_clock_overridden": False,
        "availability_clock_basis": "JSON update_date is a retrospective source-clock proxy, not certified initial publication",
        "historical_signal_time_coverage_certified": False,
        "clock_ignored_diagnostic_has_no_predictive_statistics": True,
        "known_exploration_family": len(registry),
        "full_legacy_exploration_covered": False,
        "claim_eligible": False,
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob("*.py"))
        },
    }
    (output / "run.json").write_text(
        json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    page = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>持仓历史版本与信息集复核</title><style>body{{font:16px/1.8 'Segoe UI','Microsoft YaHei',sans-serif;color:#173149;background:#f2f6f9;max-width:1250px;margin:35px auto;padding:24px}}section{{background:#fff;border-radius:12px;padding:24px;margin:24px 0}}td,th{{padding:10px;border-bottom:1px solid #dce5ec}}table{{font-size:13px;border-collapse:collapse;white-space:nowrap}}.scroll{{overflow:auto}}.notice{{background:#fff1df;padding:20px}}a{{color:#147d92}}</style><h1>历史版本未找回，已发布历史可以用于后续信号</h1><p class="notice">原 44 日期的时钟继续保留；不把未来更新文件回填当日。新规则只允许历史窗口中所有记录在当前信号截止之前已知，仍要求同合约连续 21 个完整交易日。保留原版，另登记一次信息集比较。</p><section><h2>原规则与当前截止规则</h2>{table(baseline_summary)}{table(current)}<p>旧版全部 {int(existing.sum())} 个合格信号数值保持一致，新增 {int(paired.newly_admissible_with_current_cutoff.sum())} 个品种日期在其窗口数据标注更新时间之后才纳入。前者是单条信号一致性，增加截面品种后每日相关仍可能变化。完整已登记探索 {len(registry)} 项；历史已查看，不能称独立发现。</p></section><section><h2>为什么可以恢复后续窗口</h2><p>例如历史记录标注十二月十六日更新：十月信号不能使用该版；十二月十六日更新之后的信号，只要其二十日历史窗口所需记录均已知，就可使用它。原版要求每条历史记录在其所属日已知，更保守；当前版按信号当天的信息集核对，仍不使用未来文件。</p></section><section><h2>覆盖和年度</h2>{table(coverage)}{table(annual)}<p>忽略时钟一行仅估算证据齐全时的覆盖上限，未计算该设定的 IC、p 值或收益。不能将其数值当成可用历史或优选方案。</p></section><section><h2>可核查明细</h2><a href="selected_panel.csv">逐信号来源窗口与发布时间</a> · <a href="late_file_dates.csv">全部 44 日期</a> · <a href="registered_exploration.csv">全部登记探索</a> · <a href="run.json">冻结方案和源码校验</a><p>JSON 更新时间是本轮回溯可用性代理。别名接口对照日的 HTTP 文件修改时间还晚于 JSON 时钟；两者均不能认证首次发布或数值修订。当前接口及文件时钟仍不能认证全部历史初发版本，所有投资主张资格继续为 false。</p></section></html>"""
    (output / "index.html").write_text(page, encoding="utf-8")
    return output
