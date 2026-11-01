"""Fixed, retrospective contract-characteristic research. No assumed trade at settlement."""

import hashlib
import json
import platform
import tomllib
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .curve_factors import CurveRules, contract_curves
from .factor_definitions import FACTORS, definitions
from .fundamentals import seasonal_inventory
from .inference import by_adjust, hac_mean

HYPOTHESES = {item.name: item.controls for item in FACTORS if item.role == "candidate"}


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_receipt(path: Path) -> dict:
    receipt_path = path.parent / "manifest.json"
    if not receipt_path.exists():
        raise ValueError(f"Missing input manifest for {path.name}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("normalized_sha256") != file_hash(path):
        raise ValueError(f"Input no longer matches its manifest: {path.name}")
    return {
        "file": str(path.resolve()),
        "sha256": file_hash(path),
        "manifest_sha256": file_hash(receipt_path),
        "receipt": receipt,
    }


def future_contract_labels(
    curves: pd.DataFrame,
    bars: pd.DataFrame,
    calendar: pd.DataFrame,
    lag: int = 1,
    horizon: int = 20,
) -> pd.DataFrame:
    """A fixed contract's future settlement change, not a trade execution return."""
    if lag < 1 or horizon < 1:
        raise ValueError("Prediction targets must begin after the signal")
    dates = pd.DatetimeIndex(pd.to_datetime(calendar.date))
    if not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError("Prediction calendar must be sorted and unique")
    data = bars.copy()
    data["date"] = pd.to_datetime(data.date)
    if data.duplicated(["date", "contract"]).any():
        raise ValueError("Duplicate contract label marks")
    marks = data.set_index(["date", "contract"]).settle
    positions = {day: i for i, day in enumerate(dates)}
    rows = []
    for row in curves.itertuples():
        i = positions[row.date]
        item = {
            "date": row.date,
            "code": row.code,
            "target": np.nan,
            "entry_date": pd.NaT,
            "exit_date": pd.NaT,
            "status": "no_known_curve",
        }
        contract = getattr(row, "front_contract", None)
        if not isinstance(contract, str):
            rows.append(item)
            continue
        item["contract"] = contract
        if i + lag + horizon >= len(dates):
            item["status"] = "outside_future_calendar"
        else:
            start, end = dates[i + lag], dates[i + lag + horizon]
            item.update(entry_date=start, exit_date=end)
            if end >= row.front_expiry:
                item["status"] = "expiry_before_target_end"
            elif (start, contract) not in marks.index or (end, contract) not in marks.index:
                item["status"] = "missing_same_contract_mark"
            else:
                first, last = marks.loc[(start, contract)], marks.loc[(end, contract)]
                if not np.isfinite([first, last]).all() or min(first, last) <= 0:
                    item["status"] = "invalid_target_mark"
                else:
                    item.update(target=float(last / first - 1), status="ok")
        rows.append(item)
    return pd.DataFrame(rows)


def partial_rank_ic(
    frame: pd.DataFrame, signal: str, controls: tuple[str, ...], min_assets: int = 6
) -> dict:
    subset = frame[[signal, "target", *controls]].replace([np.inf, -np.inf], np.nan).dropna()
    result = {
        "rank_ic": np.nan,
        "partial_rank_ic": np.nan,
        "assets": len(subset),
        "status": "insufficient_cross_section",
    }
    if len(subset) < max(min_assets, len(controls) + 3):
        return result
    ranked = subset.rank(method="average")
    x, y = ranked[signal].to_numpy(), ranked.target.to_numpy()
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        result["status"] = "constant_rank"
        return result
    result["rank_ic"] = float(np.corrcoef(x, y)[0, 1])
    design = np.column_stack([np.ones(len(subset)), ranked[list(controls)].to_numpy()])
    if np.linalg.matrix_rank(design) < design.shape[1]:
        result["status"] = "rank_deficient_controls"
        return result
    x = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    y = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    if min(np.linalg.norm(x), np.linalg.norm(y)) < 1e-10:
        result["status"] = "no_incremental_rank_variation"
        return result
    result.update(
        partial_rank_ic=float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y))), status="ok"
    )
    return result


def audit_contract_data(bars: pd.DataFrame, codes: list[str]) -> pd.DataFrame:
    if bars.duplicated(["date", "contract"]).any() or set(bars.code) != set(codes):
        raise ValueError("Duplicate marks or missing/unexpected pilot products")
    records = []
    for code, group in bars.groupby("code"):
        price_cols = [c for c in ("open", "high", "low", "close", "settle") if c in group]
        values = group[price_cols].to_numpy(dtype=float)
        bad = ~np.isfinite(values).all(axis=1) | (values <= 0).any(axis=1)
        range_bad = group.high.lt(group[["open", "close", "low"]].max(axis=1)) | group.low.gt(
            group[["open", "close", "high"]].min(axis=1)
        )
        records.append(
            {
                "code": code,
                "rows": len(group),
                "sessions": group.date.nunique(),
                "contracts": group.contract.nunique(),
                "start": str(group.date.min().date()),
                "end": str(group.date.max().date()),
                "invalid_price_rows": int(bad.sum()),
                "inconsistent_ohlc_rows": int(range_bad.sum()),
                "zero_volume_rows": int(group.volume.eq(0).sum()),
            }
        )
    return pd.DataFrame(records)


def run_contract_research(config_path: Path, without_warrants: bool = False) -> Path:
    from .contract_report import write_contract_report

    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    if set(raw) != {"paths", "study", "curve"}:
        raise ValueError("Expected paths/study/curve protocol sections")
    base, study = config_path.resolve().parent.parent, raw["study"]
    paths = {k: (base / v).resolve() for k, v in raw["paths"].items()}
    rules = CurveRules(**raw["curve"])
    if (
        study["min_assets"] < 6
        or study["hac_lags"] < study["holding_sessions"]
        or study["min_inference_obs"] < 60
    ):
        raise ValueError("Insufficient cross-section, HAC horizon or inference history")
    if any(
        paths["output"].is_relative_to(paths[k].parent) for k in ("bars", "metadata", "warrants")
    ):
        raise ValueError("Research output must be outside source directories")
    inputs = [verify_receipt(paths["bars"]), verify_receipt(paths["metadata"])]
    bars = pd.read_csv(paths["bars"], parse_dates=["date"])
    bars = bars.loc[
        bars.code.isin(study["codes"]) & bars.date.between(study["start"], study["end"])
    ].copy()
    quality = audit_contract_data(bars, study["codes"])
    metadata = pd.read_csv(paths["metadata"])
    unknown_clocks = int(metadata.metadata_available_at.isna().sum())
    metadata = metadata.loc[metadata.metadata_available_at.notna()].copy()
    dates = pd.DatetimeIndex(sorted(bars.date.unique()))
    calendar = pd.DataFrame(
        {
            "date": dates,
            "signal_cutoff": [
                f"{d:%Y-%m-%d}T{study['information_cutoff_local']}+08:00" for d in dates
            ],
        }
    )
    print("Building as-of contract curves and same-contract historical return chains", flush=True)
    curves = contract_curves(
        bars, metadata, calendar, rules, retrospective=study["retrospective_bar_availability"]
    )
    warrants_status = "explicitly_omitted" if without_warrants else "missing_source"
    if not without_warrants and paths["warrants"].exists():
        inputs.append(verify_receipt(paths["warrants"]))
        observations = pd.read_csv(paths["warrants"])
        observations = observations.loc[observations.code.isin(study["codes"])].copy()
        print("Calculating publication-aware seasonal warrant percentiles", flush=True)
        stock = seasonal_inventory(
            observations,
            calendar,
            min_seasons=study["inventory_min_seasons"],
            min_observations=study["inventory_min_observations"],
        )
        curves = curves.merge(
            stock,
            on=["date", "code"],
            how="left",
            suffixes=("", "_inventory"),
            validate="one_to_one",
        )
        warrants_status = "loaded_registered_warrants"
    else:
        curves["inventory_scarcity"] = np.nan
    print(
        "Computing fixed same-contract prediction targets and incremental diagnostics", flush=True
    )
    labels = future_contract_labels(
        curves, bars, calendar, study["entry_lag_sessions"], study["holding_sessions"]
    )
    data = curves.merge(
        labels[["date", "code", "target"]], on=["date", "code"], validate="one_to_one"
    )
    base_mask = data[["carry", "momentum_126d", "basis_momentum"]].notna().all(axis=1)
    diagnostics = []
    for date, group in data.groupby("date"):
        common = group.loc[base_mask.loc[group.index]]
        for name, controls in HYPOTHESES.items():
            diagnostics.append(
                {
                    "date": date,
                    "signal": name,
                    **partial_rank_ic(common, name, controls, study["min_assets"]),
                }
            )
    daily = pd.DataFrame(diagnostics)
    summary = []
    for name, controls in HYPOTHESES.items():
        series = daily.loc[daily.signal.eq(name)].set_index("date")
        series = series.loc[series.index >= pd.Timestamp(study["evaluation_start"])]
        stats = hac_mean(series.partial_rank_ic, study["hac_lags"], study["min_inference_obs"])
        summary.append(
            {
                "signal": name,
                "controls": ";".join(controls),
                "days": stats["nobs"],
                "mean_partial_rank_ic": stats["mean"],
                "hac_t": stats["t"],
                "hac_p": stats["p"],
                "inference_status": stats["status"],
                "mean_rank_ic": series.rank_ic.mean(),
                "claim_eligible": False,
            }
        )
    summary = pd.DataFrame(summary)
    summary["by_q"] = by_adjust(summary.hac_p.to_numpy())
    summary["positive_predictive_candidate"] = summary.by_q.le(
        0.05
    ) & summary.mean_partial_rank_ic.gt(0)
    evaluation = daily.loc[daily.date.ge(pd.Timestamp(study["evaluation_start"]))].copy()
    evaluation["year"] = evaluation.date.dt.year
    annual = (
        evaluation.groupby(["signal", "year"])
        .agg(days=("partial_rank_ic", "count"), mean_partial_rank_ic=("partial_rank_ic", "mean"))
        .reset_index()
    )
    # Descriptive stability on the same already-valid primary dates. No new
    # p-values or best-product selection from these post-protocol diagnostics.
    print("Checking descriptive year and leave-one-product stability", flush=True)
    leave_one_out = []
    for signal, controls in HYPOTHESES.items():
        valid_dates = set(
            evaluation.loc[
                evaluation.signal.eq(signal) & evaluation.partial_rank_ic.notna(), "date"
            ]
        )
        scores = {code: [] for code in study["codes"]}
        for date, group in data.loc[data.date.isin(valid_dates)].groupby("date"):
            common = group.loc[base_mask.loc[group.index]]
            for omitted in study["codes"]:
                value = partial_rank_ic(
                    common.loc[common.code.ne(omitted)], signal, controls, study["min_assets"] - 1
                )["partial_rank_ic"]
                scores[omitted].append(value)
        for omitted, values in scores.items():
            finite = np.asarray(values, dtype=float)
            finite = finite[np.isfinite(finite)]
            leave_one_out.append(
                {
                    "signal": signal,
                    "omitted_product": omitted,
                    "days": len(finite),
                    "mean_partial_rank_ic": float(finite.mean()) if len(finite) else np.nan,
                    "exploratory_stability_only": True,
                }
            )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = paths["output"] / stamp
    output.mkdir(parents=True, exist_ok=False)
    for name, frame in (
        ("curves", curves),
        ("labels", labels),
        ("quality", quality),
        ("daily_ic", daily),
        ("summary", summary),
        ("calendar_assumption", calendar),
        ("yearly_diagnostics", annual),
        ("leave_one_product_out", pd.DataFrame(leave_one_out)),
    ):
        frame.to_csv(output / f"{name}.csv", index=False, encoding="utf-8-sig")
    status = (
        curves.groupby("code")
        .agg(
            curve_days=("carry", "count"),
            momentum_days=("momentum_126d", "count"),
            basis_momentum_days=("basis_momentum", "count"),
            inventory_days=("inventory_scarcity", "count"),
        )
        .reset_index()
    )
    status.to_csv(output / "coverage.csv", index=False)
    collection = []
    for item in inputs:
        receipt = item["receipt"]
        records = receipt.get("records", [])
        planned = len(receipt.get("planned_dates", [])) or len(records)
        collection.append(
            {
                "input": Path(item["file"]).stem,
                "source_kind": receipt["kind"],
                "input_archives": len(receipt.get("inputs", [])),
                "planned_requests": planned,
                "attempted_requests": len(records),
                "successful_requests": sum(r["status"] == "ok" for r in records),
                "failed_requests": sum(r["status"] != "ok" for r in records),
                "unattempted_requests": max(0, planned - len(records)),
                "point_in_time_certified": receipt.get("point_in_time_certified", False),
            }
        )
    pd.DataFrame(collection).to_csv(output / "source_collection.csv", index=False)
    label_counts = labels.groupby(["code", "status"]).size().rename("rows").reset_index()
    label_counts.to_csv(output / "label_status.csv", index=False)
    run = {
        "version": __version__,
        "python": platform.python_version(),
        "created_utc": stamp,
        "synthetic": bool(inputs[0]["receipt"].get("synthetic", False)),
        "kind": "retrospective_contract_characteristic_diagnostics",
        "protocol": raw,
        "protocol_sha256": file_hash(config_path),
        "inputs": inputs,
        "source_collection": collection,
        "rules": asdict(rules),
        "planned_primary_hypotheses": len(HYPOTHESES),
        "warrants_status": warrants_status,
        "unknown_metadata_clocks_excluded": unknown_clocks,
        "executed_investment_backtests": 0,
        "confirmed_investment_findings": 0,
        "protocol_revision_note": "30-day maturity-buffer draft retained; revised to 60 calendar days for horizon coverage. Protocol exploration is not covered by current three-test BY correction.",
        "limitations": [
            "年包是后来导出的历史版本，未认证历史修订及当时可用性。",
            "日终信息截止时间为回溯假设；逐月合约参数可能漏掉月中修订。",
            "仓单只含仓库注册仓单，排除厂库仓单；来源缺口和过期观察不填补。",
            f"本轮 {len(study['codes'])} 个上期所品种不代表整个商品市场，相关性与小截面限制统计稳定性。",
            "预测标签和部分秩相关不等于可成交收益、投资 Alpha 或因果证据。",
            "真实交易还需历史开盘日历、费用、保证金和成交条件的时点证据。",
            "协议修订和此前探索均已发生，本轮校正不能恢复为独立确认性研究。",
        ],
        "source_sha256": {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
    }
    (output / "run.json").write_text(
        json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "factor_definitions.json").write_text(
        json.dumps(definitions(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_contract_report(output, summary, status, daily, quality, label_counts, run)
    return output
