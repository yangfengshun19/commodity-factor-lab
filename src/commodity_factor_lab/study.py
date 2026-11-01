"""Controlled retrospective replication on one canonical price source.

All signals share the same causal eligibility mask, execution calendar, folds
and costs. These snapshots lack verified contract rolls/publication timestamps;
no statistical result from this command qualifies as verified investment alpha.
"""

import hashlib
import json
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .catalog import FAMILIES, SECTOR_MAP, code_from_filename
from .data import OHLC, normalize_snapshot, read_csv
from .engine import MissingHeldPrice, performance, quantile_groups, signal_weights
from .inference import by_adjust, hac_alpha, hac_mean, paired_block_interval
from .portfolio import Simulation, simulate
from .protocol import SIGNALS, StudyConfig


def canonical_data(config: StudyConfig) -> tuple[pd.DataFrame, list[dict], list[dict]]:
    frames, quality, inputs = [], [], []
    for path in sorted((config.data / "trend").glob("*.csv")):
        inputs.append(
            {
                "file": path.relative_to(config.data).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
        row = {"file": path.name, "status": "ok", "source": "trend_snapshot"}
        try:
            code = code_from_filename(path.name)
            row["code"] = code
            if SECTOR_MAP[code] == "financial":
                row["status"] = "excluded_financial"
                quality.append(row)
                continue
            raw = read_csv(path)
            frame, audit = normalize_snapshot(raw, ())
            row.update(audit)
            if "volume" in raw:
                dates = pd.to_datetime(raw["date"], errors="coerce")
                volume = pd.Series(
                    pd.to_numeric(raw["volume"], errors="coerce").to_numpy(), index=dates
                )
                frame["volume"] = frame["date"].map(volume)
            else:
                frame["volume"] = np.nan
            frame = frame.loc[frame["date"] <= config.end].copy()
            frame["code"] = code
            frame["sector"] = SECTOR_MAP[code]
            row["rows"] = len(frame)
            row["missing_volume_rows"] = int(frame["volume"].isna().sum())
            frames.append(frame)
        except (ValueError, KeyError, pd.errors.ParserError) as error:
            row.update(status="rejected", error=str(error))
        quality.append(row)
    if not frames:
        raise ValueError("No canonical trend OHLC files. Run demo first or import local snapshots.")
    panel = pd.concat(frames, ignore_index=True)
    if panel.duplicated(["date", "code"]).any():
        raise ValueError("Duplicate canonical date/code keys")
    return panel.sort_values(["date", "code"]), quality, inputs


def source_comparison(panel: pd.DataFrame, root: Path) -> pd.DataFrame:
    """Audit price agreement; never repair one source with another source's price."""
    canonical = panel.set_index(["date", "code"])[OHLC]
    records = []
    for family in FAMILIES:
        if family == "trend":
            continue
        for path in sorted((root / family).glob("*.csv")):
            record = {
                "family": family,
                "file": path.name,
                "overlap_rows": 0,
                "disagree_rows": 0,
                "status": "no_overlap",
            }
            try:
                code = code_from_filename(path.name)
                frame, _ = normalize_snapshot(read_csv(path), ())
                frame["code"] = code
                other = frame.set_index(["date", "code"])[OHLC]
                keys = canonical.index.intersection(other.index)
                if len(keys):
                    a, b = canonical.loc[keys].to_numpy(), other.loc[keys].to_numpy()
                    comparable = np.isfinite(a) & np.isfinite(b)
                    disagreement = (comparable & ~np.isclose(a, b, rtol=1e-6, atol=1e-6)).any(
                        axis=1
                    )
                    record.update(
                        overlap_rows=int(comparable.any(axis=1).sum()),
                        disagree_rows=int(disagreement.sum()),
                        status="compared",
                    )
            except (ValueError, KeyError, pd.errors.ParserError) as error:
                record.update(status="rejected", error=str(error))
            records.append(record)
    return pd.DataFrame(
        records, columns=["family", "file", "overlap_rows", "disagree_rows", "status", "error"]
    )


def build_signals(
    panel: pd.DataFrame, config: StudyConfig
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    calendar = pd.DatetimeIndex(sorted(panel["date"].unique()))
    opens = panel.pivot(index="date", columns="code", values="open").reindex(calendar)
    output = {name: pd.DataFrame(np.nan, index=calendar, columns=opens.columns) for name in SIGNALS}
    eligible = pd.DataFrame(False, index=calendar, columns=opens.columns)
    for code, subset in panel.groupby("code"):
        bars = subset.set_index("date").reindex(calendar)
        valid = bars[OHLC].notna().all(axis=1) & bars[OHLC].gt(0).all(axis=1)
        valid &= bars["high"].ge(bars[["open", "low", "close"]].max(axis=1))
        valid &= bars["low"].le(bars[["open", "high", "close"]].min(axis=1))
        close = bars["close"].where(valid)
        output["momentum_126d"][code] = close / close.shift(126) - 1
        output["trend_10_60"][code] = close.rolling(10).mean() / close.rolling(60).mean() - 1
        output["reversal_5d"][code] = -(close / close.shift(5) - 1)
        history = valid.rolling(config.history_sessions).sum().eq(config.history_sessions)
        liquid = pd.Series(True, index=calendar)
        if config.min_median_volume > 0:
            volume = pd.to_numeric(bars["volume"], errors="coerce").where(bars["volume"] >= 0)
            liquid = volume.rolling(20, min_periods=20).median().ge(config.min_median_volume)
            liquid &= volume.gt(0)
        eligible[code] = history & liquid
    # Joint availability is decided from information already observed at the signal date.
    for values in output.values():
        eligible &= values.notna()
    for name in output:
        output[name] = output[name].where(eligible)
    return output, opens, eligible


def walk_forward_folds(calendar: pd.DatetimeIndex, config: StudyConfig) -> pd.DataFrame:
    rows = []
    horizon = 1 + config.publication_lag_sessions + config.rebalance_sessions
    for number, first in enumerate(
        range(config.training_sessions, len(calendar) - 1, config.test_sessions)
    ):
        last = min(first + config.test_sessions - 1, len(calendar) - 2)
        train_first = max(0, first - config.training_sessions)
        train_last = first - horizon - 1
        if train_last < train_first:
            continue
        rows.append(
            {
                "fold": number,
                "train_first_pos": train_first,
                "train_last_pos": train_last,
                "test_first_pos": first,
                "test_last_pos": last,
                "train_start": calendar[train_first],
                "train_last_signal": calendar[train_last],
                "train_last_label_end": calendar[train_last + horizon],
                "test_entry_start": calendar[first],
                "test_entry_end": calendar[last],
                "purged_signal_sessions": horizon,
            }
        )
    return pd.DataFrame(rows)


def rank_ic(values: pd.DataFrame, labels: pd.DataFrame, min_assets: int) -> pd.Series:
    paired = values.notna() & labels.notna()
    a = values.where(paired).rank(axis=1, method="average")
    b = labels.where(paired).rank(axis=1, method="average")
    a, b = a.sub(a.mean(axis=1), axis=0), b.sub(b.mean(axis=1), axis=0)
    denominator = np.sqrt(a.pow(2).sum(axis=1) * b.pow(2).sum(axis=1)).replace(0, np.nan)
    return ((a * b).sum(axis=1) / denominator).where(paired.sum(axis=1) >= min_assets)


def aligned_targets(
    plan: pd.DataFrame, opens: pd.DataFrame, steps: int, long_only: bool = False
) -> pd.DataFrame:
    entry = plan.shift(steps - 1).fillna(0).where(opens.notna(), 0)
    if long_only:
        entry = entry.div(entry.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    else:
        positive, negative = entry.clip(lower=0), -entry.clip(upper=0)
        both = positive.sum(axis=1).gt(0) & negative.sum(axis=1).gt(0)
        entry = 0.5 * positive.div(positive.sum(axis=1).replace(0, np.nan), axis=0).fillna(
            0
        ) - 0.5 * negative.div(negative.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
        entry.loc[~both] = 0
    return entry.shift(1).fillna(0)


def run_study(config: StudyConfig) -> Path:
    from .study_report import write_study_report

    panel, quality, inputs = canonical_data(config)
    snapshot_metadata = None
    if (config.data / "snapshot.json").exists():
        snapshot_metadata = json.loads((config.data / "snapshot.json").read_text(encoding="utf-8"))
        expected_hashes = {
            r["normalized_file"]: r["normalized_sha256"]
            for r in snapshot_metadata["records"]
            if r["status"] == "ok"
        }
        actual_hashes = {item["file"]: item["sha256"] for item in inputs}
        if expected_hashes != actual_hashes:
            raise ValueError("Downloaded snapshot no longer matches its recorded normalized hashes")
    signals, opens, eligible = build_signals(panel, config)
    # Retain pre-start price history for causal warm-up, restrict only the experiment calendar.
    calendar = opens.index[opens.index >= pd.Timestamp(config.start)]
    opens = opens.reindex(calendar)
    eligible = eligible.reindex(calendar)
    signals = {name: frame.reindex(calendar) for name, frame in signals.items()}
    folds = walk_forward_folds(calendar, config)
    if folds.empty:
        raise ValueError("Not enough sessions for the declared walk-forward protocol")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = config.output / stamp
    directory.mkdir(parents=True, exist_ok=False)
    for name in ("daily", "weights", "ic"):
        (directory / name).mkdir()
    synthetic = (config.data / "SYNTHETIC_DATA.json").exists()
    comparisons = source_comparison(panel, config.data)
    comparisons.to_csv(directory / "source_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(quality).to_csv(directory / "quality.csv", index=False, encoding="utf-8-sig")
    folds.to_csv(directory / "folds.csv", index=False)
    breadth = pd.DataFrame({"eligible_assets": eligible.sum(axis=1)})
    breadth.to_csv(directory / "universe.csv", index_label="signal_date")
    steps = 2 + config.publication_lag_sessions
    first_entry = int(folds.iloc[0]["test_first_pos"])
    positions = np.arange(len(calendar))
    evaluation = positions >= first_entry + 1
    dates = calendar[evaluation]
    # Schedule is attached to entry opens; the ledger is indexed by ending opens.
    rebalanced = pd.Series(
        (positions - first_entry - 1) % config.rebalance_sessions == 0, index=calendar
    ).loc[dates]
    entry_dates = pd.Series(calendar, index=calendar).shift(1).loc[dates]
    scenarios, summaries, fold_results, paired_intervals = [], [], [], []
    primary_ledgers = {}
    for sector in config.sectors:
        codes = [
            code
            for code in opens.columns
            if sector == "all_commodities" or SECTOR_MAP[code] == sector
        ]
        prices = opens[codes]
        asset_returns = prices / prices.shift(1) - 1
        # Five-session predictive label matches the intended rebalancing horizon.
        labels = (
            prices.shift(-(steps - 1 + config.rebalance_sessions)) / prices.shift(-(steps - 1)) - 1
        )
        sector_eligible = eligible[codes]
        breadth_ok = sector_eligible.sum(axis=1).ge(config.min_assets)
        ew = sector_eligible.astype(float).where(breadth_ok, 0)
        ew = ew.div(ew.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
        plans = {"equal_weight_long": ew}
        daily_ic = {}
        for name, values in signals.items():
            values = values[codes]
            plans[name] = signal_weights(
                quantile_groups(values, config.groups, config.min_assets), config.groups
            )
            daily_ic[name] = rank_ic(values, labels, config.min_assets)
            daily_ic[name].to_csv(
                directory / "ic" / f"{sector}__{name}.csv", index_label="signal_date"
            )
        results: dict[tuple[str, float], Simulation] = {}
        errors = {}
        for name, plan in plans.items():
            target = aligned_targets(
                plan, prices, steps, long_only=name == "equal_weight_long"
            ).loc[dates]
            for bps in config.cost_scenarios_bps:
                key = (name, bps)
                try:
                    result = simulate(
                        target, asset_returns.loc[dates], rebalanced, bps / 10000, liquidate=True
                    )
                    result.daily["entry_date"] = entry_dates
                    results[key] = result
                    record = {
                        "sector": sector,
                        "signal": name,
                        "cost_bps": bps,
                        "status": "ok",
                        "days": len(result.daily),
                        **performance(result.daily["net_return"]),
                    }
                    if not result.daily["gross_exposure"].gt(0).any():
                        record["status"] = "no_eligible_portfolio"
                    if bps == config.primary_cost_bps:
                        stem = f"{sector}__{name}"
                        result.daily.to_csv(directory / "daily" / f"{stem}.csv", index_label="date")
                        result.weights.to_csv(
                            directory / "weights" / f"{stem}.csv",
                            index_label="date",
                            float_format="%.10g",
                        )
                        primary_ledgers[(sector, name)] = result.daily
                except MissingHeldPrice as error:
                    errors[key] = str(error)
                    record = {
                        "sector": sector,
                        "signal": name,
                        "cost_bps": bps,
                        "status": "missing_held_price",
                        "days": 0,
                        "error": str(error),
                    }
                scenarios.append(record)
        for name in SIGNALS:
            primary = results.get((name, config.primary_cost_bps))
            ic = daily_ic[name]
            # OOS IC is assigned by its execution entry. Avoid including a label
            # formed after the final possible holding horizon.
            future_entry = pd.Series(calendar, index=calendar).shift(-(steps - 1))
            ic_oos = ic.where(future_entry >= calendar[first_entry])
            ic_stats = hac_mean(ic_oos, config.hac_lags, config.min_ic_obs)
            row = {
                "sector": sector,
                "signal": name,
                "status": "missing_held_price" if primary is None else "ok",
                "days": 0,
                "ic_nobs": ic_stats["nobs"],
                "rank_ic": ic_stats["mean"],
                "ic_hac_p": ic_stats["p"],
                "mean_hac_p": np.nan,
                "claim_eligible": False,
                "evidence_status": "synthetic_demo" if synthetic else "unverified_continuous_proxy",
                "error": errors.get((name, config.primary_cost_bps), ""),
            }
            if primary is not None:
                daily = primary.daily
                active = daily["gross_exposure"].gt(0)
                if not active.any():
                    row["status"] = "no_eligible_portfolio"
                row.update(
                    days=len(daily),
                    invested_days=int(active.sum()),
                    **performance(daily["net_return"]),
                )
                mean_stats = hac_mean(
                    daily["net_return"], config.hac_lags, config.min_inference_obs
                )
                row.update(
                    mean_hac_p=mean_stats["p"],
                    mean_hac_t=mean_stats["t"],
                    inference_status=mean_stats["status"],
                    annual_turnover=float(daily["turnover"].mean() * 252),
                )
                reference_name = "equal_weight_long" if name == "momentum_126d" else "momentum_126d"
                reference = results.get((reference_name, config.primary_cost_bps))
                row["comparison_benchmark"] = reference_name
                if reference is not None:
                    for block in (10, 20, 40):
                        ci = paired_block_interval(
                            daily["net_return"],
                            reference.daily["net_return"],
                            block,
                            config.bootstrap_repetitions,
                            min_obs=config.min_inference_obs,
                        )
                        paired_intervals.append(
                            {
                                "sector": sector,
                                "signal": name,
                                "benchmark": reference_name,
                                "block_sessions": block,
                                **ci,
                            }
                        )
                benchmark_names = (
                    ["equal_weight_long"]
                    if name == "momentum_126d"
                    else ["equal_weight_long", "momentum_126d"]
                )
                if all(
                    (benchmark, config.primary_cost_bps) in results for benchmark in benchmark_names
                ):
                    benchmark_returns = pd.DataFrame(
                        {
                            benchmark: results[(benchmark, config.primary_cost_bps)].daily[
                                "net_return"
                            ]
                            for benchmark in benchmark_names
                        }
                    )
                    alpha = hac_alpha(
                        daily["net_return"],
                        benchmark_returns,
                        config.hac_lags,
                        config.min_inference_obs,
                    )
                    row.update(
                        benchmark_intercept=alpha["annual_intercept"],
                        benchmark_intercept_p=alpha["p"],
                        benchmark_model_status=alpha["status"],
                    )
                for _, fold in folds.iterrows():
                    segment = daily.loc[
                        daily["entry_date"].between(
                            fold["test_entry_start"], fold["test_entry_end"]
                        )
                    ]
                    train_ic = ic.loc[fold["train_start"] : fold["train_last_signal"]]
                    fold_results.append(
                        {
                            "sector": sector,
                            "signal": name,
                            "fold": fold["fold"],
                            "days": len(segment),
                            "train_ic_days": int(train_ic.notna().sum()),
                            "train_ic_mean": float(train_ic.mean()),
                            **performance(segment["net_return"]),
                        }
                    )
            summaries.append(row)
        print(f"study {sector}: {len(codes)} source products; fixed signals evaluated", flush=True)
    summary = pd.DataFrame(summaries)
    # Includes every planned signal x sector, even untestable hypotheses.
    summary["mean_by_q"] = by_adjust(summary["mean_hac_p"].tolist())
    summary["ic_by_q"] = by_adjust(summary["ic_hac_p"].tolist())
    summary["positive_mean_candidate"] = (summary["mean_by_q"] <= 0.05) & (
        summary.get("annual_return", np.nan) > 0
    )
    summary.to_csv(directory / "summary.csv", index=False, encoding="utf-8-sig")
    cost_table = pd.DataFrame(scenarios)
    cost_table.to_csv(directory / "cost_sensitivity.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fold_results).to_csv(
        directory / "fold_results.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(paired_intervals).to_csv(
        directory / "paired_intervals.csv", index=False, encoding="utf-8-sig"
    )
    protocol = asdict(config)
    protocol.update(data=str(config.data), output=str(config.output))
    limitations = [
        "historical_prices_previously_examined",
        "contract_rolls_not_verified",
        "point_in_time_source_timestamps_missing",
        "official_exchange_calendar_missing",
        "no_margin_price_limit_or_capacity_model",
    ]
    run = {
        "version": __version__,
        "created_utc": stamp,
        "python": platform.python_version(),
        "protocol": protocol,
        "signals": list(SIGNALS),
        "inputs": inputs,
        "source_snapshot": snapshot_metadata,
        "snapshot_manifest_sha256": hashlib.sha256(
            (config.data / "snapshot.json").read_bytes()
        ).hexdigest()
        if snapshot_metadata is not None
        else None,
        "planned_primary_hypotheses": len(SIGNALS) * len(config.sectors),
        "models_fitted": False,
        "assessment": "retrospective_walk_forward_fixed_rules",
        "confirmed_investment_findings": 0,
        "synthetic": synthetic,
        "limitations": limitations,
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob("*.py"))
        },
    }
    run["protocol_sha256"] = hashlib.sha256(
        json.dumps(protocol, sort_keys=True).encode()
    ).hexdigest()
    (directory / "run.json").write_text(
        json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_study_report(directory, summary, cost_table, folds, primary_ledgers, run)
    return directory
