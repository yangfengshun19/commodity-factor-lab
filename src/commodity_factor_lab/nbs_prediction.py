"""A fixed, descriptive event forecast comparison. No investment or significance claims."""

import json
import re
import tomllib
from datetime import datetime, timezone
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .contract_report import table
from .curve_factors import utc_column
from .inference import by_adjust
from .nbs_ingest import digest

CODES = ("al", "cu", "hc", "rb", "ru", "sp", "zn")
CONTROLS = ("futures_since_previous_release", "carry", "momentum_126d", "basis_momentum")
SIGNAL = "reference_log_change"
FEATURES = (*CONTROLS, SIGNAL)
MODELS = ("futures", "augmented", "training_mean", "zero")


def period_number(period: str) -> int:
    match = re.fullmatch(r"(\d{4})-(0[1-9]|1[0-2])-(early|middle|late)", period)
    if match is None:
        raise ValueError("Invalid source period")
    year, month, part = match.groups()
    return int(year) * 36 + (int(month) - 1) * 3 + ("early", "middle", "late").index(part)


def prediction_plan(config: Path) -> tuple[dict, dict]:
    plan = tomllib.loads(config.read_text(encoding="utf-8"))
    root = config.resolve().parent.parent
    paths = {name: (root / value).resolve() for name, value in plan["paths"].items()}
    if any(not path.is_relative_to(root) for path in paths.values()):
        raise ValueError("Prediction paths must stay within the workspace")
    if set(paths) != set(plan["input_hashes"]) | {"output"}:
        raise ValueError("Every prediction input needs a frozen hash")
    study, evaluation = plan["study"], plan["evaluation"]
    required = {
        "kind": "retrospective_descriptive_event_forecast",
        "codes": list(CODES),
        "forecast_start": "2023-01-01",
        "forecast_end": "2025-12-31",
        "minimum_training_events_per_code": 60,
        "model": "separate_code_expanding_ols_train_only_standardization",
        "controls": list(CONTROLS),
        "signal": SIGNAL,
        "require_adjacent_calendar_periods": True,
        "require_identical_comparability_key": True,
        "training_target_exit_strictly_before_signal_date": True,
        "entry_lag_sessions": 1,
        "holding_sessions": 20,
        "forecast_without_observed_target": True,
        "same_training_and_scoring_rows_for_all_models": True,
        "missing_values_filled": False,
        "window_or_sign_search": False,
    }
    if study != required or any(
        evaluation[name]
        for name in (
            "formal_p_value_computed",
            "independent_new_sample",
            "historical_first_vintage_certified",
            "strict_spot_basis_claim_eligible",
            "investment_claim_eligible",
        )
    ):
        raise ValueError("Keep the fixed descriptive design and claim boundaries")
    if (
        evaluation["primary"] != "one_minus_event_mean_augmented_mse_over_event_mean_futures_mse"
        or evaluation["aggregate"] != "equal_code_within_release_then_equal_release"
        or evaluation["additional_baselines"] != ["training_mean", "zero"]
        or evaluation["breakdowns"] != ["year", "code"]
        or evaluation["registered_experiments_before"] != 14
        or evaluation["registered_experiments_after"] != 15
        or evaluation["new_experiment_p_value_status"] != "unavailable_descriptive_screen"
        or not evaluation["new_experiment_counts_in_BY_family_as_p_one"]
    ):
        raise ValueError("Do not change the fixed comparison or exploration accounting")
    output = paths["output"]
    for name, expected in plan["input_hashes"].items():
        path = paths[name]
        if output == path or output.is_relative_to(path.parent):
            raise ValueError("Prediction output must be outside input directories")
        if digest(path) != expected:
            raise ValueError(f"Frozen prediction input changed: {name}")
    return plan, paths


def build_event_panel(
    references: pd.DataFrame,
    curves: pd.DataFrame,
    labels: pd.DataFrame,
    bars: pd.DataFrame,
    calendar: pd.DataFrame,
) -> pd.DataFrame:
    """Keep every source event, distinguish input exclusions from unavailable future labels."""
    refs, futures, targets, marks, sessions = (
        frame.copy() for frame in (references, curves, labels, bars, calendar)
    )
    for frame in (futures, targets, marks, sessions):
        frame["date"] = pd.to_datetime(frame.date)
    for col in ("entry_date", "exit_date"):
        targets[col] = pd.to_datetime(targets[col])
    futures["front_expiry"] = pd.to_datetime(futures.front_expiry)
    if (
        not sessions.date.is_unique
        or not sessions.date.is_monotonic_increasing
        or futures.duplicated(["date", "code"]).any()
        or targets.duplicated(["date", "code"]).any()
        or marks.duplicated(["date", "contract"]).any()
        or refs.duplicated(["period", "related_code"]).any()
        or not refs.related_code.isin(CODES).all()
        or not refs.reference_mapping_usable.eq(True).all()
    ):
        raise ValueError("Ambiguous event, curve, target, or contract keys")
    if marks.groupby("contract").code.nunique().gt(1).any():
        raise ValueError("A contract must retain its product identity")
    cutoffs = pd.DatetimeIndex(utc_column(sessions.signal_cutoff))
    if not cutoffs.is_unique or not cutoffs.is_monotonic_increasing:
        raise ValueError("Signal cutoffs must be strictly increasing")
    refs["date"] = pd.to_datetime(refs.first_signal_date)
    refs["source_declared_published_at"] = utc_column(refs.source_declared_published_at)
    refs["period_number"] = refs.period.map(period_number)
    refs["code"] = refs.related_code
    refs["reference_price"] = pd.to_numeric(refs.reference_price, errors="raise")
    if (
        not np.isfinite(refs.reference_price).all()
        or refs.reference_price.le(0).any()
        or refs.comparability_key.isna().any()
        or refs.comparability_key.eq("").any()
    ):
        raise ValueError("Invalid reference prices or comparability identities")
    for event in refs.itertuples():
        index = cutoffs.searchsorted(event.source_declared_published_at)
        expected = sessions.iloc[index].date if index < len(sessions) else pd.NaT
        if not ((pd.isna(expected) and pd.isna(event.date)) or expected == event.date):
            raise ValueError("Reference event is not the first cutoff after declared publication")
        if index < len(sessions):
            declared = pd.Timestamp(event.first_signal_cutoff)
            if declared.tzinfo is None or declared != cutoffs[index]:
                raise ValueError("Reference cutoff differs from the fixed calendar")
    if refs.loc[refs.date.notna()].duplicated(["date", "code"]).any():
        raise ValueError("Multiple releases for one code/cutoff need a separate event protocol")
    if refs.groupby("period").source_declared_published_at.nunique().gt(1).any():
        raise ValueError("Conflicting publication clocks for one release")
    previous = refs.sort_values(["code", "period_number"]).groupby("code")
    for col in ("period", "period_number", "date", "comparability_key", "reference_price"):
        refs[f"previous_{col}"] = previous[col].shift(1)
    refs[SIGNAL] = np.nan
    refs["input_status"] = "ok"
    adjacent = refs.period_number.sub(refs.previous_period_number).eq(1)
    same = refs.comparability_key.eq(refs.previous_comparability_key)
    refs.loc[~adjacent, "input_status"] = "no_adjacent_published_reference"
    refs.loc[adjacent & ~same, "input_status"] = "comparability_break"
    refs.loc[refs.date.isna(), "input_status"] = "outside_signal_calendar"
    invalid_clock = refs.date.notna() & (
        refs.previous_date.isna() | refs.previous_date.ge(refs.date)
    )
    refs.loc[refs.input_status.eq("ok") & invalid_clock, "input_status"] = (
        "previous_release_not_strictly_earlier"
    )
    eligible = refs.input_status.eq("ok")
    refs.loc[eligible, SIGNAL] = np.log(
        refs.loc[eligible, "reference_price"] / refs.loc[eligible, "previous_reference_price"]
    )
    keep = [
        "period",
        "period_number",
        "date",
        "code",
        "reference_price",
        "comparability_key",
        "source_declared_published_at",
        "first_signal_cutoff",
        "source_url",
        "source_sha256",
        "previous_period",
        "previous_date",
        "previous_reference_price",
        "previous_comparability_key",
        SIGNAL,
        "input_status",
    ]
    panel = refs[keep].merge(
        futures[
            ["date", "code", "front_contract", "front_expiry", "reference_settle", *CONTROLS[1:]]
        ],
        on=["date", "code"],
        how="left",
        validate="many_to_one",
    )
    panel = panel.merge(
        targets[["date", "code", "target", "entry_date", "exit_date", "status", "contract"]].rename(
            columns={"status": "label_status", "contract": "label_contract"}
        ),
        on=["date", "code"],
        how="left",
        validate="many_to_one",
    )
    mark_lookup = marks.set_index(["date", "contract"])
    positions = {date: index for index, date in enumerate(sessions.date)}
    dates = list(sessions.date)
    rows = []
    for item in panel.to_dict("records"):
        item["futures_since_previous_release"] = np.nan
        item["target_verified"] = False
        date, contract = item["date"], item["front_contract"]
        if pd.isna(date):
            rows.append(item)
            continue
        known = isinstance(contract, str) and bool(contract)
        expected_target, expected_status = np.nan, "no_known_curve"
        if known:
            if item["label_contract"] != contract:
                raise ValueError("Prediction label changes the signal's selected contract")
            i = positions[date]
            if i + 21 >= len(dates):
                expected_status = "outside_future_calendar"
            else:
                entry, exit_date = dates[i + 1], dates[i + 21]
                if item["entry_date"] != entry or item["exit_date"] != exit_date:
                    raise ValueError("Prediction target does not match lag=1 and horizon=20")
                if exit_date >= item["front_expiry"]:
                    expected_status = "expiry_before_target_end"
                elif (entry, contract) not in mark_lookup.index or (
                    exit_date,
                    contract,
                ) not in mark_lookup.index:
                    expected_status = "missing_same_contract_mark"
                else:
                    first = mark_lookup.loc[(entry, contract), "settle"]
                    last = mark_lookup.loc[(exit_date, contract), "settle"]
                    if not np.isfinite([first, last]).all() or min(first, last) <= 0:
                        expected_status = "invalid_target_mark"
                    else:
                        expected_status = "ok"
                        expected_target = float(last / first - 1)
        if item["label_status"] != expected_status or not np.isclose(
            item["target"], expected_target, atol=1e-13, rtol=1e-13, equal_nan=True
        ):
            raise ValueError("Original prediction label disagrees with same-contract source marks")
        item["target_verified"] = True
        current_key = (date, contract)
        old_key = (item["previous_date"], contract)
        if known and current_key in mark_lookup.index:
            current = mark_lookup.loc[current_key]
            if current.code != item["code"] or not np.isclose(
                current.settle, item["reference_settle"], atol=1e-12, rtol=1e-12
            ):
                raise ValueError("Signal curve disagrees with its source contract mark")
        if item["input_status"] == "ok":
            if (
                not known
                or current_key not in mark_lookup.index
                or old_key not in mark_lookup.index
            ):
                item["input_status"] = "missing_same_contract_past_mark"
            else:
                first = mark_lookup.loc[old_key, "settle"]
                last = mark_lookup.loc[current_key, "settle"]
                if not np.isfinite([first, last]).all() or min(first, last) <= 0:
                    item["input_status"] = "invalid_same_contract_past_mark"
                else:
                    item["futures_since_previous_release"] = float(last / first - 1)
                    if not np.isfinite([item[name] for name in FEATURES]).all():
                        item["input_status"] = "missing_futures_controls"
        rows.append(item)
    return pd.DataFrame(rows).sort_values(["period_number", "code"]).reset_index(drop=True)


def recursive_forecasts(
    panel: pd.DataFrame,
    start: str = "2023-01-01",
    end: str = "2025-12-31",
    min_train: int = 60,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Each forecast depends on mature past labels only, including all scaling and intercepts."""
    if min_train < len(FEATURES) + 2:
        raise ValueError("Insufficient training requirement")
    forecasts, parameters = [], []
    for code, history in panel.groupby("code", sort=True):
        history = history.sort_values(["date", "period"])
        complete = history.input_status.eq("ok") & np.isfinite(history[list(FEATURES)]).all(axis=1)
        candidate = history.date.between(start, end)
        for event in history.loc[candidate].to_dict("records"):
            event.update(
                prediction_status="input_not_admissible",
                training_events=0,
                training_max_exit=pd.NaT,
                training_max_signal=pd.NaT,
                training_periods_sha256="",
                score_status="no_prediction",
            )
            for model in MODELS:
                event[f"prediction_{model}"] = np.nan
                event[f"loss_{model}"] = np.nan
            if event["input_status"] != "ok":
                forecasts.append(event)
                continue
            training = history.loc[
                complete
                & np.isfinite(history.target)
                & history.label_status.eq("ok")
                & history.exit_date.lt(event["date"])
                & history.date.lt(event["date"])
            ]
            event["training_events"] = len(training)
            if len(training) < min_train:
                event["prediction_status"] = "insufficient_mature_training"
                forecasts.append(event)
                continue
            x = training[list(FEATURES)].to_numpy(dtype=float)
            y = training.target.to_numpy(dtype=float)
            center, scale = x.mean(axis=0), x.std(axis=0, ddof=0)
            if np.any(scale < 1e-12):
                event["prediction_status"] = "constant_training_feature"
                forecasts.append(event)
                continue
            scaled = (x - center) / scale
            current = (np.array([event[name] for name in FEATURES]) - center) / scale
            fits = {}
            for model, width in (("futures", len(CONTROLS)), ("augmented", len(FEATURES))):
                design = np.column_stack([np.ones(len(x)), scaled[:, :width]])
                if np.linalg.matrix_rank(design) < design.shape[1]:
                    break
                beta = np.linalg.lstsq(design, y, rcond=None)[0]
                fits[model] = (beta, float(np.r_[1.0, current[:width]] @ beta))
            if len(fits) != 2:
                event["prediction_status"] = "rank_deficient_training"
                forecasts.append(event)
                continue
            event.update(
                prediction_status="ok",
                training_max_exit=training.exit_date.max(),
                training_max_signal=training.date.max(),
                prediction_training_mean=float(y.mean()),
                prediction_zero=0.0,
            )
            import hashlib

            membership = "\n".join(f"{row.period}|{row.code}" for row in training.itertuples())
            event["training_periods_sha256"] = hashlib.sha256(membership.encode()).hexdigest()
            for model, (beta, prediction) in fits.items():
                event[f"prediction_{model}"] = prediction
                for j, coefficient in enumerate(beta):
                    parameters.append(
                        {
                            "period": event["period"],
                            "date": event["date"],
                            "code": code,
                            "model": model,
                            "term": "intercept" if j == 0 else FEATURES[j - 1],
                            "coefficient_standardized": float(coefficient),
                            "training_center": 0.0 if j == 0 else float(center[j - 1]),
                            "training_scale": 1.0 if j == 0 else float(scale[j - 1]),
                        }
                    )
            if np.isfinite(event["target"]) and event["label_status"] == "ok":
                event["score_status"] = "scored"
                for model in MODELS:
                    event[f"loss_{model}"] = float(
                        (event["target"] - event[f"prediction_{model}"]) ** 2
                    )
            else:
                event["score_status"] = "forecast_without_observed_target"
            forecasts.append(event)
    return (
        pd.DataFrame(forecasts).sort_values(["period_number", "code"]).reset_index(drop=True),
        pd.DataFrame(
            parameters,
            columns=[
                "period",
                "date",
                "code",
                "model",
                "term",
                "coefficient_standardized",
                "training_center",
                "training_scale",
            ],
        ),
    )


def forecast_metrics(forecasts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One score per release, with equal codes within each release; no t-tests on panel rows."""
    scored = forecasts.loc[forecasts.score_status.eq("scored")].copy()
    losses = [f"loss_{name}" for name in MODELS]
    events = scored.groupby(["period", "date"], as_index=False).agg(
        **{name: (name, "mean") for name in losses}, assets=("code", "nunique")
    )
    rows = []

    def add(group: pd.DataFrame, kind: str, name: str, count: int):
        item = {"group": kind, "name": name, "events": len(group), "scored_rows": count}
        for model in MODELS:
            item[f"mse_{model}"] = float(group[f"loss_{model}"].mean())
        for comparator in ("futures", "training_mean", "zero"):
            denominator = item[f"mse_{comparator}"]
            item[f"r2_vs_{comparator}"] = (
                1 - item["mse_augmented"] / denominator if denominator > 0 else np.nan
            )
        rows.append(item)

    add(events, "overall", "all", len(scored))
    for year in (2023, 2024, 2025):
        group = events.loc[events.date.dt.year.eq(year)]
        add(group, "year", str(year), int(scored.date.dt.year.eq(year).sum()))
    for code in CODES:
        group = scored.loc[scored.code.eq(code)]
        add(group, "code", code, len(group))
    return pd.DataFrame(rows), events


def register_experiment(prior: pd.DataFrame, events: int) -> pd.DataFrame:
    if len(prior) != 14 or "nbs_reference_event_forecast" in set(prior.analysis):
        raise ValueError("Expected the frozen previous 14-experiment registry")
    current = pd.DataFrame(
        [
            *prior.to_dict("records"),
            {
                "analysis": "nbs_reference_event_forecast",
                "signal": SIGNAL,
                "controls": ",".join(CONTROLS),
                "days": events,
                "hac_p": np.nan,
                "inference_status": "unavailable_descriptive_screen",
                "claim_eligible": False,
                "reason": "Recursive event forecast comparison; no calibrated p-value or investment claim",
            },
        ]
    )
    current["family_15_by_q"] = by_adjust(current.hac_p.to_numpy(dtype=float))
    current["family_size"] = 15
    return current


def run_nbs_prediction(config: Path) -> Path:
    plan, paths = prediction_plan(config)
    started = datetime.now(timezone.utc).isoformat()
    history = json.loads(paths["history_run"].read_text(encoding="utf-8"))
    if not history["complete_normalized_history"] or history["expected_calendar_periods"] != 216:
        raise ValueError("The fixed complete source history is required")
    panel = build_event_panel(
        pd.read_csv(paths["references"]),
        pd.read_csv(paths["curves"]),
        pd.read_csv(paths["labels"]),
        pd.read_csv(paths["bars"], usecols=["date", "code", "contract", "settle"]),
        pd.read_csv(paths["calendar"]),
    )
    forecasts, parameters = recursive_forecasts(panel)
    metrics, events = forecast_metrics(forecasts)
    registry = register_experiment(pd.read_csv(paths["prior_exploration"]), len(events))
    coverage = forecasts.groupby(["code", "prediction_status", "score_status"], as_index=False).agg(
        rows=("period", "size")
    )
    exclusions = panel.groupby(["code", "input_status"], as_index=False).agg(
        rows=("period", "size")
    )
    main = metrics.iloc[0]
    summary = {
        "version": __version__,
        "kind": plan["study"]["kind"],
        "started_utc": started,
        "source_event_product_rows": len(panel),
        "input_admissible_rows": int(panel.input_status.eq("ok").sum()),
        "forecast_period_rows": len(forecasts),
        "predictions": int(forecasts.prediction_status.eq("ok").sum()),
        "scored_rows": int(forecasts.score_status.eq("scored").sum()),
        "unscored_predictions": int(
            forecasts.score_status.eq("forecast_without_observed_target").sum()
        ),
        "scored_release_events": len(events),
        "primary_r2_vs_futures": float(main.r2_vs_futures),
        "r2_vs_training_mean": float(main.r2_vs_training_mean),
        "r2_vs_zero": float(main.r2_vs_zero),
        "descriptive_experiments_added": 1,
        "formal_p_values_computed": 0,
        "registered_exploration_family_size": len(registry),
        "old_minimum_cross_section_unchanged": 8,
        "independent_new_sample": False,
        "historical_first_vintage_certified": False,
        "strict_spot_basis_claim_eligible": False,
        "investment_claim_eligible": False,
        "real_investment_backtests": 0,
        "protocol": plan,
        "protocol_sha256": digest(config),
        "inputs": {
            name: {"path": str(paths[name]), "sha256": digest(paths[name])}
            for name in plan["input_hashes"]
        },
        "code_sha256": {
            name: digest(Path(__file__).with_name(name))
            for name in ("nbs_prediction.py", "inference.py", "curve_factors.py")
        },
    }
    output = paths["output"] / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    for name, frame in {
        "event_panel": panel,
        "forecasts": forecasts,
        "parameters": parameters,
        "metrics": metrics,
        "release_losses": events,
        "forecast_coverage": coverage,
        "input_exclusions": exclusions,
        "registered_exploration": registry,
    }.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    (output / "protocol.toml").write_bytes(config.read_bytes())
    (output / "run.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_prediction_report(output, summary, metrics, coverage, exclusions)
    return output


def write_prediction_report(output, summary, metrics, coverage, exclusions):
    primary = summary["primary_r2_vs_futures"]
    direction = "降低" if primary > 0 else "增加"
    display = metrics.copy()
    for name in ("r2_vs_futures", "r2_vs_training_mean", "r2_vs_zero"):
        display[name] = display[name].map(
            lambda value: f"{value:.2%}" if np.isfinite(value) else "—"
        )
    display = display.rename(
        columns={
            "group": "分组",
            "name": "范围",
            "events": "事件数",
            "scored_rows": "评分行数",
            "mse_futures": "期货基准MSE",
            "mse_augmented": "加参考信息MSE",
            "mse_training_mean": "历史均值MSE",
            "mse_zero": "零预测MSE",
            "r2_vs_futures": "相对期货基准R²",
            "r2_vs_training_mean": "相对历史均值R²",
            "r2_vs_zero": "相对零预测R²",
        }
    )
    page = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>工业品参考信息预测诊断 · v{__version__}</title>
<style>body{{font:16px/1.85 system-ui,'Microsoft YaHei',sans-serif;max-width:1200px;margin:32px auto;padding:24px;background:#f4f7fa;color:#193348}}h1{{font-size:34px;line-height:1.4}}section{{background:white;border:1px solid #dbe5eb;border-radius:16px;padding:25px;margin:24px 0}}.scroll{{overflow:auto}}table{{border-collapse:collapse;font-size:13px;white-space:nowrap}}td,th{{padding:9px;border-bottom:1px solid #dbe5eb;text-align:left}}a{{color:#126b83}}.small{{font-size:13px;color:#526c7b}}</style>
<p>Commodity Factor Lab · v{__version__} · 固定历史递推诊断</p><h1>参考价变化有没有期货之外的预测信息？</h1>
<section><h2>本轮结果</h2><p>在{summary["scored_release_events"]}次可评分公告、{summary["scored_rows"]}个品种事件中，加参考信息的模型相对期货基准使平方预测误差<b>{direction}{abs(primary):.2%}</b>。相对历史均值R²为{summary["r2_vs_training_mean"]:.2%}，相对零预测为{summary["r2_vs_zero"]:.2%}。这些是同一个固定历史比较的描述统计，不是显著性、独立样本确认或投资收益。</p></section>
<section><h2>单一特征，固定对照</h2><p>只用同口径、相邻日历旬的参考价对数变化；缺期或元数据变化不跨越。对照为同合约的两次公告间结算涨幅，以及原carry、126日动量、252日基差动量。每品种分别用至少60个目标已结束的历史事件训练OLS；同样本比较，标准化只看训练。2023—2025递推预测，当前目标不参与拟合或预测准入。</p><p>标签是信号后下一交易日结算至20交易日后同合约结算的简单收益，已逐行回查日历、合约与源价。它不能代表真实成交或资金回报。七品种时间序列实验与旧八品种截面门槛分别保留。</p></section>
<section><h2>完整比较，不挑年份或品种</h2><div class="scroll">{table(display)}</div><p class="small">整体先对同公告品种等权，再对公告等权；品种行是该品种事件均值。R²为1−新增模型MSE/对照MSE，正值才表示误差减少。没有单独检验各切片的显著性。</p></section>
<section><h2>覆盖与没有出预测的原因</h2><p>{summary["predictions"]}条预测，其中{summary["unscored_predictions"]}条因目标未观测而不评分；源映射共{summary["source_event_product_rows"]}行，{summary["input_admissible_rows"]}行通过输入规则。没有用尾部缺标签倒过来决定是否生成预测。</p><div class="scroll">{table(coverage)}</div><div class="scroll">{table(exclusions)}</div></section>
<section><h2>能说什么，仍不能说什么</h2><p>本轮不计算正式p值或置信区间。嵌套递推模型、重叠目标及非平衡事件需要另行校准推断，不能直接套独立样本或普通DM检验。仍登记为第15个探索实验，新项p缺失，在BY族大小中按p=1占位；旧十四项文件保留。</p><p>旧行情此前已看过，不称独立新样本。源时钟仍是当前正文声明，初发版本、精确交割现货与真实投资资格均未认证。结果不能靠翻符号、挑窗口或删不利品种“修好”。</p><p><a href="event_panel.csv">全部事件与输入状态</a> · <a href="forecasts.csv">全部预测与训练截止</a> · <a href="parameters.csv">逐次参数及标准化</a> · <a href="metrics.csv">完整指标</a> · <a href="release_losses.csv">公告级误差</a> · <a href="registered_exploration.csv">十五项登记</a> · <a href="protocol.toml">执行协议</a> · <a href="run.json">输入与代码哈希</a></p><p class="small">协议指纹：{escape(summary["protocol_sha256"])}</p></section></html>"""
    (output / "index.html").write_text(page, encoding="utf-8")
