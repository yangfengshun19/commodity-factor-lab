"""Independent event/label/forecast arithmetic with statsmodels; no production imports.

This checks implementation agreement, not independent source vintages or statistical power.
"""

import argparse
import hashlib
import json
import math
import tomllib
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

FEATURES = [
    "futures_since_previous_release",
    "carry",
    "momentum_126d",
    "basis_momentum",
    "reference_log_change",
]
MODELS = ["futures", "augmented", "training_mean", "zero"]


class Checkbook:
    def __init__(self):
        self.checks = 0
        self.differences = []

    def check(self, name, passed, actual=None, expected=None):
        self.checks += 1
        if not passed:
            self.differences.append(
                {"check": name, "actual": str(actual), "expected": str(expected)}
            )

    def equal(self, name, actual, expected):
        both_missing = pd.isna(actual) and pd.isna(expected)
        self.check(name, both_missing or actual == expected, actual, expected)

    def close(self, name, actual, expected):
        self.check(
            name,
            bool(np.isclose(actual, expected, atol=1e-11, rtol=1e-10, equal_nan=True)),
            actual,
            expected,
        )


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ordinal(value):
    year, month, part = value.split("-")
    return int(year) * 36 + (int(month) - 1) * 3 + {"early": 0, "middle": 1, "late": 2}[part]


def audit_events(inputs, panel, checks):
    refs = pd.read_csv(inputs["references"])
    refs["calendar_order"] = refs.period.map(ordinal)
    refs = refs.sort_values(["related_code", "calendar_order"])
    curves = pd.read_csv(inputs["curves"], parse_dates=["date", "front_expiry"]).set_index(
        ["date", "code"]
    )
    labels = pd.read_csv(
        inputs["labels"], parse_dates=["date", "entry_date", "exit_date"]
    ).set_index(["date", "code"])
    marks = pd.read_csv(
        inputs["bars"], usecols=["date", "code", "contract", "settle"], parse_dates=["date"]
    ).set_index(["date", "contract"])
    calendar = pd.read_csv(inputs["calendar"], parse_dates=["date"])
    slots = list(calendar.date)
    cutoffs = pd.to_datetime(calendar.signal_cutoff, utc=True)
    panel_lookup = panel.set_index(["period", "code"])
    checks.check("unique panel keys", panel_lookup.index.is_unique)
    checks.equal("all reference rows retained", len(panel), len(refs))
    previous = {}
    for ref in refs.itertuples():
        code, period = ref.related_code, ref.period
        tag = f"{period}/{code}"
        key = (period, code)
        checks.check(tag + " exists", key in panel_lookup.index)
        if key not in panel_lookup.index:
            continue
        row = panel_lookup.loc[key]
        clock = pd.Timestamp(ref.source_declared_published_at)
        valid_dates = calendar.loc[cutoffs.ge(clock)]
        date = valid_dates.iloc[0].date if len(valid_dates) else pd.NaT
        checks.equal(tag + " first signal date", row.date, date)
        checks.equal(tag + " input signal date", pd.to_datetime(ref.first_signal_date), date)
        checks.close(tag + " source price", row.reference_price, ref.reference_price)
        checks.equal(tag + " identity", row.comparability_key, ref.comparability_key)
        checks.equal(tag + " source hash", row.source_sha256, ref.source_sha256)
        old = previous.get(code)
        expected_signal, status = np.nan, "ok"
        if old is None or ordinal(period) - ordinal(old.period) != 1:
            status = "no_adjacent_published_reference"
        elif old.comparability_key != ref.comparability_key:
            status = "comparability_break"
        if pd.isna(date):
            status = "outside_signal_calendar"
        old_date = pd.to_datetime(old.first_signal_date) if old is not None else pd.NaT
        if status == "ok" and (pd.isna(old_date) or old_date >= date):
            status = "previous_release_not_strictly_earlier"
        if status == "ok":
            expected_signal = math.log(ref.reference_price / old.reference_price)
        checks.equal(tag + " previous period", row.previous_period, old.period if old else np.nan)
        checks.equal(tag + " previous signal date", row.previous_date, old_date)
        checks.close(tag + " reference log change", row.reference_log_change, expected_signal)
        previous[code] = ref
        if pd.isna(date):
            checks.equal(tag + " outside input status", row.input_status, status)
            continue
        checks.equal(
            tag + " cutoff",
            pd.Timestamp(ref.first_signal_cutoff),
            cutoffs.loc[calendar.date.eq(date)].iloc[0],
        )
        curve = curves.loc[(date, code)]
        target = labels.loc[(date, code)]
        contract = curve.front_contract
        checks.equal(tag + " selected contract", row.front_contract, contract)
        known = isinstance(contract, str) and bool(contract)
        if known and (date, contract) in marks.index:
            checks.equal(tag + " source mark code", marks.loc[(date, contract), "code"], code)
            checks.close(
                tag + " selected source mark",
                curve.reference_settle,
                marks.loc[(date, contract), "settle"],
            )
        expected_target, expected_label = np.nan, "no_known_curve"
        entry, exit_date = pd.NaT, pd.NaT
        if known:
            i = slots.index(date)
            if i + 21 >= len(slots):
                expected_label = "outside_future_calendar"
            else:
                entry, exit_date = slots[i + 1], slots[i + 21]
                if exit_date >= curve.front_expiry:
                    expected_label = "expiry_before_target_end"
                elif (entry, contract) not in marks.index or (
                    exit_date,
                    contract,
                ) not in marks.index:
                    expected_label = "missing_same_contract_mark"
                else:
                    begin = marks.loc[(entry, contract), "settle"]
                    finish = marks.loc[(exit_date, contract), "settle"]
                    if not np.isfinite([begin, finish]).all() or min(begin, finish) <= 0:
                        expected_label = "invalid_target_mark"
                    else:
                        expected_label = "ok"
                        expected_target = finish / begin - 1
            checks.equal(tag + " label contract", target.contract, contract)
        for name, actual in (("panel", row), ("original", target)):
            checks.close(tag + f" {name} target", actual.target, expected_target)
            checks.equal(tag + f" {name} entry", actual.entry_date, entry)
            checks.equal(tag + f" {name} exit", actual.exit_date, exit_date)
        checks.equal(tag + " panel label status", row.label_status, expected_label)
        checks.equal(tag + " original label status", target.status, expected_label)
        past = np.nan
        if status == "ok":
            if (
                not known
                or (old_date, contract) not in marks.index
                or (date, contract) not in marks.index
            ):
                status = "missing_same_contract_past_mark"
            else:
                first = marks.loc[(old_date, contract), "settle"]
                last = marks.loc[(date, contract), "settle"]
                if not np.isfinite([first, last]).all() or min(first, last) <= 0:
                    status = "invalid_same_contract_past_mark"
                else:
                    past = last / first - 1
                    if not np.isfinite([past, *(curve[name] for name in FEATURES[1:4])]).all():
                        status = "missing_futures_controls"
        checks.close(tag + " same-contract past control", row.futures_since_previous_release, past)
        for feature in FEATURES[1:4]:
            checks.close(tag + " " + feature, row[feature], curve[feature])
        checks.equal(tag + " input status", row.input_status, status)


def audit_forecasts(
    panel, forecasts, parameters, checks, start="2023-01-01", end="2025-12-31", min_train=60
):
    expected_rows = panel.loc[panel.date.between(start, end)]
    lookup = forecasts.set_index(["period", "code"])
    params = parameters.set_index(["period", "code", "model", "term"])
    checks.check("unique forecast keys", lookup.index.is_unique)
    checks.check("unique parameter keys", params.index.is_unique)
    checks.equal("forecast row count", len(forecasts), len(expected_rows))
    expected_parameter_keys = set()
    for code, history in panel.groupby("code", sort=True):
        history = history.sort_values(["date", "period"])
        for current in history.loc[history.date.between(start, end)].itertuples():
            key = (current.period, code)
            tag = f"{current.period}/{code}"
            checks.check(tag + " forecast retained", key in lookup.index)
            if key not in lookup.index:
                continue
            row = lookup.loc[key]
            for name in ("date", "target", "input_status", "label_status", *FEATURES):
                actual = row[name]
                expected = getattr(current, name)
                if name in (*FEATURES, "target"):
                    checks.close(tag + " forecast input " + name, actual, expected)
                else:
                    checks.equal(tag + " forecast input " + name, actual, expected)
            status, count = "input_not_admissible", 0
            predictions = dict.fromkeys(MODELS, np.nan)
            train = history.iloc[0:0]
            if current.input_status == "ok":
                train = history.loc[
                    history.input_status.eq("ok")
                    & history.label_status.eq("ok")
                    & np.isfinite(history[[*FEATURES, "target"]]).all(axis=1)
                    & history.exit_date.lt(current.date)
                    & history.date.lt(current.date)
                ]
                count = len(train)
                status = "insufficient_mature_training"
                if count >= min_train:
                    averages = train[FEATURES].mean().to_numpy()
                    deviations = train[FEATURES].std(ddof=0).to_numpy()
                    status = "constant_training_feature"
                    if not (deviations < 1e-12).any():
                        x = (train[FEATURES].to_numpy() - averages) / deviations
                        now = (
                            np.array([getattr(current, name) for name in FEATURES]) - averages
                        ) / deviations
                        fits = {}
                        status = "rank_deficient_training"
                        for model, width in (("futures", 4), ("augmented", 5)):
                            design = np.column_stack([np.ones(count), x[:, :width]])
                            if np.linalg.matrix_rank(design) != width + 1:
                                break
                            # Independent library and solver, with independently reconstructed training rows.
                            fit = sm.OLS(train.target.to_numpy(), design).fit()
                            fits[model] = fit.params
                            predictions[model] = float(fit.predict(np.r_[1.0, now[:width]])[0])
                        if len(fits) == 2:
                            status = "ok"
                            predictions["training_mean"] = float(train.target.mean())
                            predictions["zero"] = 0.0
                            checks.equal(
                                tag + " latest target exit",
                                row.training_max_exit,
                                train.exit_date.max(),
                            )
                            checks.equal(
                                tag + " latest training signal",
                                row.training_max_signal,
                                train.date.max(),
                            )
                            checks.check(
                                tag + " no immature target in training",
                                train.exit_date.lt(current.date).all(),
                            )
                            membership = "\n".join(
                                f"{r.period}|{r.code}" for r in train.itertuples()
                            )
                            checks.equal(
                                tag + " training membership",
                                row.training_periods_sha256,
                                hashlib.sha256(membership.encode()).hexdigest(),
                            )
                            for model, coefficients in fits.items():
                                for index, coefficient in enumerate(coefficients):
                                    term = "intercept" if index == 0 else FEATURES[index - 1]
                                    parameter_key = (current.period, code, model, term)
                                    expected_parameter_keys.add(parameter_key)
                                    checks.check(
                                        tag + " parameter exists " + model + term,
                                        parameter_key in params.index,
                                    )
                                    if parameter_key not in params.index:
                                        continue
                                    parameter = params.loc[parameter_key]
                                    checks.close(
                                        tag + " beta " + model + term,
                                        parameter.coefficient_standardized,
                                        coefficient,
                                    )
                                    checks.close(
                                        tag + " center " + model + term,
                                        parameter.training_center,
                                        0.0 if index == 0 else averages[index - 1],
                                    )
                                    checks.close(
                                        tag + " scale " + model + term,
                                        parameter.training_scale,
                                        1.0 if index == 0 else deviations[index - 1],
                                    )
                        else:
                            predictions = dict.fromkeys(MODELS, np.nan)
            checks.equal(tag + " training count", row.training_events, count)
            checks.equal(tag + " prediction status", row.prediction_status, status)
            score = "no_prediction"
            if status == "ok":
                score = (
                    "scored"
                    if np.isfinite(current.target) and current.label_status == "ok"
                    else "forecast_without_observed_target"
                )
            checks.equal(tag + " scoring status", row.score_status, score)
            for model in MODELS:
                checks.close(
                    tag + " prediction " + model, row[f"prediction_{model}"], predictions[model]
                )
                loss = (current.target - predictions[model]) ** 2 if score == "scored" else np.nan
                checks.close(tag + " loss " + model, row[f"loss_{model}"], loss)
    checks.equal("all parameter rows accounted", len(parameters), len(expected_parameter_keys))
    checks.check(
        "parameter keys exactly match admitted fits", set(params.index) == expected_parameter_keys
    )


def audit_metrics(forecasts, metrics, release_losses, prior, registry, checks):
    scored = forecasts.loc[forecasts.score_status.eq("scored")]
    losses = [f"loss_{model}" for model in MODELS]
    by_release = scored.groupby(["period", "date"])[losses].mean()
    actual_release = release_losses.set_index(["period", "date"])
    checks.equal("release count", len(actual_release), len(by_release))
    for key, values in by_release.iterrows():
        checks.check(str(key) + " release retained", key in actual_release.index)
        if key not in actual_release.index:
            continue
        for name in losses:
            checks.close(
                str(key) + " event mean " + name, actual_release.loc[key, name], values[name]
            )
    for row in metrics.itertuples():
        if row.group == "overall":
            values, count = by_release, len(scored)
        elif row.group == "year":
            values = by_release.loc[by_release.index.get_level_values("date").year == int(row.name)]
            count = int(scored.date.dt.year.eq(int(row.name)).sum())
        else:
            values = scored.loc[scored.code.eq(row.name)]
            count = len(values)
        checks.equal(f"{row.group}/{row.name} event count", row.events, len(values))
        checks.equal(f"{row.group}/{row.name} row count", row.scored_rows, count)
        for model in MODELS:
            checks.close(
                f"{row.group}/{row.name} mse {model}",
                getattr(row, f"mse_{model}"),
                values[f"loss_{model}"].mean(),
            )
        for model in ["futures", "training_mean", "zero"]:
            denominator = values[f"loss_{model}"].mean()
            expected = 1 - values.loss_augmented.mean() / denominator if denominator > 0 else np.nan
            checks.close(
                f"{row.group}/{row.name} relative r2 {model}",
                getattr(row, f"r2_vs_{model}"),
                expected,
            )
    checks.equal("registry size", len(registry), 15)
    for i, row in prior.iterrows():
        checks.equal(f"prior experiment {i} identity", registry.iloc[i].analysis, row.analysis)
        checks.close(f"prior experiment {i} p retained", registry.iloc[i].hac_p, row.hac_p)
    checks.check("new experiment has no invented p", pd.isna(registry.iloc[-1].hac_p))
    pvalues = registry.hac_p.fillna(1).to_numpy()
    order = np.argsort(pvalues)
    harmonic = sum(1 / i for i in range(1, 16))
    last = 1.0
    for rank in range(14, -1, -1):
        index = order[rank]
        last = min(last, pvalues[index] * 15 * harmonic / (rank + 1))
        expected = last if pd.notna(registry.iloc[index].hac_p) else np.nan
        checks.close(f"family15 q {index}", registry.iloc[index].family_15_by_q, expected)


def reconcile(config, report, output, frozen_receipt=None):
    plan = tomllib.loads(config.read_text(encoding="utf-8"))
    root = config.resolve().parent.parent
    inputs = {name: root / plan["paths"][name] for name in plan["input_hashes"]}
    summary = json.loads((report / "run.json").read_text(encoding="utf-8"))
    checks = Checkbook()
    checks.equal("same protocol file", summary["protocol_sha256"], sha(config))
    checks.check("embedded protocol matches", summary["protocol"] == plan)
    fingerprints = {}
    for name, path in inputs.items():
        fingerprints[name] = sha(path)
        checks.equal(name + " frozen input hash", fingerprints[name], plan["input_hashes"][name])
        checks.equal(
            name + " run input hash", fingerprints[name], summary["inputs"][name]["sha256"]
        )
    for name, expected in summary["code_sha256"].items():
        checks.equal(
            name + " run code hash", sha(root / "src/commodity_factor_lab" / name), expected
        )
    if frozen_receipt:
        frozen = json.loads(frozen_receipt.read_text(encoding="utf-8"))
        checks.equal(
            "pre-outcome frozen protocol",
            sha(config),
            frozen["sha256"]["configs/nbs_prediction.toml"],
        )
        checks.check(
            "frozen before first computation",
            pd.Timestamp(frozen["frozen_utc"]) < pd.Timestamp(summary["started_utc"]),
        )
    panel = pd.read_csv(
        report / "event_panel.csv", parse_dates=["date", "previous_date", "entry_date", "exit_date"]
    )
    forecasts = pd.read_csv(
        report / "forecasts.csv",
        parse_dates=["date", "entry_date", "exit_date", "training_max_exit", "training_max_signal"],
    )
    parameters = pd.read_csv(report / "parameters.csv")
    metrics = pd.read_csv(report / "metrics.csv")
    losses = pd.read_csv(report / "release_losses.csv", parse_dates=["date"])
    prior, registry = (
        pd.read_csv(inputs["prior_exploration"]),
        pd.read_csv(report / "registered_exploration.csv"),
    )
    audit_events(inputs, panel, checks)
    audit_forecasts(panel, forecasts, parameters, checks)
    audit_metrics(forecasts, metrics, losses, prior, registry, checks)
    checks.equal(
        "summary scored rows",
        summary["scored_rows"],
        int(forecasts.score_status.eq("scored").sum()),
    )
    checks.equal(
        "summary predictions",
        summary["predictions"],
        int(forecasts.prediction_status.eq("ok").sum()),
    )
    checks.equal(
        "summary unscored predictions",
        summary["unscored_predictions"],
        int(forecasts.score_status.eq("forecast_without_observed_target").sum()),
    )
    checks.equal("summary release events", summary["scored_release_events"], len(losses))
    checks.close("summary primary", summary["primary_r2_vs_futures"], metrics.iloc[0].r2_vs_futures)
    for name in (
        "independent_new_sample",
        "historical_first_vintage_certified",
        "strict_spot_basis_claim_eligible",
        "investment_claim_eligible",
    ):
        checks.equal(name + " remains false", summary[name], False)
    checks.equal("no new formal p-values", summary["formal_p_values_computed"], 0)
    result = {
        "kind": "independent_nbs_event_forecast_reconciliation",
        "checks": checks.checks,
        "differences": len(checks.differences),
        "statsmodels_version": sm.__version__,
        "production_module_imported": False,
        "independent_statistical_samples": False,
        "source_vintage_certified": False,
        "inputs": fingerprints,
        "report_run_sha256": sha(report / "run.json"),
        "audit_code_sha256": sha(__file__),
        "report_files_sha256": {p.name: sha(p) for p in sorted(report.glob("*.csv"))},
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(checks.differences, columns=["check", "actual", "expected"]).to_csv(
        output / "differences.csv", index=False
    )
    page = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>工业品预测独立核对</title><style>body{{max-width:950px;margin:40px auto;padding:24px;font:17px/1.85 system-ui,'Microsoft YaHei',sans-serif;background:#f4f7fa;color:#193348}}a{{color:#126b83}}</style><h1>工业品预测诊断的独立计算核对</h1><p>{checks.checks:,}项检查，{len(checks.differences)}处差异。</p><p>不导入生产解析、因子或预测模块；另读冻结参考价、源合约结算、日历与原标签，逐事件核对相邻期、可比身份、公告时点、同合约涨幅和未来标签。训练成员与截止另行重建，使用statsmodels的OLS核对全部递推预测、参数、训练标准化及误差聚合，另算15项族调整。</p><p>这是实现与算术复核，不是独立供应商、独立统计样本或历史初发认证，也不提供本轮正式显著性。</p><p><a href="audit.json">输入、检查器和报告指纹</a> · <a href="differences.csv">差异明细</a></p><p>{escape(str(report))}</p></html>"""
    (output / "index.html").write_text(page, encoding="utf-8")
    print(json.dumps({"checks": checks.checks, "differences": len(checks.differences)}))
    return not checks.differences


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/nbs_prediction.toml"))
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-receipt", type=Path)
    args = parser.parse_args()
    raise SystemExit(
        0 if reconcile(args.config, args.report, args.output, args.frozen_receipt) else 1
    )
