"""Independent rank/OLS and HAC reference; no production module imports."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import brentq
from scipy.stats import norm, rankdata
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.sandwich_covariance import cov_hac

CONTROLS = {
    "carry": ["momentum_126d"],
    "basis_momentum": ["momentum_126d", "carry"],
    "inventory_scarcity": ["momentum_126d", "carry"],
}
FEATURES = ["momentum_126d", "carry", "basis_momentum", "inventory_scarcity"]


def reference_precision(values, lag, alpha=0.05, family=3, resolution=0.05, power=0.8):
    x = np.asarray(values, dtype=float)
    valid = np.isfinite(x)
    n, total = int(valid.sum()), len(x)
    mean = x[valid].mean()
    # Full calendar scores preserve missing-session separation. Scale the
    # independent library covariance back from T to n valid observations.
    scores = np.where(valid, x - mean, 0.0)
    fit = sm.OLS(scores, np.ones((total, 1))).fit()
    variance = cov_hac(fit, nlags=lag, use_correction=True)[0, 0]
    variance *= (total / n) ** 2 * (n / (n - 1)) / (total / (total - 1))
    se = np.sqrt(variance)
    critical = norm.ppf(1 - alpha / (2 * family))

    def probability(delta):
        return norm.sf(critical - delta / se) + norm.cdf(-critical - delta / se)

    return {
        "nobs": n,
        "mean": mean,
        "se": se,
        "t": mean / se,
        "p": 2 * norm.sf(abs(mean / se)),
        "point_low": mean - norm.ppf(1 - alpha / 2) * se,
        "point_high": mean + norm.ppf(1 - alpha / 2) * se,
        "simultaneous_low": mean - critical * se,
        "simultaneous_high": mean + critical * se,
        "normal_planning_mde": brentq(lambda d: probability(d) - power, 0.0, 1.0)
        if probability(1.0) >= power
        else np.nan,
        "power_at_resolution": probability(resolution),
    }


def verify(run_folder, root):
    run = json.loads((run_folder / "run.json").read_text(encoding="utf-8"))
    plan, a = run["protocol"], run["protocol"]["analysis"]
    paths = {k: root / v for k, v in plan["paths"].items()}
    curves, labels = pd.read_csv(paths["curves"]), pd.read_csv(paths["labels"])
    calendar = pd.read_csv(paths["calendar"])
    data = curves.merge(
        labels[["date", "code", "target", "status"]].rename(columns={"status": "label_status"}),
        on=["date", "code"],
        validate="one_to_one",
    )
    data = data.loc[data.code.isin(a["codes"]) & data.date.between(a["start"], a["end"])]
    saved_panel = pd.read_csv(run_folder / "common_panel.csv").set_index(["date", "code"])
    checks, errors = 0, []

    def check(condition, name):
        nonlocal checks
        checks += 1
        if not condition and len(errors) < 100:
            errors.append(name)

    def close(first, second, name):
        check(bool(np.isclose(first, second, rtol=1e-10, atol=1e-11, equal_nan=True)), name)

    eligible = np.isfinite(data[FEATURES]).all(axis=1)
    measurable = eligible & np.isfinite(data.target) & data.label_status.eq("ok")
    for row, known, measured in zip(data.itertuples(), eligible, measurable, strict=True):
        saved = saved_panel.loc[(row.date, row.code)]
        check(saved.input_eligible == known, f"{row.date}/{row.code}/input")
        check(saved.measurement_eligible == measured, f"{row.date}/{row.code}/measurement")
    estimates = []
    for day in calendar.loc[calendar.date.between(a["start"], a["end"]), "date"]:
        group = data.loc[data.date.eq(day) & measurable]
        results = {}
        for signal, controls in CONTROLS.items():
            value = np.nan
            if len(group) >= a["minimum_assets"]:
                arrays = {
                    col: rankdata(group[col].to_numpy(), method="average")
                    for col in [signal, "target", *controls]
                }
                design = np.column_stack([np.ones(len(group)), *[arrays[c] for c in controls]])
                if np.linalg.matrix_rank(design) == design.shape[1]:
                    x = sm.OLS(arrays[signal], design).fit().resid
                    y = sm.OLS(arrays["target"], design).fit().resid
                    if min(np.linalg.norm(x), np.linalg.norm(y)) >= 1e-10:
                        value = np.corrcoef(x, y)[0, 1]
            results[signal] = value
        good = all(np.isfinite(v) for v in results.values())
        for signal in CONTROLS:
            estimates.append(
                {
                    "date": day,
                    "signal": signal,
                    "value": results[signal] if good else np.nan,
                    "assets": len(group),
                }
            )
    reference = pd.DataFrame(estimates)
    saved_daily = pd.read_csv(run_folder / "common_daily.csv").set_index(["date", "signal"])
    for row in reference.itertuples():
        saved = saved_daily.loc[(row.date, row.signal)]
        close(saved.partial_rank_ic, row.value, f"{row.date}/{row.signal}/IC")
        close(saved.assets, row.assets, f"{row.date}/{row.signal}/assets")
    sensitivity = pd.read_csv(run_folder / "lag_sensitivity.csv")
    for row in sensitivity.itertuples():
        values = reference.loc[reference.signal.eq(row.signal), "value"]
        ref = reference_precision(
            values,
            row.hac_lags,
            a["alpha"],
            a["current_family_size"],
            a["resolution_target_rank_ic"],
            a["planning_power"],
        )
        for field, value in ref.items():
            close(getattr(row, field), value, f"{row.signal}/{row.hac_lags}/{field}")
    primary = pd.read_csv(run_folder / "primary.csv").set_index("signal")
    for row in sensitivity.loc[sensitivity.primary].itertuples():
        for col in ("mean", "se", "simultaneous_low", "simultaneous_high", "normal_planning_mde"):
            close(primary.loc[row.signal, col], getattr(row, col), f"{row.signal}/primary/{col}")
    registry = pd.read_csv(run_folder / "registered_exploration.csv")
    pvalues = registry.hac_p.fillna(1).to_numpy()
    corrected = multipletests(pvalues, method="fdr_by")[1]
    field = f"family_{a['registered_after']}_by_q"
    for i, row in registry.iterrows():
        close(row[field], corrected[i] if pd.notna(row.hac_p) else np.nan, f"registry/{i}")
    return {"checks": checks, "differences": errors}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.run, Path(__file__).resolve().parents[1])
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"checks": result["checks"], "differences": len(result["differences"])}))
    if result["differences"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
