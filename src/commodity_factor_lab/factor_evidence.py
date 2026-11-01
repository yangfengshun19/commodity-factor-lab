"""Exploratory common-sample effect sizes; never an investment certificate."""

import json
import math
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from . import __version__
from .contract_research import HYPOTHESES, partial_rank_ic
from .inference import by_adjust, hac_mean
from .market_benchmark import digest

SIGNALS = ("carry", "basis_momentum", "inventory_scarcity")
FEATURES = ("momentum_126d", *SIGNALS)
NORMAL = NormalDist()


def precision_summary(
    values, lags=40, min_obs=252, alpha=0.05, family=3, resolution=0.05, power=0.8
):
    """Normal/HAC intervals and fixed-SE planning power, conditional on assumptions."""
    if (
        not 0 < alpha < 1
        or type(family) is not int
        or family < 1
        or not 0 < resolution < 1
        or not 0.5 < power < 1
    ):
        raise ValueError("Invalid precision analysis settings")
    stats = hac_mean(values, lags, min_obs)
    stats.update(
        hac_lags=lags,
        point_low=np.nan,
        point_high=np.nan,
        simultaneous_low=np.nan,
        simultaneous_high=np.nan,
        normal_planning_mde=np.nan,
        power_at_resolution=np.nan,
        small_effect_bound_supported=False,
    )
    if stats["status"] != "ok":
        return stats
    z = NORMAL.inv_cdf(1 - alpha / 2)
    z_family = NORMAL.inv_cdf(1 - alpha / (2 * family))
    mean, se = stats["mean"], stats["se"]

    def probability(effect):
        scaled = effect / se
        return 1 - NORMAL.cdf(z_family - scaled) + NORMAL.cdf(-z_family - scaled)

    lo, hi = 0.0, 1.0
    if probability(hi) >= power:
        for _ in range(80):
            midpoint = (lo + hi) / 2
            if probability(midpoint) < power:
                lo = midpoint
            else:
                hi = midpoint
        stats["normal_planning_mde"] = hi
    stats.update(
        point_low=mean - z * se,
        point_high=mean + z * se,
        simultaneous_low=mean - z_family * se,
        simultaneous_high=mean + z_family * se,
        power_at_resolution=probability(resolution),
        small_effect_bound_supported=bool(
            mean - z_family * se > -resolution and mean + z_family * se < resolution
        ),
    )
    return stats


def common_sample(curves, labels, calendar, settings):
    """No signal selection by future availability; measurement eligibility is explicit."""
    data, targets, sessions = curves.copy(), labels.copy(), calendar.copy()
    for frame in (data, targets, sessions):
        frame["date"] = pd.to_datetime(frame.date).dt.normalize()
    if (
        data.duplicated(["date", "code"]).any()
        or targets.duplicated(["date", "code"]).any()
        or not sessions.date.is_unique
        or not sessions.date.is_monotonic_increasing
        or not data.date.isin(sessions.date).all()
    ):
        raise ValueError("Unique chronological sample keys required")
    data = data.loc[
        data.code.isin(settings["codes"]) & data.date.between(settings["start"], settings["end"])
    ].copy()
    data = data.merge(
        targets[["date", "code", "target", "status"]].rename(columns={"status": "label_status"}),
        on=["date", "code"],
        how="left",
        validate="one_to_one",
    )
    data["input_eligible"] = np.isfinite(data[list(FEATURES)]).all(axis=1)
    data["measurement_eligible"] = (
        data.input_eligible & np.isfinite(data.target) & data.label_status.eq("ok")
    )
    dates = sessions.loc[sessions.date.between(settings["start"], settings["end"]), "date"]
    groups = {d: g.loc[g.measurement_eligible] for d, g in data.groupby("date")}
    rows = []
    for day in dates:
        group = groups.get(day, data.iloc[:0])
        values = {
            s: partial_rank_ic(group, s, HYPOTHESES[s], settings["minimum_assets"]) for s in SIGNALS
        }
        common_valid = all(value["status"] == "ok" for value in values.values())
        for signal, value in values.items():
            rows.append(
                {
                    "date": day,
                    "signal": signal,
                    **value,
                    "unrestricted_partial_rank_ic": value["partial_rank_ic"],
                    "partial_rank_ic": value["partial_rank_ic"] if common_valid else np.nan,
                    "common_status": "ok" if common_valid else "common_sample_not_estimable",
                }
            )
    daily = pd.DataFrame(rows)
    valid = set(daily.loc[daily.common_status.eq("ok"), "date"])
    data["used_in_common_estimate"] = data.measurement_eligible & data.date.isin(valid)
    return data, daily


def evidence_plan(config):
    plan = tomllib.loads(config.read_text(encoding="utf-8"))
    root = config.resolve().parent.parent
    paths = {k: (root / v).resolve() for k, v in plan["paths"].items()}
    if set(paths) != set(plan["input_hashes"]) | {"output", "methodology"} or any(
        not p.is_relative_to(root) for p in paths.values()
    ):
        raise ValueError("Frozen workspace-contained evidence inputs required")
    for name, expected in plan["input_hashes"].items():
        if digest(paths[name]) != expected or paths["output"].is_relative_to(paths[name].parent):
            raise ValueError(f"Changed input or overlapping output: {name}")
    a = plan["analysis"]
    if (
        a["signals"] != list(SIGNALS)
        or a["common_features"] != list(FEATURES)
        or len(set(a["codes"])) != len(a["codes"])
        or a["minimum_assets"] < 6
        or a["current_family_size"] != 3
        or a["primary_hac_lags"] < 20
        or a["minimum_inference_observations"] < 252
        or any(lag < 20 for lag in a["sensitivity_hac_lags"])
        or a["registered_after"] != a["registered_before"] + 3
        or a["independent_new_sample"]
        or a["investment_claim_eligible"]
    ):
        raise ValueError("Invalid common-sample protocol or claim boundary")
    return plan, paths, root / "reports/protocols" / config.stem / "receipt.json"


def freeze_evidence(config):
    plan, paths, receipt = evidence_plan(config)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with receipt.open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "frozen_utc": datetime.now(timezone.utc).isoformat(),
                "config_sha256": digest(config),
                "methodology_sha256": digest(paths["methodology"]),
                "inputs": plan["input_hashes"],
                "previous_historical_results_viewed": True,
                "independent_preregistration": False,
            },
            stream,
            indent=2,
        )
    return receipt


def run_evidence(config):
    plan, paths, receipt = evidence_plan(config)
    frozen = json.loads(receipt.read_text(encoding="utf-8"))
    if frozen["config_sha256"] != digest(config) or frozen["methodology_sha256"] != digest(
        paths["methodology"]
    ):
        raise ValueError("Evidence protocol changed after freezing")
    started = datetime.now(timezone.utc)
    if started <= datetime.fromisoformat(frozen["frozen_utc"]):
        raise ValueError("Freeze must precede computation")
    a = plan["analysis"]
    old = pd.read_csv(paths["original_daily"], parse_dates=["date"])
    old_summary = pd.read_csv(paths["original_summary"])
    checks = []
    for row in old_summary.itertuples():
        values = old.loc[
            old.signal.eq(row.signal) & old.date.between(a["start"], a["end"]), "partial_rank_ic"
        ]
        stats = hac_mean(values, a["primary_hac_lags"], a["minimum_inference_observations"])
        for key, previous in (
            ("mean", row.mean_partial_rank_ic),
            ("t", row.hac_t),
            ("p", row.hac_p),
            ("nobs", row.days),
        ):
            if not math.isclose(stats[key], previous, rel_tol=1e-11, abs_tol=1e-13):
                raise ValueError(f"Original diagnostic not reproduced: {row.signal}/{key}")
            checks.append(
                {
                    "signal": row.signal,
                    "field": key,
                    "original": previous,
                    "reconstructed": stats[key],
                    "difference": stats[key] - previous,
                }
            )
    panel, daily = common_sample(
        pd.read_csv(paths["curves"]),
        pd.read_csv(paths["labels"]),
        pd.read_csv(paths["calendar"]),
        a,
    )
    results = []
    for lag in [a["primary_hac_lags"], *a["sensitivity_hac_lags"]]:
        for signal in SIGNALS:
            values = daily.loc[daily.signal.eq(signal), "partial_rank_ic"]
            stats = precision_summary(
                values,
                lag,
                a["minimum_inference_observations"],
                a["alpha"],
                a["current_family_size"],
                a["resolution_target_rank_ic"],
                a["planning_power"],
            )
            results.append(
                {
                    "signal": signal,
                    "controls": ";".join(HYPOTHESES[signal]),
                    "primary": lag == a["primary_hac_lags"],
                    **stats,
                    "claim_eligible": False,
                }
            )
    summary = pd.DataFrame(results)
    primary = summary.loc[summary.primary].copy()
    prior = pd.read_csv(paths["prior_registry"])
    if len(prior) != a["registered_before"]:
        raise ValueError("Exploration ledger must be complete")
    additions = pd.DataFrame(
        [
            {
                "analysis": "common_sample_effect_precision",
                "signal": r.signal,
                "controls": r.controls,
                "days": r.nobs,
                "hac_p": r.p,
                "mean_partial_rank_ic": r.mean,
                "claim_eligible": False,
                "reason": "Previously viewed history; common sample and asymptotic precision diagnostic",
            }
            for r in primary.itertuples()
        ]
    )
    registry = pd.concat([prior, additions], ignore_index=True)
    family_column = f"family_{a['registered_after']}_by_q"
    registry[family_column] = by_adjust(registry.hac_p.to_numpy())
    registry[f"family_{a['registered_after']}_size"] = a["registered_after"]
    primary["registered_family_by_q"] = registry.iloc[-3:][family_column].to_numpy()
    annual = (
        daily.assign(year=daily.date.dt.year)
        .groupby(["signal", "year"], as_index=False)
        .agg(days=("partial_rank_ic", "count"), mean_partial_rank_ic=("partial_rank_ic", "mean"))
    )
    coverage = panel.groupby("code", as_index=False).agg(
        input_days=("input_eligible", "sum"),
        measurable_days=("measurement_eligible", "sum"),
        used_days=("used_in_common_estimate", "sum"),
    )
    output = paths["output"] / started.strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    frames = {
        "common_panel": panel,
        "common_daily": daily,
        "primary": primary,
        "lag_sensitivity": summary,
        "yearly": annual,
        "coverage": coverage,
        "original_reconstruction": pd.DataFrame(checks),
        "registered_exploration": registry,
    }
    for name, frame in frames.items():
        frame.to_csv(output / f"{name}.csv", index=False, encoding="utf-8-sig")
    (output / "registered_exploration_source.csv").write_bytes(paths["prior_registry"].read_bytes())
    run = {
        "version": __version__,
        "created_utc": started.isoformat(),
        "protocol": plan,
        "protocol_sha256": digest(config),
        "freeze_receipt": frozen,
        "known_exploration_family": a["registered_after"],
        "independent_new_sample": False,
        "investment_claim_eligible": False,
        "original_diagnostics_reproduced": True,
        "formal_independent_confirmation": False,
        "source_sha256": {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")},
        "output_sha256": {
            p.relative_to(output).as_posix(): digest(p) for p in output.glob("*.csv")
        },
    }
    (output / "run.json").write_text(
        json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_evidence_report(output, primary, summary, coverage, a)
    return output


def write_evidence_report(output, primary, sensitivity, coverage, settings):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
    ordered = primary.set_index("signal").loc[list(SIGNALS)].reset_index()
    for index, row in ordered.iterrows():
        ax.errorbar(
            row["mean"],
            index,
            xerr=[[row["mean"] - row.simultaneous_low], [row.simultaneous_high - row["mean"]]],
            fmt="o",
            color="#146c94",
            capsize=5,
            linewidth=2,
        )
    target = settings["resolution_target_rank_ic"]
    ax.axvspan(-target, target, color="#dcefe7", label="Declared statistical resolution band")
    ax.axvline(0, color="#516776", linewidth=1)
    ax.set(
        yticks=range(3),
        yticklabels=SIGNALS,
        xlabel="Mean partial rank correlation",
        title="Common sample: approximate 95% simultaneous intervals",
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=8, frameon=False)
    ax.grid(axis="x", alpha=0.2)
    fig.savefig(output / "effect_intervals.png", dpi=160)
    plt.close(fig)
    names = {
        "carry": "期限结构carry",
        "basis_momentum": "基差动量",
        "inventory_scarcity": "季节仓单稀缺",
    }
    display = primary[
        [
            "signal",
            "nobs",
            "mean",
            "se",
            "p",
            "point_low",
            "point_high",
            "simultaneous_low",
            "simultaneous_high",
            "normal_planning_mde",
        ]
    ].copy()
    display["signal"] = display.signal.map(names)
    display.columns = [
        "候选",
        "有效日数",
        "平均相关",
        "标准误",
        "近似p",
        "点态下限",
        "点态上限",
        "同时下限",
        "同时上限",
        "80%规划最小效应",
    ]
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>因子效应与不确定性</title><style>body{font:16px/1.8 system-ui,'Microsoft YaHei';max-width:1250px;margin:36px auto;padding:24px;background:#f2f6fa;color:#19354a}section{background:white;padding:24px;margin:22px 0;border-radius:12px}table{border-collapse:collapse;white-space:nowrap;font-size:13px}td,th{padding:9px;border-bottom:1px solid #ddd}.scroll{overflow:auto}img{max-width:100%}.notice{background:#fff0d5;padding:20px}</style><h1>同一批品种与日期：效应有多大，误差有多宽</h1><p class="notice">使用已看过的历史。区间是HAC/正态近似，不能恢复独立发现资格；绿色区域是统计分辨率目标，不是盈利门槛。不显著不能直接解释为无效。</p>"""
    page += f'<section><h2>固定40日HAC主结果</h2><div class="scroll">{display.to_html(index=False, float_format=lambda x: f"{x:.4f}")}</div><img src="effect_intervals.png" alt="共同样本效应及同时区间"></section>'
    page += '<section><p>有效日期不是独立样本数量。80%规划最小效应只描述固定估计误差下的正态模型分辨率，不是未来功效保证。完整保留20/80日敏感性，不能挑较好的结果。</p><a href="primary.csv">主结果</a> · <a href="lag_sensitivity.csv">全部滞后敏感性</a> · <a href="coverage.csv">品种覆盖</a> · <a href="run.json">方法与输入记录</a></section></html>'
    (output / "index.html").write_text(page, encoding="utf-8")
