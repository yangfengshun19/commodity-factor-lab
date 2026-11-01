"""Registered incremental hypotheses, with source-gated tests kept in the family."""

import hashlib
import json
import platform
import tomllib
from datetime import datetime, timezone
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .contract_report import table
from .contract_research import partial_rank_ic
from .factor_audit import load_decisions
from .inference import by_adjust, hac_mean


def verify_frozen_inputs(config: dict, project: Path) -> tuple[dict, list[dict]]:
    paths = {k: (project / v).resolve() for k, v in config["paths"].items()}
    sources = {
        "curves": paths["baseline_run"] / "curves.csv",
        "labels": paths["baseline_run"] / "labels.csv",
        "run": paths["baseline_run"] / "run.json",
        "prior_exploration": paths["prior_exploration"],
    }
    receipts = []
    for name, source in sources.items():
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if config["input_hashes"].get(name) != digest:
            raise ValueError(f"Frozen extension input changed: {name}")
        receipts.append({"file": str(source), "sha256": digest})
    return paths, receipts


def evaluate_extension(
    data: pd.DataFrame, hypotheses: list[dict], study: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    names = [h["name"] for h in hypotheses]
    if len(names) != len(set(names)) or not names:
        raise ValueError("Hypothesis names must be unique and nonempty")
    if study["min_assets"] < 8 or study["hac_lags"] < study["holding_sessions"]:
        raise ValueError("Extension requires the fixed minimum cross-section and overlap-aware HAC")
    diagnostics = []
    for hypothesis in hypotheses:
        name = hypothesis["name"]
        if hypothesis["direction"] != 1:
            raise ValueError("This registered extension permits no retrospective sign reversal")
        status = hypothesis["status"]
        if status == "ready":
            if name not in data or set(hypothesis["controls"]) - set(data):
                raise ValueError("Ready hypothesis has no declared source feature/control")
            for date, group in data.groupby("date"):
                diagnostics.append(
                    {
                        "date": date,
                        "signal": name,
                        **partial_rank_ic(
                            group, name, tuple(hypothesis["controls"]), study["min_assets"]
                        ),
                    }
                )
        elif status == "gated_source_validation":
            for date in sorted(data.date.unique()):
                diagnostics.append(
                    {
                        "date": date,
                        "signal": name,
                        "rank_ic": np.nan,
                        "partial_rank_ic": np.nan,
                        "assets": 0,
                        "status": status,
                    }
                )
        else:
            raise ValueError("Unknown source readiness state")
    daily = pd.DataFrame(diagnostics)
    summary = []
    for hypothesis in hypotheses:
        subset = daily.loc[
            daily.signal.eq(hypothesis["name"])
            & daily.date.ge(pd.Timestamp(study["evaluation_start"]))
        ].copy()
        stats = hac_mean(subset.partial_rank_ic, study["hac_lags"], study["min_inference_obs"])
        good = subset.partial_rank_ic.notna()
        summary.append(
            {
                "signal": hypothesis["name"],
                "controls": ";".join(hypothesis["controls"]),
                "source_status": hypothesis["status"],
                "days": stats["nobs"],
                "mean_partial_rank_ic": stats["mean"],
                "hac_t": stats["t"],
                "hac_p": stats["p"],
                "inference_status": stats["status"]
                if hypothesis["status"] == "ready"
                else hypothesis["status"],
                "min_assets_used": int(subset.loc[good, "assets"].min()) if good.any() else 0,
                "mean_assets_used": subset.loc[good, "assets"].mean(),
                "reason": hypothesis["reason"],
                "claim_eligible": False,
            }
        )
    summary = pd.DataFrame(summary)
    summary["by_q"] = by_adjust(summary.hac_p.to_numpy())
    return daily, summary


def run_extension(config_path: Path) -> Path:
    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    project = config_path.resolve().parent.parent
    paths, receipts = verify_frozen_inputs(raw, project)
    load_decisions(paths["registry"])
    curves = pd.read_csv(paths["baseline_run"] / "curves.csv", parse_dates=["date"])
    labels = pd.read_csv(paths["baseline_run"] / "labels.csv", parse_dates=["date"])
    baseline = json.loads((paths["baseline_run"] / "run.json").read_text(encoding="utf-8"))
    for field in ("holding_sessions", "entry_lag_sessions"):
        if raw["study"][field] != baseline["protocol"]["study"][field]:
            raise ValueError("Frozen prediction labels do not match the extension horizon")
    data = curves.merge(
        labels[["date", "code", "target"]], on=["date", "code"], validate="one_to_one"
    )
    daily, summary = evaluate_extension(data, raw["hypotheses"], raw["study"])
    prior = pd.read_csv(paths["prior_exploration"])
    current = summary.copy()
    current["analysis"] = "registered_other_factor_extension"
    registry = pd.concat([prior, current], ignore_index=True)
    registry["registered_exploration_by_q"] = by_adjust(registry.hac_p.to_numpy())
    lookup = (
        registry.loc[registry.analysis.eq("registered_other_factor_extension")]
        .set_index("signal")
        .registered_exploration_by_q
    )
    summary["registered_exploration_by_q"] = summary.signal.map(lookup)
    summary["positive_predictive_candidate"] = summary.by_q.le(
        0.05
    ) & summary.mean_partial_rank_ic.gt(0)
    evaluation = daily.loc[daily.date.ge(pd.Timestamp(raw["study"]["evaluation_start"]))].copy()
    evaluation["year"] = evaluation.date.dt.year
    annual = (
        evaluation.groupby(["signal", "year"])
        .agg(days=("partial_rank_ic", "count"), mean_partial_rank_ic=("partial_rank_ic", "mean"))
        .reset_index()
    )
    omit = []
    for hypothesis in raw["hypotheses"]:
        if hypothesis["status"] != "ready":
            continue
        name, controls = hypothesis["name"], tuple(hypothesis["controls"])
        days = set(
            evaluation.loc[evaluation.signal.eq(name) & evaluation.partial_rank_ic.notna(), "date"]
        )
        for code in sorted(data.code.unique()):
            values = [
                partial_rank_ic(
                    g.loc[g.code.ne(code)], name, controls, raw["study"]["min_assets"] - 1
                )["partial_rank_ic"]
                for _, g in data.loc[data.date.isin(days)].groupby("date")
            ]
            finite = np.array(values, dtype=float)
            finite = finite[np.isfinite(finite)]
            omit.append(
                {
                    "signal": name,
                    "omitted_product": code,
                    "days": len(finite),
                    "mean_partial_rank_ic": float(finite.mean()) if len(finite) else np.nan,
                    "exploratory_only": True,
                }
            )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = paths["output"] / stamp
    output.mkdir(parents=True, exist_ok=False)
    for name, frame in (
        ("daily_ic", daily),
        ("summary", summary),
        ("registered_exploration", registry),
        ("yearly_diagnostics", annual),
        ("leave_one_product_out", pd.DataFrame(omit)),
    ):
        frame.to_csv(output / f"{name}.csv", index=False, encoding="utf-8-sig")
    run = {
        "version": __version__,
        "python": platform.python_version(),
        "created_utc": stamp,
        "kind": "registered_retrospective_factor_extension",
        "protocol": raw,
        "protocol_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "inputs": receipts,
        "registry_sha256": hashlib.sha256(paths["registry"].read_bytes()).hexdigest(),
        "current_family": len(summary),
        "known_exploration_family": len(registry),
        "gated_hypotheses_count_in_family": True,
        "earlier_exploration_fully_covered": False,
        "economic_effectiveness_confirmed": False,
        "claim_eligible": False,
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob("*.py"))
        },
    }
    (output / "run.json").write_text(
        json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    reasons = "".join(
        f"<li>{escape(h['name'])}：{escape(h['reason'])}</li>" for h in raw["hypotheses"]
    )
    page = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>其他因子的登记检验</title><style>body{{font:15px/1.8 "Segoe UI","Microsoft YaHei",sans-serif;background:#f1f5f8;color:#173149;max-width:1350px;margin:35px auto;padding:24px}}section{{background:#fff;padding:24px;border-radius:12px;margin:24px 0}}.notice{{background:#fff1df;padding:18px}}.scroll{{overflow:auto}}table{{border-collapse:collapse;white-space:nowrap;font-size:12px}}td,th{{padding:11px;border-bottom:1px solid #dce5ec;text-align:left}}a{{color:#147d92}}</style><h1>其他因子：登记、准入与增量检验</h1><p class="notice">本轮只对数据已就绪的相对基差代理计算新增诊断。持仓与真实现货基差因输入未达准入条件保持不可检验，仍占完整三项校正族。历史已被查看，本次登记不构成独立样本验证。</p><section><h2>完整三项与已登记探索</h2>{table(summary)}<p>本轮三项 BY 包括两项数据不足；已登记探索族共 {len(registry)} 项，仍不覆盖全部旧研究搜索。不能只挑本轮最小 p 值宣传发现，也不把不可检验写成因子无效。</p><ul>{reasons}</ul></section><section><h2>公式与范围</h2><p>相对基差代理为近/次近与次近/第三合约年化对数斜率之差，控制 126 日动量、carry 和 252 日基差动量；最少八品种、未来二十交易日同合约结算标签、四十日 HAC。与论文公式结构一致，但本地六十日到期筛选、日频信息及样本不同，未完整复现论文实证。</p><p>持仓候选要求同合约、完整排名、单边持仓量与发布时间。经纪会员名单不等于商业套保者；没有披露的合约不补零。现货基差要求等级、地区、税费、单位、合约匹配和发布时点，旧派生列不能代替这些证据。</p></section><section><h2>年度与留一品种</h2>{table(annual)}<p>留一品种只作描述性检查，日期可能因秩不足而变化，不追加事后显著性。</p><a href="leave_one_product_out.csv">完整留一品种</a></section><section><h2>可复核结果</h2><a href="summary.csv">主检验与准入状态</a> · <a href="daily_ic.csv">逐日结果</a> · <a href="registered_exploration.csv">全部登记探索</a> · <a href="run.json">冻结协议和输入/源码哈希</a><p>没有生成真实历史投资收益或策略净值。</p></section></html>"""
    (output / "index.html").write_text(page, encoding="utf-8")
    return output
