"""Strict SHFE broker-rank source profiling, without treating missing ranks as zero."""

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def parse_position_ranks(payload: bytes, expected_date: str) -> pd.DataFrame:
    obj = json.loads(payload)
    if str(obj.get("report_date")) != expected_date:
        raise ValueError("Position ranks are not for the requested report date")
    clock = obj.get("update_date", "")
    if not isinstance(clock, str) or ":" not in clock:
        raise ValueError("Position publication clock is unknown")
    available = pd.Timestamp(clock)
    if available.tzinfo is None:
        available = available.tz_localize("Asia/Shanghai")
    available = available.tz_convert("UTC")
    if available.tz_convert("Asia/Shanghai").date() < pd.Timestamp(expected_date).date():
        raise ValueError("Position publication precedes its report date")
    groups = {}
    for row in obj["o_cursor"]:
        contract = str(row["INSTRUMENTID"]).strip()
        if not re.fullmatch(r"[a-z]+\d{4}", contract):
            continue  # Product aggregates, total rows and options are not contracts.
        groups.setdefault(contract, [])
        rank = float(row["RANK"])
        if rank.is_integer() and 1 <= rank <= 20:
            groups[contract].append(row)
    results = []
    for contract, rows in sorted(groups.items()):
        ranks = [int(row["RANK"]) for row in rows]
        complete = len(ranks) == 20 and set(ranks) == set(range(1, 21))
        totals = {}
        for side, suffix in (("long", "2"), ("short", "3")):
            members = [str(row.get("PARTICIPANTID" + suffix, "")).strip() for row in rows]
            values = pd.to_numeric(
                pd.Series([row.get("CJ" + suffix) for row in rows], dtype=object),
                errors="coerce",
            )
            valid = (
                len(members) == 20
                and len(set(members)) == 20
                and all(re.fullmatch(r"\d+", member) for member in members)
                and np.isfinite(values).all()
                and values.ge(0).all()
                and values.mod(1).eq(0).all()
            )
            complete &= valid
            totals[side + "_top20"] = float(values.sum()) if valid else np.nan
        if not complete:
            totals = {key: np.nan for key in totals}
        results.append(
            {
                "date": pd.Timestamp(expected_date),
                "contract": contract,
                "available_at": available,
                "ranks": len(ranks),
                "status": "complete_top20" if complete else "incomplete_or_invalid_ranks",
                **totals,
            }
        )
    return pd.DataFrame(
        results,
        columns=[
            "date",
            "contract",
            "available_at",
            "ranks",
            "status",
            "long_top20",
            "short_top20",
        ],
    )


def profile_position_sources(raw: Path, curves: Path, output: Path) -> Path:
    if output.exists():
        raise FileExistsError("Position profile output must be new")
    manifest_path = raw / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected = pd.read_csv(curves, parse_dates=["date"])
    if selected.duplicated(["date", "code"]).any():
        raise ValueError("Duplicate selected product dates")
    parsed, inputs, requested = [], [], []
    for item in manifest["records"]:
        if item["status"] != "ok":
            continue
        date = item["date"]
        if not re.fullmatch(r"\d{8}", date) or date in requested:
            raise ValueError("Invalid or duplicate source date")
        requested.append(date)
        source = raw / f"positions_{date}.json"
        content = source.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if digest != item["sha256"]:
            raise ValueError("Position response no longer matches its manifest")
        parsed.append(parse_position_ranks(content, date))
        inputs.append({"file": str(source.resolve()), "sha256": digest})
    if not parsed:
        raise ValueError("No successful ranking responses")
    ranks = pd.concat(parsed, ignore_index=True)
    subset = selected.loc[selected.date.isin(pd.to_datetime(requested))].copy()
    subset = subset[["date", "code", "front_contract", "signal_cutoff"]]
    joined = subset.merge(
        ranks,
        left_on=["date", "front_contract"],
        right_on=["date", "contract"],
        how="left",
        validate="one_to_one",
    )
    joined["status"] = joined.status.fillna("not_disclosed")
    clocks = pd.to_datetime(joined.available_at, utc=True)
    cutoff = pd.to_datetime(joined.signal_cutoff, utc=True)
    joined.loc[joined.status.eq("complete_top20") & clocks.gt(cutoff), "status"] = (
        "published_after_cutoff"
    )
    joined["usable_at_cutoff"] = joined.status.eq("complete_top20") & clocks.le(cutoff)
    summary = (
        joined.groupby("date")
        .agg(selected=("code", "count"), complete_at_cutoff=("usable_at_cutoff", "sum"))
        .reset_index()
    )
    output.mkdir(parents=True)
    ranks.to_csv(output / "contract_top20.csv", index=False, encoding="utf-8-sig")
    joined.to_csv(output / "selected_contract_coverage.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output / "coverage_by_date.csv", index=False)
    receipt = {
        "kind": "bounded_source_readiness_cases_not_representative_coverage",
        "inputs": inputs,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "curves_sha256": hashlib.sha256(curves.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "selected_cases": len(joined),
        "complete_at_cutoff": int(joined.usable_at_cutoff.sum()),
        "limitations": [
            "Five selected dates do not estimate population-wide completeness",
            "Current downloadable publication clocks do not certify initial historical versions",
            "Changing top20 broker membership is not a fixed trader cohort or COT classification",
            "No complete same-contract lagged panel or open-interest normalization evaluated",
        ],
        "claim_eligible": False,
    }
    (output / "profile.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    page = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>持仓来源准入核查</title><style>body{{font:16px/1.8 'Segoe UI','Microsoft YaHei',sans-serif;max-width:1100px;margin:40px auto;padding:24px;color:#173149}}table{{border-collapse:collapse}}td,th{{padding:12px;border-bottom:1px solid #ddd}}a{{color:#147d92}}</style><h1>持仓排名：来源通了，研究输入尚未完整</h1><p>五个固定日期的 {len(joined)} 个所选合约案例中，{int(joined.usable_at_cutoff.sum())} 个在截止时有完整 Top20。缺失不填零，不根据可披露合约重选原选约规则；此小样本不代表总体覆盖率。</p>{summary.to_html(index=False)}<p>检查报告日期、发布时间、合约身份、1–20 唯一排名、多空双方的唯一数字会员编号与非负整数持仓；排除品种汇总、总计和期权。不采用经纪会员中文名称作为稳定身份。</p><p>下一准入条件是完整逐日同合约历史、单边总持仓与发布版本核查。会员排名不等于商业套保者，榜单进出会影响净持仓变化的解释。本轮没有生成持仓因子的收益检验。</p><p><a href="selected_contract_coverage.csv">全部案例与缺失</a> · <a href="contract_top20.csv">原合约排名合计</a> · <a href="profile.json">来源校验与边界</a></p></html>"""
    (output / "index.html").write_text(page, encoding="utf-8")
    return output
