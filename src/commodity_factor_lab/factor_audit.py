"""Evidence inventory for every legacy indicator; never ranks candidates by profit."""

import hashlib
import json
import math
import tomllib
from collections import defaultdict
from datetime import datetime, timezone
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

from .catalog import FAMILIES, SECTOR_MAP, code_from_filename
from .data import read_csv

DECISIONS = {
    "rebuild": "重建",
    "merge": "并入价格基准",
    "retire": "撤出收益检验",
    "risk_only": "风险用途",
}


def load_decisions(path: Path) -> list[dict]:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    rows = raw["factors"]
    expected = {(family, name) for family, item in FAMILIES.items() for name in item.factors}
    keys = [(r["family"], r["legacy_id"]) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Decision registry must cover every legacy indicator exactly once")
    for row in rows:
        if row["decision"] not in DECISIONS or not all(
            row.get(k) for k in ("reason", "next_gate", "source_anchor", "legacy_formula")
        ):
            raise ValueError(
                "Every decision requires a role, source, reason and next evidence gate"
            )
    return rows


def valid_dates(frame: pd.DataFrame) -> pd.Series:
    source = frame.get("date", frame.get("Unnamed: 0", pd.Series(index=frame.index, dtype=object)))
    text = source.astype(str)
    values = pd.to_datetime(
        text.where(text.str.fullmatch(r"\d{4}-\d{2}-\d{2}(?: 00:00:00)?")), errors="coerce"
    )
    return values.where(values.ge("2000-01-01"))


def arithmetic_check(frame: pd.DataFrame, name: str, expected: pd.Series, source: str) -> dict:
    observed = pd.to_numeric(frame[name], errors="coerce")
    actual_ok, expected_ok = np.isfinite(observed), np.isfinite(expected)
    common = actual_ok & expected_ok
    errors = (observed.loc[common] - expected.loc[common]).abs()
    return {
        "file": source,
        "factor": name,
        "comparable_rows": int(common.sum()),
        "matches_rounding_tolerance": int(errors.le(5.1e-7).sum()),
        "differences": int(errors.gt(5.1e-7).sum()),
        "legacy_present_reference_missing": int((actual_ok & ~expected_ok).sum()),
        "reference_present_legacy_missing": int((~actual_ok & expected_ok).sum()),
        "max_absolute_error": float(errors.max()) if len(errors) else math.nan,
        "policy": "strict no-fill arithmetic on retained rows; missing prehistory is not a legacy error",
    }


def audit_factors(data: Path, registry: Path, output: Path) -> Path:
    if output.exists():
        raise FileExistsError("Factor audit output must be a new directory")
    if not data.is_dir():
        raise ValueError("Legacy source directory is missing")
    decisions = load_decisions(registry)
    project = registry.resolve().parent.parent
    inputs, evidence, arithmetic, coverage = [], [], [], []
    by_family = defaultdict(list)
    for row in decisions:
        path = project / row["source_file"]
        content = path.read_text(encoding="utf-8-sig")
        lines = [
            i for i, line in enumerate(content.splitlines(), 1) if row["source_anchor"] in line
        ]
        if not lines:
            raise ValueError(f"Source evidence anchor changed: {row['legacy_id']}")
        evidence.append(
            {
                **row,
                "source_line": lines[0],
                "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
        by_family[row["family"]].append(row)
    for family, rows in by_family.items():
        stats = {
            r["legacy_id"]: {
                "files": 0,
                "missing_column_files": 0,
                "finite_values": 0,
                "finite_values_with_valid_ohlc": 0,
                "invalid_date_values": 0,
                "products": set(),
                "dates": set(),
            }
            for r in rows
        }
        for path in sorted((data / family).glob("*.csv")):
            code = code_from_filename(path.name)
            financial = SECTOR_MAP[code] == "financial"
            inputs.append(
                {
                    "file": str(path.resolve()),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "financial_excluded": financial,
                }
            )
            if financial:
                continue
            frame = read_csv(path)
            dates = valid_dates(frame)
            prices = frame.reindex(columns=["open", "high", "low", "close"]).apply(
                pd.to_numeric, errors="coerce"
            )
            valid_price = np.isfinite(prices).all(axis=1) & prices.gt(0).all(axis=1)
            valid_price &= prices.high.ge(
                prices[["open", "close", "low"]].max(axis=1)
            ) & prices.low.le(prices[["open", "close", "high"]].min(axis=1))
            for name, item in stats.items():
                item["files"] += 1
                if name not in frame:
                    item["missing_column_files"] += 1
                    continue
                values = pd.to_numeric(frame[name], errors="coerce")
                present = np.isfinite(values)
                valid = present & dates.notna()
                item["finite_values"] += int(valid.sum())
                item["finite_values_with_valid_ohlc"] += int((valid & valid_price).sum())
                item["invalid_date_values"] += int((present & dates.isna()).sum())
                if valid.any():
                    item["products"].add(code)
                    item["dates"].update(dates.loc[valid].dt.strftime("%Y-%m-%d"))
            clean = frame.loc[dates.notna()].copy()
            clean["audit_date"] = dates.loc[dates.notna()]
            clean = clean.sort_values("audit_date")
            if family == "value":
                close = pd.to_numeric(clean.close, errors="coerce")
                for name, window in (("Val_halfyear", 126), ("Val_oneyear", 252)):
                    if name in clean:
                        expected = close.div(close.shift(window)).sub(1)
                        result = arithmetic_check(clean, name, expected, path.name)
                        # Retained files omit warm-up history; do not flag those first rows.
                        result["legacy_present_reference_missing"] = int(
                            (
                                np.isfinite(pd.to_numeric(clean[name], errors="coerce"))
                                & ~np.isfinite(expected)
                                & (np.arange(len(clean)) >= window)
                            ).sum()
                        )
                        arithmetic.append(result)
            if family == "basis" and "spot_price" in clean:
                close = pd.to_numeric(clean.close, errors="coerce")
                spot = pd.to_numeric(clean.spot_price, errors="coerce")
                expected = {
                    "fs_basis": spot - close,
                    "fs_carry": (spot - close) / close,
                    "fs_exmom": close.pct_change(fill_method=None)
                    - spot.pct_change(fill_method=None),
                }
                for name, values in expected.items():
                    if name in clean:
                        result = arithmetic_check(clean, name, values, path.name)
                        if name == "fs_exmom" and len(clean):
                            result["legacy_present_reference_missing"] -= int(
                                np.isfinite(pd.to_numeric(clean[name], errors="coerce").iloc[0])
                            )
                        arithmetic.append(result)
        for name, item in stats.items():
            days = sorted(item.pop("dates"))
            products = sorted(item.pop("products"))
            coverage.append(
                {
                    "family": family,
                    "legacy_id": name,
                    **item,
                    "data_status": "missing_source"
                    if not item["files"]
                    else (
                        "no_finite_values" if not item["finite_values"] else "observed_unverified"
                    ),
                    "product_count": len(products),
                    "products": ";".join(products),
                    "observed_dates": len(days),
                    "first_date": days[0] if days else None,
                    "last_date": days[-1] if days else None,
                }
            )
    if not inputs:
        raise ValueError("No legacy source files were audited")
    output.mkdir(parents=True)
    summary = pd.DataFrame(evidence).merge(
        pd.DataFrame(coverage), on=["family", "legacy_id"], validate="one_to_one"
    )
    summary.to_csv(output / "factor_decisions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(arithmetic).to_csv(
        output / "arithmetic_checks.csv", index=False, encoding="utf-8-sig"
    )
    receipt = {
        "kind": "complete_legacy_factor_evidence_inventory",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "indicators": len(summary),
        "families": len(by_family),
        "input_files": len(inputs),
        "registry_sha256": hashlib.sha256(registry.read_bytes()).hexdigest(),
        "audit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "decisions": summary.decision.value_counts().to_dict(),
        "inputs": inputs,
        "source_evidence": [
            {"file": r["source_file"], "sha256": r["source_sha256"]} for r in evidence
        ],
        "limitations": [
            "finite legacy values do not certify source provenance or tradability",
            "retained-row arithmetic cannot restore omitted warmup or missing calendar dates",
            "no performance-based selection or new investment claim",
        ],
        "claim_eligible": False,
    }
    (output / "audit.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    displayed = summary[
        [
            "family",
            "legacy_id",
            "decision",
            "canonical_family",
            "product_count",
            "observed_dates",
            "first_date",
            "last_date",
            "reason",
        ]
    ].copy()
    displayed.decision = displayed.decision.map(DECISIONS)
    displayed = displayed.rename(
        columns={
            "family": "原类别",
            "legacy_id": "原指标",
            "decision": "处理",
            "canonical_family": "归入方向",
            "product_count": "品种数",
            "observed_dates": "观测日期数",
            "first_date": "开始",
            "last_date": "结束",
            "reason": "依据",
        }
    )
    page = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>七类指标的去留与证据</title><style>body{{font:15px/1.8 "Segoe UI","Microsoft YaHei",sans-serif;background:#f1f5f8;color:#173149;margin:35px auto;max-width:1450px;padding:24px}}section{{background:white;padding:24px;border-radius:12px;margin:24px 0}}.scroll{{overflow:auto}}table{{border-collapse:collapse;font-size:12px}}td,th{{padding:12px;border-bottom:1px solid #dce5ec;text-align:left;min-width:90px}}td:last-child{{min-width:340px}}a{{color:#147d92}}.notice{{background:#fff1df;padding:18px}}</style><h1>七类、26 个旧指标：逐项处理与证据</h1><p class="notice">按经济含义和数据证据决定角色，没有按回测收益选赢家。覆盖统计仅描述原快照，金融期货默认排除；非空数量不等于有效独立样本。</p><section><h2>每个指标都有明确去向</h2><p>{escape(str(receipt["decisions"]))}</p><div class="scroll">{displayed.fillna("—").to_html(index=False, border=0, escape=True)}</div></section><section><h2>可以继续核查的明细</h2><p><a href="factor_decisions.csv">完整定义、来源行号、去留与准入条件</a> · <a href="arithmetic_checks.csv">价值/动量及现期公式重算</a> · <a href="audit.json">原文件与配置校验</a></p><p>旧半年和一年“价值”归入价格动量族，不代表两个期限数值相同；新同合约链也不与旧连续价格列混称等价。持仓与现货候选需要完整原始输入和时间证据，不能从旧派生值反推成真实基本面。</p></section></html>"""
    (output / "index.html").write_text(page, encoding="utf-8")
    return output
