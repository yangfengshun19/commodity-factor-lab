"""Independent raw-JSON arithmetic audit; imports no production factor/parser code."""

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


def raw_totals(payload: dict, date: str) -> dict:
    if str(payload["report_date"]) != date:
        raise ValueError("Wrong raw ranking report date")
    clock = payload["update_date"]
    parsed = None
    for pattern in ("%Y%m%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            parsed = datetime.strptime(clock, pattern).replace(tzinfo=timezone(timedelta(hours=8)))
            break
        except ValueError:
            continue
    if parsed is None:
        raise ValueError("Unsupported raw publication clock")
    groups = {}
    for item in payload["o_cursor"]:
        contract = item["INSTRUMENTID"].strip()
        if not re.fullmatch(r"[a-z]+\d{4}", contract):
            continue
        rank = float(item["RANK"])
        if rank.is_integer() and 1 <= rank <= 20:
            groups.setdefault(contract, []).append(item)
    output = {}
    for contract, items in groups.items():
        if sorted(float(item["RANK"]) for item in items) != list(range(1, 21)):
            continue
        sums = []
        for side in ("2", "3"):
            members = [str(item.get("PARTICIPANTID" + side, "")).strip() for item in items]
            if len(set(members)) != 20 or not all(
                re.fullmatch(r"\d+", member) for member in members
            ):
                break
            try:
                values = [float(item["CJ" + side]) for item in items]
                if not all(
                    math.isfinite(value) and value >= 0 and value.is_integer() for value in values
                ):
                    break
            except (TypeError, ValueError, KeyError):
                break
            sums.append(sum(values))
        if len(sums) == 2:
            output[contract] = (sums[0], sums[1], parsed)
    return output


def reconcile(
    raw: Path,
    report: Path,
    bars_file: Path,
    calendar_file: Path,
    daily_raw: Path,
    output: Path,
    cutoff_policy: str = "own_day",
    qualified_report: Path | None = None,
) -> dict:
    if cutoff_policy not in {"own_day", "current_signal"}:
        raise ValueError("Unknown independent cutoff policy")
    qualified_file = (qualified_report or report) / "qualified_contract_ranks.csv"
    panel_file = report / (
        "selected_panel.csv" if cutoff_policy == "current_signal" else "selected_position_panel.csv"
    )
    if output.exists():
        raise FileExistsError("Independent audit output must be new")
    manifest_path = raw / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sources, receipts = {}, []
    for item in manifest["records"]:
        if item["status"] != "ok":
            continue
        path = raw / "responses" / f"{item['date']}.json"
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if digest != item["sha256"]:
            raise ValueError("Raw response hash differs from receipt")
        sources[item["date"]] = raw_totals(json.loads(content), item["date"])
        receipts.append({"date": item["date"], "sha256": digest})
    bars = pd.read_csv(bars_file)
    oi = {(row.date.replace("-", ""), row.contract): row.open_interest for row in bars.itertuples()}
    calendar = pd.read_csv(calendar_file)
    cutoffs = {
        row.date.replace("-", ""): datetime.fromisoformat(row.signal_cutoff)
        for row in calendar.itertuples()
    }
    dates = manifest["planned_dates"]
    positions = {date: i for i, date in enumerate(dates)}
    counts, differences = Counter(), []

    def check(kind, key, expected, observed):
        counts[kind] += 1
        equal = pd.isna(expected) and pd.isna(observed)
        if not equal and not pd.isna(expected) and not pd.isna(observed):
            equal = math.isclose(float(expected), float(observed), rel_tol=1e-12, abs_tol=1e-12)
        if not equal:
            differences.append(
                {"kind": kind, "key": key, "expected": expected, "observed": observed}
            )

    def net_at(date, contract, asof=None):
        record = sources.get(date, {}).get(contract)
        value = oi.get((date, contract), math.nan)
        if (
            record is None
            or not math.isfinite(value)
            or value <= 0
            or not float(value).is_integer()
        ):
            return math.nan
        long, short, clock = record
        if max(long, short) > value or clock > (asof if asof is not None else cutoffs[date]):
            return math.nan
        return long - short

    qualified = pd.read_csv(qualified_file)
    for row in qualified.itertuples():
        date = row.date.replace("-", "")
        original = sources.get(date, {}).get(row.contract)
        for column, i in (("long_top20", 0), ("short_top20", 1)):
            check(
                "raw_top20_sums",
                date + "/" + row.contract + "/" + column,
                original[i] if original else math.nan,
                getattr(row, column),
            )
        check(
            "qualified_net_and_timing",
            date + "/" + row.contract,
            net_at(date, row.contract),
            row.net_top20,
        )
    panel = pd.read_csv(panel_file)
    for row in panel.itertuples():
        date = row.date.replace("-", "")
        i, contract = positions[date], row.contract
        expected = math.nan
        if i >= 20 and isinstance(contract, str):
            asof = cutoffs[date] if cutoff_policy == "current_signal" else None
            history = [net_at(d, contract, asof) for d in dates[i - 20 : i + 1]]
            if all(math.isfinite(value) for value in history):
                expected = (history[-1] - history[0]) / oi[date, contract]
        check(
            "same_contract_twenty_session_flow",
            date + "/" + str(contract),
            expected,
            row.normalized_position_flow,
        )
    days_used, daily_receipts = [], []
    for source in sorted(daily_raw.glob("[0-9]" * 8 + ".json")):
        date = source.stem
        if date not in sources:
            continue
        obj = json.loads(source.read_bytes())
        if str(obj["report_date"]) != date:
            raise ValueError("Wrong independent daily report date")
        days_used.append(date)
        daily_receipts.append(
            {"date": date, "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
        )
        for item in obj["o_curinstrument"]:
            code, month = str(item["PRODUCTGROUPID"]).strip(), str(item["DELIVERYMONTH"]).strip()
            contract = code + month
            if re.fullmatch(r"[a-z]+\d{4}", contract) and (date, contract) in oi:
                check(
                    "daily_export_vs_annual_oi",
                    date + "/" + contract,
                    float(item["OPENINTEREST"]),
                    oi[date, contract],
                )
    output.mkdir(parents=True)
    pd.DataFrame(differences, columns=["kind", "key", "expected", "observed"]).to_csv(
        output / "differences.csv", index=False
    )
    result = {
        "kind": "independent_raw_position_arithmetic",
        "checks": sum(counts.values()),
        "checks_by_kind": dict(counts),
        "differences": len(differences),
        "historical_window_cutoff_policy": cutoff_policy,
        "raw_responses": receipts,
        "independent_daily_cases": days_used,
        "independent_daily_receipts": daily_receipts,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "inputs": {
            str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                manifest_path,
                bars_file,
                calendar_file,
                report / "run.json",
                qualified_file,
                panel_file,
            )
        },
        "scope": "same-exchange raw exports and independent arithmetic, not historical-vintage or execution certification",
        "claim_eligible": False,
    }
    (output / "audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    page = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>持仓独立对账</title><style>body{{font:16px/1.8 'Segoe UI','Microsoft YaHei',sans-serif;max-width:1050px;margin:40px auto;color:#173149}}td,th{{padding:12px;text-align:left}}</style><h1>持仓原始算术与单边总持仓核对</h1><p>{result["checks"]:,} 项检查，{result["differences"]} 项差异。检查高度相关，不是独立研究样本。</p>{pd.DataFrame(list(counts.items()), columns=["核对范围", "次数"]).to_html(index=False)}<p>独立读取原 JSON 并逐项相加，不调用生产解析、滚动窗口或因子函数；重算连续二十交易日同合约变化。另使用 {len(days_used)} 个已保存官方单日导出核对年包总持仓。</p><p>这只能支持算术及同交易所不同导出的一致性，不能认证历史初发版本或可成交条件。</p><a href="differences.csv">所有差异</a> · <a href="audit.json">来源校验与范围</a></html>"""
    (output / "index.html").write_text(page, encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("raw", "report", "bars", "calendar", "daily-raw", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--cutoff-policy", choices=["own_day", "current_signal"], default="own_day")
    parser.add_argument("--qualified-report", type=Path)
    args = parser.parse_args()
    result = reconcile(
        args.raw,
        args.report,
        args.bars,
        args.calendar,
        args.daily_raw,
        args.output,
        args.cutoff_policy,
        args.qualified_report,
    )
    print({key: result[key] for key in ("checks", "differences", "checks_by_kind")})
    raise SystemExit(int(result["differences"] != 0))
