"""Independent reference arithmetic and raw-source reconciliation.

This module never calls the production curve, fundamental or source parsers.
Agreement verifies the stated implementation, not the economic hypothesis or a
point-in-time market-data vintage. Daily and annual files are two SHFE exports,
not two independent data providers.
"""

import calendar
import hashlib
import json
import math
import re
import zipfile
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from html import escape
from pathlib import Path

import pandas as pd

LOCAL = timezone(timedelta(hours=8))
WAREHOUSE_NAMES = {
    "铜": "cu",
    "铝": "al",
    "锌": "zn",
    "螺纹钢": "rb",
    "螺纹钢仓库": "rb",
    "热轧卷板": "hc",
    "热轧卷板仓库": "hc",
    "天然橡胶": "ru",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clock(text):
    if not isinstance(text, str) or not re.search(r"\d{2}:\d{2}:\d{2}", text):
        return None
    for pattern in ("%Y%m%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, pattern).replace(tzinfo=LOCAL).astimezone(timezone.utc)
        except ValueError:
            pass
    value = datetime.fromisoformat(text)
    return (value if value.tzinfo else value.replace(tzinfo=LOCAL)).astimezone(timezone.utc)


def number(value):
    try:
        return float(str(value).replace(",", ""))
    except (ValueError, TypeError):
        return math.nan


def workbook_identity(cell, current):
    """Merged blanks inherit identity; an option/header/total starts a new block."""
    if pd.isna(cell) or str(cell).strip() == "":
        return current
    label = str(cell).strip().lower()
    return label if re.fullmatch(r"[a-z]+\d{4}", label) else None


def raw_metadata(directory: Path, receipt: dict) -> tuple[list[dict], list[dict]]:
    result, sources = [], []
    for record in receipt["records"]:
        day = record["query_date"]
        path = directory / f"{day}.json"
        if sha(path) != record["raw_sha256"]:
            raise ValueError("Raw metadata hash mismatch")
        obj = json.loads(path.read_bytes())
        if str(obj["report_date"]) != day:
            raise ValueError("Metadata report date mismatch")
        top_clock = clock(obj.get("update_date"))
        for row in obj["ContractBaseInfo"]:
            available = [v for v in (top_clock, clock(row.get("UPDATE_DATE"))) if v is not None]
            contract = str(row["INSTRUMENTID"]).strip().lower()
            if available:
                result.append(
                    {
                        "contract": contract,
                        "expiry": datetime.strptime(row["EXPIREDATE"], "%Y%m%d").date(),
                        "published": max(available),
                        "source": record["url"],
                    }
                )
        sources.append({"file": str(path), "sha256": record["raw_sha256"]})
    return result, sources


def reference_selection(
    bars: pd.DataFrame, metadata: list[dict], sessions: pd.DataFrame, rules: dict
) -> tuple[dict, list[dict]]:
    histories = defaultdict(list)
    for row in metadata:
        histories[row["contract"]].append(row)
    for history in histories.values():
        history.sort(key=lambda r: r["published"])
    grouped = {(d, c): g.to_dict("records") for (d, c), g in bars.groupby(["date", "code"])}
    marks = {(r.date, r.contract): r.settle for r in bars.itertuples()}
    selected, output = {}, []
    for code in sorted(bars.code.unique()):
        previous = []
        for session in sessions.itertuples():
            candidates = []
            for bar in grouped.get((session.date, code), []):
                known = [
                    r for r in histories[bar["contract"]] if r["published"] <= session.signal_cutoff
                ]
                if not known:
                    continue
                expiry = known[-1]["expiry"]
                days = (expiry - session.date.date()).days
                if (
                    rules["min_days"] <= days <= rules["max_days"]
                    and math.isfinite(bar["settle"])
                    and bar["settle"] > 0
                    and bar["volume"] >= rules["min_volume"]
                    and bar["open_interest"] >= rules["min_open_interest"]
                ):
                    candidates.append({**bar, "expiry": expiry})
            chosen, used = [], set()
            for candidate in sorted(candidates, key=lambda r: (r["expiry"], r["contract"])):
                if candidate["expiry"] not in used:
                    chosen.append(candidate)
                    used.add(candidate["expiry"])
                if len(chosen) == 3:
                    break
            selected[(session.date, code)] = chosen
            row = {
                "date": session.date,
                "code": code,
                "carry": math.nan,
                "rank1_return": math.nan,
                "rank2_return": math.nan,
                "front_contract": None,
                "next_contract": None,
                "relative_basis": math.nan,
            }
            for rank, old in enumerate(previous[:2], 1):
                today = marks.get((session.date, old["contract"]), math.nan)
                if math.isfinite(today) and today > 0:
                    row[f"rank{rank}_return"] = today / old["settle"] - 1
            if len(chosen) >= 2:
                first, second = chosen[:2]
                row.update(
                    front_contract=first["contract"],
                    next_contract=second["contract"],
                    carry=math.log(first["settle"] / second["settle"])
                    * 365.25
                    / (second["expiry"] - first["expiry"]).days,
                )
                if len(chosen) == 3:
                    third = chosen[2]
                    row["relative_basis"] = (
                        row["carry"]
                        - math.log(second["settle"] / third["settle"])
                        * 365.25
                        / (third["expiry"] - second["expiry"]).days
                    )
            output.append(row)
            previous = chosen
    return selected, output


def compounded(values: list[float], window: int) -> float:
    """Use sums of log returns, separately from production rolling products."""
    if len(values) < window or not all(math.isfinite(x) and x > -1 for x in values[-window:]):
        return math.nan
    return math.expm1(math.fsum(math.log1p(x) for x in values[-window:]))


def reference_season(
    records: list[dict], date: pd.Timestamp, cutoff: datetime, minimum: int, observations: int
) -> dict:
    known = {}
    for row in sorted(records, key=lambda r: r["published"]):
        if row["published"] <= cutoff and row["date"] <= date.date():
            known[row["date"]] = row
    result = {"expected": math.nan, "seasonal_observations": 0, "seasons": 0}
    if not known:
        return result
    current = known[max(known)]
    result.update(current_date=current["date"], current_value=current["value"])
    if (date.date() - current["date"]).days > 10:
        return result
    values, seasons = [], set()
    for year in range(current["date"].year - 5, current["date"].year):
        day = min(current["date"].day, calendar.monthrange(year, current["date"].month)[1])
        anchor = current["date"].replace(year=year, day=day)
        for period, row in known.items():
            if abs((period - anchor).days) <= 30:
                values.append(row["value"])
                seasons.add(year)
    result.update(seasonal_observations=len(values), seasons=len(seasons))
    if len(values) >= observations and len(seasons) >= minimum:
        lower = sum(v < current["value"] for v in values)
        equal = sum(v == current["value"] for v in values)
        result.update(lower=lower, tied=equal, expected=0.5 - (lower + equal / 2) / len(values))
    return result


class Checks:
    def __init__(self):
        self.rows = []

    def add(self, kind, date, code, observed, expected, contract="", tolerance=2e-10):
        if isinstance(expected, bool):
            good, difference = observed == expected, math.nan
        elif isinstance(expected, (int, float)):
            actual = number(observed)
            both_missing = math.isnan(actual) and math.isnan(expected)
            difference = abs(actual - expected)
            good = both_missing or (
                math.isfinite(difference) and difference <= tolerance * max(1, abs(expected))
            )
        else:
            actual = None if pd.isna(observed) else observed
            good, difference = actual == expected, math.nan
        self.rows.append(
            {
                "check": kind,
                "date": date,
                "code": code,
                "contract": contract,
                "observed": observed,
                "expected": expected,
                "absolute_error": difference,
                "status": "match" if good else "difference",
            }
        )


def reconcile_sources(baseline: Path, plan_path: Path, daily_directory: Path, output: Path) -> Path:
    if output.exists():
        raise FileExistsError("Reconciliation output must be a new directory")
    run = json.loads((baseline / "run.json").read_text(encoding="utf-8"))
    if run["synthetic"] or not run["protocol"]["study"]["retrospective_bar_availability"]:
        raise ValueError("This raw-source audit is for the retrospective official-data pilot")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    inputs = {Path(x["file"]).stem: x for x in run["inputs"]}
    sources = []
    for item in inputs.values():
        path = Path(item["file"])
        if (
            sha(path) != item["sha256"]
            or sha(path.parent / "manifest.json") != item["manifest_sha256"]
        ):
            raise ValueError("Baseline input or manifest hash changed")
        sources.append({"file": str(path), "sha256": sha(path)})
    bars = pd.read_csv(inputs["bars"]["file"], parse_dates=["date"])
    curves = pd.read_csv(baseline / "curves.csv", parse_dates=["date"])
    sessions = pd.read_csv(baseline / "calendar_assumption.csv", parse_dates=["date"])
    sessions["signal_cutoff"] = pd.to_datetime(sessions.signal_cutoff, utc=True)
    meta_receipt = inputs["metadata"]["receipt"]
    original_meta = Path(meta_receipt["input_manifest"]).parent / "responses"
    metadata, meta_sources = raw_metadata(original_meta, meta_receipt)
    sources.extend(meta_sources)
    selected, reference = reference_selection(bars, metadata, sessions, run["rules"])
    actual = curves.set_index(["date", "code"])
    checks, history, legs = Checks(), defaultdict(lambda: [[], []]), []
    case_dates = {pd.Timestamp(x) for x in plan["case_dates"]}
    cases = []
    for row in reference:
        date, code = row["date"], row["code"]
        target = actual.loc[(date, code)]
        first, second = history[code]
        first.append(row["rank1_return"])
        second.append(row["rank2_return"])
        row["momentum_126d"] = compounded(first, run["rules"]["momentum_sessions"])
        row["basis_momentum"] = compounded(
            first, run["rules"]["basis_momentum_sessions"]
        ) - compounded(second, run["rules"]["basis_momentum_sessions"])
        for field in (
            "front_contract",
            "next_contract",
            "carry",
            "relative_basis",
            "rank1_return",
            "rank2_return",
            "momentum_126d",
            "basis_momentum",
        ):
            checks.add(field, date, code, target.get(field), row[field])
        legs.append(
            {
                k: row[k]
                for k in (
                    "date",
                    "code",
                    "front_contract",
                    "next_contract",
                    "rank1_return",
                    "rank2_return",
                )
            }
        )
        if date in case_dates:
            chosen = selected[(date, code)]
            cases.append(
                {
                    **row,
                    "front_expiry": chosen[0]["expiry"] if chosen else None,
                    "next_expiry": chosen[1]["expiry"] if len(chosen) > 1 else None,
                    "front_settle": chosen[0]["settle"] if chosen else math.nan,
                    "next_settle": chosen[1]["settle"] if len(chosen) > 1 else math.nan,
                }
            )

    # Direct workbook row walk: no production header/identity parser is reused.
    fields = {
        "open": "开盘价",
        "high": "最高价",
        "low": "最低价",
        "close": "收盘价",
        "settle": "结算价",
        "volume": "成交量",
        "open_interest": "持仓量",
    }
    sample = bars.loc[bars.date.isin(case_dates)]
    keys = {(r.date, r.contract): r for r in sample.itertuples()}
    raw_marks = {}
    for archive_receipt in inputs["bars"]["receipt"]["inputs"]:
        archive = Path(archive_receipt["file"])
        if sha(archive) != archive_receipt["sha256"]:
            raise ValueError("Official annual archive hash changed")
        sources.append({"file": str(archive), "sha256": archive_receipt["sha256"]})
        with zipfile.ZipFile(archive) as zipped:
            for name in sorted(
                set(sample.loc[sample.date.dt.year.eq(int(archive.stem)), "source_member"])
            ):
                book = pd.ExcelFile(zipped.open(name), engine="calamine")
                for sheet in book.sheet_names:
                    frame = book.parse(sheet, header=None)
                    header = next(
                        (
                            i
                            for i, r in enumerate(frame.head(12).itertuples(index=False, name=None))
                            if "合约" in r and ("日期" in r or "交易日期" in r)
                        ),
                        None,
                    )
                    if header is None:
                        continue
                    columns = {str(v).strip(): i for i, v in enumerate(frame.iloc[header])}
                    date_col = columns.get("日期", columns.get("交易日期"))
                    contract = None
                    for row in frame.iloc[header + 1 :].itertuples(index=False, name=None):
                        contract = workbook_identity(row[columns["合约"]], contract)
                        text = str(row[date_col]).removesuffix(".0")
                        if re.fullmatch(r"\d{8}", text):
                            date = pd.Timestamp(datetime.strptime(text, "%Y%m%d"))
                            if (date, contract) in keys:
                                if (date, contract) in raw_marks:
                                    raise ValueError(
                                        "Duplicate futures key in independent workbook audit"
                                    )
                                raw_marks[(date, contract)] = {
                                    k: number(row[columns[v]]) for k, v in fields.items()
                                }
    for (date, contract), row in keys.items():
        expected = raw_marks.get((date, contract), {})
        checks.add("annual_row_present", date, row.code, bool(expected), True, contract)
        for field in fields:
            checks.add(
                f"annual_{field}",
                date,
                row.code,
                getattr(row, field),
                expected.get(field, math.nan),
                contract,
                1e-12,
            )

    daily_receipt = json.loads((daily_directory / "manifest.json").read_text())
    marks = bars.set_index(["date", "contract"])
    daily_fields = dict(
        zip(
            fields,
            (
                "OPENPRICE",
                "HIGHESTPRICE",
                "LOWESTPRICE",
                "CLOSEPRICE",
                "SETTLEMENTPRICE",
                "VOLUME",
                "OPENINTEREST",
            ),
            strict=True,
        )
    )
    for record in daily_receipt["records"]:
        if record["status"] != "ok":
            continue
        path = daily_directory / f"{record['date']}.json"
        if sha(path) != record["sha256"]:
            raise ValueError("Daily check response hash mismatch")
        sources.append({"file": str(path), "sha256": record["sha256"]})
        obj = json.loads(path.read_bytes())
        date = pd.Timestamp(record["date"])
        for row in obj["o_curinstrument"]:
            code = str(row["PRODUCTID"]).strip().removesuffix("_f")
            month = str(row["DELIVERYMONTH"]).strip()
            if code not in set(bars.code) or not re.fullmatch(r"\d{4}", month):
                continue
            contract = code + month
            exists = (date, contract) in marks.index
            checks.add("daily_row_present", date, code, exists, True, contract)
            if exists:
                for field, key in daily_fields.items():
                    checks.add(
                        f"daily_{field}",
                        date,
                        code,
                        marks.loc[(date, contract), field],
                        number(row[key]),
                        contract,
                        1e-12,
                    )

    warehouse_receipt = inputs["warrants"]["receipt"]
    observations = pd.read_csv(inputs["warrants"]["file"])
    observations["period_date"] = pd.to_datetime(observations.period_date)
    stock = observations.set_index(["period_date", "code"])
    stock_reference = defaultdict(list)
    components = []
    for record in warehouse_receipt["records"]:
        if record["status"] != "ok":
            continue
        path = Path(warehouse_receipt["source_response_directory"]) / f"{record['date']}.json"
        if sha(path) != record["raw_sha256"]:
            raise ValueError("Raw warehouse hash mismatch")
        sources.append({"file": str(path), "sha256": record["raw_sha256"]})
        obj = json.loads(path.read_bytes())
        published = clock(obj["update_date"])
        date = pd.Timestamp(record["date"])
        by_name = defaultdict(list)
        for row in obj["o_cursor"]:
            name = str(row["VARNAME"]).split("$$")[0].strip()
            if name in WAREHOUSE_NAMES:
                by_name[name].append(row)
        for name, rows in by_name.items():
            code = WAREHOUSE_NAMES[name]
            totals = [r for r in rows if str(r["WHABBRNAME"]).split("$$")[0].strip() == "总计"]
            if len(totals) != 1:
                raise ValueError("Nonunique warehouse grand total in raw audit")
            total = number(totals[0]["WRTWGHTS"])
            detail = math.fsum(number(r["WRTWGHTS"]) for r in rows if str(r["ROWSTATUS"]) == "0")
            checks.add("warehouse_details_sum", date, code, detail, total, tolerance=1e-12)
            checks.add("warehouse_total", date, code, stock.loc[(date, code), "value"], total)
            unit = "SHFE_WGHTUNIT_" + str(totals[0]["WGHTUNIT"]).removesuffix(".0")
            checks.add("warehouse_unit", date, code, stock.loc[(date, code), "unit"], unit)
            checks.add(
                "warehouse_clock",
                date,
                code,
                str(pd.Timestamp(stock.loc[(date, code), "available_at"])),
                str(pd.Timestamp(published)),
            )
            stock_reference[code].append(
                {"date": date.date(), "published": published, "value": total}
            )
            components.append(
                {
                    "date": date,
                    "code": code,
                    "source_name": name,
                    "warehouse_detail_sum": detail,
                    "unique_grand_total": total,
                    "factory_included": False,
                }
            )
    seasonal_rows = []
    for session in sessions.loc[sessions.date.isin(case_dates)].itertuples():
        for code in sorted(bars.code.unique()):
            result = reference_season(
                stock_reference[code],
                session.date,
                session.signal_cutoff,
                run["protocol"]["study"]["inventory_min_seasons"],
                run["protocol"]["study"]["inventory_min_observations"],
            )
            observed = actual.loc[(session.date, code), "inventory_scarcity"]
            checks.add("seasonal_inventory", session.date, code, observed, result["expected"])
            seasonal_rows.append(
                {"date": session.date, "code": code, "observed": observed, **result}
            )
    output.mkdir(parents=True)
    frame = pd.DataFrame(checks.rows)
    summary = frame.groupby(["check", "status"]).size().unstack(fill_value=0).reset_index()
    for name, data in (
        ("checks", frame),
        ("summary", summary),
        ("manual_cases", pd.DataFrame(cases)),
        ("reference_daily_chain", pd.DataFrame(legs)),
        ("warehouse_components", pd.DataFrame(components)),
        ("seasonal_cases", pd.DataFrame(seasonal_rows)),
    ):
        data.to_csv(output / f"{name}.csv", index=False, encoding="utf-8-sig")
    differences = int(frame.status.eq("difference").sum())
    daily_ok = sum(r["status"] == "ok" for r in daily_receipt["records"])
    receipt = {
        "kind": "independent_arithmetic_and_raw_source_reconciliation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "baseline": str(baseline),
        "baseline_run_sha256": sha(baseline / "run.json"),
        "baseline_curves_sha256": sha(baseline / "curves.csv"),
        "plan_sha256": sha(plan_path),
        "checks": len(frame),
        "differences": differences,
        "daily_sources_ok": daily_ok,
        "daily_sources_planned": len(plan["daily_source_checks"]),
        "status": "differences_require_review"
        if differences
        else (
            "matched_with_scope_limits"
            if daily_ok == len(plan["daily_source_checks"])
            else "incomplete_daily_sources"
        ),
        "sources": sources,
        "audit_source_sha256": sha(Path(__file__)),
        "limitations": [
            "same-exchange export agreement does not certify historic vintages",
            "reference arithmetic checks definitions, not economic effectiveness",
            "raw annual price reconciliation is limited to the declared case dates",
            "no fee, margin, fill or actual opening certification",
        ],
        "claim_eligible": False,
    }
    (output / "audit.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    page = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>因子原始记录对账</title><style>body{{font:15px/1.8 "Segoe UI","Microsoft YaHei",sans-serif;color:#173149;background:#f3f6f8;max-width:1150px;margin:35px auto;padding:24px}}section{{background:#fff;padding:24px;border-radius:10px;margin:20px 0}}table{{border-collapse:collapse;width:100%}}td,th{{padding:9px;border-bottom:1px solid #dce5ec;text-align:left}}a{{color:#147d92}}.notice{{background:#fff1df;padding:18px}}.scroll{{overflow:auto}}</style><h1>因子与原始记录对账</h1><p class="notice">本页独立重算选约、同合约收益链和仓单季节分位，不调用生产因子或原数据解析函数。检查一致不等于因子有效，也不等于历史数据未修订。</p><section><h2>{len(frame):,} 项检查 · {differences:,} 项差异</h2><p>对照 {len(case_dates)} 个预先记录的案例日期；原始日行情成功 {daily_ok}/{len(plan["daily_source_checks"])}。年度和日度接口都来自上期所，不是独立供应商认证。状态：{escape(receipt["status"])}。</p><div class="scroll">{summary.to_html(index=False, border=0)}</div></section><section><h2>可以逐条核对的明细</h2><p><a href="manual_cases.csv">选约、期限、结算价与因子</a> · <a href="seasonal_cases.csv">仓单历史分位重算</a> · <a href="warehouse_components.csv">仓库明细之和与唯一总计</a> · <a href="reference_daily_chain.csv">独立逐日收益链</a> · <a href="checks.csv">全部一致与差异</a> · <a href="audit.json">来源与校验记录</a></p><p>换月日只对昨天持有的同一合约估值。季节仓单保留发布时间、历史季节数、样本数和并列分位。没有填缺、反转符号或按结果筛选案例。</p></section></html>"""
    (output / "index.html").write_text(page, encoding="utf-8")
    return output
