"""Official weekly sampled registered warehouse warrants; not total inventories."""

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .official import PILOT_CODES, SHFE_CODES, read_public

PRODUCTS = {
    "铜": "cu",
    "铝": "al",
    "锌": "zn",
    "铅": "pb",
    "镍": "ni",
    "锡": "sn",
    "螺纹钢": "rb",
    "线材": "wr",
    "热轧卷板": "hc",
    "天然橡胶": "ru",
    "沥青": "bu",
    "石油沥青": "bu",
    "燃料油": "fu",
    "纸浆": "sp",
    "白银": "ag",
    "不锈钢": "ss",
    "氧化铝": "ao",
    "丁二烯橡胶": "br",
}
# The exchange renamed warehouse series when factory warrants became separate rows.
# Factory warrants are a different measure and must not enter this stock series.
PRODUCTS.update({name + "仓库": code for name, code in list(PRODUCTS.items())})


def parse_warrants(
    payload: bytes, expected_date: str, source: str, codes: tuple[str, ...] = PILOT_CODES
) -> pd.DataFrame:
    if not codes or set(codes) - SHFE_CODES:
        raise ValueError("Warehouse products must be registered SHFE codes")
    obj = json.loads(payload)
    if str(obj.get("report_date")) != expected_date:
        raise ValueError("Warrant response is not for the requested historical date")
    clock = obj.get("update_date", "")
    if not isinstance(clock, str) or ":" not in clock:
        raise ValueError("Warehouse publication clock is missing; do not assume midnight")
    available = pd.Timestamp(clock)
    if available.tzinfo is None:
        available = available.tz_localize("Asia/Shanghai")
    available = available.tz_convert("UTC")
    rows = []
    for item in obj["o_cursor"]:
        name = str(item["VARNAME"]).split("$$")[0].strip()
        # Do not add warehouses, regional subtotals and bonded/tax subtotals twice.
        total = str(item["WHABBRNAME"]).split("$$")[0].strip()
        if name not in PRODUCTS or PRODUCTS[name] not in codes or total != "总计":
            continue
        value = float(str(item["WRTWGHTS"]).replace(",", ""))
        if not np.isfinite(value) or value < 0:
            raise ValueError("Invalid warehouse total")
        unit = str(item["WGHTUNIT"]).strip().removesuffix(".0")
        if not unit or unit in {"nan", "None"}:
            raise ValueError("Missing source weight-unit code")
        rows.append(
            {
                "code": PRODUCTS[name],
                "period_date": pd.Timestamp(expected_date),
                "available_at": available,
                "value": value,
                "unit": f"SHFE_WGHTUNIT_{unit}",
                "source": source,
                "measure": "registered_warehouse_warrants",
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty or frame.code.duplicated().any():
        raise ValueError("Missing or duplicate unique warehouse grand totals")
    return frame


def reparse_warrants(source: Path, output: Path, codes: tuple[str, ...] = PILOT_CODES) -> Path:
    """Derive a corrected table from immutable, hash-checked public responses."""
    if output.exists():
        raise FileExistsError("Reparsed warehouse directory must be new")
    manifest_path = source / "manifest.json"
    receipt = json.loads(manifest_path.read_text(encoding="utf-8"))
    records, frames = [], []
    for previous in receipt["records"]:
        record = dict(previous)
        record["original_status"] = previous["status"]
        if previous.get("error"):
            record["original_error"] = previous["error"]
        record["status"] = "failed"
        try:
            raw = (source / "responses" / f"{record['date']}.json").read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            if previous.get("raw_sha256") and digest != previous["raw_sha256"]:
                raise ValueError("Raw warehouse response hash changed")
            frame = parse_warrants(raw, record["date"], record["url"], codes)
            frames.append(frame)
            record.pop("error", None)
            record.update(
                status="ok",
                raw_sha256=digest,
                products=len(frame),
                missing_products=sorted(set(codes) - set(frame.code)),
            )
        except (ValueError, KeyError, OSError) as error:
            record["error"] = str(error)
        records.append(record)
    if not frames:
        raise ValueError("No warehouse responses can be parsed")
    output.mkdir(parents=True)
    destination = output / "warrants.csv"
    pd.concat(frames, ignore_index=True).to_csv(destination, index=False)
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "kind": "official_reparsed_weekly_warehouse_warrants",
                "codes": list(codes),
                "planned_dates": receipt.get("planned_dates", []),
                "records": records,
                "input_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                "source_response_directory": str((source / "responses").resolve()),
                "derived_utc": datetime.now(timezone.utc).isoformat(),
                "measure": "registered warehouse-only warrants; factory warrants excluded",
                "source_unit_policy": "preserve exchange weight code; within-product seasonal ranks only",
                "point_in_time_certified": False,
                "measure_is_total_social_inventory": False,
                "timestamp_timezone_assumption": "source local clocks are Asia/Shanghai",
                "normalized_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output


def fetch_warrants(dates: list[str], output: Path, codes: tuple[str, ...] = PILOT_CODES) -> Path:
    if output.exists():
        raise FileExistsError("Warehouse snapshot directory must be new")
    days = sorted(set(dates))
    if not days or len(days) > 400:
        raise ValueError("Use at most 400 predeclared weekly sample dates per snapshot")
    for day in days:
        if len(day) != 8 or not day.isdigit():
            raise ValueError("Warehouse query dates must be YYYYMMDD")
        pd.to_datetime(day, format="%Y%m%d", errors="raise")
    output.mkdir(parents=True)
    (output / "responses").mkdir()
    records, frames, failures = [], [], 0
    try:
        for number, day in enumerate(days, 1):
            url = f"https://www.shfe.com.cn/data/tradedata/future/dailydata/{day}dailystock.dat"
            record = {"date": day, "url": url, "status": "failed"}
            try:
                raw = read_public(url, maximum=2_000_000)
                (output / "responses" / f"{day}.json").write_bytes(raw)
                record["raw_sha256"] = hashlib.sha256(raw).hexdigest()
                frame = parse_warrants(raw, day, url, codes)
                frames.append(frame)
                record.update(
                    status="ok",
                    products=len(frame),
                    missing_products=sorted(set(codes) - set(frame.code)),
                )
                failures = 0
            except (ValueError, KeyError, OSError) as error:
                record["error"] = str(error)
                failures += 1
            record["retrieved_utc"] = datetime.now(timezone.utc).isoformat()
            records.append(record)
            print(f"Weekly warrants {number}/{len(days)} {day}: {record['status']}", flush=True)
            if failures >= 5:
                print(
                    "Stopped after five consecutive source failures; no missing values filled",
                    flush=True,
                )
                break
            time.sleep(0.35)
    finally:
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(output / "warrants.csv", index=False)
        (output / "manifest.json").write_text(
            json.dumps(
                {
                    "kind": "official_weekly_sampled_registered_warrants",
                    "codes": list(codes),
                    "planned_dates": days,
                    "records": records,
                    "source_unit_policy": "preserve exchange weight code; within-product seasonal ranks only",
                    "point_in_time_certified": False,
                    "measure_is_total_social_inventory": False,
                    "timestamp_timezone_assumption": "source local clocks are Asia/Shanghai",
                    "normalized_sha256": hashlib.sha256(
                        (output / "warrants.csv").read_bytes()
                    ).hexdigest()
                    if frames
                    else None,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return output
