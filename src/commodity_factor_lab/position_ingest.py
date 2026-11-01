"""Resumable, rate-limited public SHFE ranking collection with immutable responses."""

import hashlib
import http.client
import json
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .positions import parse_position_ranks

ENDPOINT = "https://www.shfe.com.cn/data/tradedata/future/dailydata/pm{date}.dat"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def position_plan(path: Path) -> tuple[dict, dict, list[str]]:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    root = path.resolve().parent.parent
    paths = {name: (root / value).resolve() for name, value in raw["paths"].items()}
    for name, expected in raw["input_hashes"].items():
        if sha256(paths[name]) != expected:
            raise ValueError(f"Frozen position input changed: {name}")
    study, collection = raw["study"], raw["collection"]
    if (
        collection["endpoint"] != ENDPOINT
        or collection["minimum_interval_seconds"] < 0.5
        or not 1 <= collection["maximum_response_bytes"] <= 8_000_000
        or not 1 <= collection["maximum_attempts_per_date"] <= 2
        or collection["stop_after_consecutive_failures"] != 3
    ):
        raise ValueError("Collection must retain its bounded official-source policy")
    if (
        study["lookback_sessions"] != 20
        or not study["require_all_window_sessions"]
        or not study["historical_rows_require_own_cutoff"]
        or study["direction"] != 1
        or study["min_assets"] < 8
        or study["min_inference_obs"] < 252
        or study["hac_lags"] < study["holding_sessions"]
    ):
        raise ValueError("Position study requires the registered window and inference gates")
    calendar = pd.read_csv(paths["calendar"], parse_dates=["date"])
    if calendar.date.duplicated().any() or not calendar.date.is_monotonic_increasing:
        raise ValueError("Position calendar must be unique and sorted")
    sample = calendar.loc[calendar.date.between(study["sample_start"], study["sample_end"])]
    first = sample.date.searchsorted(pd.Timestamp(study["evaluation_start"]))
    if first != study["lookback_sessions"]:
        raise ValueError("Sample must contain exactly the declared pre-evaluation warmup")
    dates = sample.date.dt.strftime("%Y%m%d").tolist()
    return raw, paths, dates


class PublicRankingConnection:
    """One ordinary verified HTTPS connection; never executes provider responses."""

    def __init__(self):
        self.connection = None

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def read(self, date: str, maximum: int) -> tuple[int, bytes]:
        if len(date) != 8 or not date.isdecimal():
            raise ValueError("Invalid ranking request date")
        if self.connection is None:
            self.connection = http.client.HTTPSConnection("www.shfe.com.cn", timeout=25)
        self.connection.request(
            "GET",
            f"/data/tradedata/future/dailydata/pm{date}.dat",
            headers={
                "User-Agent": "CommodityFactorLab/0.6 public research",
                "Accept": "application/json",
            },
        )
        response = self.connection.getresponse()
        start, size, chunks = time.monotonic(), 0, []
        while content := response.read(262144):
            size += len(content)
            if size > maximum or time.monotonic() - start > 60:
                self.close()
                raise ValueError("Ranking response exceeds its size/time budget")
            chunks.append(content)
        return response.status, b"".join(chunks)


def _checkpoint(path: Path, record: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def fetch_positions(config: Path, limit: int | None = None) -> dict:
    raw, paths, dates = position_plan(config)
    if limit is not None and limit < 1:
        raise ValueError("Collection limit must be positive")
    destination = paths["raw"]
    destination.mkdir(parents=True, exist_ok=True)
    responses = destination / "responses"
    responses.mkdir(exist_ok=True)
    receipt = destination / "manifest.json"
    protocol_hash = sha256(config)
    if receipt.exists():
        manifest = json.loads(receipt.read_text(encoding="utf-8"))
        if manifest["protocol_sha256"] != protocol_hash or manifest["planned_dates"] != dates:
            raise ValueError("Cannot resume a different collection protocol in this directory")
        for item in manifest["records"]:
            if (
                item["status"] == "ok"
                and sha256(responses / f"{item['date']}.json") != item["sha256"]
            ):
                raise ValueError("A saved ranking response has changed")
    else:
        manifest = {
            "kind": "official_daily_broker_ranks",
            "protocol_sha256": protocol_hash,
            "protocol": raw,
            "planned_dates": dates,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "records": [],
        }
        seed = paths["seed"] / "manifest.json"
        if seed.exists():
            seed_manifest = json.loads(seed.read_text(encoding="utf-8"))
            for item in seed_manifest["records"]:
                date = item["date"]
                if item["status"] != "ok" or date not in dates:
                    continue
                source = paths["seed"] / f"positions_{date}.json"
                if sha256(source) != item["sha256"]:
                    raise ValueError("Seed ranking response no longer matches its receipt")
                parse_position_ranks(source.read_bytes(), date)
                target = responses / f"{date}.json"
                if target.exists():
                    raise FileExistsError("Unregistered ranking response already exists")
                target.write_bytes(source.read_bytes())
                manifest["records"].append({**item, "reused_seed": True})
        _checkpoint(receipt, manifest)
    done = {item["date"] for item in manifest["records"]}
    budget, consecutive, last_start = 0, 0, 0.0
    transport = PublicRankingConnection()
    try:
        for date in dates:
            if date in done or limit is not None and budget >= limit:
                continue
            item = {"date": date, "url": ENDPOINT.format(date=date), "attempts": []}
            for attempt in range(raw["collection"]["maximum_attempts_per_date"]):
                time.sleep(
                    max(
                        0,
                        raw["collection"]["minimum_interval_seconds"]
                        - (time.monotonic() - last_start),
                    )
                )
                last_start = time.monotonic()
                attempt_record = {"retrieved_utc": datetime.now(timezone.utc).isoformat()}
                try:
                    status, content = transport.read(
                        date, raw["collection"]["maximum_response_bytes"]
                    )
                    attempt_record.update(
                        http_status=status,
                        bytes=len(content),
                        sha256=hashlib.sha256(content).hexdigest(),
                    )
                    if status != 200:
                        failed = responses / f"{date}.attempt{attempt + 1}.bin"
                        if failed.exists():
                            raise FileExistsError("Failed response already exists")
                        failed.write_bytes(content)
                        raise ValueError(f"Official endpoint returned HTTP {status}")
                    target = responses / f"{date}.json"
                    if target.exists():
                        raise FileExistsError("Unregistered ranking response already exists")
                    target.write_bytes(content)
                    parsed = parse_position_ranks(content, date)
                    obj = json.loads(content)
                    item.update(
                        status="ok",
                        sha256=attempt_record["sha256"],
                        bytes=len(content),
                        report_date=obj["report_date"],
                        update_date=obj["update_date"],
                        parsed_contracts=len(parsed),
                        retrieved_utc=attempt_record["retrieved_utc"],
                    )
                except (
                    OSError,
                    ValueError,
                    KeyError,
                    TypeError,
                    http.client.HTTPException,
                ) as error:
                    attempt_record["error"] = f"{type(error).__name__}: {error}"
                    transport.close()
                    item["status"] = "failed"
                item["attempts"].append(attempt_record)
                # Invalid content, access controls and throttling are retained, not retried.
                if item["status"] == "ok" or "http_status" in attempt_record:
                    break
            manifest["records"].append(item)
            done.add(date)
            budget += 1
            consecutive = 0 if item["status"] == "ok" else consecutive + 1
            manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
            manifest["completed_dates"] = len(done)
            manifest["pending_dates"] = len(dates) - len(done)
            _checkpoint(receipt, manifest)
            if budget % 20 == 0 or item["status"] != "ok":
                print(
                    f"Ranking collection {len(done)}/{len(dates)}; {date}: {item['status']}",
                    flush=True,
                )
            status = item["attempts"][-1].get("http_status")
            if consecutive >= 3 or status in (403, 429):
                print(
                    "Ranking collection stopped at the registered failure/access limit", flush=True
                )
                break
    finally:
        transport.close()
    return {
        "planned": len(dates),
        "completed": len(done),
        "successful": sum(item["status"] == "ok" for item in manifest["records"]),
        "pending": len(dates) - len(done),
    }
