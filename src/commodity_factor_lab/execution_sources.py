"""Bounded official execution-source capture. Current snapshots are not first vintages."""

import http.client
import json
import re
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from .nbs_ingest import digest


def execution_plan(config: Path) -> tuple[dict, dict]:
    plan = tomllib.loads(config.read_text(encoding="utf-8"))
    root = config.resolve().parent.parent
    paths = {name: (root / value).resolve() for name, value in plan["paths"].items()}
    study, policy = plan["study"], plan["collection"]
    if any(not path.is_relative_to(root) for path in paths.values()):
        raise ValueError("Execution assay paths must stay within its workspace")
    if (
        study["kind"] != "dated_execution_source_readiness_only"
        or study["prediction_tests_added"] != 0
        or study["registered_exploration_family_size_unchanged"] != 15
        or study["historical_first_vintage_certified"]
        or study["actual_fill_certified"]
        or study["real_investment_backtests"] != 0
        or study["fee_dates"] != ["2024-05-30", "2024-05-31", "2024-06-03"]
        or study["fee_contracts"] != ["rb2408", "rb2409", "rb2410"]
        or study["margin_dates"] != ["2025-01-24", "2025-01-27", "2025-02-05", "2025-02-06"]
        or study["margin_contracts"] != ["cu2503", "rb2505"]
        or policy["minimum_interval_seconds"] < 1
        or not 1 <= policy["maximum_new_requests"] <= 14
        or not 1 <= policy["maximum_response_bytes"] <= 3_000_000
        or policy["stop_after_consecutive_failures"] != 3
    ):
        raise ValueError("Keep the bounded source-only scope and claim gates")
    seen, new = set(), 0
    for source in plan["sources"]:
        url = urlsplit(source["url"])
        if (
            url.scheme != "https"
            or url.hostname not in {"www.shfe.com.cn", "www.shfe.cn"}
            or url.port is not None
            or url.username
            or url.password
            or url.query
            or url.fragment
            or not re.fullmatch(r"[a-zA-Z0-9_]+", source["id"])
            or source["id"] in seen
            or source["kind"] not in {"notice", "current_hours", "trading", "settlement"}
        ):
            raise ValueError("Only unique, explicit official public sources are allowed")
        seen.add(source["id"])
        if "seed_file" in source:
            if Path(source["seed_file"]).name != source["seed_file"]:
                raise ValueError("Seed must be a direct file")
            if digest(paths["seed"] / source["seed_file"]) != source["seed_sha256"]:
                raise ValueError("Frozen seed response changed")
        else:
            new += 1
    if new > policy["maximum_new_requests"] or len(seen) > 16:
        raise ValueError("Too many source requests")
    expected = {
        "rb2410_april",
        "rb_offcycle_may",
        "cny2025",
        "current_hours",
        "trading_20240130_seed",
        "settlement_20240130_seed",
    }
    expected.update(f"trading_{date}" for date in ("20250124", "20250127", "20250205", "20250206"))
    expected.update(
        f"settlement_{date}"
        for date in ("20240531", "20240603", "20250124", "20250127", "20250205", "20250206")
    )
    if seen != expected:
        raise ValueError("Keep all registered sources and negative controls")
    if paths["output"].is_relative_to(paths["raw"]):
        raise ValueError("Reports must stay outside raw captures")
    for name, expected in plan["input_hashes"].items():
        if digest(paths[name]) != expected:
            raise ValueError(f"Frozen execution assay input changed: {name}")
    return plan, paths


def capture_execution_sources(config: Path) -> dict:
    plan, paths = execution_plan(config)
    return capture_public_sources(config, plan, paths)


def capture_public_sources(config: Path, plan: dict, paths: dict) -> dict:
    """Capture a caller-validated fixed official source list, preserving the original transport rules."""
    raw = paths["raw"]
    raw.mkdir(parents=True, exist_ok=True)
    manifest_path = raw / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["protocol_sha256"] != digest(config):
            raise ValueError("Cannot resume a changed source assay")
    else:
        manifest = {"protocol_sha256": digest(config), "records": [], "new_requests": 0}
    failures = 0
    for record in manifest["records"]:
        if "file" in record and digest(raw / record["file"]) != record["sha256"]:
            raise ValueError("Saved execution response changed")
        failures = 0 if record["status"] in {"http_ok", "seed_reused"} else failures + 1
        if record.get("http_status") in (403, 429) or failures >= 3:
            raise ValueError("Saved access/failure stop remains active")
    done = {item["id"] for item in manifest["records"]}
    for source in plan["sources"]:
        if source["id"] in done:
            continue
        record = {
            **source,
            "verification_or_capture_started_utc": datetime.now(timezone.utc).isoformat(),
        }
        destination = raw / (
            source["id"] + (".json" if source["kind"] in {"trading", "settlement"} else ".html")
        )
        if destination.exists():
            raise FileExistsError("Unregistered capture file exists")
        if "seed_file" in source:
            destination.write_bytes((paths["seed"] / source["seed_file"]).read_bytes())
            record.update(
                status="seed_reused", historical_retrieval_clock="not separately recorded"
            )
        else:
            if manifest["new_requests"] >= plan["collection"]["maximum_new_requests"]:
                raise ValueError("Source request budget exhausted")
            time.sleep(plan["collection"]["minimum_interval_seconds"])
            url = urlsplit(source["url"])
            connection = http.client.HTTPSConnection(url.hostname, timeout=25)
            manifest["new_requests"] += 1
            try:
                connection.request(
                    "GET",
                    url.path,
                    headers={
                        "User-Agent": "CommodityFactorLab public source assay",
                        "Accept": "text/html, application/json",
                    },
                )
                response = connection.getresponse()
                record.update(
                    http_status=response.status,
                    headers={
                        key: response.getheader(key)
                        for key in ("Date", "Content-Type", "Last-Modified", "ETag", "Location")
                    },
                )
                content = response.read(plan["collection"]["maximum_response_bytes"] + 1)
                if len(content) > plan["collection"]["maximum_response_bytes"]:
                    raise ValueError("Source response exceeds its size budget")
                destination.write_bytes(content)
                record["status"] = "http_ok" if response.status == 200 else "http_failure"
            except (OSError, http.client.HTTPException, ValueError) as error:
                record.update(status="failed", error=f"{type(error).__name__}: {error}")
            finally:
                connection.close()
        if destination.exists():
            record.update(
                file=destination.name, bytes=destination.stat().st_size, sha256=digest(destination)
            )
        record["completed_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["records"].append(record)
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(source["id"], record["status"], flush=True)
        failures = 0 if record["status"] in {"http_ok", "seed_reused"} else failures + 1
        if record.get("http_status") in (403, 429) or failures >= 3:
            break
    return manifest
