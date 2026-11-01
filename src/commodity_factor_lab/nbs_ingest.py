"""Bounded capture of the fixed NBS reference-price assay, with verified HTTPS."""

import hashlib
import http.client
import json
import re
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def nbs_plan(config: Path) -> tuple[dict, dict]:
    plan = tomllib.loads(config.read_text(encoding="utf-8"))
    root = config.resolve().parent.parent
    paths = {name: (root / value).resolve() for name, value in plan["paths"].items()}
    if any(not path.is_relative_to(root) for path in paths.values()):
        raise ValueError("NBS assay paths must stay inside its workspace")
    study, policy = plan["study"], plan["collection"]
    if (
        study["kind"] != "industrial_reference_source_readiness_only"
        or study["prediction_tests_added"] != 0
        or study["predictive_statistics_computed"]
        or study["minimum_assets_unchanged"] < 8
        or study["allow_url_date_as_publication"]
        or study["allow_missing_release_inversion"]
        or study["strict_spot_basis_claim_eligible"]
        or study["published_reference_is_executable_spot"]
        or plan["negative_controls"]["allow_reconstruction_from_later_change"]
    ):
        raise ValueError("Reference-source assay cannot relax timing or activate prediction")
    if (
        not 1 <= len(plan["sources"]) <= policy["maximum_requests"] <= 11
        or policy["minimum_interval_seconds"] < 1
        or not 1 <= policy["maximum_response_bytes"] <= 3_000_000
        or policy["stop_after_consecutive_failures"] != 3
    ):
        raise ValueError("Retain the bounded collection budget")
    identifiers = set()
    for item in plan["sources"]:
        url = urlsplit(item["url"])
        if (
            url.scheme != "https"
            or url.hostname != "www.stats.gov.cn"
            or url.port is not None
            or url.username
            or url.password
            or url.fragment
            or item["kind"] not in {"release", "methodology"}
            or not re.fullmatch(r"[a-zA-Z0-9_-]+", item["id"])
            or item["id"] in identifiers
        ):
            raise ValueError("Only unique, explicitly listed public NBS sources are allowed")
        identifiers.add(item["id"])
        if (
            item["kind"] == "release"
            and item["period"] in plan["negative_controls"]["unpublished_periods"]
        ):
            raise ValueError("Do not add a fabricated release for a cancelled period")
    for name, expected in plan["input_hashes"].items():
        if digest(paths[name]) != expected:
            raise ValueError(f"Frozen NBS input changed: {name}")
    return plan, paths


def capture_nbs(config: Path) -> dict:
    plan, paths = nbs_plan(config)
    destination = paths["raw"]
    destination.mkdir(parents=True, exist_ok=True)
    receipt = destination / "manifest.json"
    if receipt.exists():
        manifest = json.loads(receipt.read_text(encoding="utf-8"))
        if (
            manifest["protocol_sha256"] != digest(config)
            or manifest["planned_sources"] != plan["sources"]
        ):
            raise ValueError("Cannot resume a different source assay")
        for item in manifest["records"]:
            if "file" in item and digest(destination / item["file"]) != item["sha256"]:
                raise ValueError("A saved NBS response differs from its capture receipt")
    else:
        manifest = {
            "kind": plan["study"]["kind"],
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_sha256": digest(config),
            "planned_sources": plan["sources"],
            "records": [],
            "historical_first_vintage_certified": False,
        }
    failures = 0
    for item in reversed(manifest["records"]):
        if item["status"] == "http_ok":
            break
        failures += 1
    if failures >= 3 or any(item.get("http_status") in (403, 429) for item in manifest["records"]):
        raise ValueError("Previous access/failure stop remains active")
    done = {item["id"] for item in manifest["records"]}
    for source in plan["sources"]:
        if source["id"] in done:
            continue
        time.sleep(plan["collection"]["minimum_interval_seconds"])
        url = urlsplit(source["url"])
        item = {**source, "started_utc": datetime.now(timezone.utc).isoformat()}
        connection = http.client.HTTPSConnection(url.hostname, timeout=25)
        try:
            connection.request(
                "GET",
                url.path + ("?" + url.query if url.query else ""),
                headers={
                    "User-Agent": "CommodityFactorLab public reference source assay",
                    "Accept": "text/html",
                },
            )
            response = connection.getresponse()
            item.update(
                http_status=response.status,
                headers={
                    key: response.getheader(key)
                    for key in ("Date", "Content-Type", "Last-Modified", "ETag", "Location")
                },
            )
            chunks, size, started = [], 0, time.monotonic()
            while chunk := response.read(65536):
                size += len(chunk)
                if (
                    size > plan["collection"]["maximum_response_bytes"]
                    or time.monotonic() - started > 60
                ):
                    raise ValueError("NBS response exceeds its size/time budget")
                chunks.append(chunk)
            content = b"".join(chunks)
            path = destination / (source["id"] + ".html")
            if path.exists():
                raise FileExistsError("Unregistered NBS raw file already exists")
            path.write_bytes(content)
            item.update(
                file=path.name,
                bytes=len(content),
                sha256=digest(path),
                status="http_ok" if response.status == 200 else "http_failure",
            )
        except (OSError, http.client.HTTPException, ValueError) as error:
            item.update(status="failed", error=f"{type(error).__name__}: {error}")
        finally:
            connection.close()
        item["completed_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["records"].append(item)
        receipt.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        failures = 0 if item["status"] == "http_ok" else failures + 1
        print(source["id"], item["status"], item.get("bytes", 0), flush=True)
        if item.get("http_status") in (403, 429) or failures >= 3:
            break
    return manifest
