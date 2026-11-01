"""Capture a frozen, small official-source assay. Never runs response scripts."""

import argparse
import hashlib
import http.client
import json
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode, urlsplit

ROOT = Path(__file__).resolve().parents[1]
HOSTS = {"www.sge.com.cn", "www.shfe.com.cn", "shanghai.chinatax.gov.cn"}


def collect(config: Path, catalog: Path) -> dict:
    plan = tomllib.loads(config.read_text(encoding="utf-8"))
    sources = tomllib.loads(catalog.read_text(encoding="utf-8"))
    probe = plan["probe"]
    dates = probe["dates"]
    if (
        len(dates) != len(set(dates))
        or len(dates) > 12
        or probe["minimum_interval_seconds"] < 0.8
        or not 1 <= probe["maximum_response_bytes"] <= 4_000_000
        or probe["stop_after_consecutive_failures"] != 3
    ):
        raise ValueError("Retain the bounded source assay; bulk collection is not enabled")
    requests = []
    for date in dates:
        if date < probe["new_endpoint_start"]:
            url = sources["historical_pages"][date]
            mode = "historical_article"
        else:
            query = urlencode({"start_date": date, "end_date": date})
            url = sources["daily_query"]["endpoint"] + "?" + query
            mode = "dated_query"
        requests.append({"id": date, "date": date, "mode": mode, "url": url})
    requests.extend({**item, "mode": "document"} for item in sources["documents"])
    if len(requests) > 18 or len({r["id"] for r in requests}) != len(requests):
        raise ValueError("Source assay exceeds the recorded request budget")
    for item in requests:
        url = urlsplit(item["url"])
        if url.scheme != "https" or url.hostname not in HOSTS or url.port is not None:
            raise ValueError("Only ordinary HTTPS reads of the official source catalog are allowed")
        if url.username or url.password or url.fragment or "/" in item["id"]:
            raise ValueError("Unexpected request identity")
    destination = ROOT / plan["paths"]["raw"]
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "manifest.json"
    hashes = {
        "protocol_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "sources_sha256": hashlib.sha256(catalog.read_bytes()).hexdigest(),
    }
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(manifest[key] != value for key, value in hashes.items()):
            raise ValueError("A different assay cannot resume this directory")
        for item in manifest["records"]:
            if "file" in item:
                content = (destination / item["file"]).read_bytes()
                if hashlib.sha256(content).hexdigest() != item["sha256"]:
                    raise ValueError("Captured response no longer matches its receipt")
    else:
        manifest = {
            "kind": "official_spot_source_readiness_only",
            **hashes,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "planned_requests": requests,
            "records": [],
            "historical_first_release_certified": False,
        }
    done = {item["id"] for item in manifest["records"]}
    consecutive = 0
    for item in reversed(manifest["records"]):
        if item["status"] == "http_ok":
            break
        consecutive += 1
    if consecutive >= 3 or any(r.get("http_status") in (403, 429) for r in manifest["records"]):
        raise ValueError("Previous access/failure stop remains in effect")
    for request in requests:
        if request["id"] in done:
            continue
        time.sleep(probe["minimum_interval_seconds"])
        item = {**request, "started_utc": datetime.now(timezone.utc).isoformat()}
        url = urlsplit(request["url"])
        connection = http.client.HTTPSConnection(url.hostname, timeout=25)
        try:
            connection.request(
                "GET",
                url.path + ("?" + url.query if url.query else ""),
                headers={
                    "User-Agent": "CommodityFactorLab public source assay",
                    "Accept": "text/html",
                },
            )
            response = connection.getresponse()
            item["http_status"] = response.status
            item["headers"] = {
                key: response.getheader(key)
                for key in ("Date", "Last-Modified", "ETag", "Content-Type", "Location")
            }
            chunks, size, started = [], 0, time.monotonic()
            while chunk := response.read(65536):
                size += len(chunk)
                if size > probe["maximum_response_bytes"] or time.monotonic() - started > 60:
                    raise ValueError("Response exceeds its size/time budget")
                chunks.append(chunk)
            content = b"".join(chunks)
            name = request["id"] + ".html"
            path = destination / name
            if path.exists():
                raise FileExistsError("Unregistered raw file already exists")
            path.write_bytes(content)
            item.update(
                file=name,
                bytes=len(content),
                sha256=hashlib.sha256(content).hexdigest(),
                status="http_ok" if response.status == 200 else "http_failure",
            )
        except (OSError, http.client.HTTPException, ValueError) as error:
            item.update(status="failed", error=f"{type(error).__name__}: {error}")
        finally:
            connection.close()
        item["completed_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["records"].append(item)
        consecutive = 0 if item["status"] == "http_ok" else consecutive + 1
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(request["id"], item["status"], item.get("bytes", 0), flush=True)
        if item.get("http_status") in (403, 429) or consecutive >= 3:
            break
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/spot_readiness.toml")
    parser.add_argument("--sources", type=Path, default=ROOT / "configs/spot_sources.toml")
    args = parser.parse_args()
    collect(args.config, args.sources)
