"""Bounded, resumable public NBS archive discovery and immutable source capture."""

import hashlib
import http.client
import json
import re
import time
import tomllib
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlsplit

from .nbs import _plain, normalize
from .nbs_ingest import digest


def history_plan(config: Path) -> tuple[dict, dict]:
    raw = tomllib.loads(config.read_text(encoding="utf-8"))
    root = config.resolve().parent.parent
    paths = {name: (root / value).resolve() for name, value in raw["paths"].items()}
    if any(not path.is_relative_to(root) for path in paths.values()):
        raise ValueError("History paths must stay within the workspace")
    study, policy = raw["history"], raw["collection"]
    if (
        (study["first_year"], study["last_year"], study["parts"])
        != (2020, 2025, ["early", "middle", "late"])
        or study["prediction_tests_added"] != 0
        or study["predictive_statistics_computed"]
        or study["minimum_assets_unchanged"] < 8
        or study["allow_missing_release_reconstruction"]
        or study["allow_url_date_as_publication"]
        or study["allow_unregistered_universe_expansion"]
        or raw["quality"]["strict_spot_basis_claim_eligible"]
        or not raw["quality"]["complete_only_when_all_periods_accounted"]
        or not 1 <= policy["maximum_response_bytes"] <= 3_000_000
        or policy["minimum_interval_seconds"] < 1
        or not 1 <= policy["maximum_attempts_per_url"] <= 2
        or policy["stop_after_consecutive_failures"] != 3
        or not 1 <= raw["discovery"]["maximum_index_pages"] <= 160
        or not 1 <= raw["discovery"]["maximum_requests"] <= 190
        or not 1 <= policy["maximum_release_requests"] <= 230
    ):
        raise ValueError("Retain the fixed history scope, no-inference and collection safeguards")
    for name, expected in raw["input_hashes"].items():
        if digest(paths[name]) != expected:
            raise ValueError(f"Frozen history input changed: {name}")
    return raw, paths


class Links(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.links, self.current, self.text = [], None, []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.current, self.text = dict(attrs), []

    def handle_data(self, data):
        if self.current is not None:
            self.text.append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self.current is not None:
            self.links.append(
                (
                    self.current.get("href", ""),
                    self.current.get("title") or "".join(self.text).strip(),
                )
            )
            self.current = None


def official_links(content: bytes, url: str) -> list[dict]:
    parser = Links()
    parser.feed(content.decode("utf-8-sig"))
    result = []
    for href, title in parser.links:
        target = urljoin(url, href)
        parsed = urlsplit(target)
        if (
            parsed.scheme == "https"
            and parsed.hostname == "www.stats.gov.cn"
            and not parsed.fragment
        ):
            result.append({"url": target, "title": re.sub(r"\s+", "", title)})
    return result


def period_from_title(title: str) -> str | None:
    match = re.search(
        r"(20\d{2})年(\d{1,2})月([上中下])旬流通领域重要生产资料市场价格变动情况",
        re.sub(r"\s+", "", title),
    )
    if match is None or not 1 <= int(match[2]) <= 12:
        return None
    return f"{match[1]}-{int(match[2]):02d}-{dict(上='early', 中='middle', 下='late')[match[3]]}"


class HistoryCapture:
    """Single ordinary TLS connection; bounded attempts, no redirects or response execution."""

    def __init__(self, config: Path, phase: str):
        self.plan, self.paths = history_plan(config)
        if phase not in {"discovery", "releases"}:
            raise ValueError("Unknown capture phase")
        self.directory = self.paths["raw"] / phase
        self.directory.mkdir(parents=True, exist_ok=True)
        self.receipt = self.directory / "manifest.json"
        self.phase, self.connection, self.last_start = phase, None, 0.0
        self.maximum = (
            self.plan["discovery"]["maximum_requests"]
            if phase == "discovery"
            else self.plan["collection"]["maximum_release_requests"]
        )
        if self.receipt.exists():
            self.manifest = json.loads(self.receipt.read_text(encoding="utf-8"))
            if self.manifest["protocol_sha256"] != digest(config):
                raise ValueError("A different history plan cannot resume this capture")
            for item in self.manifest["records"]:
                if "file" in item and digest(self.directory / item["file"]) != item["sha256"]:
                    raise ValueError("A saved archive response has changed")
        else:
            self.manifest = {
                "protocol_sha256": digest(config),
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "phase": phase,
                "records": [],
                "requests": 0,
                "access_stop": False,
            }
        if self.manifest["access_stop"]:
            raise ValueError("An earlier access/failure stop remains active")

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def checkpoint(self):
        temporary = self.receipt.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(self.manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(self.receipt)

    def get(self, identifier: str, url: str, kind: str) -> bytes:
        parsed = urlsplit(url)
        if (
            not re.fullmatch(r"[a-zA-Z0-9_-]+", identifier)
            or parsed.scheme != "https"
            or parsed.hostname != "www.stats.gov.cn"
            or parsed.port is not None
            or parsed.username
            or parsed.password
            or parsed.fragment
        ):
            raise ValueError("Only explicit public NBS source identities are allowed")
        for item in self.manifest["records"]:
            if item["id"] == identifier:
                if item["url"] != url or item["status"] != "http_ok":
                    raise ValueError(
                        "Source identity changed or a failed capture needs explicit review"
                    )
                return (self.directory / item["file"]).read_bytes()
        if self.manifest["access_stop"]:
            raise ValueError("Access/failure stop remains active")
        record = {"id": identifier, "url": url, "kind": kind, "attempts": []}
        for attempt in range(self.plan["collection"]["maximum_attempts_per_url"]):
            if self.manifest["requests"] >= self.maximum:
                raise ValueError("Frozen request budget exhausted")
            time.sleep(
                max(
                    0,
                    self.plan["collection"]["minimum_interval_seconds"]
                    - (time.monotonic() - self.last_start),
                )
            )
            self.last_start = time.monotonic()
            detail = {"started_utc": datetime.now(timezone.utc).isoformat()}
            self.manifest["requests"] += 1
            try:
                if self.connection is None:
                    self.connection = http.client.HTTPSConnection("www.stats.gov.cn", timeout=25)
                self.connection.request(
                    "GET",
                    parsed.path + ("?" + parsed.query if parsed.query else ""),
                    headers={
                        "User-Agent": "CommodityFactorLab public statistical history research",
                        "Accept": "text/html,application/json",
                    },
                )
                response = self.connection.getresponse()
                detail.update(
                    http_status=response.status,
                    headers={
                        key: response.getheader(key)
                        for key in ("Date", "Last-Modified", "ETag", "Content-Type", "Location")
                    },
                )
                chunks, size, started = [], 0, time.monotonic()
                while chunk := response.read(65536):
                    size += len(chunk)
                    if (
                        size > self.plan["collection"]["maximum_response_bytes"]
                        or time.monotonic() - started > 60
                    ):
                        raise ValueError("Response exceeds the size/time budget")
                    chunks.append(chunk)
                content = b"".join(chunks)
                file = f"{identifier}_attempt{attempt + 1}.html"
                target = self.directory / file
                if target.exists():
                    raise FileExistsError("Unregistered response already exists")
                target.write_bytes(content)
                detail.update(
                    file=file, bytes=len(content), sha256=hashlib.sha256(content).hexdigest()
                )
                record.update(
                    detail, status="http_ok" if response.status == 200 else "http_failure"
                )
            except FileExistsError:
                self.close()
                raise
            except ValueError as error:
                self.close()
                detail.update(error=f"ValueError: {error}")
                record.update(detail, status="response_rejected")
                detail["completed_utc"] = datetime.now(timezone.utc).isoformat()
                record["attempts"].append(detail)
                break
            except (OSError, http.client.HTTPException) as error:
                self.close()
                detail.update(error=f"{type(error).__name__}: {error}")
                record.update(status="transport_failure", error=detail["error"])
            detail["completed_utc"] = datetime.now(timezone.utc).isoformat()
            record["attempts"].append(detail)
            if (
                record["status"] != "transport_failure"
                or attempt + 1 == self.plan["collection"]["maximum_attempts_per_url"]
            ):
                break
        self.manifest["records"].append(record)
        trailing = 0
        for item in reversed(self.manifest["records"]):
            if item["status"] == "http_ok":
                break
            trailing += 1
        self.manifest["access_stop"] = record.get("http_status") in (403, 429) or trailing >= 3
        self.checkpoint()
        if record["status"] != "http_ok":
            raise ValueError(
                f"Public capture failed: {identifier}; {record.get('http_status', record.get('error'))}"
            )
        return (self.directory / record["file"]).read_bytes()


def discover_static_indexes(config: Path) -> dict:
    capture = HistoryCapture(config, "discovery")
    try:
        url = capture.plan["discovery"]["index"]
        first = capture.get("main_index", url, "index")
        match = re.search(rb'createPageHTML\((\d+),\s*0,\s*"index",\s*"html"\)', first)
        if (
            match is None
            or not 1 <= int(match[1]) <= capture.plan["discovery"]["maximum_index_pages"]
        ):
            raise ValueError("Official pagination proof is missing or exceeds the frozen budget")
        count = int(match[1])
        observations = []
        for page in range(count):
            page_url = url if page == 0 else urljoin(url, f"index_{page}.html")
            identifier = "main_index" if page == 0 else f"index_{page}"
            content = first if page == 0 else capture.get(identifier, page_url, "index")
            for item in official_links(content, page_url):
                period = period_from_title(item["title"])
                if period is not None and 2020 <= int(period[:4]) <= 2025:
                    observations.append({**item, "period": period, "discovered_on": page_url})
            if page % 5 == 0 or page + 1 == count:
                print(
                    f"Official index {page + 1}/{count}; {len({item['period'] for item in observations})} distinct target periods",
                    flush=True,
                )
        # The seed URLs were already discovered and captured from primary sources in v0.9.
        seed_path = capture.paths["seed"] / "manifest.json"
        seed = (
            json.loads(seed_path.read_text(encoding="utf-8"))
            if seed_path.exists()
            else {"records": []}
        )
        for item in seed["records"]:
            if item.get("kind") == "release":
                observations.append(
                    {
                        "period": item["period"],
                        "url": item["url"],
                        "title": item["period"],
                        "discovered_on": "v0.9_captured_primary_source",
                        "seed": True,
                    }
                )
        result = {
            "pagination_source_sha256": hashlib.sha256(first).hexdigest(),
            "static_pages": count,
            "observations": observations,
            "distinct_periods": len({item["period"] for item in observations}),
        }
        path = capture.paths["raw"] / "static_discovery.json"
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result
    finally:
        capture.close()


def freeze_history_catalog(config: Path, rules: Path) -> dict:
    plan, paths = history_plan(config)
    destination = paths["raw"] / "catalog.json"
    pin = paths["raw"] / "catalog.sha256"
    if destination.exists():
        if digest(destination) != pin.read_text(encoding="ascii").strip():
            raise ValueError("Frozen catalog has changed")
        catalog = json.loads(destination.read_text(encoding="utf-8"))
        if catalog["protocol_sha256"] != digest(config) or catalog["rules_sha256"] != digest(rules):
            raise ValueError("Catalog belongs to another plan or cancellation registry")
        return catalog
    observations = json.loads((paths["raw"] / "discovered_links.json").read_text(encoding="utf-8"))[
        "observations"
    ]
    discovery = json.loads((paths["raw"] / "discovery/manifest.json").read_text(encoding="utf-8"))
    records = {item["id"]: item for item in discovery["records"]}
    cancellations = tomllib.loads(rules.read_text(encoding="utf-8"))["cancellations"]
    cancelled = {}
    for item in cancellations:
        source = records[item["schedule_id"]]
        path = paths["raw"] / "discovery" / source["file"]
        if (
            source["url"] != item["source_url"]
            or digest(path) != item["source_sha256"]
            or normalize(item["evidence_fragment"])
            not in normalize(_plain(path.read_text(encoding="utf-8")))
        ):
            raise ValueError("Cancellation claim does not match its captured official evidence")
        if item["period"] in cancelled:
            raise ValueError("Duplicate cancellation claim")
        cancelled[item["period"]] = item
    expected = [
        f"{year}-{month:02d}-{part}"
        for year in range(2020, 2026)
        for month in range(1, 13)
        for part in plan["history"]["parts"]
    ]
    releases, ledger = [], []
    for period in expected:
        candidates = [item for item in observations if item["period"] == period]
        if candidates and period in cancelled:
            raise ValueError(
                "A published article conflicts with a cancellation rule; review the evidence"
            )
        if period in cancelled:
            ledger.append(
                {
                    "period": period,
                    "status": "officially_cancelled",
                    "source_url": cancelled[period]["source_url"],
                }
            )
            continue
        if not candidates:
            raise ValueError(f"Unaccounted calendar period: {period}")
        unique = {}
        for item in candidates:
            old = unique.get(item["url"])
            if old is None or item.get("seed"):
                unique[item["url"]] = item
        candidates = sorted(
            unique.values(),
            key=lambda item: (
                not item.get("seed", False),
                "/sj/zxfb/" not in item["url"],
                item["url"],
            ),
        )
        selected = candidates[0]
        releases.append(
            {
                "period": period,
                "url": selected["url"],
                "selected_from_seed": selected.get("seed", False),
                "candidates": candidates,
            }
        )
        ledger.append({"period": period, "status": "source_located", "source_url": selected["url"]})
    catalog = {
        "kind": "frozen_2020_2025_nbs_period_catalog",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": digest(config),
        "rules_sha256": digest(rules),
        "expected_periods": expected,
        "releases": releases,
        "cancellations": cancellations,
        "period_ledger": ledger,
        "selection_rule": "verified seed first; modern data-release page next; lexical URL tie-break; no price/outcome selection",
        "metadata_inputs_sha256": {
            name: digest(paths["raw"] / name)
            for name in [
                "discovered_links.json",
                "discovery/manifest.json",
                "search_provider/capture_manifest.json",
            ]
        },
        "search_summaries_used_as_prices": False,
        "prediction_tests_added": 0,
    }
    destination.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    pin.write_text(digest(destination) + "\n", encoding="ascii")
    return catalog


def fetch_history_articles(config: Path, rules: Path) -> dict:
    catalog = freeze_history_catalog(config, rules)
    capture = HistoryCapture(config, "releases")
    pin = digest(capture.paths["raw"] / "catalog.json")
    if capture.manifest.get("catalog_sha256", pin) != pin:
        raise ValueError("Captured articles belong to a different frozen source catalog")
    capture.manifest["catalog_sha256"] = pin
    seed_path = capture.paths["seed"] / "manifest.json"
    seed_manifest = (
        json.loads(seed_path.read_text(encoding="utf-8")) if seed_path.exists() else {"records": []}
    )
    seed = {
        item["url"]: item
        for item in seed_manifest["records"]
        if item.get("kind") == "release" and item["status"] == "http_ok"
    }
    try:
        done = {item["id"] for item in capture.manifest["records"]}
        for item in catalog["releases"]:
            if item["period"] in done or item["url"] not in seed:
                continue
            original = seed[item["url"]]
            source = capture.paths["seed"] / original["file"]
            if digest(source) != original["sha256"]:
                raise ValueError("Seed source has changed")
            file = item["period"] + "_seed.html"
            target = capture.directory / file
            if target.exists():
                raise FileExistsError("Unregistered seed response already exists")
            target.write_bytes(source.read_bytes())
            capture.manifest["records"].append(
                {
                    "id": item["period"],
                    "url": item["url"],
                    "kind": "release",
                    "file": file,
                    "bytes": target.stat().st_size,
                    "sha256": digest(target),
                    "status": "http_ok",
                    "reused_seed": True,
                    "seed_manifest_sha256": digest(seed_path),
                }
            )
            capture.checkpoint()
        for index, item in enumerate(catalog["releases"]):
            capture.get(item["period"], item["url"], "release")
            if index % 10 == 0 or index + 1 == len(catalog["releases"]):
                print(
                    f"Official price articles {index + 1}/{len(catalog['releases'])}; retained responses {len(capture.manifest['records'])}",
                    flush=True,
                )
        return capture.manifest
    finally:
        capture.close()
