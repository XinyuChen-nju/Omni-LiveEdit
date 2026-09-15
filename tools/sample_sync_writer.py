#!/usr/bin/env python3
"""Single-writer consumer for training-eval row manifests stored in COS."""

import argparse
import fcntl
import json
import logging
import signal
import sys
import time
from pathlib import Path

import requests

from sample_sync import (
    SCHEMA_FIELDS,
    StateStore,
    ensure_dataset,
    load_config,
    remote_sample_ids,
)

LOG = logging.getLogger("sample-sync-writer")
STOP = False


def acquire_lock():
    handle = open("/run/xrtm-sample-sync-writer.lock", "w")
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return handle


def list_manifests(config, session):
    base = config["cos_base_url"].rstrip("/") + "/api/cos/list"
    prefix = f"cos://{config['bucket_name']}/{config['manifest_prefix'].strip('/')}/"
    marker = None
    while True:
        params = {"uri": prefix, "recursive": "true", "max_keys": 1000}
        if marker:
            params["marker"] = marker
        response = session.get(base, params=params, timeout=config["request_timeout_seconds"])
        response.raise_for_status()
        data = response.json()
        for uri in data.get("uris") or []:
            yield uri
        if not data.get("is_truncated") or not data.get("next_marker"):
            return
        marker = data["next_marker"]


def download_manifest(config, session, uri):
    response = session.get(
        config["cos_base_url"].rstrip("/") + "/api/cos/download",
        params={"uri": uri, "mode": "stream"},
        timeout=config["request_timeout_seconds"],
    )
    response.raise_for_status()
    document = response.json()
    if document.get("version") != 1 or not isinstance(document.get("row"), dict):
        raise RuntimeError(f"invalid manifest document: {uri}")
    row = document["row"]
    required = [field["name"] for field in SCHEMA_FIELDS if field.get("required")]
    missing = [name for name in required if row.get(name) is None]
    if missing:
        raise RuntimeError(f"manifest missing required fields {missing}: {uri}")
    for name in ("result_video", "source_video", "target_video", "reference_video"):
        value = row.get(name)
        if value and not value.startswith(f"cos://{config['bucket_name']}/"):
            raise RuntimeError(f"manifest {name} points outside managed bucket: {uri}")
    return row


def process_once(config, state):
    session = requests.Session()
    uris = [uri for uri in list_manifests(config, session) if not state.manifest_consumed(uri)]
    if not uris:
        return 0, 0

    dataset = config["dataset_name"]
    ensure_dataset(config, session, dataset)
    state.mark_remote_rows(dataset, remote_sample_ids(config, session, dataset))
    pending = []
    for uri in uris:
        try:
            row = download_manifest(config, session, uri)
            if state.row_uploaded(dataset, row["sample_id"]):
                state.mark_manifest_consumed(uri, row["sample_id"])
            else:
                pending.append((uri, row))
        except Exception as exc:
            LOG.error("manifest read failed %s: %s", uri, exc)

    appended = 0
    base = config["data_srv_url"].rstrip("/") + f"/api/data/{dataset}"
    batch_size = int(config["append_batch_size"])
    for offset in range(0, len(pending), batch_size):
        entries = pending[offset : offset + batch_size]
        rows = [row for _, row in entries]
        state.mark_rows(dataset, rows, "appending")
        response = session.post(
            base + "/append", json={"rows": rows}, timeout=config["request_timeout_seconds"]
        )
        if response.status_code == 429:
            state.mark_rows(dataset, rows, "pending", response.text)
            LOG.warning("Dataset rate limited: %s", response.text)
            break
        response.raise_for_status()
        result = response.json()
        if result.get("skipped") or result.get("ingested") != len(rows):
            state.mark_rows(dataset, rows, "failed", json.dumps(result))
            raise RuntimeError(f"Dataset append partially failed: {result}")
        state.mark_rows(dataset, rows, "appended")
        for uri, row in entries:
            state.mark_manifest_consumed(uri, row["sample_id"])
        appended += len(rows)

    if appended:
        response = session.post(base + "/flush", timeout=config["request_timeout_seconds"])
        response.raise_for_status()
        LOG.info("flushed %s: %s", dataset, response.text)
    return len(uris), appended


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/sample_sync.yaml"))
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(args.config)
    state = StateStore(Path(config["writer_state_db"]))
    try:
        if args.status:
            media, rows, errors = state.summary()
            print("rows:", {r["status"]: r["n"] for r in rows})
            print("consumed_manifests:", state.db.execute("SELECT COUNT(*) FROM consumed_manifests").fetchone()[0])
            return 0
        lock = acquire_lock()
        signal.signal(signal.SIGTERM, lambda *_: globals().__setitem__("STOP", True))
        signal.signal(signal.SIGINT, lambda *_: globals().__setitem__("STOP", True))
        while not STOP:
            found, appended = process_once(config, state)
            LOG.info("manifest scan found=%d appended=%d", found, appended)
            if args.once:
                break
            deadline = time.time() + float(config["scan_interval_seconds"])
            while not STOP and time.time() < deadline:
                time.sleep(1)
        del lock
        return 0
    finally:
        state.close()


if __name__ == "__main__":
    sys.exit(main())
