#!/usr/bin/env python3
"""Mirror stable training evaluation cases to COS and xrtm data_srv."""

import argparse
import fcntl
import json
import hashlib
import logging
import mimetypes
import os
import re
import signal
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, Optional, Tuple

import requests
import yaml

LOG = logging.getLogger("sample-sync")
STOP = False
STEP_RE = re.compile(r"^step_(\d+)_(.+)\.mp4$")
CAUSAL_H3_STEP_RE = re.compile(r"^step-(\d+)-eval-(\d+)\.mp4$")
CAUSAL_H3_CPU_STEP_RE = re.compile(r"^step-(\d+)-eval-(\d+)-cpu\.mp4$")

DEFAULTS = {
    "runs_root": "/opt/dlami/nvme/chenxinyu/project/Universal-Edit-Forcing/runs",
    "causal_h3_runs_root": "/opt/dlami/nvme/chenxinyu/project/CausalH3/runs",
    "cos_base_url": "http://open-api-oversea.xrtm-infra.com",
    "data_srv_url": "http://140.207.124.187",
    "dataset_name": "universal-edit-training-eval",
    "bucket_alias": "buffer",
    "bucket_name": "xrtm-buffer-1258344702",
    "cos_prefix": "training-eval",
    "manifest_prefix": "training-eval-manifests",
    "state_db": "/var/lib/xrtm-sample-sync/data_srv_state.sqlite3",
    "scan_interval_seconds": 30,
    "stable_seconds": 60,
    "workers": 2,
    "request_timeout_seconds": 120,
    "retry_base_seconds": 30,
    "retry_max_seconds": 1800,
    "append_batch_size": 50,
    "flush_interval_seconds": 300,
}

SCHEMA_FIELDS = [
    {"name": "sample_id", "type": "string", "required": True},
    {"name": "experiment", "type": "enum", "required": True},
    {"name": "run_id", "type": "string", "required": True},
    {"name": "train_step", "type": "int", "required": True},
    {"name": "case_name", "type": "enum", "required": True},
    {"name": "task_type", "type": "enum", "required": False},
    {"name": "edit_type", "type": "enum", "required": False},
    {"name": "prompt", "type": "string", "required": False},
    {"name": "source_dataset", "type": "enum", "required": False},
    {"name": "source_index", "type": "int", "required": False},
    {"name": "result_video", "type": "link", "required": True, "bucket": "xrtm-buffer-1258344702"},
    {"name": "source_video", "type": "link", "required": False, "bucket": "xrtm-buffer-1258344702"},
    {"name": "target_video", "type": "link", "required": False, "bucket": "xrtm-buffer-1258344702"},
    {"name": "reference_video", "type": "link", "required": False, "bucket": "xrtm-buffer-1258344702"},
    {"name": "created_at", "type": "datetime", "required": True},
]


@dataclass(frozen=True)
class MediaFile:
    path: Path
    cos_key: str
    size: int
    mtime_ns: int

    @property
    def fingerprint(self) -> str:
        return f"{self.size}:{self.mtime_ns}"


@dataclass(frozen=True)
class CaseRow:
    sample_id: str
    experiment: str
    run_id: str
    train_step: int
    case_name: str
    metadata: dict
    media: Dict[str, MediaFile]


def load_config(path: Path) -> dict:
    config = dict(DEFAULTS)
    if path.exists():
        loaded = yaml.safe_load(path.read_text()) or {}
        if not isinstance(loaded, dict):
            raise ValueError("configuration root must be a mapping")
        config.update(loaded)
    config["runs_root"] = str(Path(config["runs_root"]).resolve())
    config["causal_h3_runs_root"] = str(
        Path(config["causal_h3_runs_root"]).resolve()
    )
    return config


def parse_step_file(path: Path) -> Optional[Tuple[int, str, str]]:
    match = STEP_RE.match(path.name)
    if not match:
        return None
    step = int(match.group(1))
    rest = match.group(2)
    for suffix, role in (("_src_in", "source_video"), ("_tgt_in", "target_video")):
        if rest.endswith(suffix):
            return step, rest[: -len(suffix)], role
    return step, rest, "result_video"


def media_for(
    path: Path,
    config: dict,
    experiment: str,
    run_id: str,
    *,
    source_namespace: Optional[str] = None,
    relative_path: Optional[PurePosixPath] = None,
) -> MediaFile:
    stat = path.stat()
    key = PurePosixPath(config["cos_prefix"])
    if source_namespace:
        key /= source_namespace
    key /= PurePosixPath(experiment) / run_id
    key /= relative_path or PurePosixPath(path.name)
    key = str(key)
    return MediaFile(path, key, stat.st_size, stat.st_mtime_ns)


def discover_universal_rows(
    config: dict,
    root: Path,
    experiment_filter=None,
    run_filter=None,
) -> Iterable[CaseRow]:
    if not root.is_dir():
        return
    for experiment_dir in sorted(root.iterdir()):
        if not experiment_dir.is_dir() or experiment_dir.name.startswith("."):
            continue
        experiment = experiment_dir.name
        if experiment_filter and experiment != experiment_filter:
            continue
        for run_dir in sorted(experiment_dir.iterdir()):
            if run_filter and run_dir.name != run_filter:
                continue
            samples = run_dir / "samples"
            if not samples.is_dir():
                continue
            meta_path = samples / "_eval_meta.json"
            try:
                meta_stat = meta_path.stat()
                if time.time() - meta_stat.st_mtime < float(config["stable_seconds"]):
                    continue
                metadata = json.loads(meta_path.read_text()).get("samples", {})
                if not isinstance(metadata, dict):
                    continue
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                continue
            grouped: Dict[Tuple[int, str], Dict[str, Path]] = {}
            for path in samples.glob("step_*.mp4"):
                parsed = parse_step_file(path)
                if parsed:
                    step, case_name, role = parsed
                    grouped.setdefault((step, case_name), {})[role] = path
            for (step, case_name), paths in sorted(grouped.items()):
                if "result_video" not in paths:
                    continue
                for filename, role in (
                    (f"_source_{case_name}.mp4", "source_video"),
                    (f"_target_{case_name}.mp4", "target_video"),
                    (f"_ref0_{case_name}.mp4", "reference_video"),
                ):
                    candidate = samples / filename
                    if role not in paths and candidate.is_file():
                        paths[role] = candidate
                media = {
                    role: media_for(path, config, experiment, run_dir.name)
                    for role, path in paths.items()
                }
                yield CaseRow(
                    sample_id=f"{experiment}:{run_dir.name}:{step:06d}:{case_name}",
                    experiment=experiment,
                    run_id=run_dir.name,
                    train_step=step,
                    case_name=case_name,
                    metadata=metadata.get(case_name, {}),
                    media=media,
                )


def causal_h3_result_files(samples: Path) -> Dict[Tuple[int, int], Tuple[Path, PurePosixPath]]:
    results = {}
    for path in samples.glob("step-*-eval-*.mp4"):
        match = CAUSAL_H3_STEP_RE.match(path.name)
        if not match:
            continue
        status_path = path.with_suffix(".decode_status.json")
        if status_path.is_file():
            try:
                if not json.loads(status_path.read_text()).get("ok"):
                    continue
            except (json.JSONDecodeError, OSError):
                continue
        if not path.with_suffix(".pt").is_file():
            continue
        results[(int(match.group(1)), int(match.group(2)))] = (
            path,
            PurePosixPath(path.name),
        )

    cpu_dir = samples / "cpu_decoded"
    if cpu_dir.is_dir():
        for path in cpu_dir.glob("step-*-eval-*-cpu.mp4"):
            match = CAUSAL_H3_CPU_STEP_RE.match(path.name)
            if not match:
                continue
            key = (int(match.group(1)), int(match.group(2)))
            latent_path = samples / f"step-{key[0]:09d}-eval-{key[1]:06d}.pt"
            if key not in results and latent_path.is_file():
                results[key] = (
                    path,
                    PurePosixPath("cpu_decoded") / path.name,
                )
    return results


def discover_causal_h3_rows(
    config: dict,
    root: Path,
    experiment_filter=None,
    run_filter=None,
) -> Iterable[CaseRow]:
    if not root.is_dir():
        return
    for experiment_dir in sorted(root.iterdir()):
        if not experiment_dir.is_dir() or experiment_dir.name.startswith("."):
            continue
        experiment = experiment_dir.name
        if experiment_filter and experiment != experiment_filter:
            continue
        for run_dir in sorted(experiment_dir.iterdir()):
            if run_filter and run_dir.name != run_filter:
                continue
            samples = run_dir / "artifacts" / "rollout_samples"
            if not samples.is_dir():
                continue
            condition_dir = samples / "fixed_conditions"
            for (step, eval_index), (result_path, result_relative) in sorted(
                causal_h3_result_files(samples).items()
            ):
                case_name = f"eval-{eval_index:06d}"
                candidates = (
                    ("source_video", condition_dir / f"{case_name}-source.mp4"),
                    ("target_video", condition_dir / f"{case_name}-target.mp4"),
                    (
                        "reference_video",
                        condition_dir / f"{case_name}-garment-reference.jpg",
                    ),
                )
                media = {
                    "result_video": media_for(
                        result_path,
                        config,
                        experiment,
                        run_dir.name,
                        source_namespace="causal_h3",
                        relative_path=result_relative,
                    )
                }
                for role, path in candidates:
                    if path.is_file():
                        media[role] = media_for(
                            path,
                            config,
                            experiment,
                            run_dir.name,
                            source_namespace="causal_h3",
                            relative_path=PurePosixPath("fixed_conditions") / path.name,
                        )
                yield CaseRow(
                    sample_id=(
                        f"causal_h3:{experiment}:{run_dir.name}:"
                        f"{step:09d}:{case_name}"
                    ),
                    experiment=experiment,
                    run_id=run_dir.name,
                    train_step=step,
                    case_name=case_name,
                    metadata={
                        "task_type": "rv2v",
                        "edit_type": "tryon",
                        "dataset": experiment,
                        "index": eval_index,
                    },
                    media=media,
                )


def discover_rows(config: dict, experiment_filter=None, run_filter=None) -> Iterable[CaseRow]:
    yield from discover_universal_rows(
        config, Path(config["runs_root"]), experiment_filter, run_filter
    )
    yield from discover_causal_h3_rows(
        config,
        Path(config["causal_h3_runs_root"]),
        experiment_filter,
        run_filter,
    )


class StateStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS media (
              local_path TEXT PRIMARY KEY, fingerprint TEXT NOT NULL,
              first_seen REAL NOT NULL, status TEXT NOT NULL,
              cos_uri TEXT, attempts INTEGER NOT NULL DEFAULT 0,
              next_retry REAL NOT NULL DEFAULT 0, last_error TEXT);
            CREATE TABLE IF NOT EXISTS rows (
              dataset TEXT NOT NULL, sample_id TEXT NOT NULL,
              status TEXT NOT NULL, payload TEXT, last_error TEXT,
              PRIMARY KEY(dataset, sample_id));
            CREATE TABLE IF NOT EXISTS datasets (
              name TEXT PRIMARY KEY, last_flush REAL NOT NULL DEFAULT 0,
              unflushed INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS manifests (
              sample_id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL,
              cos_uri TEXT NOT NULL, status TEXT NOT NULL, last_error TEXT);
            CREATE TABLE IF NOT EXISTS consumed_manifests (
              cos_uri TEXT PRIMARY KEY, sample_id TEXT NOT NULL);
            UPDATE media SET status='pending' WHERE status='uploading';
            UPDATE rows SET status='pending' WHERE status='appending';
            """
        )
        self.db.commit()

    def close(self):
        self.db.close()

    def media_ready(self, media: MediaFile, now: float, stable: float) -> bool:
        key = str(media.path)
        row = self.db.execute("SELECT * FROM media WHERE local_path=?", (key,)).fetchone()
        if row is None:
            self.db.execute(
                "INSERT INTO media(local_path,fingerprint,first_seen,status) VALUES(?,?,?,'pending')",
                (key, media.fingerprint, now),
            )
            self.db.commit()
            return stable <= 0
        if row["fingerprint"] != media.fingerprint:
            self.db.execute(
                "UPDATE media SET fingerprint=?,first_seen=?,status='pending',cos_uri=NULL,attempts=0,next_retry=0,last_error=NULL WHERE local_path=?",
                (media.fingerprint, now, key),
            )
            self.db.commit()
            return stable <= 0
        return row["status"] == "uploaded" or (
            row["status"] in ("pending", "failed")
            and now - row["first_seen"] >= stable
            and now >= row["next_retry"]
        )

    def media_uri(self, media: MediaFile) -> Optional[str]:
        row = self.db.execute(
            "SELECT cos_uri FROM media WHERE local_path=? AND fingerprint=? AND status='uploaded'",
            (str(media.path), media.fingerprint),
        ).fetchone()
        return row[0] if row else None

    def mark_media_uploading(self, media):
        self.db.execute("UPDATE media SET status='uploading' WHERE local_path=?", (str(media.path),))
        self.db.commit()

    def mark_media_success(self, media, uri):
        self.db.execute(
            "UPDATE media SET status='uploaded',cos_uri=?,last_error=NULL,next_retry=0 WHERE local_path=?",
            (uri, str(media.path)),
        )
        self.db.commit()

    def mark_media_failure(self, media, error, config):
        row = self.db.execute("SELECT attempts FROM media WHERE local_path=?", (str(media.path),)).fetchone()
        attempts = (row[0] if row else 0) + 1
        delay = min(config["retry_max_seconds"], config["retry_base_seconds"] * 2 ** min(attempts - 1, 10))
        self.db.execute(
            "UPDATE media SET status='failed',attempts=?,next_retry=?,last_error=? WHERE local_path=?",
            (attempts, time.time() + delay, error[-2000:], str(media.path)),
        )
        self.db.commit()

    def row_uploaded(self, dataset, sample_id):
        row = self.db.execute("SELECT status FROM rows WHERE dataset=? AND sample_id=?", (dataset, sample_id)).fetchone()
        return bool(row and row[0] == "appended")

    def mark_remote_rows(self, dataset, ids):
        self.db.executemany(
            "INSERT OR REPLACE INTO rows(dataset,sample_id,status) VALUES(?,?,'appended')",
            [(dataset, sample_id) for sample_id in ids],
        )
        self.db.commit()

    def mark_rows(self, dataset, payloads, status, error=None):
        self.db.executemany(
            "INSERT OR REPLACE INTO rows(dataset,sample_id,status,payload,last_error) VALUES(?,?,?,?,?)",
            [(dataset, row["sample_id"], status, json.dumps(row, ensure_ascii=False), error) for row in payloads],
        )
        self.db.commit()

    def note_append(self, dataset, count):
        self.db.execute(
            "INSERT INTO datasets(name,unflushed) VALUES(?,?) ON CONFLICT(name) DO UPDATE SET unflushed=unflushed+excluded.unflushed",
            (dataset, count),
        )
        self.db.commit()

    def should_flush(self, dataset, config):
        row = self.db.execute("SELECT last_flush,unflushed FROM datasets WHERE name=?", (dataset,)).fetchone()
        return bool(row and row["unflushed"] > 0 and (
            row["unflushed"] >= config["append_batch_size"]
            or time.time() - row["last_flush"] >= config["flush_interval_seconds"]
        ))

    def mark_flushed(self, dataset):
        self.db.execute("UPDATE datasets SET last_flush=?,unflushed=0 WHERE name=?", (time.time(), dataset))
        self.db.commit()

    def manifest_published(self, sample_id, fingerprint):
        row = self.db.execute(
            "SELECT fingerprint,status FROM manifests WHERE sample_id=?", (sample_id,)
        ).fetchone()
        return bool(row and row[0] == fingerprint and row[1] == "published")

    def mark_manifest(self, sample_id, fingerprint, uri, status, error=None):
        self.db.execute(
            "INSERT OR REPLACE INTO manifests(sample_id,fingerprint,cos_uri,status,last_error) VALUES(?,?,?,?,?)",
            (sample_id, fingerprint, uri, status, error),
        )
        self.db.commit()

    def manifest_consumed(self, uri):
        return self.db.execute(
            "SELECT 1 FROM consumed_manifests WHERE cos_uri=?", (uri,)
        ).fetchone() is not None

    def mark_manifest_consumed(self, uri, sample_id):
        self.db.execute(
            "INSERT OR REPLACE INTO consumed_manifests(cos_uri,sample_id) VALUES(?,?)",
            (uri, sample_id),
        )
        self.db.commit()

    def retry(self, path_prefix):
        prefix = str(Path(path_prefix).resolve())
        cursor = self.db.execute(
            "UPDATE media SET status='pending',attempts=0,next_retry=0,last_error=NULL WHERE local_path=? OR local_path LIKE ?",
            (prefix, prefix.rstrip("/") + "/%"),
        )
        self.db.commit()
        return cursor.rowcount

    def summary(self):
        media = self.db.execute("SELECT status,COUNT(*) n FROM media GROUP BY status").fetchall()
        rows = self.db.execute("SELECT status,COUNT(*) n FROM rows GROUP BY status").fetchall()
        errors = self.db.execute(
            "SELECT local_path,last_error FROM media WHERE last_error IS NOT NULL ORDER BY next_retry DESC LIMIT 10"
        ).fetchall()
        return media, rows, errors


def cos_uri(config, media):
    return f"cos://{config['bucket_name']}/{media.cos_key}"


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def upload_media(config, media):
    uri = cos_uri(config, media)
    base = config["cos_base_url"].rstrip("/")
    existing = requests.get(
        base + "/api/cos/resource",
        params={"uri": uri},
        timeout=config["request_timeout_seconds"],
    )
    if existing.status_code == 200:
        info = existing.json()
        etag = str(info.get("etag", "")).strip('"')
        if info.get("size") == media.size and etag == file_md5(media.path):
            return uri
        raise RuntimeError(f"COS conflict at {uri}: existing object differs from local file")
    elif existing.status_code not in (404,):
        existing.raise_for_status()

    headers = {
        "Content-Type": mimetypes.guess_type(media.path.name)[0] or "application/octet-stream",
        "Content-Length": str(media.size),
    }
    with media.path.open("rb") as body:
        response = requests.post(
            base + "/api/cos/upload",
            params={"uri": uri, "mode": "fast"},
            headers=headers,
            data=body,
            timeout=config["request_timeout_seconds"],
        )
    response.raise_for_status()
    result = response.json()
    if result.get("cos_uri") != uri or result.get("size") != media.size:
        raise RuntimeError(f"unexpected COS upload response: {result}")
    return uri


def manifest_document(row, state):
    payload = row_payload(row, state)
    if payload is None:
        return None
    content = json.dumps(
        {"version": 1, "row": payload}, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    fingerprint = hashlib.sha256(content).hexdigest()
    return content, fingerprint


def publish_manifest(config, row, content, fingerprint):
    key = f"{config['manifest_prefix'].strip('/')}/{hashlib.sha256(row.sample_id.encode()).hexdigest()}.json"
    uri = f"cos://{config['bucket_name']}/{key}"
    base = config["cos_base_url"].rstrip("/")
    existing = requests.get(
        base + "/api/cos/resource", params={"uri": uri},
        timeout=config["request_timeout_seconds"],
    )
    digest = hashlib.md5(content).hexdigest()
    if existing.status_code == 200:
        info = existing.json()
        if info.get("size") == len(content) and str(info.get("etag", "")).strip('"') == digest:
            return uri, fingerprint
        raise RuntimeError(f"manifest conflict at {uri}")
    if existing.status_code != 404:
        existing.raise_for_status()
    response = requests.post(
        base + "/api/cos/upload",
        params={"uri": uri, "mode": "stream"},
        headers={"Content-Type": "application/json", "Content-Length": str(len(content))},
        data=content,
        timeout=config["request_timeout_seconds"],
    )
    response.raise_for_status()
    result = response.json()
    if result.get("cos_uri") != uri or result.get("size") != len(content):
        raise RuntimeError(f"unexpected manifest upload response: {result}")
    return uri, fingerprint


def schema_payload(dataset):
    return {
        "name": dataset,
        "description": f"Training evaluation samples for {dataset}",
        "labels": {"source": "training_eval", "project": "universal-edit"},
        "schema": {"dataset": dataset, "version": "1", "fields": SCHEMA_FIELDS},
    }


def ensure_dataset(config, session, dataset):
    base = config["data_srv_url"].rstrip("/") + "/api/data"
    response = session.get(f"{base}/{dataset}", timeout=config["request_timeout_seconds"])
    if response.status_code == 404:
        created = session.post(f"{base}/create", json=schema_payload(dataset), timeout=config["request_timeout_seconds"])
        created.raise_for_status()
        active = session.patch(f"{base}/{dataset}/status", params={"status": "active"}, timeout=config["request_timeout_seconds"])
        active.raise_for_status()
        return
    response.raise_for_status()
    schema = session.get(f"{base}/{dataset}/schema", timeout=config["request_timeout_seconds"])
    schema.raise_for_status()
    actual = [(f["name"], f["type"], bool(f.get("required")), f.get("bucket")) for f in schema.json()["schema"]["fields"]]
    expected = [(f["name"], f["type"], bool(f.get("required")), f.get("bucket")) for f in SCHEMA_FIELDS]
    if actual != expected:
        raise RuntimeError(f"existing Dataset {dataset} has incompatible schema")


def remote_sample_ids(config, session, dataset):
    base = config["data_srv_url"].rstrip("/") + f"/api/data/{dataset}/read"
    cursor = None
    result = set()
    while True:
        # A bounded read excludes the Redis-buffer handoff, which can block
        # after the final sealed shard on this data_srv deployment.
        params = {"limit": 2000, "to": (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()}
        if cursor:
            params["cursor"] = cursor
        response = session.get(base, params=params, timeout=config["request_timeout_seconds"])
        response.raise_for_status()
        data = response.json()
        result.update(row.get("sample_id") for row in data.get("rows", []) if row.get("sample_id"))
        if not data.get("has_more") or not data.get("next_cursor"):
            return result
        cursor = data["next_cursor"]


def row_payload(row, state):
    links = {role: state.media_uri(media) for role, media in row.media.items()}
    if not links.get("result_video"):
        return None
    meta = row.metadata
    payload = {
        "sample_id": row.sample_id,
        "experiment": row.experiment,
        "run_id": row.run_id,
        "train_step": row.train_step,
        "case_name": row.case_name,
        "task_type": meta.get("task_type") or None,
        "edit_type": meta.get("edit_type") or None,
        "prompt": meta.get("prompt") or None,
        "source_dataset": meta.get("dataset") or None,
        "source_index": meta.get("index") if isinstance(meta.get("index"), int) else None,
        "result_video": links["result_video"],
        "source_video": links.get("source_video"),
        "target_video": links.get("target_video"),
        "reference_video": links.get("reference_video"),
        "created_at": datetime.fromtimestamp(row.media["result_video"].mtime_ns / 1e9, timezone.utc).isoformat(),
    }
    return {key: value for key, value in payload.items() if value is not None}


def process_once(config, state, experiment=None, run_id=None, limit=None):
    now = time.time()
    rows = list(discover_rows(config, experiment, run_id))
    if limit:
        rows = rows[:limit]
    needed = {}
    stable_rows = []
    for row in rows:
        readiness = [state.media_ready(media, now, config["stable_seconds"]) for media in row.media.values()]
        if all(readiness):
            stable_rows.append(row)
            for media in row.media.values():
                if not state.media_uri(media):
                    needed[str(media.path)] = media
    with ThreadPoolExecutor(max_workers=config["workers"]) as pool:
        futures = {}
        for media in needed.values():
            state.mark_media_uploading(media)
            futures[pool.submit(upload_media, config, media)] = media
        for future in as_completed(futures):
            media = futures[future]
            try:
                uri = future.result()
                current = media.path.stat()
                if current.st_size != media.size or current.st_mtime_ns != media.mtime_ns:
                    raise RuntimeError("source changed during upload")
                state.mark_media_success(media, uri)
                LOG.info("uploaded %s -> %s", media.path, uri)
            except Exception as exc:
                state.mark_media_failure(media, str(exc), config)
                LOG.error("upload failed %s: %s", media.path, exc)

    publish_jobs = {}
    published = 0
    with ThreadPoolExecutor(max_workers=config["workers"]) as pool:
        for row in stable_rows:
            built = manifest_document(row, state)
            if built is None:
                continue
            _, fingerprint = built
            if state.manifest_published(row.sample_id, fingerprint):
                continue
            content, fingerprint = built
            publish_jobs[pool.submit(publish_manifest, config, row, content, fingerprint)] = (row, fingerprint)
        for future in as_completed(publish_jobs):
            row, fingerprint = publish_jobs[future]
            try:
                uri, actual_fingerprint = future.result()
                state.mark_manifest(row.sample_id, actual_fingerprint, uri, "published")
                published += 1
                LOG.info("published manifest %s -> %s", row.sample_id, uri)
            except Exception as exc:
                state.mark_manifest(row.sample_id, fingerprint, "", "failed", str(exc))
                LOG.error("manifest failed %s: %s", row.sample_id, exc)
    return len(rows), len(needed), published

def acquire_lock():
    handle = open("/run/xrtm-sample-sync.lock", "w")
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return handle


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/sample_sync.yaml"))
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--retry", metavar="PATH")
    parser.add_argument("--experiment")
    parser.add_argument("--run-id")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--stable-seconds", type=float)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(args.config)
    if args.stable_seconds is not None:
        config["stable_seconds"] = args.stable_seconds
    if args.dry_run:
        rows = list(discover_rows(config, args.experiment, args.run_id))
        if args.limit:
            rows = rows[:args.limit]
        for row in rows:
            print(row.sample_id, "->", row.experiment, sorted(row.media))
        print("rows:", len(rows))
        return 0
    state = StateStore(Path(config["state_db"]))
    try:
        if args.status:
            media, rows, errors = state.summary()
            print("media:", {r["status"]: r["n"] for r in media})
            print("rows:", {r["status"]: r["n"] for r in rows})
            for row in errors:
                print(f"error {row['local_path']}: {row['last_error']}")
            return 0
        if args.retry:
            print("reset:", state.retry(args.retry))
            return 0
        lock_handle = acquire_lock()
        signal.signal(signal.SIGTERM, lambda *_: globals().__setitem__("STOP", True))
        signal.signal(signal.SIGINT, lambda *_: globals().__setitem__("STOP", True))
        while not STOP:
            found, uploads, published = process_once(config, state, args.experiment, args.run_id, args.limit)
            LOG.info("scan rows=%d uploads=%d manifests=%d", found, uploads, published)
            if args.once:
                break
            deadline = time.time() + float(config["scan_interval_seconds"])
            while not STOP and time.time() < deadline:
                time.sleep(1)
        del lock_handle
        return 0
    finally:
        state.close()


if __name__ == "__main__":
    sys.exit(main())
