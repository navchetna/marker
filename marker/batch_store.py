from __future__ import annotations
import asyncio
import hashlib

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .batch_models import BatchJob, BatchJobFile, BatchJobStatus
from typing import Annotated, List, Optional


BATCH_STORE_PATH = Path("./batch_jobs_store/batch_jobs.json")


def _ensure_store_path() -> None:
    """Make sure the batch job store file exists and starts as an object."""
    BATCH_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not BATCH_STORE_PATH.exists() or BATCH_STORE_PATH.stat().st_size == 0:
        BATCH_STORE_PATH.write_text("{}")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, BatchJobStatus):
        return obj.value
    return str(obj)


def _read_store() -> Dict[str, Dict[str, Any]]:
    _ensure_store_path()
    try:
        raw = BATCH_STORE_PATH.read_text()
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    return {}

def _write_store(data: Dict[str, Dict[str, Any]]) -> None:
    _ensure_store_path()
    with BATCH_STORE_PATH.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=_json_default)


def calculate_pdf_checksum(pdf_path: Path | str) -> str:
    """
    Compute a SHA256 checksum for the given PDF file path.
    """
    path = Path(pdf_path)
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_store_snapshot() -> Dict[str, Dict[str, Any]]:
    """
    Return a snapshot of the stored batch jobs.
    """
    return _read_store()


def add_batch_job(job: BatchJob) -> None:
    store = _read_store()
    store[job.job_id] = job.model_dump()
    _write_store(store)


def _normalize_job_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(payload)
    normalized["job_id"] = (
        normalized.get("job_id") or normalized.get("batch_job_id") or ""
    )
    normalized["params"] = normalized.get("params") or {}
    files = []
    for entry in normalized.get("files", []):
        normalized_entry = dict(entry)
        normalized_entry["job_id"] = normalized["job_id"]
        normalized_entry["file_id"] = (
            normalized_entry.get("file_id")
            or normalized_entry.get("batch_job_file_id")
            or ""
        )
        normalized_entry["storage_path"] = normalized_entry.get(
            "storage_path"
        ) or normalized_entry.get("filepath") or ""
        normalized_entry.setdefault("params", normalized["params"])
        normalized_entry.setdefault("metadata", {})
        normalized_entry.setdefault("images", {})
        files.append(normalized_entry)
    normalized["files"] = files
    return normalized


def get_batch_job_status(job_id: str) -> Optional[BatchJob]:
    store = _read_store()
    payload = store.get(job_id)
    if not payload:
        return None
    return BatchJob(**_normalize_job_payload(payload))


def get_batch_file_status(file_id: str, job_id: Optional[str] = None) -> Optional[BatchJobFile]:
    store = _read_store()
    candidates = (
        [store[job_id]] if job_id and job_id in store else store.values()
    )
    for payload in candidates:
        files = payload.get("files", [])
        for entry in files:
            if entry.get("file_id") == file_id:
                normalized_entry = dict(entry)
                normalized_entry["job_id"] = payload.get("job_id")
                normalized_entry.setdefault("metadata", {})
                normalized_entry.setdefault("images", {})
                normalized_entry.setdefault("params", payload.get("params", {}))
                normalized_entry.setdefault(
                    "storage_path",
                    normalized_entry.get("storage_path") or normalized_entry.get("filepath") or "",
                )
                return BatchJobFile(**normalized_entry)
    return None

def _determine_job_status(files: list[dict[str, Any]]) -> BatchJobStatus:
    statuses = {file.get("status") for file in files}
    if BatchJobStatus.processing.value in statuses:
        return BatchJobStatus.processing
    if BatchJobStatus.failed.value in statuses:
        return BatchJobStatus.failed
    if BatchJobStatus.pending.value in statuses:
        return BatchJobStatus.pending
    return BatchJobStatus.completed


def update_batch_file(
    job_id: str,
    file_id: str,
    *,
    status: BatchJobStatus,
    parsed_text: str,
    metadata: Dict[str, Any],
    images: Dict[str, str],
    format: str,
) -> None:
    store = _read_store()
    job_payload = store.get(job_id)
    if not job_payload:
        return
    file_entries = job_payload.get("files", [])
    for entry in file_entries:
        if entry.get("file_id") == file_id:
            entry["status"] = status.value
            entry["parsed_text"] = parsed_text
            entry["metadata"] = metadata or {}
            entry["images"] = images or {}
            entry["format"] = format
            break
    job_payload["files"] = file_entries
    job_payload["status"] = _determine_job_status(file_entries).value
    job_payload["updated_at"] = datetime.utcnow().isoformat()
    store[job_id] = job_payload
    _write_store(store)


def claim_pending_batch_file() -> Optional[Tuple[str, Dict[str, Any]]]:
    store = _read_store()
    for job_id, job_payload in store.items():
        files = job_payload.get("files", [])
        for entry in files:
            if entry.get("status") == BatchJobStatus.pending.value:
                entry["status"] = BatchJobStatus.processing.value
                job_payload["status"] = BatchJobStatus.processing.value
                job_payload["updated_at"] = datetime.utcnow().isoformat()
                job_payload["files"] = files
                store[job_id] = job_payload
                _write_store(store)
                entry_copy = dict(entry)
                entry_copy.setdefault("params", job_payload.get("params", {}))
                entry_copy["job_id"] = job_id
                return job_id, entry_copy
    return None
