from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class BatchJobStatus(str, Enum):
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class BatchJobFile(BaseModel):
    job_id: str
    file_id: str
    user: str
    status: BatchJobStatus = Field(default=BatchJobStatus.pending)
    original_filename: Optional[str] = None
    storage_path: str
    params: Dict[str, Any] = Field(default_factory=dict)
    checksum: Optional[str] = None
    original_file_id: Optional[str] = None
    parsed_text: str = Field(default="")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    images: Dict[str, str] = Field(default_factory=dict)
    format: str = Field(default="markdown")


class BatchJob(BaseModel):
    job_id: str
    user: str
    status: BatchJobStatus = Field(default=BatchJobStatus.pending)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    params: Dict[str, Any] = Field(default_factory=dict)
    files: List[BatchJobFile] = Field(default_factory=list)

