import asyncio
import base64
import io
import json
import os
import shutil
import time
import traceback
import zipfile

from contextlib import asynccontextmanager, suppress
from datetime import datetime
from pathlib import Path
from typing import Annotated, Dict, List, Optional

import click
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from starlette.responses import HTMLResponse

from marker.batch_models import BatchJob, BatchJobFile, BatchJobStatus
from marker.batch_store import (
    add_batch_job,
    calculate_pdf_checksum,
    claim_pending_batch_file,
    get_batch_file_status,
    get_batch_job_status,
    get_store_snapshot,
    update_batch_file,
)
from marker.config.parser import ConfigParser
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from marker.settings import settings
from tree_parser.tree import Tree
from tree_parser.treeparser import TreeParser

app_data = {}



UPLOAD_DIRECTORY = Path(os.environ.get("UPLOAD_DIR", "./uploads"))
BATCH_STORAGE_DIR = Path(os.environ.get("BATCH_STORE_DIR", "./batch_jobs_store"))
BATCH_UPLOAD_DIRECTORY = BATCH_STORAGE_DIR / "uploads"
BATCH_PROCESSING_DIR = Path(os.environ.get("BATCH_PROCESSING_DIR", "./batch_processing"))
BATCH_PROCESSING_INTERVAL_SECONDS = int(os.environ.get("BATCH_INTERVAL", "10"))


def _sanitize_file_id(file_id: str) -> str:
    return file_id.replace("/", "_").replace(":", "_")


def _file_processing_dir(job_id: str, file_id: str) -> Path:
    return BATCH_PROCESSING_DIR / job_id / _sanitize_file_id(file_id)


def _file_status_response(file: BatchJobFile) -> dict:
    return {
        "job_id": file.job_id,
        "file_id": file.file_id,
        "status": file.status.value,
        "original_filename": file.original_filename,
        # "metadata": file.metadata,
        "format": file.format,
        "params": file.params,
    }


def _job_status_response(job: BatchJob) -> dict:
    return {
        "job_id": job.job_id,
        "user": job.user,
        "status": job.status.value,
        "created_at": job.created_at.isoformat(),
        "updated_at": job.updated_at.isoformat(),
        "params": job.params,
        "files": [_file_status_response(file) for file in job.files],
    }


def _write_job_status_file(job: BatchJob) -> None:
    """Persist the job's status response inside its batch_processing folder."""
    status_dir = BATCH_PROCESSING_DIR / job.job_id
    status_dir.mkdir(parents=True, exist_ok=True)
    status_file = status_dir / "status.json"
    status_file.write_text(json.dumps(_job_status_response(job), indent=2))

@asynccontextmanager
async def lifespan(app: FastAPI):
    app_data["models"] = create_model_dict()
    BATCH_PROCESSING_DIR.mkdir(parents=True, exist_ok=True)
    app_data["batch_processor_task"] = asyncio.create_task(_batch_processing_loop())

    try:
        yield
    finally:
        print("[lifespan] Cleaning up app data")
        if "models" in app_data:
            del app_data["models"]
        task = app_data.pop("batch_processor_task", None)
        if task:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    return HTMLResponse(
        """
<h1>Marker API</h1>
<ul>
    <li><a href="/docs">API Documentation</a></li>
    <li><a href="/marker">Run marker (post request only)</a></li>
</ul>
"""
    )


class CommonParams(BaseModel):
    user: str = Field(..., description="The user submitting the request")
    filepath: Annotated[
        Optional[str], Field(description="The path to the PDF file to convert.")
    ]
    page_range: Annotated[
        Optional[str],
        Field(
            description="Page range to convert, specify comma separated page numbers or ranges.  Example: 0,5-10,20",
            example=None,
        ),
    ] = None
    force_ocr: Annotated[
        bool,
        Field(
            description="Force OCR on all pages of the PDF.  Defaults to False.  This can lead to worse results if you have good text in your PDFs (which is true in most cases)."
        ),
    ] = False
    paginate_output: Annotated[
        bool,
        Field(
            description="Whether to paginate the output.  Defaults to False.  If set to True, each page of the output will be separated by a horizontal rule that contains the page number (2 newlines, {PAGE_NUMBER}, 48 - characters, 2 newlines)."
        ),
    ] = False
    output_format: Annotated[
        str,
        Field(
            description="The format to output the text in.  Can be 'markdown', 'json', or 'html'.  Defaults to 'markdown'."
        ),
    ] = "markdown"
    output_dir: Annotated[
        Optional[str],
        Field(
            description="Optional directory for storing TreeParser output files."
        ),
    ] = None


async def _convert_pdf(params: CommonParams):
    assert params.output_format in ["markdown", "json", "html", "chunks"], (
        "Invalid output format"
    )
    try:
        user_param = params.user
        options = params.model_dump()
        config_parser = ConfigParser(options)
        config_dict = config_parser.generate_config_dict()
        config_dict["pdftext_workers"] = 1
        converter_cls = PdfConverter
        converter = converter_cls(
            config=config_dict,
            artifact_dict=app_data["models"],
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
            llm_service=config_parser.get_llm_service(),
        )
        tree = Tree(params.filepath, user_param, output_dir=params.output_dir)
        tree_parser = TreeParser(user_param, params.output_dir)
        tree_parser.populate_tree(tree, converter)

        tree_parser.generate_output_text(tree)
        tree_parser.generate_output_json(tree)
        rendered = converter(params.filepath)
        text, _, images = text_from_rendered(rendered)
        metadata = rendered.metadata
    except Exception as e:
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
        }

    encoded = {}
    for k, v in images.items():
        byte_stream = io.BytesIO()
        v.save(byte_stream, format=settings.OUTPUT_IMAGE_FORMAT)
        encoded[k] = base64.b64encode(byte_stream.getvalue()).decode(
            settings.OUTPUT_ENCODING
        )

    return {
        "format": params.output_format,
        "output": text,
        "images": encoded,
        "metadata": metadata,
        "success": True,
    }


async def _save_upload_file(destination: Path, upload_file: UploadFile) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    contents = await upload_file.read()
    destination.write_bytes(contents)
    return destination


async def _extract_from_zip(destination: Path, zip_file: UploadFile) -> List[Path]:
    extracted_paths: List[Path] = []
    archived_path = destination / zip_file.filename
    await _save_upload_file(archived_path, zip_file)
    try:
        with zipfile.ZipFile(archived_path) as archive:
            for member in archive.namelist():
                if member.endswith("/") or not member.lower().endswith(".pdf"):
                    continue
                target_name = Path(member).name
                if not target_name:
                    continue
                target_path = destination / target_name
                target_path.write_bytes(archive.read(member))
                extracted_paths.append(target_path)
    finally:
        if archived_path.exists():
            archived_path.unlink()
    return extracted_paths


async def _process_batch_file(job_id: str, batch_file: BatchJobFile) -> None:
    processing_dir = _file_processing_dir(job_id, batch_file.file_id)
    input_file = processing_dir / "input" / "original.pdf"
    if not input_file.exists():
        error_text = "Original PDF not found for file"
        update_batch_file(
            job_id=job_id,
            file_id=batch_file.file_id,
            status=BatchJobStatus.failed,
            parsed_text=error_text,
            metadata={},
            images={},
            format=batch_file.params.get("output_format", "markdown"),
        )
        return

    scratch_file = processing_dir / "scratch" / "processing.pdf"
    scratch_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_file, scratch_file)

    batch_output_dir = processing_dir / "output"
    batch_output_dir.mkdir(parents=True, exist_ok=True)
    params = CommonParams(
        **batch_file.params,
        filepath=str(scratch_file),
        output_dir=str(batch_output_dir),
    )

    try:
        result = await _convert_pdf(params)
    except Exception as exc:
        traceback.print_exc()
        result = {"success": False, "error": str(exc)}

    success = result.get("success", False)
    parsed_text = result.get("output", "")
    if not success:
        parsed_text = result.get("error", parsed_text)

    format_choice = result.get(
        "format", batch_file.params.get("output_format", "markdown")
    )
    status = BatchJobStatus.completed if success else BatchJobStatus.failed

    update_batch_file(
        job_id=job_id,
        file_id=batch_file.file_id,
        status=status,
        parsed_text=parsed_text,
        metadata=result.get("metadata", {}),
        images=result.get("images", {}),
        format=format_choice,
    )
    job = get_batch_job_status(job_id)
    if job and job.status not in (
        BatchJobStatus.processing,
        BatchJobStatus.pending,
    ):
        _write_job_status_file(job)

    output_dir = batch_output_dir
    output_path = output_dir / "output.json"
    output_payload = {
        "job_id": job_id,
        "file_id": batch_file.file_id,
        "user": batch_file.user,
        "success": success,
        "format": format_choice,
        "output": result.get("output"),
        "parsed_text": parsed_text,
        "metadata": result.get("metadata", {}),
        "images": result.get("images", {}),
        "error": result.get("error"),
        "timestamp": datetime.utcnow().isoformat(),
    }
    output_path.write_text(json.dumps(output_payload, indent=2))


async def _batch_processing_loop():
    while True:
        try:
            claim = claim_pending_batch_file()
            if not claim:
                await asyncio.sleep(BATCH_PROCESSING_INTERVAL_SECONDS)
                continue
            print(f"Claimed file: {claim}")
            job_id, file_entry = claim
            batch_file = BatchJobFile(**file_entry)
            await _process_batch_file(job_id, batch_file)
        except asyncio.CancelledError:
            break
        except Exception:
            traceback.print_exc()
            await asyncio.sleep(BATCH_PROCESSING_INTERVAL_SECONDS)


@app.post("/marker")
async def convert_pdf(params: CommonParams):
    return await _convert_pdf(params)


@app.post("/marker/upload")
async def convert_pdf_upload(
    user: str = Form(..., description="The user submitting the request"),
    page_range: Optional[str] = Form(default=None),
    force_ocr: Optional[bool] = Form(default=False),
    paginate_output: Optional[bool] = Form(default=False),
    output_format: Optional[str] = Form(default="markdown"),
    file: UploadFile = File(
        ..., description="The PDF file to convert.", media_type="application/pdf"
    ),
):
    os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)
    upload_path = os.path.join(UPLOAD_DIRECTORY, file.filename)
    with open(upload_path, "wb+") as upload_file:
        file_contents = await file.read()
        upload_file.write(file_contents)

    params = CommonParams(
        user=user,
        filepath=upload_path,
        page_range=page_range,
        force_ocr=force_ocr,
        paginate_output=paginate_output,
        output_format=output_format,
    )
    results = await _convert_pdf(params)
    os.remove(upload_path)
    return results


@app.post("/marker/batch_job")
async def create_batch_job(
    user: str = Form(..., description="The user submitting the request"),
    page_range: Optional[str] = Form(default=None),
    force_ocr: Optional[bool] = Form(default=False),
    paginate_output: Optional[bool] = Form(default=False),
    output_format: Optional[str] = Form(default="markdown"),
    files: Optional[List[UploadFile]] = File(default=None),
    zip_file: Optional[UploadFile] = File(
        default=None, description="A ZIP archive containing PDF files."
    ),
):
    if not files and not zip_file:
        raise HTTPException(
            status_code=400,
            detail="Provide at least one PDF file or a ZIP archive containing PDFs.",
        )

    sanitized_user = user.replace(" ", "_")
    batch_job_id = f"{sanitized_user}-{time.time()}"
    job_upload_dir = BATCH_UPLOAD_DIRECTORY / batch_job_id
    job_upload_dir.mkdir(parents=True, exist_ok=True)

    saved_files: List[Path] = []
    for upload_file in files or []:
        destination = job_upload_dir / upload_file.filename
        saved_files.append(await _save_upload_file(destination, upload_file))

    if zip_file:
        saved_files.extend(await _extract_from_zip(job_upload_dir, zip_file))

    if not saved_files:
        raise HTTPException(
            status_code=400, detail="No PDF files were discovered for the batch."
        )

    store_snapshot = get_store_snapshot()
    checksum_lookup: Dict[str, Dict[str, str]] = {}
    for job_payload in store_snapshot.values():
        if job_payload.get("user") != user:
            continue
        for entry in job_payload.get("files", []):
            stored_checksum = entry.get("checksum")
            stored_file_id = entry.get("file_id")
            if stored_checksum and stored_file_id:
                checksum_lookup.setdefault(
                    stored_checksum, {"file_id": stored_file_id}
                )

    seen_checksums: Dict[str, Dict[str, str]] = dict(checksum_lookup)
    duplicate_files: List[Dict[str, str]] = []
    unique_saved_files: List[Dict[str, str]] = []
    for saved in saved_files:
        checksum = calculate_pdf_checksum(saved)
        existing_entry = seen_checksums.get(checksum)
        if existing_entry:
            duplicate_files.append(
                {
                    "file_name": saved.name,
                    "original_file_id": existing_entry["file_id"],
                }
            )
            continue
        file_id = f"{batch_job_id}:{saved.name}"
        seen_checksums[checksum] = {"file_id": file_id}
        unique_saved_files.append(
            {"path": saved, "checksum": checksum, "file_id": file_id}
        )

    params_template = {
        "user": user,
        "page_range": page_range,
        "force_ocr": force_ocr,
        "paginate_output": paginate_output,
        "output_format": output_format,
    }

    params_copy = params_template.copy()
    batch_files: List[BatchJobFile] = []
    for entry in unique_saved_files:
        saved_path = entry["path"]
        file_id = entry["file_id"]
        checksum = entry["checksum"]
        file_dir = _file_processing_dir(batch_job_id, file_id)
        input_dir = file_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        destination = input_dir / "original.pdf"
        shutil.copy2(saved_path, destination)

        batch_files.append(
            BatchJobFile(
                job_id=batch_job_id,
                file_id=file_id,
                user=user,
                original_filename=saved_path.name,
                storage_path=str(destination),
                params=params_copy.copy(),
                checksum=checksum,
            )
        )

    batch_job = BatchJob(
        job_id=batch_job_id,
        user=user,
        params=params_copy,
        files=batch_files,
    )
    add_batch_job(batch_job)

    shutil.rmtree(job_upload_dir, ignore_errors=True)

    return {
        "batch_job_id": batch_job_id,
        "duplicate_found": bool(duplicate_files),
        "duplicate_count": len(duplicate_files),
        "duplicate_files": duplicate_files,
        "files": [
            {"file_id": batch_file.file_id, "original_filename": batch_file.original_filename}
            for batch_file in batch_files
        ],
    }

@app.post("/marker/batch_job/status")
async def batch_job_status(
    batch_job_id: Optional[str] = Form(default=None),
    batch_job_file_id: Optional[str] = Form(default=None),
):
    if not batch_job_id and not batch_job_file_id:
        raise HTTPException(
            status_code=400,
            detail="Provide either batch_job_id or batch_job_file_id to fetch status.",
        )

    if batch_job_id:
        job = get_batch_job_status(batch_job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Batch job not found.")
        return _job_status_response(job)

    file = get_batch_file_status(batch_job_file_id)
    if not file:
        raise HTTPException(status_code=404, detail="Batch job file not found.")
    return _file_status_response(file)


@app.post("/marker/batch_job/output")
async def batch_job_output(
    batch_job_id: Optional[str] = Form(default=None),
    batch_job_file_id: Optional[str] = Form(default=None),
):
    if not batch_job_id and not batch_job_file_id:
        raise HTTPException(status_code=400, detail="Provide batch_job_id or batch_job_file_id.")

    if batch_job_file_id:
        job_for_file = batch_job_id
        if not job_for_file:
            job_for_file = batch_job_file_id.split(":", 1)[0]
        if not job_for_file:
            raise HTTPException(status_code=400, detail="Unable to infer job_id from batch_job_file_id.")
        output_file = _file_processing_dir(
            job_for_file, batch_job_file_id
        ) / "output" / "output.json"
        if not output_file.exists():
            raise HTTPException(status_code=404, detail="Output not ready yet.")
        return json.loads(output_file.read_text())

    job_dir = BATCH_PROCESSING_DIR / batch_job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Batch job not found.")

    files_output = []
    for file_dir in job_dir.iterdir():
        if not file_dir.is_dir():
            continue
        output_file = file_dir / "output" / "output.json"
        if not output_file.exists():
            continue
        try:
            files_output.append(json.loads(output_file.read_text()))
        except json.JSONDecodeError:
            continue

    if not files_output:
        raise HTTPException(status_code=404, detail="No outputs available yet.")

    return {
        "job_id": batch_job_id,
        "outputs": files_output,
    }

@click.command()
@click.option("--port", type=int, default=8000, help="Port to run the server on")
@click.option("--host", type=str, default="127.0.0.1", help="Host to run the server on")
def server_cli(port: int, host: str):
    import uvicorn

    # Run the server
    uvicorn.run(
        app,
        host=host,
        port=port,
    )

