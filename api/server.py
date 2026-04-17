"""FastAPI backend for the GitHub Pages frontend."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from agents.analyst_agent import AnalystRAGAgent
from agents.eda_agent import EDAAgent
from agents.presentation_agent import PresentationGeneratorAgent


BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs"
API_UPLOAD_DIR = OUTPUT_DIR / "api_uploads"
API_PRESENTATION_DIR = OUTPUT_DIR / "api_presentations"
API_CACHE_DIR = OUTPUT_DIR / "api_cache"
API_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
API_PRESENTATION_DIR.mkdir(parents=True, exist_ok=True)
API_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _allowed_origins() -> list[str]:
    configured = os.getenv("ALLOWED_ORIGINS", "*")
    if configured.strip() == "*":
        return ["*"]
    return [item.strip() for item in configured.split(",") if item.strip()]


app = FastAPI(title="ISE547 Analytics API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _dataset_path(dataset_id: str) -> Path | None:
    matches = sorted(API_UPLOAD_DIR.glob(f"{dataset_id}.*"))
    return matches[0] if matches else None


def _save_upload_bytes(content: bytes, filename: str, dataset_id: str | None = None) -> Path:
    suffix = Path(filename or "dataset.csv").suffix or ".csv"
    file_id = dataset_id or _bytes_hash(content)
    destination = API_UPLOAD_DIR / f"{file_id}{suffix}"
    if not destination.exists():
        destination.write_bytes(content)
    return destination


def _validate_csv(upload_path: Path) -> pd.DataFrame:
    raw = upload_path.read_bytes()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    signatures = {
        b"PK\x03\x04": "Excel workbook (.xlsx)",
        b"PAR1": "Parquet file",
        b"\xd0\xcf\x11\xe0": "legacy Excel workbook (.xls)",
    }
    for signature, label in signatures.items():
        if raw.startswith(signature):
            raise HTTPException(status_code=400, detail=f"Uploaded file appears to be a {label}, not a plain CSV.")
    if raw[:2048].count(b"\x00") / max(min(len(raw), 2048), 1) > 0.05:
        raise HTTPException(status_code=400, detail="Uploaded file looks binary, not plain CSV text.")

    encodings = ["utf-8", "utf-8-sig", "utf-16", "utf-16le", "utf-16be", "latin-1"]
    try:
        last_error: Exception | None = None
        for encoding in encodings:
            try:
                return pd.read_csv(upload_path, encoding=encoding)
            except Exception as exc:
                last_error = exc
        raise last_error or ValueError("Unknown CSV parse failure")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV file. Please export it as UTF-8 CSV and try again. Details: {exc}") from exc


def _bytes_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()[:16]


def _cache_path(dataset_id: str, key: str) -> Path:
    return API_CACHE_DIR / f"{dataset_id}_{key}.json"


def _read_cache(dataset_id: str, key: str) -> dict[str, Any] | None:
    path = _cache_path(dataset_id, key)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _write_cache(dataset_id: str, key: str, payload: dict[str, Any]) -> None:
    _cache_path(dataset_id, key).write_text(json.dumps(payload, indent=2, default=str))


def _resolve_upload(dataset_id: str | None, file: UploadFile | None) -> tuple[str, Path, str]:
    if dataset_id:
        upload_path = _dataset_path(dataset_id)
        if upload_path is None:
            raise HTTPException(status_code=404, detail="Dataset session not found. Please upload the CSV again.")
        return dataset_id, upload_path, upload_path.name

    if file is None:
        raise HTTPException(status_code=400, detail="A dataset_id or file upload is required.")

    content = file.file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    resolved_dataset_id = _bytes_hash(content)
    upload_path = _save_upload_bytes(content, file.filename or "dataset.csv", dataset_id=resolved_dataset_id)
    return resolved_dataset_id, upload_path, file.filename or upload_path.name


def _summary_payload(dataset_name: str, dataset: pd.DataFrame, summary: dict[str, Any], dataset_id: str) -> dict[str, Any]:
    schema = summary["schema"]
    summary_schema = {
        "target": schema.get("target"),
        "time": schema.get("time"),
        "customer": schema.get("primary_dimension"),
        "channel": schema.get("secondary_dimension"),
        "engagement": schema.get("primary_metric"),
        "friction": schema.get("secondary_metric"),
        "primary_dimension": schema.get("primary_dimension"),
        "secondary_dimension": schema.get("secondary_dimension"),
        "primary_metric": schema.get("primary_metric"),
        "secondary_metric": schema.get("secondary_metric"),
    }
    return {
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "rows": summary["rows"],
        "columns": summary["columns"],
        "analysis_rows": summary["analysis_rows"],
        "analysis_grain": summary["analysis_grain"],
        "source_format": summary["source_format"],
        "target_rate": summary["target_rate"],
        "top_time": summary["top_time"],
        "top_customer": summary["top_primary_dimension_value"],
        "top_channel": summary["top_secondary_dimension_value"],
        "schema": summary_schema,
        "preview": dataset.head(12).fillna("").to_dict(orient="records"),
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/datasets")
async def register_dataset(file: UploadFile = File(...)) -> dict[str, Any]:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    dataset_id = _bytes_hash(content)
    upload_path = _save_upload_bytes(content, file.filename or "dataset.csv", dataset_id=dataset_id)
    dataset = _validate_csv(upload_path)
    return {
        "dataset_id": dataset_id,
        "dataset_name": file.filename or upload_path.name,
        "rows": int(len(dataset)),
        "columns": int(len(dataset.columns)),
    }


@app.post("/api/summary")
async def summarize_dataset(
    dataset_id: str | None = Form(None),
    file: UploadFile | None = File(None),
) -> dict[str, Any]:
    resolved_dataset_id, upload_path, dataset_name = _resolve_upload(dataset_id, file)
    cached = _read_cache(resolved_dataset_id, "summary")
    if cached is not None:
        return cached
    dataset = _validate_csv(upload_path)
    presenter = PresentationGeneratorAgent(output_dir=OUTPUT_DIR)
    summary = presenter.summarize_dataset(upload_path)
    payload = _summary_payload(dataset_name, dataset, summary, resolved_dataset_id)
    _write_cache(resolved_dataset_id, "summary", payload)
    return payload


@app.post("/api/analyst")
async def analyst_answer(
    dataset_id: str | None = Form(None),
    file: UploadFile | None = File(None),
    question: str = Form(...),
    model: str = Form("meta-llama/llama-3.1-8b-instruct"),
    prompt_style: str = Form("structured_json"),
    rag_enabled: bool = Form(True),
    top_k: int = Form(5),
) -> dict:
    resolved_dataset_id, upload_path, _ = _resolve_upload(dataset_id, file)
    _validate_csv(upload_path)
    agent = AnalystRAGAgent(dataset_path=upload_path)
    result = agent.answer_question(
        question=question,
        model=model,
        prompt_style=prompt_style,
        rag_enabled=rag_enabled,
        top_k=top_k,
    )
    return {
        "dataset_id": resolved_dataset_id,
        "question": result.question,
        "model": result.model,
        "prompt_style": result.prompt_style,
        "rag_enabled": result.rag_enabled,
        "parsed_response": result.parsed_response,
        "raw_response": result.raw_response,
        "retrieved_context": result.retrieved_context,
    }


@app.post("/api/eda")
async def eda_report(
    dataset_id: str | None = Form(None),
    file: UploadFile | None = File(None),
) -> dict[str, Any]:
    resolved_dataset_id, upload_path, dataset_name = _resolve_upload(dataset_id, file)
    _validate_csv(upload_path)
    cached = _read_cache(resolved_dataset_id, "eda")
    if cached is not None:
        return cached
    report_id = resolved_dataset_id
    eda_agent = EDAAgent(output_dir=OUTPUT_DIR)
    report = eda_agent.analyze_dataset(upload_path, include_charts=True, chart_prefix=report_id, cache_key=report_id)
    payload = {
        "dataset_id": resolved_dataset_id,
        "dataset_name": dataset_name,
        "profile": report["profile"],
        "quality_checks": report["quality_checks"],
        "chart_manifest": [
            {
                **chart,
                "chart_url": f"/api/downloads/{chart['filename']}",
            }
            for chart in report["chart_manifest"]
        ],
        "suggested_questions": report["suggested_questions"],
        "handoff_summary": report["handoff_summary"],
        "retrieval_chunks": report["retrieval_chunks"],
        "key_findings": report["key_findings"],
        "preview": report["preview"],
    }
    _write_cache(resolved_dataset_id, "eda", payload)
    return payload


@app.post("/api/presentation")
async def generate_presentation(
    dataset_id: str | None = Form(None),
    file: UploadFile | None = File(None),
) -> dict[str, Any]:
    resolved_dataset_id, upload_path, _ = _resolve_upload(dataset_id, file)
    _validate_csv(upload_path)
    cached = _read_cache(resolved_dataset_id, "presentation")
    if cached is not None:
        return cached
    presenter = PresentationGeneratorAgent(output_dir=OUTPUT_DIR)
    output_filename = f"{resolved_dataset_id}_presentation.pptx"
    output_path = API_PRESENTATION_DIR / output_filename
    ppt_path, slides, chart_paths = presenter.create_presentation(upload_path, output_path, chart_prefix=resolved_dataset_id)
    payload = {
        "dataset_id": resolved_dataset_id,
        "download_url": f"/api/downloads/{ppt_path.name}",
        "slides": [
            {
                "title": slide.title,
                "bullets": slide.bullets,
                "speaker_notes": slide.speaker_notes,
                "chart_key": slide.chart_key,
                "chart_url": f"/api/downloads/{chart_paths[slide.chart_key].name}" if slide.chart_key and slide.chart_key in chart_paths else None,
            }
            for slide in slides
        ],
    }
    _write_cache(resolved_dataset_id, "presentation", payload)
    return payload


@app.get("/api/downloads/{filename}")
def download_artifact(filename: str):
    for directory in [API_PRESENTATION_DIR, OUTPUT_DIR / "charts"]:
        candidate = directory / filename
        if candidate.exists():
            media_type = None
            if candidate.suffix.lower() == ".pptx":
                media_type = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
            elif candidate.suffix.lower() == ".png":
                media_type = "image/png"
            return FileResponse(candidate, media_type=media_type, filename=candidate.name)
    raise HTTPException(status_code=404, detail="Artifact not found")
