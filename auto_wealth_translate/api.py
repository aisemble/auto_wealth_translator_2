"""
AutoWealthTranslate API
-----------------------
API server for the AutoWealthTranslate application.
"""

import os
import tempfile
import uuid
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import time
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel

from auto_wealth_translate.core.document_processor import DocumentProcessor
from auto_wealth_translate.core.translator import TranslationService
from auto_wealth_translate.core.document_rebuilder import DocumentRebuilder
from auto_wealth_translate.core.validator import OutputValidator
from auto_wealth_translate.utils.logger import setup_logger, get_logger

# Configure logging
setup_logger(logging.INFO)
logger = get_logger("api")

# Setup FastAPI app
app = FastAPI(
    title="AutoWealthTranslate API",
    description="API for translating wealth plan reports while preserving formatting",
    version="0.1.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supported languages
SUPPORTED_LANGUAGES = {
    "en": "English",
    "zh": "Chinese",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "ja": "Japanese",
    "ko": "Korean",
    "ru": "Russian",
    "ar": "Arabic",
    "it": "Italian",
    "pt": "Portuguese"
}

# Storage for translation jobs
UPLOAD_DIR = Path(tempfile.gettempdir()) / "auto_wealth_translate_uploads"
OUTPUT_DIR = Path(tempfile.gettempdir()) / "auto_wealth_translate_outputs"
JOBS = {}  # Store job status

# Create directories if they don't exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Pydantic models
class TranslationRequest(BaseModel):
    target_lang: str
    model: str = "gpt-4"

class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: float
    created_at: str
    updated_at: str
    error: Optional[str] = None
    input_file: str
    target_lang: str
    model: str
    output_file: Optional[str] = None
    validation_score: Optional[float] = None

@app.get("/", tags=["Info"])
async def root():
    """Get API information."""
    return {
        "name": "AutoWealthTranslate API",
        "version": "0.1.0",
        "description": "API for translating wealth plan reports while preserving formatting",
        "supported_languages": SUPPORTED_LANGUAGES,
    }

@app.get("/languages", tags=["Info"])
async def get_languages():
    """Get supported languages."""
    return SUPPORTED_LANGUAGES

@app.post("/translate", tags=["Translation"])
async def translate_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    target_lang: str = Form(...),
    model: str = Form("gpt-4"),
):
    """
    Translate a document.
    
    This endpoint accepts a document file (PDF/DOCX) and translates it to the specified language.
    The translation is performed asynchronously, and a job ID is returned for tracking progress.
    
    - **file**: The document file (PDF/DOCX)
    - **target_lang**: Target language code
    - **model**: Translation model to use (default: gpt-4)
    
    Returns a job ID that can be used to check status and retrieve the translated document.
    """
    # Validate target language
    if target_lang not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Unsupported language: {target_lang}")
    
    # Validate file type
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ['.pdf', '.docx']:
        raise HTTPException(status_code=400, detail="File must be PDF or DOCX")
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Save uploaded file
    input_path = UPLOAD_DIR / f"{job_id}{file_ext}"
    
    with open(input_path, "wb") as buffer:
        buffer.write(await file.read())
    
    # Create output path
    output_path = OUTPUT_DIR / f"{job_id}_{target_lang}{file_ext}"
    
    # Create job record
    JOBS[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0.0,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "error": None,
        "input_file": file.filename,
        "target_lang": target_lang,
        "model": model,
        "output_file": None,
        "validation_score": None,
    }
    
    # Start processing in the background
    background_tasks.add_task(
        process_translation,
        job_id,
        str(input_path),
        str(output_path),
        target_lang,
        model,
    )
    
    logger.info(f"Translation job {job_id} queued for {file.filename} to {target_lang}")
    
    return {"job_id": job_id, "status": "queued"}

@app.get("/jobs/{job_id}", tags=["Jobs"])
async def get_job_status(job_id: str):
    """
    Get the status of a translation job.
    
    - **job_id**: ID of the job to check
    
    Returns the current status of the job, including progress information.
    """
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    return JOBS[job_id]

@app.get("/jobs", tags=["Jobs"])
async def list_jobs(
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    status: Optional[str] = Query(None),
):
    """
    List translation jobs.
    
    - **limit**: Maximum number of jobs to return
    - **offset**: Number of jobs to skip
    - **status**: Filter by job status
    
    Returns a list of jobs matching the criteria.
    """
    jobs = list(JOBS.values())
    
    # Apply filters
    if status:
        jobs = [job for job in jobs if job["status"] == status]
    
    # Sort by creation time (newest first)
    jobs.sort(key=lambda j: j["created_at"], reverse=True)
    
    # Apply pagination
    paginated_jobs = jobs[offset:offset+limit]
    
    return {
        "total": len(jobs),
        "limit": limit,
        "offset": offset,
        "jobs": paginated_jobs,
    }

@app.get("/download/{job_id}", tags=["Downloads"])
async def download_translated_file(job_id: str):
    """
    Download a translated document.
    
    - **job_id**: ID of the completed translation job
    
    Returns the translated document file.
    """
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    job = JOBS[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job {job_id} is not completed")
    
    if not job["output_file"] or not os.path.exists(job["output_file"]):
        raise HTTPException(status_code=404, detail=f"Output file for job {job_id} not found")
    
    return FileResponse(
        job["output_file"],
        filename=Path(job["output_file"]).name,
        media_type="application/octet-stream",
    )

@app.delete("/jobs/{job_id}", tags=["Jobs"])
async def delete_job(job_id: str):
    """
    Delete a translation job and its files.
    
    - **job_id**: ID of the job to delete
    
    Returns a confirmation of deletion.
    """
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    job = JOBS[job_id]
    
    # Delete input file
    input_path = Path(UPLOAD_DIR) / f"{job_id}{Path(job['input_file']).suffix}"
    if input_path.exists():
        input_path.unlink()
    
    # Delete output file
    if job["output_file"] and os.path.exists(job["output_file"]):
        Path(job["output_file"]).unlink()
    
    # Remove job from registry
    del JOBS[job_id]
    
    return {"message": f"Job {job_id} deleted"}

async def process_translation(
    job_id: str,
    input_path: str,
    output_path: str,
    target_lang: str,
    model: str,
):
    """
    Process a translation job.
    
    This function is run in the background to perform the actual translation.
    
    Args:
        job_id: Job ID
        input_path: Path to input file
        output_path: Path to output file
        target_lang: Target language code
        model: Translation model to use
    """
    try:
        logger.info(f"Starting translation job {job_id}")
        
        # Update job status
        JOBS[job_id]["status"] = "processing"
        JOBS[job_id]["updated_at"] = datetime.now().isoformat()
        
        # Initialize core components
        JOBS[job_id]["progress"] = 0.1
        JOBS[job_id]["updated_at"] = datetime.now().isoformat()
        
        doc_processor = DocumentProcessor(input_path)
        translation_service = TranslationService(target_lang=target_lang, model=model)
        doc_rebuilder = DocumentRebuilder()
        validator = OutputValidator()
        
        # Process document
        logger.info(f"Job {job_id}: Extracting document components")
        JOBS[job_id]["progress"] = 0.2
        JOBS[job_id]["updated_at"] = datetime.now().isoformat()
        
        doc_components = doc_processor.process()
        
        # Translate components
        logger.info(f"Job {job_id}: Translating document components")
        JOBS[job_id]["progress"] = 0.4
        JOBS[job_id]["updated_at"] = datetime.now().isoformat()
        
        translated_components = translation_service.translate(doc_components)
        
        # Rebuild document
        logger.info(f"Job {job_id}: Rebuilding document with translated content")
        JOBS[job_id]["progress"] = 0.7
        JOBS[job_id]["updated_at"] = datetime.now().isoformat()
        
        rebuilt_doc = doc_rebuilder.rebuild(
            translated_components, 
            output_format=Path(input_path).suffix[1:]
        )
        
        # Validate output
        logger.info(f"Job {job_id}: Validating translation")
        JOBS[job_id]["progress"] = 0.9
        JOBS[job_id]["updated_at"] = datetime.now().isoformat()
        
        validation_result = validator.validate(doc_components, rebuilt_doc)
        
        # Save output
        logger.info(f"Job {job_id}: Saving output to {output_path}")
        rebuilt_doc.save(output_path)
        
        # Update job status
        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["progress"] = 1.0
        JOBS[job_id]["updated_at"] = datetime.now().isoformat()
        JOBS[job_id]["output_file"] = output_path
        JOBS[job_id]["validation_score"] = validation_result["score"]
        
        logger.info(f"Job {job_id} completed successfully. Validation score: {validation_result['score']:.2f}/10")
        
    except Exception as e:
        logger.error(f"Error processing job {job_id}: {str(e)}", exc_info=True)
        
        # Update job status
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["updated_at"] = datetime.now().isoformat()
        JOBS[job_id]["error"] = str(e)

def start():
    """Start the API server."""
    uvicorn.run("auto_wealth_translate.api:app", host="0.0.0.0", port=8000)

if __name__ == "__main__":
    start()
