"""
Resona Backend - FastAPI application for AI-powered audio processing
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import uuid
import shutil
import json
from typing import Optional, List, Dict
from pathlib import Path

from .audio_processor import AudioProcessor
from .youtube_downloader import YouTubeDownloader

app = FastAPI(title="Resona API", description="AI-powered audio processing API", version="1.0.0")

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ],  # Streamlit default ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize processors
audio_processor = AudioProcessor()
youtube_downloader = YouTubeDownloader()

# Ensure directories exist
os.makedirs("downloads", exist_ok=True)
os.makedirs("uploads", exist_ok=True)
os.makedirs("processed", exist_ok=True)

class YouTubeRequest(BaseModel):
    url: str

class ProcessingRequest(BaseModel):
    file_id: str
    remove_vocals: bool = False
    extract_beats: bool = False
    isolate_instruments: bool = False
    start_time: Optional[float] = None
    end_time: Optional[float] = None

# live mixer
class StemsInitiateRequest(BaseModel):
    file_id: str

class MixPreviewRequest(BaseModel):
    file_id: str
    stem_toggles: Dict[str, bool]
    stem_volumes: Dict[str, float] = {"vocals": 1.0, "drums": 1.0, "bass": 1.0, "other": 1.0}
    start_time: Optional[float] = None
    duration: Optional[float] = None

class MixExportRequest(BaseModel):
    file_id: str
    stem_toggles: Dict[str, bool]
    stem_volumes: Dict[str, float] = {"vocals": 1.0, "drums": 1.0, "bass": 1.0, "other": 1.0}
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    format: str = "wav"

@app.get("/")
async def root():
    return {"message": "Resona API is running"}

@app.post("/process-audio")
async def process_audio(request: ProcessingRequest, background_tasks: BackgroundTasks):
    """Process audio with AI models"""
    try:
        # Find the source file
        source_file = None
        for ext in ['.mp3', '.wav', '.flac', '.m4a', '.ogg']:
            upload_path = f"uploads/{request.file_id}{ext}"
            download_path = f"downloads/{request.file_id}{ext}"
            
            if os.path.exists(upload_path):
                source_file = upload_path
                break
            elif os.path.exists(download_path):
                source_file = download_path
                break
        
        if not source_file:
            raise HTTPException(status_code=404, detail="Source file not found")
        
        # Process the audio
        result = await audio_processor.process_audio(
            source_file=source_file,
            file_id=request.file_id,
            remove_vocals=request.remove_vocals,
            extract_beats=request.extract_beats,
            isolate_instruments=request.isolate_instruments,
            start_time=request.start_time,
            end_time=request.end_time
        )
        
        return {
            "success": True,
            "processed_files": result["processed_files"],
            "processing_time": result["processing_time"],
            "beat_info": result.get("beat_info")
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.get("/beat-analysis/{file_id}")
async def get_beat_analysis(file_id: str):
    """Get beat analysis data for visualization"""
    try:
        # Find the source file
        source_file = None
        for ext in ['.mp3', '.wav', '.flac', '.m4a', '.ogg']:
            upload_path = f"uploads/{file_id}{ext}"
            download_path = f"downloads/{file_id}{ext}"
            
            if os.path.exists(upload_path):
                source_file = upload_path
                break
            elif os.path.exists(download_path):
                source_file = download_path
                break
        
        if not source_file:
            raise HTTPException(status_code=404, detail="Source file not found")
        
        beat_data = audio_processor.analyze_beats(source_file)

        # Downsample beats/onsets for lightweight payload
        beats = beat_data["beats"].tolist()
        onsets = beat_data["onset_times"].tolist()
        if len(beats) > 200:
            step = max(1, len(beats) // 200)
            beats = beats[::step]
        if len(onsets) > 200:
            step_o = max(1, len(onsets) // 200)
            onsets = onsets[::step_o]

        return {
            "success": True,
            "tempo": beat_data["tempo"],
            "beats": beats,
            "onset_times": onsets,
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Beat analysis failed: {str(e)}")

@app.delete("/cleanup/{file_id}")
async def cleanup_files(file_id: str):
    """Clean up temporary files"""
    try:
        # Remove files from all directories
        directories = ["uploads", "downloads"]
        cleaned = []
        
        for directory in directories:
            for ext in ['.mp3', '.wav', '.flac', '.m4a', '.ogg']:
                file_path = f"{directory}/{file_id}{ext}"
                if os.path.exists(file_path):
                    os.remove(file_path)
                    cleaned.append(file_path)

        # Remove processed directory for this file_id (stems, previews, metadata)
        processed_dir = f"processed/{file_id}"
        if os.path.isdir(processed_dir):
            import shutil as _shutil
            _shutil.rmtree(processed_dir, ignore_errors=True)
            cleaned.append(processed_dir)
        
        return {
            "success": True,
            "cleaned_files": cleaned
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")

# ================================
# Live Mixer API Endpoints
# ================================

@app.post("/stems/initiate")
async def initiate_stems_processing(request: StemsInitiateRequest, background_tasks: BackgroundTasks):
    """Initiate stem separation processing for a file"""
    try:
        # Check if stems already exist
        status = audio_processor.get_stems_status(request.file_id)
        if status["status"] == "ready":
            return {
                "success": True,
                "message": "Stems already available",
                "status": "ready",
                **status
            }
        
        # Find the source file
        source_file = None
        for ext in ['.mp3', '.wav', '.flac', '.m4a', '.ogg']:
            upload_path = f"uploads/{request.file_id}{ext}"
            download_path = f"downloads/{request.file_id}{ext}"
            
            if os.path.exists(upload_path):
                source_file = upload_path
                break
            elif os.path.exists(download_path):
                source_file = download_path
                break
        
        if not source_file:
            raise HTTPException(status_code=404, detail="Source file not found")
        
        # Start background processing
        background_tasks.add_task(
            audio_processor.separate_and_store_stems,
            source_file,
            request.file_id
        )
        
        return {
            "success": True,
            "message": "Stem separation started",
            "file_id": request.file_id,
            "status": "processing"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start processing: {str(e)}")

@app.get("/stems/status/{file_id}")
async def get_stems_status(file_id: str):
    """Get the current status of stem processing"""
    try:
        status = audio_processor.get_stems_status(file_id)
        return {
            "success": True,
            **status
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")

@app.get("/stems/progress/{file_id}")
async def get_stems_progress(file_id: str):
    """Get real-time progress of stem processing with ETA"""
    try:
        progress_file = f"processed/{file_id}/progress.json"
        if os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                progress = json.load(f)
            return progress
        else:
            # Return default progress if file doesn't exist yet
            return {
                "step": "Initializing...",
                "percent": 0,
                "eta_seconds": None,
                "elapsed_time": 0,
                "timestamp": None
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get progress: {str(e)}")

@app.post("/mix/preview")
async def preview_mix(request: MixPreviewRequest):
    """Generate a preview mix from stems (fast, low quality)"""
    try:
        # Generate preview mix
        mixed_audio, sample_rate = audio_processor.mix_stems_live(
            file_id=request.file_id,
            stem_toggles=request.stem_toggles,
            stem_volumes=request.stem_volumes,
            use_preview=True,  # Use fast preview stems
            start_time=request.start_time,
            duration=request.duration
        )
        
        # Save temporary preview file
        preview_id = str(uuid.uuid4())
        preview_path = f"processed/preview_{preview_id}.wav"
        
        import soundfile as sf
        sf.write(preview_path, mixed_audio, sample_rate)
        
        return {
            "success": True,
            "preview_id": preview_id,
            "message": "Preview generated",
            "download_url": f"/download/preview/{preview_id}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preview generation failed: {str(e)}")

@app.post("/mix/export")
async def export_mix(request: MixExportRequest, background_tasks: BackgroundTasks):
    """Export a high-quality mix from stems"""
    try:
        export_id = str(uuid.uuid4())
        output_path = f"processed/export_{export_id}.{request.format}"
        
        # Start background export
        background_tasks.add_task(
            audio_processor.export_mix,
            request.file_id,
            request.stem_toggles,
            request.stem_volumes,
            output_path,
            request.start_time,
            request.end_time,
            request.format
        )
        
        return {
            "success": True,
            "export_id": export_id,
            "message": "Export started",
            "status": "processing"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

@app.get("/download/preview/{preview_id}")
async def download_preview(preview_id: str):
    """Download a preview mix"""
    file_path = f"processed/preview_{preview_id}.wav"
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Preview file not found")
    
    return FileResponse(
        path=file_path,
        media_type="audio/wav",
        filename=f"resona_preview_{preview_id}.wav"
    )

@app.get("/download/export/{export_id}")
async def download_export(export_id: str):
    """Download an exported mix"""
    # Check for both wav and mp3 formats
    for ext in ['wav', 'mp3']:
        file_path = f"processed/export_{export_id}.{ext}"
        if os.path.exists(file_path):
            return FileResponse(
                path=file_path,
                media_type=f"audio/{ext}",
                filename=f"resona_export_{export_id}.{ext}"
            )
    
    raise HTTPException(status_code=404, detail="Export file not found")

# Legacy endpoint - updated to trigger stems processing automatically
@app.post("/upload-audio")
async def upload_audio(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """Upload audio file and automatically start stem processing"""
    try:
        # Validate file type
        allowed_extensions = {'.mp3', '.wav', '.flac', '.m4a', '.ogg'}
        file_extension = Path(file.filename).suffix.lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        file_id = str(uuid.uuid4())
        upload_path = f"uploads/{file_id}{file_extension}"
        
        # Save uploaded file
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Get basic audio info
        info = audio_processor.get_audio_info(upload_path)
        
        # Auto-start stem processing
        if background_tasks:
            background_tasks.add_task(
                audio_processor.separate_and_store_stems,
                upload_path,
                file_id
            )
        
        return {
            "success": True,
            "file_id": file_id,
            "filename": file.filename,
            "duration": info["duration"],
            "sample_rate": info["sample_rate"],
            "processing_started": True
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Upload failed: {str(e)}")

# Legacy endpoint - updated to trigger stems processing automatically  
@app.post("/download-youtube")
async def download_youtube(request: YouTubeRequest, background_tasks: BackgroundTasks):
    """Download audio from YouTube URL and automatically start stem processing"""
    try:
        file_id = str(uuid.uuid4())
        result = await youtube_downloader.download_audio(request.url, file_id)
        
        # Auto-start stem processing
        download_path = f"downloads/{file_id}.mp3"
        background_tasks.add_task(
            audio_processor.separate_and_store_stems,
            download_path,
            file_id
        )
        
        return {
            "success": True,
            "file_id": file_id,
            "filename": result["filename"],
            "title": result["title"],
            "duration": result["duration"],
            "processing_started": True
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Download failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
