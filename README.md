# Resona - AI Beat Extractor

A Python-based AI-powered audio source separation and beat extraction application built with FastAPI and Streamlit.

## Features

- **AI-Powered Source Separation**: Uses Demucs v4 for high-quality audio stem extraction (vocals, drums, bass, other)
- **Beat Analysis**: Automatic tempo detection and beat tracking using librosa
- **YouTube Integration**: Direct audio extraction from YouTube links
- **Live Stem Mixing**: Process-once, mix-instantly workflow with real-time stem toggles and volume controls
- **Multi-Format Export**: Support for various audio formats with high-quality output
- **Real-Time Processing**: Live progress tracking with ETA and status updates

## Architecture

- **Backend**: FastAPI REST API with asynchronous processing
- **Frontend**: Streamlit web interface with interactive audio controls
- **Audio Processing**: Demucs v4 for source separation, librosa for beat analysis
- **File Management**: Organized directory structure for uploads, downloads, and processed files

## Requirements

- Python 3.9+
- FFmpeg (installed via Homebrew on macOS)
- Intel Mac (x86_64) compatible
- 8GB+ RAM recommended for audio processing

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd resona
```

2. Install FFmpeg (macOS):
```bash
brew install ffmpeg
```

3. Install Python dependencies:
```bash
uv sync
```

## Usage

1. Start the application:
```bash
uv run python run_resona.py
```

2. Access the web interface:
   - Frontend: http://localhost:8501
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

3. Upload audio or provide YouTube link to begin processing

4. Use the live stem mixer to adjust volumes and toggle stems in real-time

5. Export your final mix in high quality

## Project Structure

```
resona/
├── backend/           # FastAPI backend API
├── frontend/          # Streamlit frontend interface
├── shared/            # Shared utilities and models
├── downloads/         # Downloaded YouTube audio
├── uploads/           # User uploaded audio files
├── processed/         # Processed stems and metadata
├── pyproject.toml     # Project configuration
└── run_resona.py      # Application startup script
```

## API Endpoints

- `POST /upload-audio` - Upload and process audio files
- `POST /download-youtube` - Download and process YouTube audio
- `POST /stems/initiate` - Start stem separation process
- `GET /stems/status/{file_id}` - Get processing status
- `GET /stems/progress/{file_id}` - Get real-time progress
- `POST /mix/preview` - Generate preview mix
- `POST /mix/export` - Export high-quality mix
- `GET /download/preview/{preview_id}` - Download preview file
- `GET /download/export/{export_id}` - Download exported file

## Technical Details

- **Audio Processing**: Demucs v4 with PyTorch backend
- **Beat Detection**: librosa 0.11.0 for tempo and beat analysis
- **Audio Formats**: Supports WAV, MP3, M4A, and other common formats
- **Processing**: Background task processing with real-time progress updates
- **Caching**: Processed stems are cached for instant mixing

## Development

The application uses `uv` for dependency management and virtual environments. All dependencies are specified in `pyproject.toml` and locked in `uv.lock`.

## License

[Add your license information here]
