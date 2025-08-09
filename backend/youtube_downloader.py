"""
YouTube audio downloader using yt-dlp
"""

import yt_dlp
import os
import asyncio
from typing import Dict, Any
from pathlib import Path

class YouTubeDownloader:
    def __init__(self):
        """Initialize YouTube downloader"""
        self.download_dir = "downloads"
        os.makedirs(self.download_dir, exist_ok=True)
    
    async def download_audio(self, url: str, file_id: str) -> Dict[str, Any]:
        """Download audio from YouTube URL"""
        try:
            output_template = os.path.join(self.download_dir, f"{file_id}.%(ext)s")
            
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': output_template,
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'extractaudio': True,
                'audioformat': 'mp3',
                'noplaylist': True,
                'ignoreerrors': True,
                'no_warnings': False,
                'extractflat': False,
            }
            
            # Download in a separate thread to avoid blocking
            loop = asyncio.get_event_loop()
            info = await loop.run_in_executor(None, self._download_with_ytdl, url, ydl_opts)
            
            # Find the downloaded file
            downloaded_file = None
            for ext in ['.mp3', '.m4a', '.webm', '.wav']:
                potential_file = os.path.join(self.download_dir, f"{file_id}{ext}")
                if os.path.exists(potential_file):
                    downloaded_file = potential_file
                    break
            
            if not downloaded_file:
                raise Exception("Downloaded file not found")
            
            return {
                "filename": os.path.basename(downloaded_file),
                "file_path": downloaded_file,
                "title": info.get("title", "Unknown"),
                "duration": info.get("duration", 0),
                "uploader": info.get("uploader", "Unknown"),
                "view_count": info.get("view_count", 0)
            }
            
        except Exception as e:
            raise Exception(f"YouTube download failed: {str(e)}")
    
    def _download_with_ytdl(self, url: str, ydl_opts: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous download function for executor"""
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Extract info first
            info = ydl.extract_info(url, download=False)
            
            # Download the audio
            ydl.download([url])
            
            return info
    
    def get_video_info(self, url: str) -> Dict[str, Any]:
        """Get video information without downloading"""
        try:
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                
                return {
                    "title": info.get("title", "Unknown"),
                    "duration": info.get("duration", 0),
                    "uploader": info.get("uploader", "Unknown"),
                    "view_count": info.get("view_count", 0),
                    "description": info.get("description", ""),
                    "thumbnail": info.get("thumbnail", ""),
                    "upload_date": info.get("upload_date", ""),
                }
                
        except Exception as e:
            raise Exception(f"Failed to get video info: {str(e)}")
    
    def validate_url(self, url: str) -> bool:
        """Validate if URL is a valid YouTube URL"""
        try:
            # Basic validation
            youtube_domains = [
                'youtube.com', 'youtu.be', 'www.youtube.com', 
                'm.youtube.com', 'music.youtube.com'
            ]
            
            return any(domain in url for domain in youtube_domains)
            
        except Exception:
            return False
