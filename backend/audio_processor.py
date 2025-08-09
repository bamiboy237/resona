"""
Audio processing module using Demucs for source separation and librosa for beat analysis
"""

import librosa
import numpy as np
import soundfile as sf
import torch
from demucs.pretrained import get_model
from demucs.apply import apply_model
import time
import os
import json
from typing import Dict, Optional, Any, List, Tuple
from pathlib import Path

class AudioProcessor:
    def __init__(self):
        """Initialize the audio processor with AI models"""
        self.demucs_model = None
        self.sample_rate = 22050  # Standard sample rate for processing
        
    def _load_demucs_model(self):
        """Lazy load Demucs model to save memory"""
        if self.demucs_model is None:
            print("Loading Demucs model...")
                # Use htdemucs which is the current best model
            self.demucs_model = get_model('htdemucs')
            print("Demucs model loaded successfully")
    
    def get_audio_info(self, file_path: str) -> Dict[str, Any]:
        """Get basic audio file information with enhanced error handling"""
        try:
            # Check if file exists and is readable
            if not os.path.exists(file_path):
                raise Exception(f"Audio file not found: {file_path}")
            
            if os.path.getsize(file_path) == 0:
                raise Exception("Audio file is empty (0 bytes)")
            
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            if file_size_mb > 100:  # Limit to 100MB
                raise Exception(f"Audio file too large: {file_size_mb:.1f}MB (max 100MB)")
            
            print(f"📂 Loading audio file: {os.path.basename(file_path)} ({file_size_mb:.1f}MB)")
            
            # Try to load audio file with librosa
            try:
                y, sr = librosa.load(file_path, sr=None)
            except Exception as load_error:
                # Provide more specific error messages
                error_msg = str(load_error).lower()
                if "format" in error_msg or "codec" in error_msg or "not supported" in error_msg:
                    raise Exception(f"Unsupported audio format or codec. Please try converting to MP3 or WAV first. Original error: {load_error}")
                elif "permission" in error_msg or "access" in error_msg:
                    raise Exception(f"Cannot access file (permission denied): {load_error}")
                elif "memory" in error_msg:
                    raise Exception(f"File too large to process in memory: {load_error}")
                elif "no such file" in error_msg:
                    raise Exception(f"File not found or corrupted: {load_error}")
                else:
                    raise Exception(f"Cannot load audio file: {load_error}")
            
            # Validate loaded audio
            if y is None or len(y) == 0:
                raise Exception("Audio file loaded but contains no audio data")
            
            # Calculate duration manually (librosa.duration was removed)
            duration = len(y) / sr
            
            if duration > 300:  # Limit to 5 minutes for processing speed
                raise Exception(f"Audio too long: {duration:.1f}s (max 5 minutes for optimal processing)")
            
            print(f"✅ Audio loaded successfully: {duration:.1f}s, {sr}Hz")
            
            return {
                "duration": duration,
                "sample_rate": sr,
                "channels": 1 if y.ndim == 1 else y.shape[0],
                "samples": len(y)
            }
        except Exception as e:
            # Re-raise with more context
            raise Exception(f"Cannot access/process audio file: {str(e)}")
    
    def analyze_beats(self, file_path: str) -> Dict[str, Any]:
        """Analyze beats and tempo of an audio file"""
        try:
            # Load audio
            y, sr = librosa.load(file_path, sr=self.sample_rate)
            
            # Beat tracking
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            
            # Onset detection
            onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')
            
            # Create time axis for visualization
            duration = len(y) / sr  # Calculate duration manually
            time_axis = np.linspace(0, duration, len(y))
            
            # Convert beat frames to time
            beat_times = librosa.frames_to_time(beats, sr=sr)
            
            return {
                "tempo": float(tempo),
                "beats": beat_times,
                "onset_times": onsets,
                "time_axis": time_axis,
                "duration": duration
            }
        except Exception as e:
            raise Exception(f"Beat analysis failed: {str(e)}")
    
    def separate_sources(self, file_path: str) -> Dict[str, np.ndarray]:
        """Separate audio into different sources using Demucs"""
        try:
            self._load_demucs_model()
            
            # Load audio for Demucs (it expects specific format)
            y, sr = librosa.load(file_path, sr=44100, mono=False)
            duration = len(y[0]) / sr if y.ndim > 1 else len(y) / sr
            
            # Ensure stereo
            if y.ndim == 1:
                y = np.stack([y, y])
            
            # Convert to torch tensor and ensure proper device
            audio_tensor = torch.from_numpy(y).float().unsqueeze(0)
            
            # Ensure tensor is on the same device as model
            device = next(self.demucs_model.parameters()).device
            audio_tensor = audio_tensor.to(device)
            
            # Apply Demucs model (this is the heavy computation)
            print("🚀 Running AI source separation...")
            print(f"📊 Processing {duration:.1f} seconds of audio")
            print(f"🖥️ Using device: {device}")
            print("⏱️ This may take 1-3 minutes on CPU (faster with GPU)")
            start_time = time.time()
            
            with torch.no_grad():
                sources = apply_model(self.demucs_model, audio_tensor)[0]
            
            processing_time = time.time() - start_time
            print(f"✅ Source separation completed in {processing_time:.1f} seconds")
            
            # Debug tensor info before conversion
            print(f"🔍 Sources tensor info:")
            print(f"  - Shape: {sources.shape}")
            print(f"  - Device: {sources.device}")
            print(f"  - Dtype: {sources.dtype}")
            print(f"  - Min: {sources.min().item():.6f}")
            print(f"  - Max: {sources.max().item():.6f}")
            print(f"  - Non-zero elements: {torch.count_nonzero(sources).item()}")
            
            # Convert back to numpy
            sources = sources.cpu().numpy()
            
            # Debug numpy info after conversion
            print(f"🔍 Numpy array info:")
            print(f"  - Shape: {sources.shape}")
            print(f"  - Dtype: {sources.dtype}")
            print(f"  - Size: {sources.size}")
            print(f"  - Min: {sources.min():.6f}")
            print(f"  - Max: {sources.max():.6f}")
            print(f"  - Non-zero elements: {np.count_nonzero(sources)}")
            
            # Validate each stem
            stem_names = ["drums", "bass", "other", "vocals"]
            for i, name in enumerate(stem_names):
                if i < len(sources):
                    stem = sources[i]
                    print(f"📊 {name} stem: shape={stem.shape}, size={stem.size}, non-zero={np.count_nonzero(stem)}")
                    if stem.size == 0:
                        raise Exception(f"Stem '{name}' is empty after conversion")
                    if np.all(stem == 0):
                        print(f"⚠️ WARNING: {name} stem is all zeros (silence)")
                else:
                    raise Exception(f"Missing stem '{name}' - only {len(sources)} stems available")
            
            # Demucs outputs: [drums, bass, other, vocals]
            return {
                "drums": sources[0],
                "bass": sources[1], 
                "other": sources[2],
                "vocals": sources[3],
                "instrumental": sources[0] + sources[1] + sources[2]  # Everything except vocals
            }
        except Exception as e:
            raise Exception(f"Source separation failed: {str(e)}")
    
    def extract_audio_segment(self, y: np.ndarray, sr: int, start_time: Optional[float] = None, 
                            end_time: Optional[float] = None) -> np.ndarray:
        """Extract a specific time segment from audio"""
        if start_time is None and end_time is None:
            return y
            
        start_sample = int(start_time * sr) if start_time else 0
        end_sample = int(end_time * sr) if end_time else len(y)
        
        # Ensure valid bounds
        start_sample = max(0, start_sample)
        end_sample = min(len(y), end_sample)
        
        if y.ndim == 1:
            return y[start_sample:end_sample]
        else:
            return y[:, start_sample:end_sample]
    
    async def process_audio(self, source_file: str, file_id: str, remove_vocals: bool = False,
                          extract_beats: bool = False, isolate_instruments: bool = False,
                          start_time: Optional[float] = None, end_time: Optional[float] = None) -> Dict[str, Any]:
        """Main audio processing function"""
        start_processing = time.time()
        processed_files = []
        beat_info = None
        
        try:
            os.makedirs("processed", exist_ok=True)
            
            # Load original audio
            y_original, sr = librosa.load(source_file, sr=None)
            
            # Extract segment if specified
            if start_time is not None or end_time is not None:
                y_original = self.extract_audio_segment(y_original, sr, start_time, end_time)
            
            # Get beat analysis if requested
            if extract_beats:
                # Create temporary file for beat analysis
                temp_file = f"processed/temp_{file_id}.wav"
                sf.write(temp_file, y_original, sr)
                beat_info = self.analyze_beats(temp_file)
                os.remove(temp_file)
            
            # Process based on requirements
            if remove_vocals or isolate_instruments:
                # Need source separation
                sources = self.separate_sources(source_file)
                
                if remove_vocals:
                    # Save instrumental version
                    instrumental = sources["instrumental"]
                    
                    # Extract segment if specified
                    if start_time is not None or end_time is not None:
                        instrumental = self.extract_audio_segment(instrumental, 44100, start_time, end_time)
                    
                    # Convert to mono and resample for consistency
                    if instrumental.ndim > 1:
                        instrumental = np.mean(instrumental, axis=0)
                    
                    output_file = f"processed/{file_id}_no_vocals.wav"
                    sf.write(output_file, instrumental, 44100)
                    processed_files.append({
                        "type": "no_vocals",
                        "filename": f"{file_id}_no_vocals.wav",
                        "description": "Vocals removed"
                    })
                
                if isolate_instruments:
                    # Save individual stems
                    stem_names = ["drums", "bass", "other", "vocals"]
                    for i, stem_name in enumerate(stem_names):
                        stem_audio = sources[stem_name]
                        
                        # Extract segment if specified
                        if start_time is not None or end_time is not None:
                            stem_audio = self.extract_audio_segment(stem_audio, 44100, start_time, end_time)
                        
                        # Convert to mono
                        if stem_audio.ndim > 1:
                            stem_audio = np.mean(stem_audio, axis=0)
                        
                        output_file = f"processed/{file_id}_{stem_name}.wav"
                        sf.write(output_file, stem_audio, 44100)
                        processed_files.append({
                            "type": stem_name,
                            "filename": f"{file_id}_{stem_name}.wav",
                            "description": f"Isolated {stem_name}"
                        })
            
            if extract_beats and not (remove_vocals or isolate_instruments):
                # Just save the beat-analyzed original
                output_file = f"processed/{file_id}_beats.wav"
                sf.write(output_file, y_original, sr)
                processed_files.append({
                    "type": "beats",
                    "filename": f"{file_id}_beats.wav",
                    "description": "Beat analysis"
                })
            
            # If no specific processing requested, save processed original
            if not processed_files:
                output_file = f"processed/{file_id}_processed.wav"
                sf.write(output_file, y_original, sr)
                processed_files.append({
                    "type": "processed",
                    "filename": f"{file_id}_processed.wav",
                    "description": "Processed audio"
                })
            
            processing_time = time.time() - start_processing
            
            return {
                "processed_files": processed_files,
                "processing_time": processing_time,
                "beat_info": beat_info
            }
            
        except Exception as e:
            raise Exception(f"Audio processing failed: {str(e)}")
    
    def create_extended_loop(self, file_path: str, target_duration: float = 600) -> str:
        """Create an extended loop from a short audio segment (for meditation)"""
        try:
            y, sr = librosa.load(file_path, sr=self.sample_rate)
            current_duration = len(y) / sr  # Calculate duration manually
            
            if current_duration >= target_duration:
                return file_path
            
            # Calculate how many repetitions needed
            repetitions = int(np.ceil(target_duration / current_duration))
            
            # Create extended version
            extended = np.tile(y, repetitions)
            
            # Trim to exact target duration
            target_samples = int(target_duration * sr)
            extended = extended[:target_samples]
            
            # Save extended version
            file_id = Path(file_path).stem
            output_file = f"processed/{file_id}_extended.wav"
            sf.write(output_file, extended, sr)
            
            return output_file
            
        except Exception as e:
            raise Exception(f"Loop creation failed: {str(e)}")
    
    # ================================
    # Live Mixer 
    # ================================
    
    def create_stems_directory(self, file_id: str) -> str:
        """Create directory structure for stems storage"""
        stems_dir = f"processed/{file_id}/stems"
        preview_dir = f"processed/{file_id}/preview"
        os.makedirs(stems_dir, exist_ok=True)
        os.makedirs(preview_dir, exist_ok=True)
        return stems_dir
    
    def separate_and_store_stems(self, source_file: str, file_id: str) -> Dict[str, Any]:
        """Separate audio into stems and store each stem as individual file"""
        try:
            print(f"🎵 Starting stem separation for {file_id}")
            start_time = time.time()
            
            # Create directory structure
            stems_dir = self.create_stems_directory(file_id)
            
            # Initialize progress tracking
            def update_progress(step: str, percent: float, eta_seconds: float = None):
                """Update progress file for real-time tracking"""
                progress_file = f"processed/{file_id}/progress.json"
                progress = {
                    "step": step,
                    "percent": percent,
                    "eta_seconds": eta_seconds,
                    "timestamp": time.time(),
                    "elapsed_time": time.time() - start_time
                }
                with open(progress_file, 'w') as f:
                    json.dump(progress, f)
                print(f"Progress: {percent:.1f}% - {step}")
            
            # Create initial progress file immediately
            update_progress("Starting audio analysis...", 0)
            
            # Get audio info for ETA calculation
            y_original, sr_original = librosa.load(source_file, sr=None)
            duration = len(y_original) / sr_original
            
            # Estimate total processing time (rough estimate: 20-30x audio duration)
            estimated_total_time = duration * 25  # seconds
            
            update_progress("Initializing audio separation...", 5, estimated_total_time)
            
            # Separate sources using existing method (this is the longest step ~80% of time)
            update_progress("Separating audio sources with AI...", 10, estimated_total_time * 0.8)
            sources = self.separate_sources(source_file)
            
            # Validate source separation results
            expected_stems = ['vocals', 'drums', 'bass', 'other']
            for stem_name in expected_stems:
                if stem_name not in sources:
                    raise Exception(f"Source separation failed: missing '{stem_name}' stem")
                if sources[stem_name] is None:
                    raise Exception(f"Source separation failed: '{stem_name}' stem is None")
                if sources[stem_name].size == 0:
                    raise Exception(f"Source separation failed: '{stem_name}' stem is empty")
            
            print(f"✅ Source separation completed successfully with {len(sources)} stems")
            
            separation_time = time.time() - start_time
            remaining_time = separation_time * 0.25  # Estimate remaining 20% based on actual separation time
            update_progress("Audio separation complete, saving stems...", 80, remaining_time)
            
            # Store each stem as individual file (preserve stereo for full quality)
            stem_files = {}
            stem_names = ['vocals', 'drums', 'bass', 'other']
            
            for i, stem_name in enumerate(stem_names):
                stem_audio = sources[stem_name]
                
                # Update progress for each stem (80% -> 95%)
                stem_progress = 80 + (i * 3.75)  # 80, 83.75, 87.5, 91.25, 95
                remaining_time_stem = remaining_time * (4 - i) / 4
                update_progress(f"Saving {stem_name} stem...", stem_progress, remaining_time_stem)
                
                # Validate stem audio before saving
                if stem_audio is None:
                    raise Exception(f"Stem '{stem_name}' is None - source separation failed")
                
                if stem_audio.size == 0:
                    raise Exception(f"Stem '{stem_name}' is empty - source separation failed")
                
                print(f"📊 {stem_name} stem shape: {stem_audio.shape}, dtype: {stem_audio.dtype}")
                
                # Prepare audio for saving
                # Ensure audio is in correct format for soundfile
                if stem_audio.ndim == 2:
                    # Transpose from (channels, samples) to (samples, channels) for soundfile
                    stem_audio_for_save = stem_audio.T
                else:
                    # Mono audio - add channel dimension
                    stem_audio_for_save = stem_audio.reshape(-1, 1)
                
                print(f"📊 {stem_name} audio for save: shape={stem_audio_for_save.shape}, dtype={stem_audio_for_save.dtype}")
                print(f"📊 {stem_name} audio range: min={stem_audio_for_save.min():.6f}, max={stem_audio_for_save.max():.6f}")
                
                # Validate audio data before saving
                if np.any(np.isnan(stem_audio_for_save)):
                    raise Exception(f"Stem '{stem_name}' contains NaN values")
                if np.any(np.isinf(stem_audio_for_save)):
                    raise Exception(f"Stem '{stem_name}' contains infinite values")
                
                # Clip audio to valid range [-1, 1]
                if np.abs(stem_audio_for_save).max() > 1.0:
                    print(f"⚠️ {stem_name} audio clipped from {np.abs(stem_audio_for_save).max():.3f} to 1.0")
                    stem_audio_for_save = np.clip(stem_audio_for_save, -1.0, 1.0)
                
                # Save full-quality stem
                stem_path = f"{stems_dir}/{stem_name}.wav"
                try:
                    sf.write(stem_path, stem_audio_for_save, 44100)
                    
                    # Verify the file was written correctly
                    if os.path.getsize(stem_path) == 0:
                        raise Exception(f"Written file is empty (0 bytes)")
                    
                    # Try to read it back to verify format
                    try:
                        test_data, test_sr = sf.read(stem_path, frames=1)
                        print(f"✅ Verified {stem_name} stem can be read back")
                    except Exception as read_error:
                        raise Exception(f"Written file cannot be read back: {read_error}")
                    
                    stem_files[stem_name] = stem_path
                    print(f"✅ Saved {stem_name} stem ({stem_audio_for_save.shape}) -> {os.path.getsize(stem_path)} bytes")
                except Exception as e:
                    raise Exception(f"Failed to save {stem_name} stem: {str(e)}")
            
            # Create preview stems (downsampled for fast mixing)
            update_progress("Creating preview stems for fast mixing...", 95, remaining_time * 0.1)
            self.create_preview_stems(file_id, stem_files)
            
            # Save metadata
            final_processing_time = time.time() - start_time
            metadata = {
                "file_id": file_id,
                "source_file": source_file,
                "duration": duration,
                "sample_rate": 44100,
                "stems": stem_files,
                "stem_names": stem_names,
                "status": "ready",
                "created_at": time.time(),
                "processing_time": final_processing_time
            }
            
            metadata_path = f"processed/{file_id}/metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Final progress update
            update_progress("✅ Complete! All stems ready for mixing", 100, 0)
            
            # Clean up progress file since processing is complete
            progress_file = f"processed/{file_id}/progress.json"
            if os.path.exists(progress_file):
                os.remove(progress_file)
            
            print(f"🎉 Stem separation completed in {final_processing_time:.1f} seconds")
            
            return metadata
            
        except Exception as e:
            # Save error metadata
            error_metadata = {
                "file_id": file_id,
                "status": "error",
                "error": str(e),
                "created_at": time.time()
            }
            
            metadata_path = f"processed/{file_id}/metadata.json"
            os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
            with open(metadata_path, 'w') as f:
                json.dump(error_metadata, f, indent=2)
            
            # Clean up progress file on error
            progress_file = f"processed/{file_id}/progress.json"
            if os.path.exists(progress_file):
                os.remove(progress_file)
            
            raise Exception(f"Stem separation failed: {str(e)}")
    
    def create_preview_stems(self, file_id: str, stem_files: Dict[str, str]):
        """Create downsampled preview versions of stems for fast mixing"""
        try:
            preview_dir = f"processed/{file_id}/preview"
            
            for stem_name, stem_path in stem_files.items():
                # Load full-quality stem
                y, sr = librosa.load(stem_path, sr=22050, mono=True)  # Downsample to 22kHz mono
                
                # Save preview version
                preview_path = f"{preview_dir}/{stem_name}.wav"
                sf.write(preview_path, y, 22050)
                
                print(f"✅ Created preview for {stem_name}")
                
        except Exception as e:
            print(f"⚠️ Preview creation failed: {str(e)}")
    
    def get_stems_metadata(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for stems processing"""
        try:
            metadata_path = f"processed/{file_id}/metadata.json"
            progress_path = f"processed/{file_id}/progress.json"
            
            # Check if processing is complete (metadata exists)
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    return json.load(f)
            
            # Check if processing is in progress (progress file exists)
            elif os.path.exists(progress_path):
                with open(progress_path, 'r') as f:
                    progress = json.load(f)
                    
                # Return a temporary metadata indicating processing status
                return {
                    "status": "processing",
                    "file_id": file_id,
                    "current_step": progress.get('step', 'Processing...'),
                    "percent": progress.get('percent', 0),
                    "eta_seconds": progress.get('eta_seconds'),
                    "elapsed_time": progress.get('elapsed_time', 0)
                }
            
            return None
        except Exception:
            return None
    
    def mix_stems_live(self, file_id: str, stem_toggles: Dict[str, bool], 
                      stem_volumes: Dict[str, float], use_preview: bool = True,
                      start_time: Optional[float] = None, duration: Optional[float] = None) -> Tuple[np.ndarray, int]:
        """Mix stems in real-time based on user toggles and volumes"""
        try:
            metadata = self.get_stems_metadata(file_id)
            if not metadata or metadata.get('status') != 'ready':
                raise Exception("Stems not ready for mixing")
            
            # Choose preview or full quality stems
            stems_dir = f"processed/{file_id}/preview" if use_preview else f"processed/{file_id}/stems"
            sample_rate = 22050 if use_preview else 44100
            
            # Load and mix stems
            mixed_audio = None
            
            for stem_name in ['vocals', 'drums', 'bass', 'other']:
                if stem_toggles.get(stem_name, False):
                    stem_path = f"{stems_dir}/{stem_name}.wav"
                    
                    if os.path.exists(stem_path):
                        # Load stem
                        y, sr = librosa.load(stem_path, sr=sample_rate, mono=True)
                        
                        # Apply time selection if specified
                        if start_time is not None or duration is not None:
                            start_sample = int(start_time * sr) if start_time else 0
                            end_sample = start_sample + int(duration * sr) if duration else len(y)
                            end_sample = min(end_sample, len(y))
                            y = y[start_sample:end_sample]
                        
                        # Apply volume
                        volume = stem_volumes.get(stem_name, 1.0)
                        y = y * volume
                        
                        # Add to mix
                        if mixed_audio is None:
                            mixed_audio = y.copy()
                        else:
                            # Ensure same length (pad shorter one)
                            if len(y) > len(mixed_audio):
                                mixed_audio = np.pad(mixed_audio, (0, len(y) - len(mixed_audio)))
                            elif len(mixed_audio) > len(y):
                                y = np.pad(y, (0, len(mixed_audio) - len(y)))
                            
                            mixed_audio += y
            
            # Return silence if no stems selected
            if mixed_audio is None:
                duration_samples = int(duration * sample_rate) if duration else int(metadata['duration'] * sample_rate)
                mixed_audio = np.zeros(duration_samples)
            
            # Normalize to prevent clipping
            max_val = np.max(np.abs(mixed_audio))
            if max_val > 0.95:
                mixed_audio = mixed_audio * (0.95 / max_val)
            
            return mixed_audio, sample_rate
            
        except Exception as e:
            raise Exception(f"Live mixing failed: {str(e)}")
    
    def export_mix(self, file_id: str, stem_toggles: Dict[str, bool], 
                  stem_volumes: Dict[str, float], output_path: str,
                  start_time: Optional[float] = None, end_time: Optional[float] = None,
                  format: str = "wav") -> str:
        """Export high-quality mix to file"""
        try:
            duration = None
            if start_time is not None and end_time is not None:
                duration = end_time - start_time
            
            # Use full-quality stems for export
            mixed_audio, sample_rate = self.mix_stems_live(
                file_id=file_id,
                stem_toggles=stem_toggles, 
                stem_volumes=stem_volumes,
                use_preview=False,  # Use full quality
                start_time=start_time,
                duration=duration
            )
            
            # Save to file
            sf.write(output_path, mixed_audio, sample_rate)
            
            return output_path
            
        except Exception as e:
            raise Exception(f"Export failed: {str(e)}")
    
    def get_stems_status(self, file_id: str) -> Dict[str, Any]:
        """Get current status of stems processing"""
        metadata = self.get_stems_metadata(file_id)
        
        if not metadata:
            return {
                "status": "not_found",
                "message": "No processing found for this file"
            }
        
        status = metadata.get('status', 'unknown')
        
        result = {
            "status": status,
            "file_id": file_id
        }
        
        if status == "ready":
            result.update({
                "duration": metadata.get('duration'),
                "stems_available": list(metadata.get('stem_names', [])),
                "processing_time": metadata.get('processing_time'),
                "preview_ready": True  # We always create previews
            })
        elif status == "processing":
            result.update({
                "current_step": metadata.get('current_step'),
                "percent": metadata.get('percent', 0),
                "eta_seconds": metadata.get('eta_seconds'),
                "elapsed_time": metadata.get('elapsed_time', 0)
            })
        elif status == "error":
            result["error"] = metadata.get('error')
        
        return result
