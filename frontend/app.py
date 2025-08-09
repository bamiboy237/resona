"""
Resona Frontend - Live Stem Mixer Interface
Process once, mix forever!
"""

import streamlit as st
import requests
import time
import os
import plotly.graph_objects as go
import numpy as np
from typing import Optional, Dict, Any
import io

# Configure Streamlit page
st.set_page_config(
    page_title="Resona - Live Stem Mixer",
    page_icon="🎛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Backend API base URL
API_BASE_URL = "http://localhost:8000"

# Custom CSS for mixer interface
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1e40af;
        text-align: center;
        margin-bottom: 1rem;
    }
    .mixer-section {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        padding: 2rem;
        border-radius: 1rem;
        border: 2px solid #e2e8f0;
        margin: 1rem 0;
    }
    .stem-control {
        background: white;
        padding: 1.5rem;
        border-radius: 0.75rem;
        border: 1px solid #e2e8f0;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .processing-card {
        background: #dbeafe;
        border: 2px solid #3b82f6;
        border-radius: 1rem;
        padding: 2rem;
        text-align: center;
    }
    .ready-card {
        background: #d1fae5;
        border: 2px solid #10b981;
        border-radius: 1rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .upload-section {
        background: #f9fafb;
        border: 2px dashed #d1d5db;
        border-radius: 1rem;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state for the mixer"""
    if 'current_file_id' not in st.session_state:
        st.session_state.current_file_id = None
    if 'stems_status' not in st.session_state:
        st.session_state.stems_status = None
    if 'audio_info' not in st.session_state:
        st.session_state.audio_info = None
    if 'stem_toggles' not in st.session_state:
        st.session_state.stem_toggles = {
            'vocals': True,
            'drums': True, 
            'bass': True,
            'other': True
        }
    if 'stem_volumes' not in st.session_state:
        st.session_state.stem_volumes = {
            'vocals': 1.0,
            'drums': 1.0,
            'bass': 1.0,
            'other': 1.0
        }

def show_main_header():
    """Display the main application header"""
    st.markdown('<h1 class="main-header">🎛️ Resona Live Mixer</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem; color: #6b7280;">
        Process once, mix forever - Your audio stems at your fingertips
    </div>
    """, unsafe_allow_html=True)

def upload_section():
    """Audio upload section with auto-processing"""
    if st.session_state.current_file_id:
        return  # Skip if already have a file

    st.markdown("## 📁 Upload Your Audio")

    tab1, tab2 = st.tabs(["📁 Upload File", "🔗 YouTube"])

    with tab1:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['mp3', 'wav', 'flac', 'm4a', 'ogg'],
            help="Upload any audio file - stems will be automatically processed"
    )
    
    if uploaded_file is not None:
        if st.button("🚀 Upload & Process", type="primary", use_container_width=True):
            with st.spinner("Uploading and starting stem processing..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post(f"{API_BASE_URL}/upload-audio", files=files)
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.session_state.current_file_id = result["file_id"]
                        st.session_state.audio_info = result
                        st.success("✅ Upload successful! Processing stems...")
                        st.rerun()
                    else:
                        st.error(f"❌ Upload failed: {response.json().get('detail', 'Unknown error')}")
                        
                except Exception as e:
                    st.error(f"❌ Upload error: {str(e)}")

    st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        youtube_url = st.text_input(
            "YouTube URL",
            placeholder="https://www.youtube.com/watch?v=...",
            help="Paste any YouTube URL - audio will be downloaded and stems processed automatically"
        )
        
        if youtube_url and st.button("🔗 Download & Process", type="primary", use_container_width=True):
            with st.spinner("Downloading from YouTube and starting stem processing..."):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/download-youtube",
                        json={"url": youtube_url}
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.session_state.current_file_id = result["file_id"]
                        st.session_state.audio_info = result
                        st.success(f"✅ Downloaded: {result['title']}")
                        st.rerun()
                    else:
                        st.error(f"❌ Download failed: {response.json().get('detail', 'Unknown error')}")
                        
                except Exception as e:
                    st.error(f"❌ Download error: {str(e)}")

        st.markdown('</div>', unsafe_allow_html=True)

def check_stems_status():
    """Check and update stems processing status"""
    if not st.session_state.current_file_id:
        return
    
    try:
        response = requests.get(f"{API_BASE_URL}/stems/status/{st.session_state.current_file_id}")
        if response.status_code == 200:
            st.session_state.stems_status = response.json()
        else:
            st.session_state.stems_status = {"status": "error", "error": "Failed to get status"}
    except Exception as e:
        st.session_state.stems_status = {"status": "error", "error": str(e)}

def show_processing_status():
    """Show real-time processing status with exact progress and ETA"""
    if not st.session_state.stems_status:
        return
    
    status = st.session_state.stems_status.get('status', 'unknown')
    
    if status == "processing":
        st.markdown('<div class="processing-card">', unsafe_allow_html=True)
        st.markdown("### 🔄 Processing Stems...")
        st.markdown(f"**Track**: {st.session_state.audio_info.get('title', st.session_state.audio_info.get('filename', 'Unknown'))}")
        
        # Real-time progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        eta_text = st.empty()
        elapsed_text = st.empty()
        
        # Poll for real progress
        try:
            response = requests.get(f"{API_BASE_URL}/stems/progress/{st.session_state.current_file_id}")
            if response.status_code == 200:
                progress = response.json()
                
                # Update progress bar
                progress_percent = min(progress.get('percent', 0), 100)
                progress_bar.progress(progress_percent / 100)
                
                # Update status text
                current_step = progress.get('step', 'Processing...')
                status_text.markdown(f"**{current_step}** ({progress_percent:.1f}%)")
                
                # Update ETA if available
                eta_seconds = progress.get('eta_seconds')
                if eta_seconds and eta_seconds > 0:
                    mins, secs = divmod(int(eta_seconds), 60)
                    if mins > 0:
                        eta_text.markdown(f"⏱️ **Estimated time remaining: {mins}m {secs}s**")
                    else:
                        eta_text.markdown(f"⏱️ **Estimated time remaining: {secs}s**")
                else:
                    eta_text.markdown("⏱️ **Calculating time remaining...**")
                
                # Show elapsed time
                elapsed_time = progress.get('elapsed_time', 0)
                elapsed_mins, elapsed_secs = divmod(int(elapsed_time), 60)
                if elapsed_mins > 0:
                    elapsed_text.markdown(f"🕐 **Elapsed: {elapsed_mins}m {elapsed_secs}s**")
                else:
                    elapsed_text.markdown(f"🕐 **Elapsed: {elapsed_secs}s**")
                
                # Check if complete
                if progress_percent >= 100:
                    check_stems_status()  # Update main status
                    if st.session_state.stems_status.get('status') == 'ready':
                        st.markdown('</div>', unsafe_allow_html=True)
                        return
                        
        except Exception as e:
            status_text.markdown(f"**Processing... (Status check failed: {str(e)})**")
            eta_text.markdown("⏱️ **Estimated time: 1-3 minutes**")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Auto-refresh every 2 seconds for real-time updates
        time.sleep(2)
        st.rerun()
    
    elif status == "ready":
        st.markdown('<div class="ready-card">', unsafe_allow_html=True)
        st.markdown("### ✅ Stems Ready!")
        st.markdown(f"**Processing completed in {st.session_state.stems_status.get('processing_time', 0):.1f} seconds**")
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif status == "error":
        st.markdown('<div class="processing-card">', unsafe_allow_html=True)
        st.markdown("### ❌ Processing Failed")
        error_msg = st.session_state.stems_status.get('error', 'Unknown error')
        st.error(f"**Error**: {error_msg}")
        
        # Show helpful error context
        if "vocals.wav" in error_msg or "Format not recognised" in error_msg:
            st.info("💡 **Tip**: This looks like an audio format issue. Try uploading a different audio file or check if the file is corrupted.")
        elif "source separation failed" in error_msg:
            st.info("💡 **Tip**: The AI model had trouble processing this audio. Try a shorter clip or different audio file.")
        elif "Unsupported audio format" in error_msg or "codec" in error_msg:
            st.info("💡 **Format Issue**: Try converting your file to MP3 or WAV format. Supported: MP3, WAV, FLAC, M4A, OGG")
        elif "too large" in error_msg:
            st.info("💡 **File Size**: Please use a smaller file (max 100MB) or shorter audio clip (max 5 minutes)")
        elif "Cannot access" in error_msg or "permission" in error_msg:
            st.info("💡 **Access Issue**: Check if the file is corrupted or try uploading a different file.")
        elif "empty" in error_msg or "no audio data" in error_msg:
            st.info("💡 **Empty File**: The uploaded file appears to be empty or corrupted. Try a different file.")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Try Again", type="primary"):
                # Clear error state and restart
                st.session_state.current_file_id = None
                st.session_state.stems_status = None
                st.session_state.audio_info = None
                st.rerun()
        with col2:
            if st.button("📁 Upload Different File", type="secondary"):
                # Clear everything and go back to upload
                st.session_state.current_file_id = None
                st.session_state.stems_status = None
                st.session_state.audio_info = None
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

def show_live_mixer():
    """Show the live stem mixer interface"""
    if not st.session_state.stems_status or st.session_state.stems_status.get('status') != 'ready':
        return
    
    st.markdown("## 🎛️ Live Stem Mixer")
    
    # Track info
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        track_name = st.session_state.audio_info.get('title', st.session_state.audio_info.get('filename', 'Unknown Track'))
        st.markdown(f"**🎵 {track_name}**")
    with col2:
        duration = st.session_state.stems_status.get('duration', 0)
        st.markdown(f"**⏱️ {int(duration // 60)}:{int(duration % 60):02d}**")
    with col3:
        st.markdown("**🎚️ Mix Live**")
    
    st.markdown('<div class="mixer-section">', unsafe_allow_html=True)
    
    # Stem controls in a 2x2 grid
    col1, col2 = st.columns(2)
    
    with col1:
        # Vocals control
        st.markdown('<div class="stem-control">', unsafe_allow_html=True)
        st.markdown("### 🎤 Vocals")
        vocals_on = st.toggle("Enable Vocals", value=st.session_state.stem_toggles['vocals'], key="vocals_toggle")
        vocals_vol = st.slider("Volume", 0.0, 2.0, st.session_state.stem_volumes['vocals'], key="vocals_volume")
        st.session_state.stem_toggles['vocals'] = vocals_on
        st.session_state.stem_volumes['vocals'] = vocals_vol
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Bass control
        st.markdown('<div class="stem-control">', unsafe_allow_html=True)
        st.markdown("### 🎸 Bass")
        bass_on = st.toggle("Enable Bass", value=st.session_state.stem_toggles['bass'], key="bass_toggle")
        bass_vol = st.slider("Volume", 0.0, 2.0, st.session_state.stem_volumes['bass'], key="bass_volume")
        st.session_state.stem_toggles['bass'] = bass_on
        st.session_state.stem_volumes['bass'] = bass_vol
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Drums control
        st.markdown('<div class="stem-control">', unsafe_allow_html=True)
        st.markdown("### 🥁 Drums")
        drums_on = st.toggle("Enable Drums", value=st.session_state.stem_toggles['drums'], key="drums_toggle")
        drums_vol = st.slider("Volume", 0.0, 2.0, st.session_state.stem_volumes['drums'], key="drums_volume")
        st.session_state.stem_toggles['drums'] = drums_on
        st.session_state.stem_volumes['drums'] = drums_vol
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Other instruments control
        st.markdown('<div class="stem-control">', unsafe_allow_html=True)
        st.markdown("### 🎹 Other")
        other_on = st.toggle("Enable Other", value=st.session_state.stem_toggles['other'], key="other_toggle")
        other_vol = st.slider("Volume", 0.0, 2.0, st.session_state.stem_volumes['other'], key="other_volume")
        st.session_state.stem_toggles['other'] = other_on
        st.session_state.stem_volumes['other'] = other_vol
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Preview and Export controls
    st.markdown("### 🎧 Preview & Export")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("🎧 Preview Current Mix", type="secondary", use_container_width=True):
            preview_mix()
    
    with col2:
        preset_buttons()
    
    with col3:
        if st.button("💾 Export High Quality", type="primary", use_container_width=True):
            export_mix()

def preset_buttons():
    """Quick preset buttons for common mixes"""
    preset = st.selectbox("Quick Presets", [
        "Custom",
        "🎤 Karaoke (No Vocals)",
        "🥁 Drums Only", 
        "🎸 Instrumental",
        "🎵 Vocals + Drums"
    ])
    
    if preset != "Custom":
        if st.button("Apply Preset", type="secondary", use_container_width=True):
            if preset == "🎤 Karaoke (No Vocals)":
                st.session_state.stem_toggles = {'vocals': False, 'drums': True, 'bass': True, 'other': True}
                st.session_state.stem_volumes = {'vocals': 0.0, 'drums': 1.0, 'bass': 1.0, 'other': 1.0}
            elif preset == "🥁 Drums Only":
                st.session_state.stem_toggles = {'vocals': False, 'drums': True, 'bass': False, 'other': False}
                st.session_state.stem_volumes = {'vocals': 0.0, 'drums': 1.0, 'bass': 0.0, 'other': 0.0}
            elif preset == "🎸 Instrumental":
                st.session_state.stem_toggles = {'vocals': False, 'drums': True, 'bass': True, 'other': True}
                st.session_state.stem_volumes = {'vocals': 0.0, 'drums': 1.0, 'bass': 1.0, 'other': 1.0}
            elif preset == "🎵 Vocals + Drums":
                st.session_state.stem_toggles = {'vocals': True, 'drums': True, 'bass': False, 'other': False}
                st.session_state.stem_volumes = {'vocals': 1.0, 'drums': 1.0, 'bass': 0.0, 'other': 0.0}
            st.rerun()

def preview_mix():
    """Generate and play a preview mix"""
    try:
        with st.spinner("Generating preview..."):
            response = requests.post(f"{API_BASE_URL}/mix/preview", json={
            "file_id": st.session_state.current_file_id,
                "stem_toggles": st.session_state.stem_toggles,
                "stem_volumes": st.session_state.stem_volumes,
                "start_time": 30.0,  # Preview from 30 seconds
                "duration": 15.0     # 15 second preview
            })
        
        if response.status_code == 200:
            result = response.json()
            preview_id = result["preview_id"]

            # Get the preview file
            download_response = requests.get(f"{API_BASE_URL}/download/preview/{preview_id}")
            if download_response.status_code == 200:
                st.audio(download_response.content, format="audio/wav")
                st.success("🎧 Preview generated! Listen above ⬆️")
            else:
                st.error("Failed to download preview")
        else:
            st.error(f"Preview failed: {response.json().get('detail', 'Unknown error')}")
            
    except Exception as e:
        st.error(f"Preview error: {str(e)}")

def export_mix():
    """Export high-quality mix"""
    try:
        with st.spinner("Starting high-quality export..."):
            response = requests.post(f"{API_BASE_URL}/mix/export", json={
                "file_id": st.session_state.current_file_id,
                "stem_toggles": st.session_state.stem_toggles,
                "stem_volumes": st.session_state.stem_volumes,
                "format": "wav"
            })
            
            if response.status_code == 200:
                result = response.json()
                export_id = result["export_id"]
                
                st.success("🚀 Export started! This may take a few minutes...")
                
                # Show download link
                download_url = f"{API_BASE_URL}/download/export/{export_id}"
                st.markdown(f"**[📥 Download your mix when ready]({download_url})**")
                
                # TODO: Add export status polling
                
            else:
                st.error(f"Export failed: {response.json().get('detail', 'Unknown error')}")
                
    except Exception as e:
        st.error(f"Export error: {str(e)}")

def sidebar_info():
    """Sidebar with info and controls"""
    with st.sidebar:
        st.markdown("## 🎛️ Mixer Info")
        
        if st.session_state.current_file_id:
            st.markdown(f"**File ID**: `{st.session_state.current_file_id[:8]}...`")
            
            if st.session_state.stems_status:
                status = st.session_state.stems_status.get('status', 'unknown')
                if status == "ready":
                    st.success("✅ Stems Ready")
                    st.markdown(f"**Duration**: {st.session_state.stems_status.get('duration', 0):.1f}s")
                    st.markdown(f"**Processing Time**: {st.session_state.stems_status.get('processing_time', 0):.1f}s")
                elif status == "processing":
                    st.info("🔄 Processing...")
                else:
                    st.error(f"❌ {status}")
            
            if st.button("🗑️ Clear & Start Over"):
                st.session_state.current_file_id = None
                st.session_state.stems_status = None
                st.session_state.audio_info = None
                st.rerun()
        
        st.markdown("---")
        
        st.markdown("## 🎵 How It Works")
        st.markdown("""
        1. **Upload** your audio or paste YouTube URL
        2. **Wait** 1-3 minutes for AI stem separation  
        3. **Mix** stems instantly with toggles & sliders
        4. **Preview** your mix in real-time
        5. **Export** high-quality final mix
        """)
        
        st.markdown("---")
        
        st.markdown("## 📁 Supported Formats")
        st.markdown("""
        **✅ Audio Files:**
        - MP3 (recommended)
        - WAV (best quality)  
        - FLAC (lossless)
        - M4A/AAC
        - OGG/Vorbis
        
        **📏 Limits:**
        - Max size: 100MB
        - Max length: 5 minutes
        """)
        
        st.markdown("---")
        
        st.markdown("## ⚡ Features")
        st.markdown("""
        - **🤖 AI-Powered**: Demucs v4 separation
        - **⚡ Real-time**: Instant mixing preview
        - **🎚️ Full Control**: Volume & toggle each stem
        - **🎧 Preview**: Test before export
        - **💾 High Quality**: 44.1kHz WAV export
        """)

def main():
    """Main application function"""
    initialize_session_state()
    show_main_header()
    
    # Check stems status if we have a file
    if st.session_state.current_file_id:
        check_stems_status()
    
    # Main content based on current state
    if not st.session_state.current_file_id:
        # Show upload section
        upload_section()
    elif st.session_state.stems_status and st.session_state.stems_status.get('status') in ['processing']:
        # Show processing status
        show_processing_status()
    elif st.session_state.stems_status and st.session_state.stems_status.get('status') == 'ready':
        # Show live mixer
        show_live_mixer()
    elif st.session_state.stems_status and st.session_state.stems_status.get('status') == 'error':
        # Show error status
        show_processing_status()  # This function handles error display
    else:
        # Fallback: check status
        with st.spinner("Checking stems status..."):
            check_stems_status()
            time.sleep(1)
            st.rerun()
    
    # Sidebar
    sidebar_info()

if __name__ == "__main__":
    main()