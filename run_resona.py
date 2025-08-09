#!/usr/bin/env python3
"""
Resona Startup Script

This script starts both the FastAPI backend and Streamlit frontend
"""

import subprocess
import sys
import time
import os
import signal
import atexit
from pathlib import Path

def cleanup_processes():
    """Clean up background processes on exit"""
    global backend_process, frontend_process
    
    print("\n🛑 Shutting down Resona...")
    
    if 'backend_process' in globals() and backend_process:
        backend_process.terminate()
        print("✅ Backend stopped")
    
    if 'frontend_process' in globals() and frontend_process:
        frontend_process.terminate()
        print("✅ Frontend stopped")

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        # Use uv run to check dependencies in the virtual environment
        import subprocess
        result = subprocess.run([
            "uv", "run", "python", "-c", 
            "import fastapi, streamlit, librosa, demucs, torch, yt_dlp; print('✅ All dependencies are installed')"
        ], capture_output=True, text=True, cwd="backend")
        
        if result.returncode == 0:
            print(result.stdout.strip())
            return True
        else:
            print(f"❌ Dependency check failed: {result.stderr}")
            print("Please run: uv sync")
            return False
    except Exception as e:
        print(f"❌ Dependency check error: {e}")
        print("Please run: uv sync")
        return False

def start_backend():
    """Start the FastAPI backend"""
    print("🚀 Starting FastAPI backend...")
    
    try:
        # Start uvicorn from project root using module path
        cmd = [
            "uv", "run", "uvicorn", 
            "backend.main:app", 
            "--host", "0.0.0.0", 
            "--port", "8000",
            "--reload"
        ]
        
        process = subprocess.Popen(
            cmd,
            cwd=Path("."),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Give the backend time to start
        time.sleep(3)
        
        if process.poll() is None:
            print("✅ Backend started on http://localhost:8000")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Backend failed to start:")
            print(f"STDOUT: {stdout.decode()}")
            print(f"STDERR: {stderr.decode()}")
            return None
            
    except Exception as e:
        print(f"❌ Error starting backend: {e}")
        return None

def start_frontend():
    """Start the Streamlit frontend"""
    print("🎨 Starting Streamlit frontend...")
    
    try:
        cmd = [
            "uv", "run", "streamlit", 
            "run", "frontend/app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ]
        
        process = subprocess.Popen(
            cmd,
            cwd=Path(".")
        )
        
        print("✅ Frontend started on http://localhost:8501")
        return process
        
    except Exception as e:
        print(f"❌ Error starting frontend: {e}")
        return None

def main():
    """Main startup function"""
    global backend_process, frontend_process
    
    # Register cleanup function
    atexit.register(cleanup_processes)
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
    signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))
    
    print("🎵 Starting Resona - AI Beat Extractor")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Create necessary directories
    os.makedirs("downloads", exist_ok=True)
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("processed", exist_ok=True)
    
    # Start backend
    backend_process = start_backend()
    if not backend_process:
        print("❌ Failed to start backend")
        sys.exit(1)
    
    # Start frontend
    frontend_process = start_frontend()
    if not frontend_process:
        print("❌ Failed to start frontend")
        backend_process.terminate()
        sys.exit(1)
    
    print("=" * 50)
    print("🎉 Resona is now running!")
    print("📱 Frontend: http://localhost:8501")
    print("🔧 Backend API: http://localhost:8000")
    print("📖 API Docs: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop")
    print("=" * 50)
    
    try:
        # Keep the script running
        while True:
            time.sleep(1)
            
            # Check if processes are still running
            if backend_process.poll() is not None:
                print("❌ Backend process died")
                break
                
            if frontend_process.poll() is not None:
                print("❌ Frontend process died")
                break
                
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    
    cleanup_processes()

if __name__ == "__main__":
    main()
