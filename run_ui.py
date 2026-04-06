#!/usr/bin/env python3
"""
Run the RAG File Search UI.

This script starts the FastAPI server and opens the web UI in your default browser.
Simply run: python run_ui.py

The UI will be available at: http://localhost:8000/ui
"""

import sys
import time
import threading
import webbrowser
from pathlib import Path

# Add the package to the path
sys.path.insert(0, str(Path(__file__).parent))

def open_browser():
    """Open the UI in the default browser after a short delay."""
    time.sleep(1.5)  # Wait for server to start
    webbrowser.open("http://localhost:8000/ui")
    print("\n[OK] UI opened in your browser!")
    print("  If it didn't open automatically, visit: http://localhost:8000/ui")
    print("\n" + "="*60)
    print("Press Ctrl+C to stop the server")
    print("="*60)

def main():
    """Start the server and open the UI."""
    print("="*60)
    print("  RAG File Search - Starting...")
    print("="*60)
    
    # Try to import required dependencies
    try:
        import uvicorn
        from fastapi import FastAPI
        from fastapi.staticfiles import StaticFiles
        from fastapi.responses import FileResponse
    except ImportError as e:
        print(f"\n[ERROR] Missing dependency: {e}")
        print("\nPlease install required packages:")
        print("  pip install fastapi uvicorn python-multipart")
        sys.exit(1)
    
    # Import our API
    from rag_file_search.api.endpoints import app
    
    # Mount the UI static files
    ui_path = Path(__file__).parent / "rag_file_search" / "ui"
    
    @app.get("/ui")
    async def serve_ui():
        """Serve the main UI page."""
        return FileResponse(ui_path / "index.html")
    
    # Also serve at root for convenience
    @app.get("/")
    async def serve_root():
        """Redirect to UI."""
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/ui")
    
    print("\n[OK] Dependencies loaded successfully")
    print(f"[OK] UI directory: {ui_path}")
    
    print("[OK] UI-only startup enabled (no automatic indexing)")
    print("[TIP] Allowed roots are managed by API config (RAG_ALLOWED_ROOTS)")
    print("[TIP] UI starts indexing from configured defaults automatically")
    
    # Start browser opener in background
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    # Start the server
    print("\n[INFO] Starting server on http://localhost:8000")
    print("  (This may take a moment...)\n")
    
    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
        )
    except KeyboardInterrupt:
        print("\n\n[OK] Server stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
