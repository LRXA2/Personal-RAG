# RAG File Search - Quick Start Guide

## Overview

A metadata-first file search system with a simple web UI. The system searches your D: and E: drives using only filenames, paths, and metadata (no content indexing by default), keeping sensitive files safe while still being searchable.

## Architecture

```
User Query → Search API (/search) → Metadata Index → Results Cards
                                           ↓
                                    Download API (/download) → File Stream
```

**Two Simple APIs:**
1. `POST /search` - Search files by query, returns metadata cards
2. `GET /download?path=...` - Download a specific file (with safety checks)

## Installation

```bash
# Install dependencies
pip install fastapi uvicorn pydantic python-multipart

# Navigate to project directory
cd rag_file_search
```

## Usage

### Step 1: Start the API Server

```bash
uvicorn api.endpoints:app --reload --host 0.0.0.0 --port 8000
```

The server will start at `http://localhost:8000`

### Step 2: Index Your Drives (First Time Only)

Open another terminal and run:

```bash
curl -X POST "http://localhost:8000/index/scan" \
  -H "Content-Type: application/json" \
  -d '{"directories": ["D:/", "E:/"]}'
```

Or use Python:

```python
import requests

response = requests.post(
    "http://localhost:8000/index/scan",
    json={"directories": ["D:/", "E:/"]}
)
print(response.json())
```

### Step 3: Open the Web UI

Simply open `ui/index.html` in your browser:

```bash
# Option 1: Direct file open
start ui/index.html  # Windows
open ui/index.html   # macOS
xdg-open ui/index.html  # Linux

# Option 2: Serve with Python (recommended)
cd ui
python -m http.server 8080
# Then visit: http://localhost:8080
```

### Step 4: Search and Download

1. Type your search query in the search bar
2. Results appear as cards with:
   - File icon and name
   - File type (colored tag)
   - Size and modified date
   - Full path (truncated)
   - Relevance score
3. Click **Download** to download the file
4. Click **Copy Path** to copy the full path to clipboard

## API Endpoints

### Search Files
```bash
POST http://localhost:8000/search
Content-Type: application/json

{
  "query": "project documentation",
  "max_results": 50
}
```

Response:
```json
{
  "results": [
    {
      "filename": "README.md",
      "path": "D:/Projects/MyApp/README.md",
      "extension": "md",
      "size": 2048,
      "modified_date": "2024-03-15T10:30:00",
      "relevance_score": 0.95
    }
  ],
  "search_time_ms": 45.2,
  "total_found": 1
}
```

### Download File
```bash
GET http://localhost:8000/download?path=D:/Projects/MyApp/README.md
```

Returns the file as a download attachment.

## Safety Features

The system includes built-in safety controls:

- **Path validation**: Prevents directory traversal attacks (`../`)
- **File size limits**: Max 100MB per file (configurable)
- **Extension filtering**: Blocks potentially dangerous file types
- **Read-only access**: Cannot modify or delete files
- **Policy checks**: Every download is validated before serving

## Configuration

Edit `api/endpoints.py` to customize:

```python
policy = SafetyPolicy(
    allowed_roots=["D:/", "E:/"],  # Restrict to specific drives
    max_file_size_bytes=100 * 1024 * 1024,  # 100MB limit
    blocked_extensions=[".exe", ".bat", ".sh"],  # Block executables
)
```

## Troubleshooting

### "Service not initialized"
- Make sure you've scanned directories first using `/index/scan`

### "Access denied"
- The file path may be outside allowed roots
- File extension may be blocked
- File may be too large

### UI not loading results
- Check that the API server is running on port 8000
- Verify CORS settings if serving UI from different port
- Check browser console for errors

## Advanced Usage

### Programmatic Search

```python
import requests

# Search
response = requests.post(
    "http://localhost:8000/search",
    json={"query": "Python scripts from last month"}
)

for file in response.json()["results"]:
    print(f"{file['filename']} - {file['size']} bytes")
    
    # Download
    download_url = f"http://localhost:8000/download?path={file['path']}"
    # Use requests.get(download_url) to download
```

### Health Check

```bash
GET http://localhost:8000/health
```

### Index Statistics

```bash
GET http://localhost:8000/stats
```

## Project Structure

```
rag_file_search/
├── api/
│   └── endpoints.py       # FastAPI server with /search and /download
├── core/
│   ├── policy.py          # Safety policies
│   └── retrieval_service.py  # Search logic
├── indexer/
│   └── metadata_indexer.py   # File scanning
├── ui/
│   └── index.html         # Web interface
└── README.md              # This file
```

## License

MIT License - Free for personal and commercial use.
