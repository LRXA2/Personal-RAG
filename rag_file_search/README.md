# RAG File Search System

A **metadata-first, two-stage retrieval system** for safely searching files on your local machine. This implementation follows the design principle: **"Let the model plan retrieval, but let the retrieval service control what gets searched and opened."**

## Key Features

- **Metadata-First Retrieval**: Searches filenames, paths, dates, and extensions before reading any content
- **Two-Stage Pipeline**: 
  1. Stage 1: File-level retrieval using metadata only
  2. Stage 2: Content-level retrieval for shortlisted, policy-approved files
- **Safety Controls**: Path sanitization, directory blocklists, extension filters, size limits
- **Progressive Exposure**: Broad metadata visibility, narrow content visibility
- **No Embeddings Required (v1)**: Uses lexical matching, fuzzy search, and filters

## Architecture

```
user query → LLM planner → metadata retrieval → shortlisted files → policy filter → content reader → chunks → answer
```

### Components

1. **LLM Planner** (`core/planner.py`): Parses natural language queries into structured search intent
2. **Metadata Indexer** (`indexer/metadata_indexer.py`): Scans and indexes file metadata
3. **Safety Policy** (`core/policy.py`): Enforces access controls and safety rules
4. **Retrieval Service** (`core/retrieval_service.py`): Two-stage retrieval pipeline
5. **Content Extractor** (`indexer/content_extractor.py`): Reads and chunks file content
6. **API Endpoints** (`api/endpoints.py`): FastAPI REST API

## Installation

```bash
pip install fastapi uvicorn pydantic python-multipart
```

## Quick Start

### Option 1: Run with UI (Recommended)

Simply run the Python script to start the server and open the UI in your browser:

```bash
python run_ui.py
```

This will:
- Start the FastAPI server on `http://localhost:8000`
- Automatically open your default browser to the UI
- Display search results in cards with metadata and download options

The UI will be available at: **http://localhost:8000** or **http://localhost:8000/ui**

### Option 2: Python API

```python
from rag_file_search import RagFileSearch

# Initialize with your directories
searcher = RagFileSearch(
    allowed_roots=["D:/", "E:/"],
    max_files_to_read=10,
    enable_content_grounding=True,
)

# Index your files
searcher.index()

# Search with natural language
results = searcher.search("Find the Atlas onboarding video from March")
for result in results:
    print(f"{result.metadata.filename} - {result.metadata.full_path}")

# Get formatted answers
answer = searcher.answer("What documentation do I have for the project?")
print(answer)
```

### Option 3: REST API

```bash
# Start the server
uvicorn rag_file_search.api.endpoints:app --reload --host 0.0.0.0 --port 8000

# Visit the interactive docs
open http://localhost:8000/docs
```

#### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/search` | POST | **Simple search** - Returns file metadata with relevance scores |
| `/download` | GET | **Download a file** - Streams file content after safety checks |
| `/retrieve/files` | POST | Search metadata only (legacy) |
| `/read/file` | POST | Read a specific file (with policy checks) |
| `/extract/chunks` | POST | Extract text chunks from a file |
| `/answer` | POST | Full pipeline: retrieve + answer |
| `/stats` | GET | Index statistics |
| `/index/scan` | POST | Scan and index directories |
| `/health` | GET | Health check |

## Safety Features

The system enforces multiple layers of safety:

### Path Controls
- **Allowed Roots**: Only scan specified root directories
- **Blocked Directories**: Automatically skip sensitive folders (Windows, AppData, .git, etc.)
- **Traversal Protection**: Blocks `../` path traversal attempts

### Content Controls
- **Extension Filtering**: Only read safe file types (.txt, .md, .py, etc.)
- **Blocked Extensions**: Never open executables or scripts (.exe, .dll, .bat, etc.)
- **Size Limits**: Maximum file size for reading (default: 10 MB)
- **Query Limits**: Max files opened per query, max extracted text

### Sensitive File Detection
- Filename pattern matching (password, secret, credential, private, key, .env, .pem)
- Directory name blocking (secrets, credentials, .ssh, .gnupg)

## Query Capabilities

The planner understands:

- **File types**: "Find PDF documents", "Show me Python files"
- **Date ranges**: "Files from March", "Last week's documents", "2024 reports"
- **Folder constraints**: "In the projects folder", "Under Documents"
- **Content questions**: "What does the readme say about installation?"

## Design Philosophy

### Why Metadata-First?

Since your files are well-organized by name and path, most queries can be answered without reading content:

> "Find the Atlas onboarding video from March"

Only needs: filename matching, folder matching, date filters, type filters — no transcript needed.

### Why Two-Stage Retrieval?

1. **Safety**: The system sees all names, but not all contents
2. **Efficiency**: No need to chunk and embed entire disk
3. **Control**: Content reading happens only after file selection

### Why Not Full Embeddings (v1)?

Your naming scheme is strong enough that lexical search + filters provide excellent results. Embeddings can be added later for chunk-level retrieval within shortlisted files.

## Project Structure

```
rag_file_search/
├── __init__.py              # Main entry point (RagFileSearch class)
├── core/
│   ├── models.py            # Data models (FileMetadata, SearchResult, Chunk)
│   ├── policy.py            # Safety policy enforcement
│   ├── planner.py           # Query parsing and answer formatting
│   └── retrieval_service.py # Two-stage retrieval pipeline
├── indexer/
│   ├── metadata_indexer.py  # File scanning and metadata indexing
│   └── content_extractor.py # Text extraction and chunking
├── api/
│   └── endpoints.py         # FastAPI REST endpoints
└── examples/
    └── demo.py              # Usage demonstration
```

## Customization

### Adjust Safety Policy

```python
from rag_file_search import SafetyPolicy

policy = SafetyPolicy(
    allowed_roots=["D:/Documents", "E:/Projects"],
    blocked_dirs=["windows", "secrets", ".git"],
    max_file_size_bytes=5 * 1024 * 1024,  # 5 MB
    max_files_per_query=5,
)
```

### Configure Retrieval

```python
from rag_file_search import RetrievalConfig

config = RetrievalConfig(
    max_metadata_results=100,
    max_files_to_read=10,
    max_chunks_per_file=5,
    enable_content_grounding=True,
)
```

## Future Enhancements

- [ ] Add embedding-based chunk retrieval for Stage 2
- [ ] Support PDF/DOCX text extraction (requires PyPDF2, python-docx)
- [ ] Image/video metadata extraction (duration, resolution, EXIF)
- [ ] Local LLM integration for better query understanding
- [ ] Persistent index storage (SQLite, LanceDB)
- [ ] Web UI for browsing and searching

## License

MIT
