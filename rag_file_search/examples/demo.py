"""
Example usage of the RAG File Search system.

This script demonstrates:
1. Setting up the search system
2. Indexing directories
3. Running various types of queries
4. Using the API server
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rag_file_search import RagFileSearch


def main():
    """Demonstrate the RAG file search system."""
    
    print("=" * 60)
    print("RAG File Search System - Demo")
    print("=" * 60)
    
    # Initialize the search system
    # For demo, we'll use the current workspace directory
    # In production, you would use: allowed_roots=["D:/", "E:/"]
    demo_dir = str(Path(__file__).parent.parent)
    
    print(f"\nInitializing search system...")
    searcher = RagFileSearch(
        allowed_roots=[demo_dir],
        max_files_to_read=5,
        enable_content_grounding=True,
    )
    
    # Index the directory
    print(f"\nIndexing directory: {demo_dir}")
    count = searcher.index(verbose=True)
    print(f"Indexed {count} files")
    
    # Show stats
    stats = searcher.get_stats()
    print(f"\nIndex Statistics:")
    print(f"  Total files: {stats['total_files']}")
    print(f"  Unique extensions: {stats['unique_extensions']}")
    print(f"  Extensions: {', '.join(stats['extensions'][:10])}...")
    
    # Example queries
    queries = [
        "Find Python files in this project",
        "Show me documentation or readme files",
        "What files were created recently?",
        "Find configuration files like yaml or json",
    ]
    
    print("\n" + "=" * 60)
    print("Running Example Queries")
    print("=" * 60)
    
    for query in queries:
        print(f"\n\nQuery: '{query}'")
        print("-" * 40)
        
        results = searcher.search(query, max_results=5)
        
        if not results:
            print("  No results found.")
            continue
        
        for i, result in enumerate(results, 1):
            meta = result.metadata
            print(f"\n  {i}. {meta.filename}")
            print(f"     Path: {meta.full_path}")
            print(f"     Type: {meta.file_type} ({meta.extension})")
            print(f"     Score: {result.score:.2f}")
            print(f"     Match: {result.match_type}")
            
            if result.snippets:
                snippet = result.snippets[0][:150].replace('\n', ' ')
                print(f"     Preview: {snippet}...")
    
    # Demonstrate the answer format
    print("\n" + "=" * 60)
    print("Formatted Answer Example")
    print("=" * 60)
    
    answer_query = "Find all Python code files in the core module"
    print(f"\nQuery: '{answer_query}'\n")
    
    answer = searcher.answer(answer_query)
    print(answer)
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print("\nTo run the API server:")
    print("  uvicorn rag_file_search.api.endpoints:app --reload --host 0.0.0.0 --port 8000")
    print("\nThen visit: http://localhost:8000/docs")


if __name__ == "__main__":
    main()
