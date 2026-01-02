#!/usr/bin/env python3
"""
Quick start script for the PDF Q&A FastAPI server.
"""
import os
import sys
import uvicorn

def main():
    """Run the FastAPI server."""
    print("=" * 60)
    print("PDF Q&A API Server")
    print("=" * 60)
    print()
    print("Starting FastAPI server...")
    print()
    print("Once running, access:")
    print("  - Interactive docs (Swagger): http://localhost:8000/docs")
    print("  - Alternative docs (ReDoc):   http://localhost:8000/redoc")
    print("  - Health check:               http://localhost:8000/health")
    print()
    print("Press CTRL+C to stop the server")
    print("=" * 60)
    print()

    # Create necessary directories
    os.makedirs("data/chroma_db", exist_ok=True)
    os.makedirs("data/temp", exist_ok=True)

    # Run the server
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()
