#!/usr/bin/env python3
"""
Run script for the PDF Research Assistant Streamlit app
"""
import os
import subprocess
import sys


def main():
    print("Starting PDF Research Assistant...")

    # Ensure necessary directories exist
    os.makedirs("data/chroma_db", exist_ok=True)
    os.makedirs("data/temp", exist_ok=True)
    os.makedirs("pages", exist_ok=True)

    # Launch the Streamlit app
    try:
        subprocess.run(["streamlit", "run", "app.py"], check=True)
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error running Streamlit app: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
