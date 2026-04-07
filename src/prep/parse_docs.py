"""
Backward-compatible entrypoint for document ingestion.

Historically this project started with `parse_docs.py`. The current ingestion
implementation lives in `ingest_moodle_docs.py`.
"""

from src.prep.ingest_moodle_docs import main


if __name__ == "__main__":
    main()
