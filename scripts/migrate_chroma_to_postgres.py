"""
CLI migration script.
Run: python -m scripts.migrate_chroma_to_postgres
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from src.migration import migrate_chroma_to_postgres


def main() -> None:
    migrated = migrate_chroma_to_postgres()
    logger.info(f"Migration complete: {migrated:,} chunks moved to PostgreSQL")


if __name__ == "__main__":
    main()
