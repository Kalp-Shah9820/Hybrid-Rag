"""
CLI Ingestion Script
Run: python -m scripts.ingest
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
from src.ingestion import run_ingestion


def main():
    logger.info("=" * 60)
    logger.info("  RAG Stock Market — Data Ingestion")
    logger.info("=" * 60)

    try:
        run_ingestion()
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Ingestion interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Ingestion failed: {e}")
        raise


if __name__ == "__main__":
    main()
