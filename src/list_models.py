"""
CLI helper to list Gemini models visible to the configured API key.

Run with:
    python -m src.list_models
"""

from __future__ import annotations

from dotenv import load_dotenv

from src.gemini_client import print_available_models


def main() -> None:
    load_dotenv()
    print_available_models()


if __name__ == "__main__":
    main()
