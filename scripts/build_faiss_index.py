"""Build or rebuild the FAISS index for the product catalog.

Usage:
  # Build index (uses FAISS_INDEX_PATH env var or defaults to ./faiss_index)
  python scripts/build_faiss_index.py

  # Force-rebuild (delete existing index first)
  python scripts/build_faiss_index.py --force

  # Specify a custom index path
  FAISS_INDEX_PATH=./data/faiss_index python scripts/build_faiss_index.py
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
import sys

# Ensure the repository root is on sys.path so `app` can be imported when running this script
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.recommender import load_catalog, SemanticCatalogIndex, ROOT as REPO_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or rebuild the FAISS index")
    parser.add_argument("--force", action="store_true", help="Delete existing index and rebuild")
    parser.add_argument("--index-path", type=str, default=os.getenv("FAISS_INDEX_PATH", str(ROOT / "faiss_index")), help="Path to save/load the FAISS index")
    args = parser.parse_args()

    index_path = Path(args.index_path)

    if args.force and index_path.exists():
        print(f"Removing existing index at {index_path}")
        shutil.rmtree(index_path)

    products = load_catalog()
    print(f"Loaded {len(products)} products from catalog")

    # SemanticCatalogIndex will attempt to load existing index or build+save it
    index = SemanticCatalogIndex(products)
    print(f"Index initialized. FAISS index path: {index_path}")


if __name__ == "__main__":
    main()
