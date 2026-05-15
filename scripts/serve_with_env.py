from __future__ import annotations
import os
from pathlib import Path
import sys

from dotenv import load_dotenv

# load .env and force it to win over already-exported values
dotenv_path = Path(__file__).resolve().parents[1] / '.env'
load_dotenv(dotenv_path=dotenv_path, override=True)

# ensure repo root importable
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import uvicorn

if __name__ == '__main__':
    uvicorn.run('app.main:app', host='127.0.0.1', port=8000)
