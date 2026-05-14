from __future__ import annotations
import os
from pathlib import Path
import sys

# Load .env if present (simple parser)
dotenv_path = Path(__file__).resolve().parents[1] / '.env'
if dotenv_path.exists():
    with open(dotenv_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' not in line:
                continue
            k, v = line.split('=', 1)
            k = k.strip()
            v = v.strip()
            # don't overwrite existing env vars
            os.environ.setdefault(k, v)

print('ENV ENABLE_LLM=', os.getenv('ENABLE_LLM'))
print('ENV OPENROUTER_API_KEY present=', bool(os.getenv('OPENROUTER_API_KEY')))

try:
    # Ensure repository root is importable
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from app.llm import build_openrouter_chat_model
    model = build_openrouter_chat_model()
    print('build_openrouter_chat_model() ->', type(model).__name__ if model is not None else None)
except Exception as e:
    print('error calling build_openrouter_chat_model():', e)
