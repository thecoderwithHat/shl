"""SHL conversational recommender package."""

from pathlib import Path

from dotenv import load_dotenv


env_path = Path(__file__).resolve().parents[1] / ".env"
# Force .env to win over any already-exported values for local/dev runs.
load_dotenv(dotenv_path=env_path, override=True)
