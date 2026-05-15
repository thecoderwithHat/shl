"""SHL conversational recommender package."""

from pathlib import Path
try:
	# Load a .env file from the repository root if present (makes os.getenv pick values)
	from dotenv import load_dotenv

	env_path = Path(__file__).resolve().parents[1] / ".env"
	if env_path.exists():
		load_dotenv(env_path)
except Exception:
	# Keep import-time failure silent (app can still read env vars from the environment)
	pass
