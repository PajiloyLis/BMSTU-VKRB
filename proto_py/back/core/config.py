from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GRAMMAR_PATH = PROJECT_ROOT / "grammar.json"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
PARSES_DIR = ARTIFACTS_DIR / "parses"
ARTIFACTS_URL_PREFIX = "/artifacts"

