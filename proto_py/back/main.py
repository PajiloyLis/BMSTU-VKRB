import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from back.core.config import ARTIFACTS_DIR, ARTIFACTS_URL_PREFIX, PARSES_DIR, PROJECT_ROOT


if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from back.controllers.parse_controller import router as parse_router
from back.repositories.artifact_repository import ArtifactRepository
from back.services.parser_service import ParserService


artifact_repository = ArtifactRepository(PARSES_DIR, f"{ARTIFACTS_URL_PREFIX}/parses")
parser_service = ParserService(artifact_repository)

app = FastAPI(title="Russian CYK Parser", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount(ARTIFACTS_URL_PREFIX, StaticFiles(directory=ARTIFACTS_DIR), name="artifacts")
app.include_router(parse_router)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

