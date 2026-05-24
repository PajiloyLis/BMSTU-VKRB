from typing import Any

from pydantic import BaseModel, Field


class ParseRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)
    maxTrees: int = Field(default=5, ge=1, le=20)


class ArtifactLinks(BaseModel):
    jsonUrl: str
    bracketUrl: str | None = None
    dotUrls: list[str] = Field(default_factory=list)
    pngUrls: list[str] = Field(default_factory=list)


class ParseTree(BaseModel):
    index: int
    bracket: str
    tree: dict[str, Any]
    dotUrl: str | None = None
    pngUrl: str | None = None


class ParseResponse(BaseModel):
    id: str
    text: str
    tokens: list[str]
    parsed: bool
    rootSymbol: str = "IP"
    treeCount: int
    returnedTreeCount: int
    trees: list[ParseTree]
    artifacts: ArtifactLinks
    diagnostics: dict[str, int] = Field(default_factory=dict)
    message: str

