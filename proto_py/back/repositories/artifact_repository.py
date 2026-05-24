import json
import subprocess
from pathlib import Path
from typing import Any

from tree_utils import tree_to_dot


class ArtifactRepository:
    def __init__(self, artifacts_dir: Path, public_prefix: str) -> None:
        self.artifacts_dir = artifacts_dir
        self.public_prefix = public_prefix.rstrip("/")
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def create_parse_dir(self, parse_id: str) -> Path:
        path = self.artifacts_dir / parse_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_json(self, parse_id: str, payload: dict[str, Any]) -> str:
        path = self.create_parse_dir(parse_id) / "result.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return self._public_url(path)

    def save_brackets(self, parse_id: str, brackets: list[str]) -> str:
        path = self.create_parse_dir(parse_id) / "trees.bracket.txt"
        path.write_text("\n\n".join(brackets), encoding="utf-8")
        return self._public_url(path)

    def save_tree_artifacts(self, parse_id: str, tree_index: int, tree: dict[str, Any]) -> tuple[str, str | None]:
        nodes, edges = tree_to_dot(tree)
        dot_content = "digraph Tree {\n" + "\n".join(nodes) + "\n" + "\n".join(edges) + "\n}"
        parse_dir = self.create_parse_dir(parse_id)
        dot_path = parse_dir / f"tree_{tree_index}.dot"
        dot_path.write_text(dot_content, encoding="utf-8")

        png_path = parse_dir / f"tree_{tree_index}.png"
        png_url: str | None = None
        try:
            subprocess.run(
                ["dot", "-Tpng", str(dot_path), "-o", str(png_path)],
                check=True,
                capture_output=True,
                text=True,
            )
            png_url = self._public_url(png_path)
        except (FileNotFoundError, subprocess.CalledProcessError):
            png_url = None

        return self._public_url(dot_path), png_url

    def _public_url(self, path: Path) -> str:
        relative = path.relative_to(self.artifacts_dir)
        return f"{self.public_prefix}/{relative.as_posix()}"

