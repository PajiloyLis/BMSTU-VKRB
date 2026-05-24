from dataclasses import dataclass
from uuid import uuid4

from parser_service import ParserEngine

from back.core.config import GRAMMAR_PATH
from back.repositories.artifact_repository import ArtifactRepository
from back.schemas.parse import ArtifactLinks, ParseResponse, ParseTree


@dataclass
class ParserService:
    artifact_repository: ArtifactRepository
    root_symbol: str = "IP"

    def __post_init__(self) -> None:
        self.engine = ParserEngine(GRAMMAR_PATH, root_symbol=self.root_symbol)

    def parse(self, text: str, max_trees: int) -> ParseResponse:
        parse_id = uuid4().hex
        result = self.engine.parse_sentence(text, max_trees=max_trees)
        trees_payload: list[ParseTree] = []
        bracket_strings: list[str] = []
        dot_urls: list[str] = []
        png_urls: list[str] = []

        for parsed_tree in result.trees:
            dot_url, png_url = self.artifact_repository.save_tree_artifacts(
                parse_id, parsed_tree.index, parsed_tree.tree
            )
            bracket_strings.append(f"===== Дерево {parsed_tree.index} =====\n{parsed_tree.bracket}")
            dot_urls.append(dot_url)
            if png_url is not None:
                png_urls.append(png_url)
            trees_payload.append(
                ParseTree(
                    index=parsed_tree.index,
                    bracket=parsed_tree.bracket,
                    tree=parsed_tree.tree,
                    dotUrl=dot_url,
                    pngUrl=png_url,
                )
            )

        bracket_url = self.artifact_repository.save_brackets(parse_id, bracket_strings) if bracket_strings else None
        response_payload = {
            "id": parse_id,
            "text": result.text,
            "tokens": result.tokens,
            "parsed": result.parsed,
            "rootSymbol": result.root_symbol,
            "treeCount": result.tree_count,
            "returnedTreeCount": len(trees_payload),
            "trees": [self._dump_model(tree) for tree in trees_payload],
            "diagnostics": result.diagnostics,
            "message": self._message(result.parsed, result.tree_count, len(trees_payload)),
        }
        json_url = self.artifact_repository.save_json(parse_id, response_payload)

        return ParseResponse(
            **response_payload,
            artifacts=ArtifactLinks(
                jsonUrl=json_url,
                bracketUrl=bracket_url,
                dotUrls=dot_urls,
                pngUrls=png_urls,
            ),
        )

    @staticmethod
    def _message(parsed: bool, tree_count: int, returned_tree_count: int) -> str:
        if not parsed:
            return "Разбор не найден"
        if tree_count == returned_tree_count:
            return f"Найдено деревьев: {tree_count}"
        return f"Найдено деревьев: {tree_count}; возвращено: {returned_tree_count}"

    @staticmethod
    def _dump_model(model):
        if hasattr(model, "model_dump"):
            return model.model_dump()
        return model.dict()

