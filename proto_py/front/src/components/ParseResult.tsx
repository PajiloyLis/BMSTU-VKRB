import type { ParseResponse } from "../api/types";
import { TokenList } from "./TokenList";
import { TreeCard } from "./TreeCard";

interface ParseResultProps {
  result: ParseResponse | null;
}

export function ParseResult({ result }: ParseResultProps) {
  if (result === null) {
    return (
      <section className="panel muted">
        Введите предложение и запустите разбор. Результаты будут сохранены в публичной папке артефактов.
      </section>
    );
  }

  return (
    <section className="panel result">
      <div className="result-header">
        <div>
          <p className={result.parsed ? "status ok" : "status fail"}>
            {result.parsed ? "Разбор найден" : "Разбор не найден"}
          </p>
          <h2>{result.message}</h2>
        </div>
        <a href={result.artifacts.jsonUrl}>JSON результата</a>
      </div>

      <h3>Токены</h3>
      <TokenList tokens={result.tokens} />

      {result.artifacts.bracketUrl && (
        <p>
          <a href={result.artifacts.bracketUrl}>Скачать скобочную запись</a>
        </p>
      )}

      <div className="tree-list">
        {result.trees.map((tree) => (
          <TreeCard key={tree.index} tree={tree} />
        ))}
      </div>
    </section>
  );
}
