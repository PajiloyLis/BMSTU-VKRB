import type { ParseTree } from "../api/types";

interface TreeCardProps {
  tree: ParseTree;
}

export function TreeCard({ tree }: TreeCardProps) {
  return (
    <article className="tree-card">
      <header>
        <h3>Дерево {tree.index}</h3>
        <div className="links">
          {tree.dotUrl && <a href={tree.dotUrl}>DOT</a>}
          {tree.pngUrl && <a href={tree.pngUrl}>PNG</a>}
        </div>
      </header>
      <pre>{tree.bracket}</pre>
      {tree.pngUrl && <img src={tree.pngUrl} alt={`Дерево ${tree.index}`} />}
    </article>
  );
}
