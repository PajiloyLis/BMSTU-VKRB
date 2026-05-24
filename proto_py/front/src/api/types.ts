export interface ParseRequest {
  text: string;
  maxTrees: number;
}

export interface ArtifactLinks {
  jsonUrl: string;
  bracketUrl: string | null;
  dotUrls: string[];
  pngUrls: string[];
}

export interface ParseTree {
  index: number;
  bracket: string;
  tree: unknown;
  dotUrl: string | null;
  pngUrl: string | null;
}

export interface ParseResponse {
  id: string;
  text: string;
  tokens: string[];
  parsed: boolean;
  rootSymbol: string;
  treeCount: number;
  returnedTreeCount: number;
  trees: ParseTree[];
  artifacts: ArtifactLinks;
  diagnostics: Record<string, number>;
  message: string;
}
