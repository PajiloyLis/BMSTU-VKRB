import { useState } from "react";

import { parseSentence } from "../api/parserClient";
import type { ParseResponse } from "../api/types";

interface ParserState {
  result: ParseResponse | null;
  isLoading: boolean;
  error: string | null;
}

export function useParser() {
  const [state, setState] = useState<ParserState>({
    result: null,
    isLoading: false,
    error: null
  });

  async function submit(sentence: string, maxTrees: number) {
    setState((previous) => ({ ...previous, isLoading: true, error: null }));
    try {
      const result = await parseSentence({ text: sentence, maxTrees });
      setState({ result, isLoading: false, error: null });
    } catch (error) {
      setState({
        result: null,
        isLoading: false,
        error: error instanceof Error ? error.message : "Неизвестная ошибка"
      });
    }
  }

  return { ...state, submit };
}
