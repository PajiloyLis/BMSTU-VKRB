import type { ParseRequest, ParseResponse } from "./types";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "";

export async function parseSentence(request: ParseRequest): Promise<ParseResponse> {
  const response = await fetch(`${API_BASE_URL}/api/parse`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(request)
  });

  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `HTTP ${response.status}`);
  }

  return response.json() as Promise<ParseResponse>;
}
