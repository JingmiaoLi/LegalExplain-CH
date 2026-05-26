export type LlmMode = "prompt_only" | "openai_compatible";

export interface AskRequest {
  query: string;
  llm_mode: LlmMode;
  top_k: number;
  candidate_k: number;
  enable_reranker: boolean;
}

export interface SourceItem {
  source_label: string;
  article_number: string;
  title_path: string;
  text: string;
  score: number;
  rank: number | null;
  chunk_id: string;
  metadata: Record<string, unknown>;
}

export interface AskResponse {
  answer: string;
  sources: SourceItem[];
  debug_info: Record<string, unknown>;
}

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

export async function askLawRag(payload: AskRequest): Promise<AskResponse> {
  const response = await fetch(`${API_BASE_URL}/ask`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Request failed: ${response.status} ${errorText}`);
  }

  return response.json() as Promise<AskResponse>;
}