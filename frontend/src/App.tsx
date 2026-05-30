import { useMemo, useState } from "react";
import type { ReactNode } from "react";

import { askLawRag } from "./api";
import type { AskResponse, LlmMode, SourceItem } from "./api";

import { LegalReasoningGraph } from "./components/legal_reasoning_graph/legal_reasoning_graph";

import "./App.css";

const exampleQuestions = [
  "Can my employer dismiss me immediately without notice?",
  "Can I leave my job immediately without notice?",
  "What does Swiss employment law say about salary payment?",
  "What are the employee's duties of loyalty and care?",
];

function getSourceUrl(source: SourceItem): string | null {
  const sourceUrl = source.metadata?.source_url;

  if (typeof sourceUrl === "string" && sourceUrl.length > 0) {
    return sourceUrl;
  }

  return null;
}

function renderAnswerWithSourceLinks(
  answer: string,
  sources: AskResponse["sources"],
  keyPrefix = "answer",
): ReactNode[] {
  const sourceUrlByArticle = new Map<string, string>();

  for (const source of sources) {
    const sourceUrl = getSourceUrl(source);

    if (!sourceUrl) {
      continue;
    }

    if (source.article_number) {
      sourceUrlByArticle.set(source.article_number.toLowerCase(), sourceUrl);
    }

    if (source.source_label) {
      sourceUrlByArticle.set(source.source_label.toLowerCase(), sourceUrl);
    }
  }

  const articlePattern = /\bArt\.?\s+\d+[a-z]?(?:\s+bis)?\b/g;
  const parts: ReactNode[] = [];
  let lastIndex = 0;

  for (const match of answer.matchAll(articlePattern)) {
    const matchedText = match[0];
    const matchIndex = match.index ?? 0;

    if (matchIndex > lastIndex) {
      parts.push(answer.slice(lastIndex, matchIndex));
    }

    const normalizedArticle = matchedText
      .toLowerCase()
      .replace("art.", "")
      .replace("art", "")
      .trim();

    const url =
      sourceUrlByArticle.get(normalizedArticle) ??
      sourceUrlByArticle.get(matchedText.toLowerCase());

    if (url) {
      parts.push(
        <a
          key={`${keyPrefix}-${matchedText}-${matchIndex}`}
          href={url}
          target="_blank"
          rel="noreferrer"
          className="answerCitationLink"
        >
          {matchedText}
        </a>,
      );
    } else {
      parts.push(matchedText);
    }

    lastIndex = matchIndex + matchedText.length;
  }

  if (lastIndex < answer.length) {
    parts.push(answer.slice(lastIndex));
  }

  return parts;
}

function splitAnswerForDisplay(answer: string): {
  shortAnswer: string;
  detailedAnswer: string;
} {
  const cleanedAnswer = answer.trim();

  if (!cleanedAnswer) {
    return {
      shortAnswer: "",
      detailedAnswer: "",
    };
  }

  if (cleanedAnswer.startsWith("PROMPT_ONLY_MODE")) {
    return {
      shortAnswer: cleanedAnswer,
      detailedAnswer: "",
    };
  }

  const paragraphs = cleanedAnswer
    .split(/\n\s*\n/)
    .map((paragraph) => paragraph.trim())
    .filter(Boolean);

  if (paragraphs.length <= 1) {
    const sentences = cleanedAnswer.match(/[^.!?]+[.!?]+(?:\s|$)/g);

    if (!sentences || sentences.length <= 1) {
      return {
        shortAnswer: cleanedAnswer,
        detailedAnswer: "",
      };
    }

    return {
      shortAnswer: sentences[0].trim(),
      detailedAnswer: cleanedAnswer.slice(sentences[0].length).trim(),
    };
  }

  return {
    shortAnswer: paragraphs[0],
    detailedAnswer: paragraphs.slice(1).join("\n\n"),
  };
}

function AnswerDisplay({ result }: { result: AskResponse }) {
  const { shortAnswer, detailedAnswer } = useMemo(
    () => splitAnswerForDisplay(result.answer),
    [result.answer],
  );

  const isPromptOnlyMode = result.answer.startsWith("PROMPT_ONLY_MODE");

  if (isPromptOnlyMode) {
    return (
      <div className="answerText">
        {renderAnswerWithSourceLinks(result.answer, result.sources, "prompt")}
      </div>
    );
  }

  return (
    <>
      {shortAnswer && (
        <div className="answerText answerTextPrimary">
          {renderAnswerWithSourceLinks(
            shortAnswer,
            result.sources,
            "short-answer",
          )}
        </div>
      )}

      {detailedAnswer && (
        <div className="answerText answerTextDetailBeforeGraph">
          {renderAnswerWithSourceLinks(
            detailedAnswer,
            result.sources,
            "detailed-answer",
          )}
        </div>
      )}

      <LegalReasoningGraph reasoningMap={result.reasoning_map} />
    </>
  );
}

function App() {
  const [query, setQuery] = useState(exampleQuestions[0]);
  const [submittedQuery, setSubmittedQuery] = useState("");
  const [llmMode, setLlmMode] = useState<LlmMode>("openai_compatible");
  const [result, setResult] = useState<AskResponse | null>(null);
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const hasConversation = Boolean(submittedQuery || isLoading || result || error);

  async function handleAsk() {
    const cleanedQuery = query.trim();

    if (!cleanedQuery) {
      setError("Please enter a question.");
      return;
    }

    setIsLoading(true);
    setError("");
    setResult(null);
    setSubmittedQuery(cleanedQuery);

    try {
      const response = await askLawRag({
        query: cleanedQuery,
        llm_mode: llmMode,
        top_k: 5,
        candidate_k: 20,
        enable_reranker: true,
      });

      setResult(response);
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "Unknown request error.",
      );
    } finally {
      setIsLoading(false);
    }
  }

  function handleExampleClick(example: string) {
    setQuery(example);
  }

  return (
    <main className="appRoot">
      <header className="topBar">
        <div>
          <p className="eyebrow">LegalExplain-CH</p>
          <h1>Swiss employment-law assistant</h1>
        </div>

        <div className="topBarMeta">
          <label className="modeSelectLabel">
            Mode
            <select
              value={llmMode}
              onChange={(event) => setLlmMode(event.target.value as LlmMode)}
              className="modeSelect"
            >
              <option value="prompt_only">prompt_only</option>
              <option value="openai_compatible">openai_compatible</option>
            </select>
          </label>
        </div>
      </header>

      {!hasConversation && (
        <section className="landingPanel">
          <div className="welcomeBadge">Source-grounded legal RAG</div>
          <h2>Ask a Swiss employment-law question.</h2>
          <p>
            Get a clear answer grounded in Swiss legal sources, with article
            references linked back to official Fedlex pages.
          </p>

          <div className="welcomeExamples">
            {exampleQuestions.map((example) => (
              <button
                key={example}
                type="button"
                className="welcomeExampleButton"
                onClick={() => handleExampleClick(example)}
              >
                {example}
              </button>
            ))}
          </div>
        </section>
      )}

      {hasConversation && (
        <section className="workspace singleColumnWorkspace">
          <section className="conversationPanel">
            {submittedQuery && (
              <div className="messageRow userRow">
                <div className="avatar userAvatar">You</div>
                <div className="messageBubble userBubble">{submittedQuery}</div>
              </div>
            )}

            {isLoading && (
              <div className="messageRow assistantRow">
                <div className="avatar assistantAvatar">AI</div>
                <div className="messageBubble assistantBubble loadingBubble">
                  <span className="loadingDot" />
                  Searching legal sources and preparing an answer...
                </div>
              </div>
            )}

            {result && (
              <div className="messageRow assistantRow">
                <div className="avatar assistantAvatar">AI</div>
                <div className="messageBubble assistantBubble">
                  <div className="messageHeader">
                    {result.answer.startsWith("PROMPT_ONLY_MODE")
                      ? "Grounded prompt preview"
                      : "Answer"}
                  </div>

                  <AnswerDisplay result={result} />
                </div>
              </div>
            )}

            {error && (
              <div className="messageRow assistantRow">
                <div className="avatar assistantAvatar">AI</div>
                <div className="messageBubble errorBubble">{error}</div>
              </div>
            )}
          </section>
        </section>
      )}

      <footer className="composerBar">
        <div className="composerInner">
          <textarea
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            rows={1}
            className="chatInput"
            placeholder="Ask about Swiss employment law..."
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                void handleAsk();
              }
            }}
          />

          <button
            type="button"
            className="sendButton"
            onClick={handleAsk}
            disabled={isLoading}
          >
            {isLoading ? "..." : "Ask"}
          </button>
        </div>
      </footer>
    </main>
  );
}

export default App;