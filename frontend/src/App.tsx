import { useState } from "react";
import type { ReactNode } from "react";

import { askLawRag } from "./api";
import type { AskResponse, LlmMode, SourceItem } from "./api";

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

function findSourceByArticle(
  sources: AskResponse["sources"],
  articleNumber: string,
): SourceItem | null {
  return (
    sources.find(
      (source) => source.article_number.toLowerCase() === articleNumber,
    ) ?? null
  );
}

function ArticleChip({ source }: { source: SourceItem }) {
  const sourceUrl = getSourceUrl(source);

  if (sourceUrl) {
    return (
      <a
        href={sourceUrl}
        target="_blank"
        rel="noreferrer"
        className="graphArticleChip"
      >
        {source.source_label}
      </a>
    );
  }

  return <span className="graphArticleChip">{source.source_label}</span>;
}

function LegalReasoningGraph({
  sources,
}: {
  sources: AskResponse["sources"];
}) {
  const art337 = findSourceByArticle(sources, "337");
  const art337c = findSourceByArticle(sources, "337c");
  const art336c = findSourceByArticle(sources, "336c");

  if (!art337 && !art337c && !art336c) {
    return null;
  }

  return (
    <section className="legalGraph">
      <div className="legalGraphHeader">
        <span className="legalGraphBadge">Legal reasoning map</span>
        <p>
          A visual path showing the main legal condition and possible outcomes.
        </p>
      </div>

      <div className="graphCanvas">

        <div className="graphNode graphNodeIssue">
          <span className="graphNodeLabel">Legal issue</span>
          <h3>Immediate termination by the employer</h3>
          <p>
            The key issue is whether the employer can end the employment
            relationship immediately without ordinary notice.
          </p>
        </div>

        <div className="graphArrow">↓</div>

        {art337 && (
          <div className="graphNode graphNodeCondition">
            <span className="graphNodeLabel">Condition</span>
            <h3>Is there good cause?</h3>
            <p>
              Immediate termination depends on whether good cause exists. Good
              cause means that continuing the employment relationship would be
              unreasonable in good faith.
            </p>
            <ArticleChip source={art337} />
          </div>
        )}

        <div className="graphBranches">
          <div className="graphBranch">
            <div className="branchLabel branchYes">Yes</div>
            <div className="graphNode graphNodeOutcome">
              <span className="graphNodeLabel">Possible outcome</span>
              <h3>Immediate termination may be justified</h3>
              <p>
                If good cause exists, immediate termination may be justified,               
                subject to the specific facts and court assessment.
              </p>
              {art337 && <ArticleChip source={art337} />}
            </div>
          </div>

          <div className="graphBranch">
            <div className="branchLabel branchNo">No</div>
            <div className="graphNode graphNodeOutcome graphNodeConsequence">
              <span className="graphNodeLabel">Consequence</span>
              <h3>Damages or compensation may apply</h3>
              <p>
                If the employer terminates immediately without good cause,
                damages and possible compensation may apply.
              </p>
              {art337c && <ArticleChip source={art337c} />}
            </div>
          </div>
        </div>

        {art336c && (
          <>
            <div className="graphArrow">↓</div>

            <div className="graphNode graphNodeRelated">
              <span className="graphNodeLabel">Additional check</span>
              <h3>Protected timing rules may matter</h3>
              <p>
                If the timing falls within a protected period, termination restrictions may
                also need to be checked.
              </p>
              <ArticleChip source={art336c} />
            </div>
          </>
        )}
      </div>
    </section>
  );
}


function renderAnswerWithSourceLinks(
  answer: string,
  sources: AskResponse["sources"],
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
          key={`${matchedText}-${matchIndex}`}
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

                  <div className="answerText">
                    {renderAnswerWithSourceLinks(result.answer, result.sources)}
                  </div>

                  <LegalReasoningGraph sources={result.sources} />

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