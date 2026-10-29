import { useState } from "react"
import { errorMessage } from "../api"
import { blockedMessage, isBlocked, splitAnswer, timingSummary } from "../format"

function Answer({ result }) {
  const parts = splitAnswer(result.answer, result.sources)
  return (
    <p className="answer-text">
      {parts.map((p, i) =>
        p.type === "text" ? (
          <span key={i}>{p.text}</span>
        ) : p.valid ? (
          <a key={i} className="cite" href={`#source-${p.n}`} aria-label={`Source ${p.n}`}>{p.n}</a>
        ) : (
          <span key={i} className="cite invalid" title="The model cited a source that does not exist">{p.n}</span>
        ),
      )}
    </p>
  )
}

function ResultCard({ result }) {
  const blockedText = !result.answerable ? blockedMessage(result.blocked_reason) : null
  const label = result.answerable ? "Answered" : result.blocked_reason === "all_blocked" ? "Blocked" : "Not answered"
  const tone = result.answerable ? "success" : result.blocked_reason === "all_blocked" ? "danger" : "muted"
  const hasInvalid = result.invalid_citations?.length > 0
  const uncited = result.answerable && (result.citations?.length ?? 0) === 0

  return (
    <div className="card result-card">
      <span className={`badge ${tone}`}>{label}</span>
      {result.rewritten_query && (
        <p className="notice" data-testid="rewritten">Searched for: <em>{result.rewritten_query}</em></p>
      )}
      {result.answerable ? <Answer result={result} /> : <p className="answer-text">{blockedText ?? result.answer}</p>}
      {hasInvalid && <p className="notice warn">Some citations point to sources that do not exist. Check this answer.</p>}
      {uncited && <p className="notice warn">This answer has no citations. Treat it with care.</p>}
      {!result.llm_called && !result.answerable && <p className="notice">No model was called for this question.</p>}

      {result.sources?.length > 0 && result.answerable && (
        <div className="sources">
          <p className="sources-label">Sources</p>
          {result.sources.map(src => (
            <div key={src.chunk_id} id={`source-${src.n}`} className="source-card">
              <div className="source-header">
                <span>[{src.n}] {src.doc_name}</span>
                <span>Chunk {src.chunk_index}</span>
              </div>
              <div className="source-meta">Score {src.score.toFixed(3)} · Trust {Math.round(src.trust)}%</div>
              <p className="source-excerpt">{src.excerpt}</p>
            </div>
          ))}
        </div>
      )}
      {result.timings_ms && <p className="timing">{timingSummary(result.timings_ms)}</p>}
    </div>
  )
}

const MAX_TURNS = 3

export default function ChatPage({ api, documents }) {
  const [turns, setTurns] = useState([])
  const [selected, setSelected] = useState([])
  const [question, setQuestion] = useState("")
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState("")

  function newConversation() {
    setTurns([])
    setResult(null)
    setError("")
  }

  const toggle = id => setSelected(prev => (prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]))
  const answerable = documents.filter(d => !isBlocked(d))

  async function ask() {
    const q = question.trim()
    if (!q) return
    setLoading(true)
    setResult(null)
    setError("")
    try {
      const res = await api.query(q, selected, turns.slice(-MAX_TURNS))
      setResult(res)
      // Only answered turns are remembered, so refusals do not pollute later rewrites.
      if (res.answerable) setTurns(prev => [...prev, { question: q, answer: res.answer }])
      setQuestion("")
    } catch (err) {
      setError(errorMessage(err, "Something went wrong. Please try again."))
    } finally {
      setLoading(false)
    }
  }

  return (
    <section>
      <div className="page-header">
        <h2 className="page-title">Chat</h2>
        <p className="page-sub">Ask questions about your indexed documents.</p>
      </div>

      <div className="card">
        <div className="field-group">
          <div className="doc-select-header">
            <span className="field-label">
              Documents
              <span className="doc-select-count">
                {selected.length === 0 ? "— searching all" : `${selected.length} of ${documents.length} selected`}
              </span>
            </span>
            <div className="doc-select-actions">
              <button className="link-btn small" onClick={() => setSelected(answerable.map(d => d.id))}>Select all</button>
              <span className="doc-select-divider">·</span>
              <button className="link-btn small" onClick={() => setSelected([])}>Clear</button>
            </div>
          </div>
          <div className="doc-checkbox-list">
            {documents.length === 0 ? (
              <div className="state-placeholder">No documents uploaded yet.</div>
            ) : (
              documents.map(doc => (
                <label key={doc.id} className={`doc-checkbox-row ${selected.includes(doc.id) ? "checked" : ""}`}>
                  <input type="checkbox" checked={selected.includes(doc.id)} onChange={() => toggle(doc.id)}
                    aria-label={doc.name} />
                  <div className="doc-checkbox-info">
                    <span className="doc-checkbox-name">{doc.name}</span>
                    <span className="doc-checkbox-meta">{doc.chunk_count} chunks</span>
                  </div>
                  <span className={`badge ${isBlocked(doc) ? "danger" : ""}`}>
                    {isBlocked(doc) ? "Blocked" : `${Math.round(doc.trust_score)}% trust`}
                  </span>
                </label>
              ))
            )}
          </div>
        </div>

        <div className="field-group">
          <div className="row-between">
            <label className="field-label" htmlFor="question">Question</label>
            {turns.length > 0 && (
              <button className="link-btn small" onClick={newConversation}>
                New conversation ({turns.length} {turns.length === 1 ? "turn" : "turns"} remembered)
              </button>
            )}
          </div>
          <div className="chat-row">
            <input id="question" className="field-input" value={question}
              onChange={e => setQuestion(e.target.value)} onKeyDown={e => e.key === "Enter" && ask()}
              placeholder={selected.length === 0 ? "Ask anything across all documents…"
                : `Ask anything across ${selected.length} selected document${selected.length > 1 ? "s" : ""}…`} />
            <button className="primary-btn" onClick={ask} disabled={!question.trim() || loading}>
              {loading ? <span className="spinner" /> : "Ask"}
            </button>
          </div>
        </div>
      </div>

      {error && <div className="status-strip error" role="alert">{error}</div>}
      {result && <ResultCard result={result} />}
    </section>
  )
}
