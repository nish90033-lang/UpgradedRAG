import { useEffect, useState } from "react"
import { errorMessage } from "../api"

export default function HistoryPage({ api }) {
  const [items, setItems] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState("")

  useEffect(() => {
    let live = true
    api.history()
      .then(h => live && setItems(h))
      .catch(err => live && setError(errorMessage(err, "Could not load history.")))
      .finally(() => live && setLoading(false))
    return () => { live = false }
  }, [api])

  return (
    <section>
      <div className="page-header">
        <h2 className="page-title">History</h2>
        <p className="page-sub">Your past queries and answers.</p>
      </div>
      {error && <div className="status-strip error" role="alert">{error}</div>}
      {loading ? (
        <div className="state-placeholder">Loading history…</div>
      ) : items.length === 0 ? (
        <div className="state-placeholder">No history yet. Ask a question in Chat to get started.</div>
      ) : (
        <div className="history-list">
          {items.map(item => (
            <div key={item.id} className="history-card">
              <p className="history-question">{item.question}</p>
              <p className="history-answer">{item.answer}</p>
              <span className="history-date">
                {new Date(item.created_at).toLocaleString()}
                {!item.answerable && " · not answered"}
                {item.sources?.length > 0 && ` · ${item.sources.length} sources`}
              </span>
            </div>
          ))}
        </div>
      )}
    </section>
  )
}
