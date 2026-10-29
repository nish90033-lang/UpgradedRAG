import { useRef, useState } from "react"
import { errorMessage } from "../api"
import { describeUpload, isBlocked } from "../format"

function UploadCard({ api, onChanged }) {
  const [mode, setMode] = useState("pdf")
  const [file, setFile] = useState(null)
  const [name, setName] = useState("")
  const [text, setText] = useState("")
  const [busy, setBusy] = useState(false)
  const [status, setStatus] = useState(null)
  const fileRef = useRef(null)

  async function run(action, reset) {
    setBusy(true)
    setStatus(null)
    try {
      const res = await action()
      const flagged = res.patterns?.length
        ? ` Suspected injection removed (${res.patterns.join(", ")}). This document is blocked from answers.`
        : ""
      setStatus({ success: true, warn: Boolean(flagged), message: describeUpload(res) + flagged })
      reset()
      onChanged()
    } catch (err) {
      setStatus({ success: false, message: errorMessage(err, "Upload failed.") })
    } finally {
      setBusy(false)
    }
  }

  const uploadPdf = () =>
    run(() => api.uploadPdf(file, name.trim() || undefined), () => {
      setFile(null)
      setName("")
      if (fileRef.current) fileRef.current.value = ""
    })
  const uploadText = () =>
    run(() => api.uploadText(name.trim() || "Untitled text", text.trim()), () => {
      setText("")
      setName("")
    })

  const words = text.trim().split(/\s+/).filter(Boolean).length

  return (
    <div className="card">
      <p className="card-label">Upload Document</p>
      <div className="tab-row" style={{ marginBottom: 16 }}>
        <button className={`tab-btn ${mode === "pdf" ? "active" : ""}`} onClick={() => { setMode("pdf"); setStatus(null) }}>
          PDF file
        </button>
        <button className={`tab-btn ${mode === "text" ? "active" : ""}`} onClick={() => { setMode("text"); setStatus(null) }}>
          Paste text
        </button>
      </div>

      {mode === "pdf" ? (
        <div className="upload-row">
          <input ref={fileRef} type="file" accept=".pdf" aria-label="PDF file" className="file-input"
            onChange={e => setFile(e.target.files[0] || null)} />
          <button className="primary-btn" onClick={uploadPdf} disabled={!file || busy}>
            {busy ? <><span className="spinner" /> Indexing…</> : "Upload & index"}
          </button>
        </div>
      ) : (
        <div className="stack">
          <input className="field-input" placeholder="Document name" aria-label="Document name"
            value={name} onChange={e => setName(e.target.value)} />
          <textarea className="field-input mono" rows={8} aria-label="Document text"
            placeholder="Paste notes, articles, or any text to index."
            value={text} onChange={e => setText(e.target.value)} />
          <div className="row-between">
            <span className="word-count">{words} words</span>
            <button className="primary-btn" onClick={uploadText} disabled={!text.trim() || busy}>
              {busy ? <><span className="spinner" /> Indexing…</> : "Index text"}
            </button>
          </div>
        </div>
      )}

      {status && (
        <div className={`status-strip ${status.success ? (status.warn ? "warn" : "success") : "error"}`} role="status">
          {status.message}
        </div>
      )}
    </div>
  )
}

function DocRow({ doc, api, onChanged, setNote }) {
  const [sharing, setSharing] = useState(false)
  const [email, setEmail] = useState("")
  const blocked = isBlocked(doc)

  async function remove() {
    if (!window.confirm(`Delete "${doc.name}" and all its indexed chunks?`)) return
    try {
      await api.deleteDocument(doc.id)
      setNote({ success: true, message: `Deleted "${doc.name}".` })
      onChanged()
    } catch (err) {
      setNote({ success: false, message: errorMessage(err, "Could not delete the document.") })
    }
  }

  async function share() {
    try {
      await api.shareDocument(doc.id, email.trim())
      setNote({ success: true, message: `Shared "${doc.name}" with ${email.trim()}.` })
      setSharing(false)
      setEmail("")
    } catch (err) {
      setNote({ success: false, message: errorMessage(err, "Could not share. Check the email address.") })
    }
  }

  return (
    <div className="doc-row">
      <div className="doc-main">
        <div className="doc-name">{doc.name}</div>
        <div className="doc-meta">
          {doc.chunk_count} chunks · {new Date(doc.created_at).toLocaleDateString()}
          {doc.shared && " · shared with you"}
          {doc.patterns?.length > 0 && ` · flagged: ${doc.patterns.join(", ")}`}
        </div>
        {sharing && (
          <div className="share-row">
            <input className="field-input" type="email" placeholder="Their email address" aria-label="Share with email"
              value={email} onChange={e => setEmail(e.target.value)} onKeyDown={e => e.key === "Enter" && email && share()} />
            <button className="primary-btn small" onClick={share} disabled={!email.trim()}>Share</button>
            <button className="link-btn small" onClick={() => setSharing(false)}>Cancel</button>
          </div>
        )}
      </div>
      <div className="doc-actions">
        <span className={`badge ${blocked ? "danger" : ""}`}>
          {blocked ? `Blocked · ${Math.round(doc.trust_score)}% trust` : `${Math.round(doc.trust_score)}% trust`}
        </span>
        {!doc.shared && (
          <>
            <button className="ghost-btn" onClick={() => setSharing(s => !s)} aria-label={`Share ${doc.name}`}>Share</button>
            <button className="ghost-btn danger" onClick={remove} aria-label={`Delete ${doc.name}`}>Delete</button>
          </>
        )}
      </div>
    </div>
  )
}

export default function DocumentsPage({ api, documents, loading, onChanged }) {
  const [note, setNote] = useState(null)
  return (
    <section>
      <div className="page-header">
        <h2 className="page-title">Documents</h2>
        <p className="page-sub">Upload PDFs or paste text to index and query.</p>
      </div>
      <UploadCard api={api} onChanged={onChanged} />
      {note && <div className={`status-strip ${note.success ? "success" : "error"}`} role="status">{note.message}</div>}
      {loading ? (
        <div className="state-placeholder">Loading documents…</div>
      ) : documents.length === 0 ? (
        <div className="state-placeholder">No documents yet. Upload a PDF or paste text above to get started.</div>
      ) : (
        <div className="doc-list">
          {documents.map(doc => (
            <DocRow key={doc.id} doc={doc} api={api} onChanged={onChanged} setNote={setNote} />
          ))}
        </div>
      )}
    </section>
  )
}
