import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import "./App.css"
import { createApi } from "./api"
import { clearSession, loadSession, saveSession } from "./session"
import AuthScreen from "./components/AuthScreen"
import ChatPage from "./components/ChatPage"
import DocumentsPage from "./components/DocumentsPage"
import HistoryPage from "./components/HistoryPage"
import Logo from "./components/Logo"

const PAGES = [
  { id: "documents", label: "Documents" },
  { id: "chat", label: "Chat" },
  { id: "history", label: "History" },
]

export default function App({ apiFactory = createApi }) {
  const [session, setSession] = useState(() => loadSession())
  const [page, setPage] = useState("documents")
  const [documents, setDocuments] = useState([])
  const [docsLoading, setDocsLoading] = useState(false)
  const [notice, setNotice] = useState("")
  const tokenRef = useRef(session?.token ?? null)

  const signOut = useCallback((message = "") => {
    clearSession()
    tokenRef.current = null
    setSession(null)
    setDocuments([])
    setNotice(message)
  }, [])

  const api = useMemo(
    () => apiFactory(() => tokenRef.current, () => signOut("Your session expired. Please sign in again.")),
    [apiFactory, signOut],
  )

  const refreshDocuments = useCallback(async () => {
    setDocsLoading(true)
    try {
      setDocuments(await api.listDocuments())
    } catch {
      /* a 401 is handled by the client, other errors leave the previous list in place */
    } finally {
      setDocsLoading(false)
    }
  }, [api])

  useEffect(() => {
    if (session) refreshDocuments()
  }, [session, refreshDocuments])

  function startSession(next, remember) {
    tokenRef.current = next.token
    saveSession(next, remember)
    setNotice("")
    setSession(next)
  }

  if (!session) return <AuthScreen api={api} onSession={startSession} notice={notice} />

  return (
    <div className="app-layout">
      <aside className="sidebar">
        <div className="sidebar-brand">
          <Logo />
          <span>SmartRAG</span>
        </div>
        <nav className="sidebar-nav">
          {PAGES.map(item => (
            <button key={item.id} className={`nav-link ${page === item.id ? "active" : ""}`} onClick={() => setPage(item.id)}>
              {item.label}
            </button>
          ))}
        </nav>
        <div className="sidebar-footer">
          <div className="user-pill">
            <span className="user-avatar">{session.user.email[0].toUpperCase()}</span>
            <span className="user-email">{session.user.email}</span>
          </div>
          <button className="signout-btn" onClick={() => signOut()}>Sign out</button>
        </div>
      </aside>

      <main className="main-content">
        {page === "documents" && (
          <DocumentsPage api={api} documents={documents} loading={docsLoading} onChanged={refreshDocuments} />
        )}
        {page === "chat" && <ChatPage api={api} documents={documents} />}
        {page === "history" && <HistoryPage api={api} />}
      </main>
    </div>
  )
}
