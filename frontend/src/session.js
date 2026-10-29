const KEY = "smartrag_session"

export function loadSession() {
  for (const store of [sessionStorage, localStorage]) {
    const raw = store.getItem(KEY)
    if (!raw) continue
    try {
      const s = JSON.parse(raw)
      if (s?.token && s?.user?.email) return s
    } catch {
      /* fall through and clear */
    }
    store.removeItem(KEY)
  }
  return null
}

export function saveSession(session, remember) {
  clearSession()
  ;(remember ? localStorage : sessionStorage).setItem(KEY, JSON.stringify(session))
}

export function clearSession() {
  sessionStorage.removeItem(KEY)
  localStorage.removeItem(KEY)
}
