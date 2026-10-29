import axios from "axios"

export const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000"

/** Blocked chunks never reach retrieval. Mirrors Settings.trust_threshold in the backend. */
export const TRUST_THRESHOLD = 50

/** Turn an axios error into a message a person can read. */
export function errorMessage(err, fallback) {
  const data = err?.response?.data
  const detail = data?.detail
  if (typeof detail === "string") return detail
  if (Array.isArray(detail) && detail.length) {
    return detail.map(d => d.msg).filter(Boolean).join(". ") || fallback
  }
  if (!err?.response) return "Cannot reach the server. Check your connection and try again."
  return fallback
}

/**
 * Build the API client. getToken supplies the current bearer token and onUnauthorized runs when
 * the server rejects it, so an expired session returns the user to the sign-in screen.
 */
export function createApi(getToken, onUnauthorized, baseURL = API_URL) {
  const http = axios.create({ baseURL })
  http.interceptors.request.use(config => {
    const token = getToken()
    if (token) config.headers.Authorization = `Bearer ${token}`
    return config
  })
  http.interceptors.response.use(
    r => r,
    err => {
      const isAuthCall = err?.config?.url?.startsWith("/auth/")
      if (err?.response?.status === 401 && !isAuthCall) onUnauthorized?.()
      return Promise.reject(err)
    },
  )
  return {
    login: (email, password) => http.post("/auth/login", { email, password }).then(r => r.data),
    signup: (email, password) => http.post("/auth/signup", { email, password }).then(r => r.data),
    listDocuments: () => http.get("/documents").then(r => r.data.documents ?? []),
    uploadPdf: (file, name) => {
      const form = new FormData()
      form.append("file", file)
      if (name) form.append("name", name)
      return http.post("/documents/pdf", form).then(r => r.data)
    },
    uploadText: (name, text) => http.post("/documents/text", { name, text }).then(r => r.data),
    deleteDocument: id => http.delete(`/documents/${id}`).then(() => undefined),
    shareDocument: (id, email) =>
      http.post(`/documents/${id}/share`, { email }).then(() => undefined),
    query: (question, docIds, history = []) =>
      http
        .post("/query", {
          question,
          doc_ids: docIds && docIds.length ? docIds : null,
          history,
        })
        .then(r => r.data),
    history: () => http.get("/history").then(r => r.data.history ?? []),
  }
}
