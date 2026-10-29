import { useState } from "react"
import { errorMessage } from "../api"
import Logo from "./Logo"

export default function AuthScreen({ api, onSession, notice }) {
  const [mode, setMode] = useState("login")
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [showPassword, setShowPassword] = useState(false)
  const [remember, setRemember] = useState(false)
  const [error, setError] = useState("")
  const [loading, setLoading] = useState(false)

  async function submit() {
    setLoading(true)
    setError("")
    try {
      const data = mode === "login" ? await api.login(email, password) : await api.signup(email, password)
      onSession({ token: data.token, user: data.user }, remember)
    } catch (err) {
      setError(errorMessage(err, "Authentication failed. Please try again."))
    } finally {
      setLoading(false)
    }
  }

  function switchMode(next) {
    setMode(next)
    setError("")
  }

  return (
    <div className="auth-page">
      <div className="auth-panel">
        <div className="auth-brand">
          <Logo className="auth-logo" />
          <h1 className="auth-wordmark">SmartRAG</h1>
          <p className="auth-tagline">Secure document intelligence</p>
        </div>

        <div className="tab-row">
          <button className={`tab-btn ${mode === "login" ? "active" : ""}`} onClick={() => switchMode("login")}>
            Sign In
          </button>
          <button className={`tab-btn ${mode === "signup" ? "active" : ""}`} onClick={() => switchMode("signup")}>
            Create Account
          </button>
        </div>

        {notice && <div className="info-box" role="status">{notice}</div>}

        <div className="field-group">
          <label className="field-label" htmlFor="email">Email address</label>
          <input
            id="email" type="email" className="field-input" value={email} autoComplete="email"
            onChange={e => setEmail(e.target.value)} onKeyDown={e => e.key === "Enter" && submit()}
          />
        </div>

        <div className="field-group">
          <label className="field-label" htmlFor="password">Password</label>
          <div className="pw-wrap">
            <input
              id="password" type={showPassword ? "text" : "password"} className="field-input"
              value={password} autoComplete={mode === "login" ? "current-password" : "new-password"}
              onChange={e => setPassword(e.target.value)} onKeyDown={e => e.key === "Enter" && submit()}
            />
            <button
              type="button" className="pw-toggle" onClick={() => setShowPassword(v => !v)}
              aria-label={showPassword ? "Hide password" : "Show password"}
            >
              {showPassword ? "Hide" : "Show"}
            </button>
          </div>
          {mode === "signup" && <p className="field-hint">At least 8 characters.</p>}
        </div>

        <label className="checkbox-row">
          <input type="checkbox" className="checkbox" checked={remember} onChange={e => setRemember(e.target.checked)} />
          <span className="checkbox-label">Keep me signed in on this device</span>
        </label>

        {error && <div className="auth-error" role="alert">{error}</div>}

        <button className="primary-btn full-width" onClick={submit} disabled={loading || !email || !password}>
          {loading ? <span className="spinner" /> : mode === "login" ? "Sign In" : "Create Account"}
        </button>
      </div>
    </div>
  )
}
