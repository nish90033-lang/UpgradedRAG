import { render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { beforeEach, describe, expect, it, vi } from "vitest"
import App from "./App"

const DOC = { id: "d1", name: "plants.txt", trust_score: 100, patterns: [], chunk_count: 5,
  source_type: "text", shared: false, created_at: "2026-10-01T10:00:00Z" }
const BAD = { ...DOC, id: "d2", name: "evil.txt", trust_score: 40, patterns: ["ignore_previous"] }
const SHARED = { ...DOC, id: "d3", name: "theirs.txt", shared: true }

let api, captured

function makeApi(overrides = {}) {
  return {
    login: vi.fn().mockResolvedValue({ token: "tok", user: { id: "u1", email: "me@x.y" } }),
    signup: vi.fn().mockResolvedValue({ token: "tok", user: { id: "u1", email: "me@x.y" } }),
    listDocuments: vi.fn().mockResolvedValue([DOC, BAD, SHARED]),
    uploadPdf: vi.fn(),
    uploadText: vi.fn().mockResolvedValue({ status: "created", document: DOC, chunks_total: 5, chunks_embedded: 5, patterns: [] }),
    deleteDocument: vi.fn().mockResolvedValue(undefined),
    shareDocument: vi.fn().mockResolvedValue(undefined),
    query: vi.fn(),
    history: vi.fn().mockResolvedValue([]),
    ...overrides,
  }
}

function setup(overrides) {
  api = makeApi(overrides)
  captured = {}
  const apiFactory = (getToken, onUnauthorized) => { captured = { getToken, onUnauthorized }; return api }
  return { user: userEvent.setup(), ...render(<App apiFactory={apiFactory} />) }
}

const goto = (user, name) => user.click(screen.getByRole("button", { name }))

beforeEach(() => vi.restoreAllMocks())

describe("authentication", () => {
  it("signs in, stores the session for the tab only, and loads documents", async () => {
    const { user } = setup()
    const buttons = screen.getAllByRole("button", { name: "Sign In" })
    await user.type(screen.getByLabelText("Email address"), "me@x.y")
    await user.type(screen.getByLabelText("Password"), "password123")
    await user.click(buttons[buttons.length - 1])
    expect(await screen.findByText("plants.txt")).toBeInTheDocument()
    expect(api.login).toHaveBeenCalledWith("me@x.y", "password123")
    expect(sessionStorage.getItem("smartrag_session")).toContain("tok")
    expect(localStorage.getItem("smartrag_session")).toBeNull()
    expect(captured.getToken()).toBe("tok")
  })

  it("remembers the session on the device when asked", async () => {
    const { user } = setup()
    await user.type(screen.getByLabelText("Email address"), "me@x.y")
    await user.type(screen.getByLabelText("Password"), "password123")
    await user.click(screen.getByLabelText("Keep me signed in on this device"))
    const buttons = screen.getAllByRole("button", { name: "Sign In" })
    await user.click(buttons[buttons.length - 1])
    await screen.findByText("plants.txt")
    expect(localStorage.getItem("smartrag_session")).toContain("tok")
  })

  it("shows the server's error message when sign-in fails", async () => {
    const { user } = setup({ login: vi.fn().mockRejectedValue({ response: { status: 401, data: { detail: "Incorrect email or password." } } }) })
    await user.type(screen.getByLabelText("Email address"), "me@x.y")
    await user.type(screen.getByLabelText("Password"), "wrongwrong")
    const buttons = screen.getAllByRole("button", { name: "Sign In" })
    await user.click(buttons[buttons.length - 1])
    expect(await screen.findByRole("alert")).toHaveTextContent("Incorrect email or password.")
  })

  it("creates an account in sign-up mode", async () => {
    const { user } = setup()
    await user.click(screen.getByRole("button", { name: "Create Account" }))
    expect(screen.getByText("At least 8 characters.")).toBeInTheDocument()
    await user.type(screen.getByLabelText("Email address"), "new@x.y")
    await user.type(screen.getByLabelText("Password"), "password123")
    const buttons = screen.getAllByRole("button", { name: "Create Account" })
    await user.click(buttons[buttons.length - 1])
    await screen.findByText("plants.txt")
    expect(api.signup).toHaveBeenCalledWith("new@x.y", "password123")
  })

  it("restores a stored session, and signs out cleanly", async () => {
    sessionStorage.setItem("smartrag_session", JSON.stringify({ token: "old", user: { id: "u1", email: "me@x.y" } }))
    const { user } = setup()
    expect(await screen.findByText("plants.txt")).toBeInTheDocument()
    expect(captured.getToken()).toBe("old")
    await user.click(screen.getByRole("button", { name: "Sign out" }))
    expect(screen.getByLabelText("Email address")).toBeInTheDocument()
    expect(sessionStorage.getItem("smartrag_session")).toBeNull()
  })

  it("ignores a corrupt stored session", () => {
    sessionStorage.setItem("smartrag_session", "{not json")
    setup()
    expect(screen.getByLabelText("Email address")).toBeInTheDocument()
    expect(sessionStorage.getItem("smartrag_session")).toBeNull()
  })

  it("returns to sign-in with a notice when the server rejects the token", async () => {
    sessionStorage.setItem("smartrag_session", JSON.stringify({ token: "old", user: { id: "u1", email: "me@x.y" } }))
    setup()
    await screen.findByText("plants.txt")
    captured.onUnauthorized()
    expect(await screen.findByText("Your session expired. Please sign in again.")).toBeInTheDocument()
    expect(screen.getByLabelText("Email address")).toBeInTheDocument()
  })
})

describe("documents", () => {
  beforeEach(() => sessionStorage.setItem("smartrag_session", JSON.stringify({ token: "t", user: { id: "u1", email: "me@x.y" } })))

  it("marks blocked documents and shows which patterns flagged them", async () => {
    setup()
    const row = (await screen.findByText("evil.txt")).closest(".doc-row")
    expect(within(row).getByText("Blocked · 40% trust")).toBeInTheDocument()
    expect(row).toHaveTextContent("flagged: ignore_previous")
  })

  it("indexes pasted text, reports the result, and refreshes the list", async () => {
    const { user } = setup()
    await screen.findByText("plants.txt")
    await user.click(screen.getByRole("button", { name: "Paste text" }))
    await user.type(screen.getByLabelText("Document name"), "notes")
    await user.type(screen.getByLabelText("Document text"), "some words to index")
    expect(screen.getByText("4 words")).toBeInTheDocument()
    await user.click(screen.getByRole("button", { name: "Index text" }))
    expect(api.uploadText).toHaveBeenCalledWith("notes", "some words to index")
    expect(await screen.findByText(/Indexed "plants.txt": 5 chunks, trust 100%/)).toBeInTheDocument()
    expect(api.listDocuments).toHaveBeenCalledTimes(2)
  })

  it("warns when an upload was flagged for injection", async () => {
    const { user } = setup({ uploadText: vi.fn().mockResolvedValue({
      status: "created", document: BAD, chunks_total: 3, chunks_embedded: 3, patterns: ["ignore_previous"] }) })
    await screen.findByText("plants.txt")
    await user.click(screen.getByRole("button", { name: "Paste text" }))
    await user.type(screen.getByLabelText("Document text"), "evil words")
    await user.click(screen.getByRole("button", { name: "Index text" }))
    const strip = await screen.findByText(/Suspected injection removed \(ignore_previous\)/)
    expect(strip).toHaveClass("warn")
  })

  it("shows how many chunks an edit re-embedded", async () => {
    const { user } = setup({ uploadText: vi.fn().mockResolvedValue({
      status: "updated", document: DOC, chunks_total: 5, chunks_embedded: 1, patterns: [] }) })
    await screen.findByText("plants.txt")
    await user.click(screen.getByRole("button", { name: "Paste text" }))
    await user.type(screen.getByLabelText("Document text"), "edited")
    await user.click(screen.getByRole("button", { name: "Index text" }))
    expect(await screen.findByText(/1 of 5 chunks re-embedded/)).toBeInTheDocument()
  })

  it("shows an upload error from the server", async () => {
    const { user } = setup({ uploadText: vi.fn().mockRejectedValue({ response: { status: 409, data: { detail: "This content is already stored as 'a'." } } }) })
    await screen.findByText("plants.txt")
    await user.click(screen.getByRole("button", { name: "Paste text" }))
    await user.type(screen.getByLabelText("Document text"), "dup")
    await user.click(screen.getByRole("button", { name: "Index text" }))
    expect(await screen.findByText("This content is already stored as 'a'.")).toHaveClass("error")
  })

  it("deletes only after confirmation", async () => {
    const { user } = setup()
    await screen.findByText("plants.txt")
    vi.spyOn(window, "confirm").mockReturnValueOnce(false)
    await user.click(screen.getByRole("button", { name: "Delete plants.txt" }))
    expect(api.deleteDocument).not.toHaveBeenCalled()
    vi.spyOn(window, "confirm").mockReturnValueOnce(true)
    await user.click(screen.getByRole("button", { name: "Delete plants.txt" }))
    await waitFor(() => expect(api.deleteDocument).toHaveBeenCalledWith("d1"))
    expect(await screen.findByText('Deleted "plants.txt".')).toBeInTheDocument()
  })

  it("shares a document by email and reports failures", async () => {
    const { user } = setup()
    await screen.findByText("plants.txt")
    await user.click(screen.getByRole("button", { name: "Share plants.txt" }))
    await user.type(screen.getByLabelText("Share with email"), "friend@x.y")
    await user.click(screen.getAllByRole("button", { name: "Share" }).at(-1))
    await waitFor(() => expect(api.shareDocument).toHaveBeenCalledWith("d1", "friend@x.y"))
    expect(await screen.findByText('Shared "plants.txt" with friend@x.y.')).toBeInTheDocument()
  })

  it("offers no delete or share on documents shared with you", async () => {
    setup()
    const row = (await screen.findByText("theirs.txt")).closest(".doc-row")
    expect(row).toHaveTextContent("shared with you")
    expect(within(row).queryByRole("button")).toBeNull()
  })
})

describe("chat", () => {
  beforeEach(() => sessionStorage.setItem("smartrag_session", JSON.stringify({ token: "t", user: { id: "u1", email: "me@x.y" } })))

  const answered = {
    answer: "Chlorophyll absorbs light [1]. Also [7].", answerable: true, blocked_reason: null,
    citations: [1], invalid_citations: [7], llm_called: true,
    sources: [{ n: 1, doc_id: "d1", doc_name: "plants.txt", chunk_id: "c1", chunk_index: 0, excerpt: "Chlorophyll absorbs red and blue light", score: 0.5123, trust: 100 }],
    timings_ms: { total_ms: 210, llm_ms: 90 },
  }

  it("sends the selected documents and renders linked citations, sources, and timings", async () => {
    const { user } = setup({ query: vi.fn().mockResolvedValue(answered) })
    await screen.findByText("plants.txt")
    await goto(user, "Chat")
    await user.click(screen.getByLabelText("plants.txt"))
    await user.type(screen.getByLabelText("Question"), "What does chlorophyll absorb?")
    await user.click(screen.getByRole("button", { name: "Ask" }))
    expect(api.query).toHaveBeenCalledWith("What does chlorophyll absorb?", ["d1"], [])
    const link = await screen.findByRole("link", { name: "Source 1" })
    expect(link).toHaveAttribute("href", "#source-1")
    expect(screen.getByText("7")).toHaveClass("invalid")
    expect(screen.getByText(/citations point to sources that do not exist/)).toBeInTheDocument()
    expect(screen.getByText("Score 0.512 · Trust 100%")).toBeInTheDocument()
    expect(screen.getByText("210 ms total · model 90 ms")).toBeInTheDocument()
  })

  it("sends recent answered turns with a follow-up and shows what was searched", async () => {
    const second = { ...answered, answer: "It sits in chloroplasts [1].", invalid_citations: [],
      rewritten_query: "where is chlorophyll found" }
    const { user } = setup({ query: vi.fn().mockResolvedValueOnce({ ...answered, invalid_citations: [] }).mockResolvedValueOnce(second) })
    await screen.findByText("plants.txt")
    await goto(user, "Chat")
    await user.type(screen.getByLabelText("Question"), "What does chlorophyll absorb?")
    await user.click(screen.getByRole("button", { name: "Ask" }))
    await screen.findByText("Answered")
    expect(screen.queryByTestId("rewritten")).toBeNull()
    await user.type(screen.getByLabelText("Question"), "Where is it?")
    await user.click(screen.getByRole("button", { name: "Ask" }))
    await waitFor(() => expect(api.query).toHaveBeenCalledTimes(2))
    expect(api.query.mock.calls[1][2]).toEqual([
      { question: "What does chlorophyll absorb?", answer: "Chlorophyll absorbs light [1]. Also [7]." },
    ])
    expect(await screen.findByTestId("rewritten")).toHaveTextContent("Searched for: where is chlorophyll found")
  })

  it("remembers at most three turns, skips refusals, and starts over on request", async () => {
    const ok = n => ({ ...answered, answer: `A${n} [1].`, invalid_citations: [] })
    const refused = { answer: "x", answerable: false, blocked_reason: "low_relevance", sources: [], citations: [], invalid_citations: [], llm_called: true, timings_ms: {} }
    const q = vi.fn()
      .mockResolvedValueOnce(ok(1)).mockResolvedValueOnce(refused).mockResolvedValueOnce(ok(2))
      .mockResolvedValueOnce(ok(3)).mockResolvedValueOnce(ok(4)).mockResolvedValue(ok(5))
    const { user } = setup({ query: q })
    await screen.findByText("plants.txt")
    await goto(user, "Chat")
    const ask = async text => {
      await user.type(screen.getByLabelText("Question"), text)
      await user.click(screen.getByRole("button", { name: "Ask" }))
      await waitFor(() => expect(screen.getByRole("button", { name: "Ask" })).toBeDisabled())
    }
    for (const text of ["Q1", "Q2", "Q3", "Q4", "Q5"]) await ask(text)
    await waitFor(() => expect(q).toHaveBeenCalledTimes(5))
    expect(q.mock.calls[4][2].map(t => t.question)).toEqual(["Q1", "Q3", "Q4"])
    await user.click(await screen.findByRole("button", { name: /New conversation/ }))
    await user.type(screen.getByLabelText("Question"), "Q6")
    await user.click(screen.getByRole("button", { name: "Ask" }))
    await waitFor(() => expect(q).toHaveBeenCalledTimes(6))
    expect(q.mock.calls[5][2]).toEqual([])
  })

  it("explains a safety-filter block and that no model was called", async () => {
    const { user } = setup({ query: vi.fn().mockResolvedValue({
      answer: "x", answerable: false, blocked_reason: "all_blocked", sources: [], citations: [],
      invalid_citations: [], llm_called: false, timings_ms: { total_ms: 3 } }) })
    await screen.findByText("plants.txt")
    await goto(user, "Chat")
    await user.type(screen.getByLabelText("Question"), "anything")
    await user.click(screen.getByRole("button", { name: "Ask" }))
    expect(await screen.findByText(/blocked by the safety filter/)).toBeInTheDocument()
    expect(screen.getByText("Blocked", { selector: ".result-card .badge" })).toHaveClass("danger")
    expect(screen.getByText("No model was called for this question.")).toBeInTheDocument()
    expect(screen.queryByText("Sources")).toBeNull()
  })

  it("warns when an answer has no citations", async () => {
    const { user } = setup({ query: vi.fn().mockResolvedValue({ ...answered, answer: "No cites here.", citations: [], invalid_citations: [] }) })
    await screen.findByText("plants.txt")
    await goto(user, "Chat")
    await user.type(screen.getByLabelText("Question"), "q")
    await user.click(screen.getByRole("button", { name: "Ask" }))
    expect(await screen.findByText(/no citations/)).toBeInTheDocument()
  })

  it("select all skips blocked documents", async () => {
    const { user } = setup()
    await screen.findByText("plants.txt")
    await goto(user, "Chat")
    await user.click(screen.getByRole("button", { name: "Select all" }))
    expect(screen.getByLabelText("plants.txt")).toBeChecked()
    expect(screen.getByLabelText("theirs.txt")).toBeChecked()
    expect(screen.getByLabelText("evil.txt")).not.toBeChecked()
  })

  it("shows an alert when the request fails", async () => {
    const { user } = setup({ query: vi.fn().mockRejectedValue(new Error("offline")) })
    await screen.findByText("plants.txt")
    await goto(user, "Chat")
    await user.type(screen.getByLabelText("Question"), "q")
    await user.click(screen.getByRole("button", { name: "Ask" }))
    expect(await screen.findByRole("alert")).toHaveTextContent("Cannot reach the server")
  })

  it("keeps Ask disabled for an empty question", async () => {
    const { user } = setup()
    await screen.findByText("plants.txt")
    await goto(user, "Chat")
    expect(screen.getByRole("button", { name: "Ask" })).toBeDisabled()
  })
})

describe("history", () => {
  it("lists past questions with their source counts", async () => {
    sessionStorage.setItem("smartrag_session", JSON.stringify({ token: "t", user: { id: "u1", email: "me@x.y" } }))
    const { user } = setup({ history: vi.fn().mockResolvedValue([
      { id: 2, question: "Q2", answer: "A2", answerable: false, created_at: "2026-10-02T00:00:00Z", sources: [] },
      { id: 1, question: "Q1", answer: "A1", answerable: true, created_at: "2026-10-01T00:00:00Z", sources: [{}, {}] },
    ]) })
    await screen.findByText("plants.txt")
    await goto(user, "History")
    expect(await screen.findByText("Q1")).toBeInTheDocument()
    expect(screen.getByText(/not answered/)).toBeInTheDocument()
    expect(screen.getByText(/2 sources/)).toBeInTheDocument()
  })
})
