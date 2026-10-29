// @vitest-environment node
import http from "node:http"
import { afterAll, beforeAll, beforeEach, describe, expect, it, vi } from "vitest"
import { createApi, errorMessage } from "./api"

let server, base
let seen = []
let respond = () => ({ status: 200, body: {} })

beforeAll(async () => {
  server = http.createServer((req, res) => {
    const chunks = []
    req.on("data", c => chunks.push(c))
    req.on("end", () => {
      const raw = Buffer.concat(chunks).toString()
      seen.push({ method: req.method, url: req.url, auth: req.headers.authorization, type: req.headers["content-type"], raw })
      const { status, body } = respond(req)
      res.writeHead(status, { "content-type": "application/json" })
      res.end(JSON.stringify(body))
    })
  })
  await new Promise(r => server.listen(0, "127.0.0.1", r))
  base = `http://127.0.0.1:${server.address().port}`
})
afterAll(() => server.close())
beforeEach(() => { seen = []; respond = () => ({ status: 200, body: {} }) })

describe("errorMessage", () => {
  it("prefers a string detail, joins validation messages, and handles network failures", () => {
    expect(errorMessage({ response: { data: { detail: "Bad" } } }, "x")).toBe("Bad")
    expect(errorMessage({ response: { data: { detail: [{ msg: "a" }, { msg: "b" }] } } }, "x")).toBe("a. b")
    expect(errorMessage({ response: { status: 500, data: {} } }, "fallback")).toBe("fallback")
    expect(errorMessage(new Error("net"), "x")).toContain("Cannot reach the server")
  })
})

describe("createApi", () => {
  it("sends the bearer token on every request", async () => {
    respond = () => ({ status: 200, body: { documents: [] } })
    const api = createApi(() => "tok123", vi.fn(), base)
    await api.listDocuments()
    expect(seen[0].auth).toBe("Bearer tok123")
  })

  it("sends no Authorization header without a token", async () => {
    respond = () => ({ status: 200, body: { token: "t", user: {} } })
    await createApi(() => null, vi.fn(), base).login("a@x.y", "pw")
    expect(seen[0].auth).toBeUndefined()
  })

  it("calls onUnauthorized on a 401 from a protected route but not from login", async () => {
    const onUnauthorized = vi.fn()
    respond = () => ({ status: 401, body: { detail: "nope" } })
    const api = createApi(() => "t", onUnauthorized, base)
    await expect(api.listDocuments()).rejects.toBeTruthy()
    expect(onUnauthorized).toHaveBeenCalledTimes(1)
    await expect(api.login("a@x.y", "bad")).rejects.toBeTruthy()
    expect(onUnauthorized).toHaveBeenCalledTimes(1)
  })

  it("sends doc_ids as null when nothing is selected and as a list otherwise", async () => {
    respond = () => ({ status: 200, body: { answer: "x", sources: [] } })
    const api = createApi(() => "t", vi.fn(), base)
    await api.query("q", [])
    await api.query("q", ["d1", "d2"])
    expect(JSON.parse(seen[0].raw)).toEqual({ question: "q", doc_ids: null, history: [] })
    expect(JSON.parse(seen[1].raw)).toEqual({ question: "q", doc_ids: ["d1", "d2"], history: [] })
  })

  it("sends conversation history with a query", async () => {
    respond = () => ({ status: 200, body: { answer: "x", sources: [] } })
    const turns = [{ question: "a", answer: "b" }]
    await createApi(() => "t", vi.fn(), base).query("c", ["d1"], turns)
    expect(JSON.parse(seen[0].raw).history).toEqual(turns)
  })

  it("uses the v2 routes and shapes", async () => {
    respond = () => ({ status: 200, body: { documents: [], history: [] } })
    const api = createApi(() => "t", vi.fn(), base)
    await api.uploadText("n", "body")
    await api.deleteDocument("abc")
    await api.shareDocument("abc", "b@x.y")
    await api.history()
    expect(seen.map(s => `${s.method} ${s.url}`)).toEqual([
      "POST /documents/text", "DELETE /documents/abc", "POST /documents/abc/share", "GET /history",
    ])
    expect(JSON.parse(seen[0].raw)).toEqual({ name: "n", text: "body" })
    expect(JSON.parse(seen[2].raw)).toEqual({ email: "b@x.y" })
  })

  it("uploads a PDF as multipart form data with an optional name", async () => {
    respond = () => ({ status: 201, body: { status: "created" } })
    const file = new File([new Uint8Array([37, 80, 68, 70])], "a.pdf", { type: "application/pdf" })
    await createApi(() => "t", vi.fn(), base).uploadPdf(file, "custom")
    expect(seen[0].url).toBe("/documents/pdf")
    expect(seen[0].type).toMatch(/^multipart\/form-data/)
    expect(seen[0].raw).toContain('name="file"')
    expect(seen[0].raw).toContain("custom")
  })
})
