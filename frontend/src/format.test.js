import { describe, expect, it } from "vitest"
import { blockedMessage, describeUpload, isBlocked, splitAnswer, timingSummary } from "./format"

const sources = [{ n: 1 }, { n: 2 }]

describe("splitAnswer", () => {
  it("separates text from citations and marks unknown sources invalid", () => {
    const parts = splitAnswer("Light drives it [1]. Also [9].", sources)
    expect(parts.map(p => p.type)).toEqual(["text", "cite", "text", "cite", "text"])
    expect(parts[1]).toMatchObject({ n: 1, valid: true })
    expect(parts[3]).toMatchObject({ n: 9, valid: false })
  })
  it("handles adjacent citations, empty input, and missing sources", () => {
    expect(splitAnswer("A [1][2]", sources).filter(p => p.type === "cite")).toHaveLength(2)
    expect(splitAnswer("", sources)).toEqual([])
    expect(splitAnswer(undefined, undefined)).toEqual([])
    expect(splitAnswer("x [1]", undefined)[1].valid).toBe(false)
  })
  it("does not treat other bracketed text as a citation", () => {
    expect(splitAnswer("see [a] and [1x]", sources).every(p => p.type === "text")).toBe(true)
  })
})

describe("helpers", () => {
  it("flags documents below the trust threshold", () => {
    expect(isBlocked({ trust_score: 40 })).toBe(true)
    expect(isBlocked({ trust_score: 50 })).toBe(false)
    expect(isBlocked({ trust_score: 100 })).toBe(false)
  })
  it("describes each upload outcome", () => {
    const doc = { name: "notes", trust_score: 100 }
    expect(describeUpload({ document: doc, status: "created", chunks_total: 5 })).toBe(
      'Indexed "notes": 5 chunks, trust 100%.')
    expect(describeUpload({ document: doc, status: "updated", chunks_total: 5, chunks_embedded: 1 })).toBe(
      'Updated "notes": 1 of 5 chunks re-embedded, trust 100%.')
    expect(describeUpload({ document: doc, status: "unchanged", chunks_total: 5 })).toContain("unchanged")
  })
  it("maps blocked reasons to messages and leaves unknown reasons empty", () => {
    expect(blockedMessage("all_blocked")).toContain("safety filter")
    expect(blockedMessage("no_documents")).toContain("Upload one first")
    expect(blockedMessage("something_new")).toBeNull()
    expect(blockedMessage(null)).toBeNull()
  })
  it("summarises timings", () => {
    expect(timingSummary({ total_ms: 311.6, rerank_ms: 120.2, llm_ms: 160.4 })).toBe(
      "312 ms total · rerank 120 ms · model 160 ms")
    expect(timingSummary({ total_ms: 5 })).toBe("5 ms total")
    expect(timingSummary(undefined)).toBe("")
  })
})
