import { TRUST_THRESHOLD } from "./api"

const CITATION = /(\[\d+\])/g

/**
 * Split an answer into text and citation parts. A citation is valid only when the response
 * actually contains a source with that number.
 */
export function splitAnswer(answer, sources) {
  const known = new Set((sources ?? []).map(s => s.n))
  return (answer ?? "")
    .split(CITATION)
    .filter(part => part !== "")
    .map(part => {
      const m = /^\[(\d+)\]$/.exec(part)
      if (!m) return { type: "text", text: part }
      const n = Number(m[1])
      return { type: "cite", n, valid: known.has(n), text: part }
    })
}

export function isBlocked(doc) {
  return doc.trust_score < TRUST_THRESHOLD
}

export function describeUpload(res) {
  const { document: doc, status, chunks_total: total, chunks_embedded: embedded } = res
  const trust = Math.round(doc.trust_score)
  if (status === "unchanged") return `"${doc.name}" is unchanged. Nothing was re-indexed.`
  if (status === "updated") {
    return `Updated "${doc.name}": ${embedded} of ${total} chunks re-embedded, trust ${trust}%.`
  }
  return `Indexed "${doc.name}": ${total} chunks, trust ${trust}%.`
}

const BLOCK_MESSAGES = {
  no_documents: "There are no documents to search yet. Upload one first.",
  all_blocked:
    "The selected documents were blocked by the safety filter because they contain " +
    "suspected prompt injection.",
  low_relevance: "No sufficiently relevant content was found in your documents.",
  no_relevant_content: "No sufficiently relevant content was found in your documents.",
  model_abstained: "Your documents do not contain enough information to answer that.",
}

export function blockedMessage(reason) {
  return BLOCK_MESSAGES[reason] ?? null
}

export function timingSummary(timings) {
  if (!timings || timings.total_ms == null) return ""
  const parts = [`${Math.round(timings.total_ms)} ms total`]
  if (timings.rerank_ms != null) parts.push(`rerank ${Math.round(timings.rerank_ms)} ms`)
  if (timings.llm_ms != null) parts.push(`model ${Math.round(timings.llm_ms)} ms`)
  return parts.join(" · ")
}
