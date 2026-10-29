import "@testing-library/jest-dom/vitest"
import { afterEach } from "vitest"
import { cleanup } from "@testing-library/react"

afterEach(() => {
  cleanup()
  // The API client tests run in a Node environment that has no browser storage.
  if (typeof sessionStorage !== "undefined") sessionStorage.clear()
  if (typeof localStorage !== "undefined") localStorage.clear()
})
