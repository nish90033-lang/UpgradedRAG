import { defineConfig } from "vite"
import react from "@vitejs/plugin-react"

// End-to-end tests run the real UI against a running backend (python -m smartrag.demo).
export default defineConfig({
  plugins: [react()],
  test: {
    environment: "jsdom",
    environmentOptions: { jsdom: { url: "http://localhost:3000" } },
    setupFiles: ["./src/test/setup.js"],
    include: ["src/**/*.e2e.test.jsx"],
    globals: true,
    testTimeout: 20000,
  },
})
