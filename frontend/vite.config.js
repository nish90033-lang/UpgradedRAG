import { defineConfig } from "vite"
import react from "@vitejs/plugin-react"

export default defineConfig({
  plugins: [react()],
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test/setup.js"],
    include: ["src/**/*.test.jsx", "src/**/*.test.js"],
    exclude: ["src/**/*.e2e.test.jsx"],
    globals: true,
  },
})
