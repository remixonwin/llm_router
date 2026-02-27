import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// Determine if we're building for container/production
const isContainer = process.env.CONTAINER_BUILD === "true";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  // In container: serve at root (/)
  // In development: serve at /llm/ (as mounted in unified gateway)
  base: isContainer ? "/" : "/llm/",
  build: {
    outDir: "dist",
  },
  server: {
    port: 3000,
    proxy: {
      "/api": {
        target: "http://localhost:7542",
        changeOrigin: true,
        rewrite: path => path.replace(/^\/api/, ""),
      },
    },
  },
});
