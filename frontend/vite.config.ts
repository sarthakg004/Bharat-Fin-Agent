import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { fileURLToPath } from "node:url";

// ESM-compatible replacement for `path.resolve(__dirname, "./src")`.
// `__dirname` is not defined under `"type": "module"`; we derive an absolute
// directory from `import.meta.url` instead. Requires @types/node for the
// `node:url` declaration.
const srcDir = fileURLToPath(new URL("./src", import.meta.url));

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": srcDir,
    },
  },
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
});
