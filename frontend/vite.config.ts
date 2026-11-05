import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `npm run dev` proxies the API to the Python server; `npm run build` writes into the Python package.
export default defineConfig({
  plugins: [react()],
  server: { port: 5180, proxy: { "/api": "http://127.0.0.1:8600" } },
  build: { outDir: "../raglab/static", emptyOutDir: true },
});
