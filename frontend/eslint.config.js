// Narrow on purpose. This exists for one rule: `react-hooks/rules-of-hooks`.
//
// A `useState` declared below an early return shipped a blank page — five hooks on the first
// render, six on the next, and React tears the tree down with no message the user can see. Neither
// `tsc` nor the build catches it, and it only shows on the one page that was not opened during
// review. The linter catches it exactly.
//
// `exhaustive-deps` is a warning rather than an error: several effects here deliberately omit deps
// and say so with an eslint-disable-line comment. Turning it into a failure would mean either
// churning those or burying real findings.
import js from "@eslint/js";
import reactHooks from "eslint-plugin-react-hooks";
import tseslint from "typescript-eslint";

export default tseslint.config(
  { ignores: ["dist", "node_modules", "../raglab/static"] },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  {
    files: ["src/**/*.{ts,tsx}"],
    plugins: { "react-hooks": reactHooks },
    languageOptions: {
      parserOptions: { ecmaFeatures: { jsx: true } },
      globals: { window: "readonly", document: "readonly", localStorage: "readonly", fetch: "readonly",
                 console: "readonly", setTimeout: "readonly", clearTimeout: "readonly",
                 setInterval: "readonly", clearInterval: "readonly", requestAnimationFrame: "readonly",
                 cancelAnimationFrame: "readonly", matchMedia: "readonly", alert: "readonly",
                 ResizeObserver: "readonly", URL: "readonly", Blob: "readonly", File: "readonly",
                 TextDecoder: "readonly", AbortController: "readonly", Event: "readonly", Date: "readonly" },
    },
    rules: {
      "react-hooks/rules-of-hooks": "error",
      "react-hooks/exhaustive-deps": "warn",
      // The codebase uses a deliberate `Any` alias with the escape hatch documented in api.ts.
      "@typescript-eslint/no-explicit-any": "off",
      "@typescript-eslint/no-unused-vars": ["error", { argsIgnorePattern: "^_", varsIgnorePattern: "^_" }],
    },
  },
);
