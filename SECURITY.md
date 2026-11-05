# Security

## Reporting a vulnerability

Please report security problems privately through GitHub: open the repository's **Security** tab and choose
**Report a vulnerability**. Do not open a public issue for them.

Include what you found, how to reproduce it, and what an attacker could do with it. Fixes land on
`main`; there are no maintained release branches yet.

## What RAG Lab Advanced is built to be

It is a single-user lab that runs on your own machine. Knowing that helps tell a vulnerability from a
design limit.

- **No authentication.** Anyone who can reach the server can use every endpoint: read and change the corpus,
  build and delete indexes, change the model settings and run queries on your models. It binds to
  `127.0.0.1` by default for that reason. Starting it with `--host 0.0.0.0` exposes all of this to your
  network.
- **API keys in plain text.** A key typed into the Models page is written to `raglab.config.json`
  (ignored by git). The API never returns a stored key in full. Prefer the "API key env var" field, which
  keeps the key out of the file.
- **Your documents go to your models.** Queries, passages and, during index builds, whole documents are sent
  to whichever endpoints the four model roles point at. With a cloud provider, that provider sees them.

Problems in scope include anything that breaks those boundaries: reading files outside the app's own
folders, reaching the server from another origin, a stored key leaking through the API or the UI, or a
crafted document or model response that runs code.
