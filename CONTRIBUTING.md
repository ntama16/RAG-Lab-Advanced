# Contributing

Issues and pull requests are welcome. For anything larger than a bug fix, open an issue first so the idea
can be discussed before you spend time on it.

## Set up

```bash
git clone https://github.com/ntama16/RAG-Lab-Advanced.git && cd RAG-Lab-Advanced
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
pytest
```

The tests run on deterministic stand-in models (`RAGLAB_MOCK=1`, set by the test suite itself), so they
need no model server and no network.

## Changing the UI

The UI source is in `frontend/`; the built copy in `raglab/static/` is committed so that users do not need
Node. If you change anything under `frontend/src`, rebuild and commit the result with your change:

```bash
cd frontend
npm ci
npm run dev     # live reload on :5180, proxying /api to a running `raglab` on :8600
npm run build   # lints, type-checks and writes raglab/static
```

CI rebuilds the UI and fails if the committed bundle does not match the source.

## Before you open a pull request

- Run `pytest` and, for UI changes, `npm run build` in `frontend/`.
- Add a test that fails without your change. Most past defects here were a fix applied in one place and
  missed in another, so [CHECKLIST.md](CHECKLIST.md) lists the places to check when you change a metric, a
  validator, a control or anything a background job reads.
- Write commit messages in the [Conventional Commits](https://www.conventionalcommits.org) style
  (`fix:`, `feat:`, `docs:`, ...) and say in the body why the change is needed, not only what it does.

## Licence

RAG Lab Advanced is licensed under Apache-2.0. Unless you say otherwise, anything you submit for inclusion is
licensed under the same terms, as section 5 of the [licence](LICENSE) describes.
