# Before calling a change done

Four rounds of review on this codebase found the same shape of defect every time: the mechanism was
fixed and a surface that displays it was not. None of them were subtle once pointed at. They were
the cases nobody enumerated.

Run through this when changing a metric, a validator, a control, or anything a job reads.

## Entry points

A rule enforced in one of these is not enforced.

- [ ] **HTTP API** — every endpoint that accepts the thing, not just the one you were looking at.
      `/api/query`, `/api/evalruns`, `/api/sweeps`, `/api/indexes`, `/api/chunk-preview`,
      `/api/indexes/{id}/hnsw`, `/api/indexes/{id}/probe`.
- [ ] **Direct Python** — `execute()`, `evaluate()`, `Pipeline`. A guard that lives only at the API
      boundary is not a guard; a script, a test or the next endpoint walks past it.
- [ ] **The UI** — client-side checking is a courtesy. The server still has to refuse.

## Inputs

For every value a caller can send:

- [ ] A valid one.
- [ ] Out of range, both ends.
- [ ] **Absent** — must keep using the default.
- [ ] **Explicitly null** — must be refused. These are not the same thing, and treating them as one
      let `max_chunks: null` past a check that rejects `max_chunks: 0`.
- [ ] `NaN` and infinity. `Number("typo")` is `NaN`, and `JSON.stringify(NaN)` is `null`, so a typo
      in any text box arrives as null.
- [ ] The wrong type — including a **word where a number is expected, and a number where a word is**.
      A blanket numeric check once refused every categorical setting the code declared sweepable.

## Displays

A number that changed meaning has to change wherever it is drawn.

- [ ] **Evaluation and Sweeps** both. They show the same metrics and have drifted apart twice.
- [ ] Metric names come from one table (`frontend/src/stages.ts`), not from each screen.
- [ ] The cut-off in a label is the cut-off that was applied. `@k` on a whole-prompt stage is a lie.
- [ ] **Saved runs from before the change** keep their own labels, behind a warning. Relabelling
      yesterday's numbers with today's meaning is the same defect you are fixing.

## What a run records

- [ ] Anything a job reads from shared state is **frozen when the job starts** — models,
      normalisation rules — or it can change under a run that is already going.
- [ ] Frozen **per index**, when the run touches more than one. One set recorded for a comparison
      describes one side of it. Same for the corpus, and the revision.
- [ ] Nothing that was not measured reads as a measured zero. `None` means "not measured"; `0.0`
      means "measured, and it was zero".
- [ ] No secrets in a manifest.

## Rendering

- [ ] **Open every page in a browser.** `tsc` and the build pass a component that crashes on
      render; a hook below an early return shipped a blank page on the one tab nobody opened.
- [ ] `npm run build` runs `eslint src --quiet` first, and `react-hooks/rules-of-hooks` is an
      error. Do not silence it.
- [ ] Every page, twice: **with no corpus or index**, and with them. A fresh install is a state,
      and a guard that waits for data renders nothing without it.
- [ ] **No component returns `null` for its whole body.** A page that renders nothing cannot be
      told apart from one that crashed. Say "loading", or say what failed.
- [ ] Every `fetch` that gates rendering has a `.catch` that puts something on the screen.

## Destructive actions

- [ ] **Say what it actually destroys, before it happens.** An index keeps its own copy of every
      chunk, so removing a document from the corpus does not remove it from an index. "Remove" that
      leaves the content answering questions is the opposite of what the button says.
- [ ] Name the collateral — which indexes, how many chunks — rather than a generic "are you sure".
- [ ] Confirmation is inline, not `window.confirm`: a native dialog cannot name the consequence,
      and it blocks the page.
- [ ] Any cached in-memory view of what was deleted is invalidated (`indexer._RUNTIME`).

## Navigation

- [ ] Results survive leaving the tab. Remembering the controls and forgetting the results leaves
      the form describing a comparison that is gone.
- [ ] What is displayed is bound to the configuration that produced it, not to the live controls.

## Tests

- [ ] A regression test per fixed defect, and confirm it **fails against the old code**.
- [ ] Hand-calculated expectations, not captured output — a captured value agrees with a plausible
      wrong implementation.
- [ ] Anything written outside the test's own rows is put back (`tests/conftest.py`). One test
      writing a global row has broken a later module three times.
- [ ] The suite passes in forward, reverse and single-module order.
