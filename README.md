# LitDesk

A personal daily digest of new biochemistry literature, ranked by *your*
interests and learned from thumbs up/down feedback. Runs locally on a
schedule, writes to a local SQLite database, renders a static HTML digest
you open in a browser. No account, no cloud, no subscription.

See [`litdesk-spec.md`](./litdesk-spec.md) for the full design doc this is
built from — data sources, schema, ranking approach, build phases.

## Status

- [x] **Phase 1 — Ingest and dedup.** Europe PMC + bioRxiv/medRxiv clients with
      on-disk caching and rate limiting, deduped into SQLite, every run logged.
- [x] **Phase 2 — Embeddings and cold-start ranking.** MiniLM embeddings, seed
      papers loaded by DOI or BibTeX, candidates ranked by similarity to the
      seed centroid, rendered to a static HTML digest with per-paper
      explainability ("closest to: ...") and score bands.
- [x] **Phase 3 — Feedback loop.** `litdesk serve` runs a local FastAPI server
      with working thumbs up/down buttons in the digest; once you've rated
      enough papers, a logistic regression classifier takes over from the
      seed centroid automatically. `litdesk retrain` reports held-out accuracy.
- [x] **Phase 4 — Claude summaries.** Off by default (`llm.enabled: false`).
      When on: a two-sentence TLDR + one-line relevance note for the top N
      papers in each digest (`claude-haiku-4-5-20251001`), plus a
      `litdesk synthesis` command for a weekly cross-paper synthesis
      (`claude-sonnet-5`). Every LLM call degrades to "skip it" on any
      failure — a bad key or a rate limit never breaks a digest run.
- [ ] Phase 5 — Watchlists, scoop alarm, retraction flags, exports

## Setup

Requires Python 3.11+.

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev,llm]"   # drop `llm` if you don't want the Anthropic SDK yet

litdesk init      # writes config/config.yaml from the example template
```

Then edit `config/config.yaml` — at minimum, replace the placeholder
`queries` with your actual standing topics, and set `seeds.dois` to 30-60
papers you already care about (see "Configuration" below).

**First run note:** the first `litdesk embed` (or `litdesk digest`/`litdesk
run`, which call it for you) downloads a few hundred MB of sentence-transformers
model weights. Nothing in Phase 1 needs that.

## Usage

```bash
litdesk ingest      # Phase 1: fetch, dedup, store. Prints fetched/new/deduped per source.
litdesk seeds       # Phase 2: load seeds.dois / seeds.bibtex_path from config as seed papers
litdesk embed       # Phase 2: embed every paper missing a vector under the configured model
litdesk digest      # Phase 2/3: rank candidates, render today's HTML digest (needs seeds + embeddings)
litdesk run         # ingest + seeds + embed + digest in one shot — the crontab entry
litdesk serve       # Phase 3: serve the digest at http://127.0.0.1:8000 with working rating buttons
litdesk retrain     # Phase 3: held-out accuracy of the trained ranker on your ratings so far
litdesk synthesis   # Phase 4: weekly cross-paper synthesis (needs llm.enabled: true) — a separate weekly cron entry
litdesk runs        # Show recent run history (useful after a cron job, or when debugging)
```

The digest you open from `litdesk digest`/`litdesk run` is a plain static
file, so its thumbs up/down buttons only work while `litdesk serve` is also
running (they POST to `http://127.0.0.1:<server.port>/rate`) — that's the
one thing in this whole tool that isn't a single command. A natural setup:
`litdesk run` on a schedule, `litdesk serve` running whenever you're actually
triaging.

`litdesk seeds` and `litdesk embed` are idempotent no-ops on a rerun once
everything's loaded, so `litdesk run` is cheap to call on every scheduled
tick rather than chaining the steps yourself. Every `ingest` run writes one
row per source to the `runs` table — even on total failure (see "Offline
development" below) — specifically so a broken API or a bad cron setup
doesn't fail silently. Check `litdesk runs` (or the process exit code, which
is non-zero if any source errored) if a digest seems stale.

## Configuration

`config/config.yaml` (gitignored — it's yours) overrides only the fields you
set; anything you omit falls back to the defaults in `litdesk/config.py`.
See `config/config.example.yaml` for every available field with comments.

The most important fields to set for yourself:

- `queries` — your standing topics/techniques/proteins/pathways, as Europe
  PMC query strings. Prototype each one at
  https://europepmc.org/advancesearch before adding it here.
- `journals_always_include` — journals you want surfaced regardless of score.
- `seeds.dois` (or `seeds.bibtex_path`, pointed at a Zotero/Paperpile BibTeX
  export) — papers you already care about, used to bootstrap ranking before
  you've rated anything. Seeds are fetched directly by DOI if they aren't
  already in your database (they usually predate your standing queries), so
  this needs network access the first time.

## Data sources

- **Europe PMC** (primary): one index over MEDLINE, PMC full text, and
  preprints (including bioRxiv/medRxiv). No key required.
- **bioRxiv / medRxiv** (secondary): polled directly for same-day coverage,
  since Europe PMC ingests preprints via Crossref roughly a day later.

Both clients cache every response on disk (24h TTL, `data/cache/`) and rate
limit themselves, so re-running during development doesn't hammer either API.

## Dedup

Papers are matched (in order) by normalized DOI, PMID, then normalized title
+ first-author surname + year. A preprint and its later journal version
collapse into a single row: the row's `doi` never changes after creation
(stable identity), the journal DOI (once known) lives in `published_doi`,
journal metadata always wins once it arrives regardless of ingestion order,
and `date_first_seen` keeps the earlier (preprint) date even after the
journal version's metadata takes over. See `litdesk/dedup.py` and
`litdesk/ingest.py::upsert_paper` for the exact rules, and
`tests/test_ingest_merge.py` for the scenarios this is tested against.

## Ranking (cold start)

Every paper is embedded as `title + "[SEP]" + abstract` (title alone if the
abstract is missing — a paper is never dropped just because a source didn't
give us one) using `sentence-transformers/all-MiniLM-L6-v2` by default.
`config.yaml`'s `embeddings.model` is the documented upgrade path to
`allenai/specter2` once you want it (spec's rationale: purpose-built for
scientific papers, but needs the finicky `adapters` library — get MiniLM
working first). The model name travels with every stored vector, so
switching models re-embeds everything rather than silently comparing
incompatible vector spaces.

Before you've rated ~30 papers, candidates are ranked by cosine similarity
to the centroid of your seed papers' embeddings — "cold start" in spec
section 5. Each ranked paper records its single nearest seed and that
similarity score, shown in the digest as "closest to: ...", because
unexplainable rankings are how these tools stop getting used. Score bands
(high/medium/low) are assigned by rank position within the digest, not by an
absolute score threshold, since cosine similarity and the Phase 3 classifier's
predicted probability live on different scales. `journals_always_include`
papers that don't score into the top N are appended anyway, tagged
"always include" instead of a score band.

A digest never repeats a paper: candidates are papers that have never been
rated, were never a seed, and never appeared in a previous `digests` row —
independent of any date window, so a lagged ingest can't cause a repeat or a
silent drop.

## Feedback loop

`litdesk serve` runs a small local-only FastAPI server (binds `127.0.0.1`,
not configurable — the `/rate` endpoint has no auth) that serves the latest
digest and accepts thumbs up/down. Once you've rated at least
`ranker.min_ratings_for_classifier` papers (default 30, config.yaml) *with
at least one thumbs-up and one thumbs-down*, `litdesk digest`/`litdesk run`
automatically switch from centroid ranking to a
`sklearn.linear_model.LogisticRegression` trained on:

- your thumbs-up as positives
- your thumbs-down as explicit negatives (weighted higher — a thumbs-down is
  a much stronger signal than "probably not interesting")
- `ranker.n_random_negatives` randomly sampled unrated papers as implicit
  negatives, to regularize the decision boundary

It retrains from scratch on every digest run (seconds, not worth caching)
and falls back to centroid ranking automatically if training can't proceed
yet (e.g. you have 30+ ratings but they're all thumbs-up so far). Nearest-
neighbor explainability switches too: once the classifier is active, "closest
to" points at your nearest *rated* paper (up or down) instead of a seed.
`litdesk retrain` reports accuracy/precision/recall on a held-out split of
your ratings, without affecting the ranker actually used for digests.

## Claude summaries

Off by default (`llm.enabled: false` in config.yaml — flip it on, and set an
`ANTHROPIC_API_KEY`; `uv pip install -e ".[llm]"` if you skipped the extra at
setup). When enabled, `litdesk digest`/`litdesk run` ask
`claude-haiku-4-5-20251001` for a two-sentence plain-language TLDR plus a
one-line relevance note for the top `llm.tldr_top_n` (default 10) papers in
each digest — cheap enough for daily, per-paper volume. Both are stored in
`rankings` alongside the score, so they survive independent of the rendered
HTML. `litdesk synthesis` (a separate weekly cron entry, not part of `litdesk
run`) asks `claude-sonnet-5` for a short synthesis across the last week's
ranked papers — themes, contradictions, anything worth reading in full —
and writes it to `digests/synthesis-<date>.md`.

Every LLM call is wrapped so a failure degrades to "skip it," never a hard
error: no `anthropic` package installed, no resolvable API key, a rate limit,
a network blip — any of these just mean that paper's TLDR (or that week's
synthesis) doesn't happen, logged as a warning. A digest run never fails
because of the LLM layer. `llm.provider: claude_code` (billing through a
Claude Pro/Max subscription via `claude -p` instead of a metered API key) is
in config as a placeholder for the option the spec raised, but isn't wired up
yet — see the open questions below.

## Offline development

The entire pipeline (minus LLM calls and minus the one-time model weight
download) runs against on-disk cached data, so you can develop without
network access. Run the test suite instead of hitting live APIs or loading
the real embedding model:

```bash
pytest
```

Tests exercise the Europe PMC / bioRxiv response parsing, ranking math, and
digest rendering against recorded fixture JSON and synthetic embeddings
(`tests/fixtures/`, `conftest.py`'s `insert_embedding` fixture) — none of it
needs network or the real sentence-transformers model.

## Project layout

```
litdesk/
  config.py          # config schema + YAML loader
  db.py               # SQLite schema (papers, embeddings, ratings, seeds, digests, rankings, runs, notifications)
  cache.py            # on-disk HTTP response cache (24h TTL)
  ratelimit.py         # sleep-based rate limiter
  http_client.py        # cache + rate limit + retry, shared by source clients
  models.py              # RawPaper — normalized shape every source client returns
  dedup.py                # identity resolution / merge-direction rules
  ingest.py                 # Phase 1 orchestration
  sources/
    europepmc.py             # Europe PMC REST client
    biorxiv.py                 # bioRxiv/medRxiv details API client
  embeddings.py                 # sentence-transformers wrapper, vector <-> BLOB
  seeds.py                       # seed loading from DOIs / BibTeX
  ranking.py                      # cosine similarity to seed centroid + nearest-seed explainability
  classifier.py                    # Phase 3: logistic regression trained on ratings, held-out eval
  server.py                         # Phase 3: local FastAPI server (digest + /rate endpoint)
  digest.py                          # candidate selection, ranking persistence, HTML rendering
  templates/
    digest.html.jinja                # the digest itself, incl. rating buttons and TLDRs
  llm/
    client.py                          # Phase 4: Anthropic client, never raises — returns None on any failure
    summarize.py                        # per-paper TLDR + weekly synthesis
  cli.py                                # `litdesk` command entrypoint
```
