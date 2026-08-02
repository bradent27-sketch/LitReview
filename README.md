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
- [ ] Phase 2 — Embeddings and cold-start ranking, HTML digest
- [ ] Phase 3 — Feedback loop (ratings, trained ranker)
- [ ] Phase 4 — Claude summaries
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
`queries` with your actual standing topics (see "Configuration" below).

**First run note:** once Phase 2 lands, the first embedding run downloads a
few hundred MB of model weights (sentence-transformers). Nothing in Phase 1
needs that.

## Usage

```bash
litdesk ingest      # Phase 1: fetch, dedup, store. Prints fetched/new/deduped per source.
litdesk runs        # Show recent run history (useful after a cron job, or when debugging)
```

Every `ingest` run writes one row per source to the `runs` table — even on
total failure (see "Offline development" below) — specifically so a broken
API or a bad cron setup doesn't fail silently. Check `litdesk runs` (or the
process exit code, which is non-zero if any source errored) if a digest
seems stale.

## Configuration

`config/config.yaml` (gitignored — it's yours) overrides only the fields you
set; anything you omit falls back to the defaults in `litdesk/config.py`.
See `config/config.example.yaml` for every available field with comments.

The most important fields to set for yourself:

- `queries` — your standing topics/techniques/proteins/pathways, as Europe
  PMC query strings. Prototype each one at
  https://europepmc.org/advancesearch before adding it here.
- `journals_always_include` — journals you want surfaced regardless of score.
- `seeds.dois` (or `seeds.bibtex_path`) — once Phase 2 lands, papers you
  already care about, used to bootstrap ranking before you've rated anything.

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

## Offline development

The entire pipeline (minus LLM calls, once Phase 4 lands) runs against
on-disk cached data, so you can develop without network access. Run the test
suite instead of hitting live APIs:

```bash
pytest
```

Tests exercise the Europe PMC / bioRxiv response parsing and pagination
against recorded fixture JSON in `tests/fixtures/`, so they don't need
network either.

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
  cli.py                        # `litdesk` command entrypoint
```
