# LitDesk — Personal Literature Triage Agent

**Hand this file to Claude Code at the start of a new session.** Suggested opening prompt:

> Read `litdesk-spec.md`. Ask me the questions in "Open Decisions," then build Phase 1 only and show me output before continuing.

---

## 1. What this is

A local daily digest of new biochemistry literature, ranked by *my* interests, learned from thumbs up/down. Runs on a schedule, writes to a local SQLite DB, renders an HTML digest I open in a browser. No account, no cloud, no subscription.

**The thing that makes it worth building:** ranking that learns from feedback, plus PubMed/MEDLINE coverage. Existing tools do one or the other. Google Scholar alerts and PubMed My NCBI alerts are keyword-only — no learning, heavy duplicates. Scholar Inbox has the learned ranking but indexes arXiv/bioRxiv/medRxiv/ChemRxiv and open-access CS proceedings — it is not a PubMed replacement. For wet-lab biochem, the peer-reviewed journal literature is most of what matters, so combining both is the actual gap.

**Non-goals:** a search engine, a reference manager, a PDF reader, anything multi-user.

---

## 2. Stack

- Python 3.11+, single repo, `uv` or `venv`
- **SQLite** for everything (papers, embeddings as BLOBs, ratings, run log)
- **sentence-transformers** for embeddings, run locally on CPU
- **scikit-learn** for the ranker (logistic regression — do not reach for anything heavier)
- **Jinja2** → static HTML digest. Add a tiny **FastAPI** server only for the rating buttons (Phase 3)
- **Anthropic API** for summaries (Phase 4, optional)
- Scheduling: `cron` on Linux, `launchd` on macOS. Not a daemon.

Keep every external API behind a thin client module with its own cache. These APIs change; I want a contained edit when one does.

---

## 3. Data sources — exact details

### 3.1 Europe PMC (primary)

`https://www.ebi.ac.uk/europepmc/webservices/rest/search`

Best single source: one index over PubMed/MEDLINE (`MED`), PMC full text (`PMC`), and preprints (`PPR`, incl. bioRxiv/medRxiv). No API key, no registration.

- Lucene-ish syntax with field qualifiers, **case-sensitive**: `TITLE:`, `ABSTRACT:`, `AUTH:`, `AFFILIATION:`, `JOURNAL:`, `MESH:`, `DOI:`, `OPEN_ACCESS:`, `FIRST_PDATE:`, plus `AND`/`OR`/`NOT` and quoted phrases
- `resultType=core` for abstract + MeSH + full-text links; `lite` for fast scans
- `format=json`, paginate with `cursorMark` (not offset)
- Add `sort_date:y` inside the `query` param to sort by date
- Filter to a source with `SRC:MED`, `SRC:PPR`, etc.
- Prototype every query at https://europepmc.org/advancesearch before hardcoding it

### 3.2 bioRxiv / medRxiv API (secondary)

`https://api.biorxiv.org/details/{server}/{start}/{end}/{cursor}`

Europe PMC ingests preprints via Crossref within roughly 24h. Poll bioRxiv directly for same-day coverage.

- `server` = `biorxiv` or `medrxiv`; interval = two `YYYY-MM-DD` dates, or `Nd` for last N days
- 100 results per page; increment `cursor` by 100 until empty
- Returns: `doi`, `title`, `authors`, `date`, `category`, `abstract`, `published` (journal DOI once published), `jatsxml`
- Optional `?category=biochemistry` (underscores for spaces)
- No key, no documented rate limit — still throttle yourself to ~1 req/s
- The `published` field is how I link preprint → journal version. Use it.

### 3.3 PubMed E-utilities (optional, for MeSH-precise queries)

`https://eutils.ncbi.nlm.nih.gov/entrez/eutils/{esearch,efetch}.fcgi`

- 3 req/s without a key, 10 req/s with a free NCBI key (`api_key=`). Always send `tool=` and `email=`
- ESearch caps at 10,000 records (`retstart + retmax <= 10000`) — chunk by date if a query is broader than that
- Only add this if Europe PMC's MeSH handling proves insufficient. Don't build it on day one.

### 3.4 Enrichment (Phase 5, all optional)

- **Semantic Scholar Recommendations API** — `POST https://api.semanticscholar.org/recommendations/v1/papers` takes `positivePaperIds` **and** `negativePaperIds`. Feed it my thumbs data directly. Unauthenticated is a shared pool; a free key gives ~1 RPS dedicated. Requires exponential backoff.
- **OpenAlex** — citation counts, topics. Note: OpenAlex announced in Jan 2026 that API keys are required as of Feb 13, 2026 and retired the `mailto` polite pool. Verify current state at developers.openalex.org before wiring it in.
- **Unpaywall** — resolve DOI → legal OA PDF link.

---

## 4. Schema (starting point)

```sql
papers(
  id INTEGER PRIMARY KEY,
  doi TEXT UNIQUE,              -- normalized: lowercase, no https://doi.org/ prefix
  pmid TEXT, pmcid TEXT,
  source TEXT,                  -- 'europepmc' | 'biorxiv' | 'medrxiv' | 'pubmed'
  is_preprint INTEGER,
  published_doi TEXT,           -- set when a preprint later appears in a journal
  title TEXT, abstract TEXT,
  authors TEXT,                 -- JSON array
  journal TEXT,
  date_published TEXT,          -- ISO date
  date_ingested TEXT,
  url TEXT, pdf_url TEXT,
  mesh_terms TEXT,              -- JSON array
  raw JSON
)
embeddings(paper_id INTEGER PRIMARY KEY, model TEXT, vec BLOB)
ratings(paper_id INTEGER, label INTEGER, rated_at TEXT)   -- +1 / -1
seeds(paper_id INTEGER, note TEXT)                        -- cold-start positives
digests(id, run_date, paper_ids JSON, model_version TEXT)
runs(id, started_at, finished_at, source, n_fetched, n_new, error TEXT)
```

Dedup, in this order: normalized DOI → PMID → normalized title (lowercase, strip punctuation, collapse whitespace) + first author surname + year. Preprint and journal version of the same work collapse into one row via `published_doi`; keep the journal version's metadata, keep the preprint's earlier date as `date_first_seen`.

---

## 5. The ranker

Use the approach validated by Scholar Inbox (arXiv:2504.08385), which is simpler than it sounds and works well:

1. Embed `title + "[SEP]" + abstract` for every paper.
   - Preferred model: `allenai/specter2` — purpose-built for scientific papers, trained on 6M citation triplets across 23 fields, and it beats general sentence embedders on paper-similarity tasks. **Caveat:** it needs the `adapters` library and is finicky about transformers versions. Pin your versions.
   - Fallback: `sentence-transformers/all-MiniLM-L6-v2` (384-dim, fast, no adapters). Build against MiniLM first, swap in SPECTER2 once the pipeline works. Store the model name alongside every vector so mixed embeddings never get compared.
   - Note SPECTER-family embeddings are trained with L2 distance, not cosine. Match the metric to the model.
2. Train `sklearn.linear_model.LogisticRegression` on the embeddings: my thumbs-up as positives, thumbs-down as explicit negatives, **plus ~5,000 randomly sampled unrated papers as implicit negatives** to regularize the boundary.
3. Handle the class imbalance with weighted loss (`class_weight`), with a separate tunable weight distinguishing explicit negatives from random ones. Explicit negatives should count for more.
4. Retrain on every run. It's a logistic regression on a few thousand vectors — it takes seconds.
5. Rank the day's candidates by predicted probability. Color-code by score band in the digest.

**Cold start.** Before any ratings exist, seed with 30–60 papers I actually care about: my own publications, my lab's, the ones in my quals reading list, the key papers from my project's intro section. Score by similarity to the seed centroid until I have ~30 real ratings, then switch to the trained classifier.

**Explainability.** For each ranked paper, show the nearest seed/rated paper and the similarity. If I can't tell why something surfaced, I stop trusting the list and the whole thing dies.

---

## 6. Build phases

**Phase 1 — Ingest and dedup.** Europe PMC + bioRxiv clients with on-disk caching. Pull the last 7 days for 2–3 queries. Write to SQLite. Print counts: fetched, new, deduped. Stop here and show me. No ranking, no UI. If dedup is wrong, everything downstream is wrong.

**Phase 2 — Embeddings and cold-start ranking.** Embed everything. Load my seed papers. Rank by centroid similarity. Render a static HTML digest: title, journal, date, abstract (collapsed), score, link, and "closest to: [seed paper]".

**Phase 3 — Feedback loop.** Minimal FastAPI server on localhost, thumbs up/down buttons in the digest, writes to `ratings`. Swap centroid similarity for the trained logistic regression once ratings ≥ 30. Add a `--retrain` command that reports classifier accuracy on held-out ratings.

**Phase 4 — Claude summaries.** For the top N (default 10) only:
- A two-sentence TLDR in plain language
- One line on why it's relevant given my stated research profile
- Once a week, a synthesis across the week's top papers: themes, anything that contradicts something else, anything I should read in full

Two billing paths, pick one:
- `claude -p` (Claude Code headless). Currently draws from my Pro subscription's usage limits. Anthropic announced on May 14, 2026 that Agent SDK and `claude -p` usage would move to a separate monthly credit from June 15, then **paused that change on June 15**. It could return — don't architect around it being free forever.
- A direct Anthropic API key, billed per token. Predictable, and it's what Anthropic points developers toward for programmatic use. Use `claude-haiku-4-5-20251001` for per-paper TLDRs (cheap, high volume) and `claude-sonnet-5` for the weekly synthesis.

Put the model name in config. Make the whole LLM layer skippable with a flag.

**Phase 5 — The things that make me keep using it.**
- **Watchlists:** specific authors/labs, and "anything citing paper X"
- **Scoop alarm:** flag anything above a similarity threshold to my active project description, separately and loudly
- **Retraction/correction flags** from Crossref/Europe PMC metadata
- **Preprint→published notifier:** tell me when a preprint I rated up gets published
- **Catch-up digest:** after a gap, aggregate and show only the top N, not 400 items
- **Weekly export:** BibTeX of everything I thumbed up

---

## 7. Gotchas

- Never call these APIs from inside a loop over papers without a rate limiter and an on-disk cache. Cache by URL, TTL 24h. This makes re-runs during development free.
- Every source has different date semantics (posted vs. e-pub vs. print). Normalize to one `date_published` and record which kind it was.
- First run downloads a few hundred MB of model weights. Say so in the README.
- Abstracts are frequently missing or truncated. Rank on title alone rather than crashing.
- Store the raw API response in `raw`. When a schema changes, backfill instead of refetch.
- Log every run to `runs`. A silent cron failure that goes unnoticed for three weeks is the most likely way this project dies.

---

## 8. Acceptance criteria

- One command produces today's digest end to end
- Runs unattended on a schedule and writes a run record every time
- Rating a paper changes tomorrow's ranking in a way I can observe
- Zero duplicates across sources in a 30-day window
- The full pipeline (minus LLM calls) runs offline against cached data, so I can develop on a plane

---

## 9. Open decisions — ask me before building

1. What are my 3–5 standing queries? (topics, techniques, specific proteins/pathways)
2. Which journals do I always want to see regardless of score?
3. Digest frequency — daily or weekday mornings only?
4. Do I want it in the browser, emailed to me, or both?
5. Am I supplying seed papers as DOIs, a BibTeX export from Zotero/Paperpile, or a folder of PDFs?
6. Claude API key or `claude -p`?
