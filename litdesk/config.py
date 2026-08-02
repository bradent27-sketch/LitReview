"""Config loading. A YAML file overrides these defaults field-by-field, so
config.yaml only needs to contain what the user wants to change."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class SourcesConfig:
    europepmc: bool = True
    biorxiv: bool = True
    medrxiv: bool = False


@dataclass
class SeedsConfig:
    dois: list[str] = field(default_factory=list)
    bibtex_path: str | None = None


@dataclass
class EmbeddingsConfig:
    # Fallback model per spec 5.1 — no `adapters` dependency, fast on CPU.
    # Swap to "allenai/specter2" once the pipeline is validated; the model
    # name travels with every stored vector so mixed embeddings never compare.
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    metric: str = "cosine"  # SPECTER-family models: use "l2" instead
    batch_size: int = 32


@dataclass
class RankerConfig:
    min_ratings_for_classifier: int = 30
    n_random_negatives: int = 5000
    # sample weights fed to LogisticRegression.fit(sample_weight=...)
    weight_positive: float = 1.0
    weight_explicit_negative: float = 3.0
    weight_random_negative: float = 1.0
    held_out_fraction: float = 0.2


@dataclass
class DigestConfig:
    top_n: int = 25
    output_dir: str = "digests"
    # "catch-up" mode (spec 5, Phase 5): if the gap since the last digest
    # exceeds this many days, collapse everything since then to top_n_catchup
    # instead of showing every item.
    catchup_gap_days: int = 3
    catchup_top_n: int = 15


@dataclass
class WatchlistsConfig:
    authors: list[str] = field(default_factory=list)
    labs: list[str] = field(default_factory=list)  # matched against affiliation text


@dataclass
class ScoopAlarmConfig:
    enabled: bool = False
    threshold: float = 0.9  # cosine similarity to project_description embedding


@dataclass
class LLMConfig:
    enabled: bool = False
    provider: str = "api"  # "api" (direct Anthropic key) | "claude_code" (`claude -p`)
    tldr_model: str = "claude-haiku-4-5-20251001"
    synthesis_model: str = "claude-sonnet-5"
    tldr_top_n: int = 10


@dataclass
class RateLimitsConfig:
    europepmc_per_sec: float = 3.0
    biorxiv_per_sec: float = 1.0


@dataclass
class Config:
    db_path: str = "data/litdesk.db"
    cache_dir: str = "data/cache"
    cache_ttl_hours: int = 24
    log_dir: str = "data/logs"

    # Open decision #1 (spec 9.1): replace with your real standing queries.
    # These are placeholders so the pipeline is runnable/testable out of the box.
    queries: list[str] = field(default_factory=lambda: [
        'ABSTRACT:"enzyme kinetics" AND ABSTRACT:mechanism',
        'ABSTRACT:"protein folding" AND ABSTRACT:chaperone',
        'MESH:"Protein Kinases" AND ABSTRACT:structure',
    ])
    lookback_days: int = 7
    biorxiv_categories: list[str] = field(default_factory=list)  # empty = no filter

    # Open decision #2: journals to always include regardless of score.
    journals_always_include: list[str] = field(default_factory=list)

    # Used by the scoop alarm (Phase 5) to flag close matches to active work.
    project_description: str = ""

    sources: SourcesConfig = field(default_factory=SourcesConfig)
    seeds: SeedsConfig = field(default_factory=SeedsConfig)
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    ranker: RankerConfig = field(default_factory=RankerConfig)
    digest: DigestConfig = field(default_factory=DigestConfig)
    watchlists: WatchlistsConfig = field(default_factory=WatchlistsConfig)
    scoop_alarm: ScoopAlarmConfig = field(default_factory=ScoopAlarmConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    rate_limits: RateLimitsConfig = field(default_factory=RateLimitsConfig)


def _merge_into_dataclass(instance: Any, overrides: dict) -> Any:
    """Recursively apply a raw dict onto a dataclass instance, field by field."""
    for f in fields(instance):
        if f.name not in overrides:
            continue
        value = overrides[f.name]
        current = getattr(instance, f.name)
        if is_dataclass(current) and isinstance(value, dict):
            setattr(instance, f.name, _merge_into_dataclass(current, value))
        else:
            setattr(instance, f.name, value)
    return instance


def load_config(path: str | Path | None) -> Config:
    cfg = Config()
    if path is None:
        return cfg
    path = Path(path)
    if not path.exists():
        return cfg
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    return _merge_into_dataclass(cfg, raw)


def resolve_path(cfg_relative: str) -> Path:
    """Config paths are relative to the repo root unless already absolute."""
    p = Path(cfg_relative)
    return p if p.is_absolute() else REPO_ROOT / p


def default_config_dict() -> dict:
    """Round-trippable dict for writing an example config."""
    def _dc_to_dict(obj):
        if is_dataclass(obj):
            return {f.name: _dc_to_dict(getattr(obj, f.name)) for f in fields(obj)}
        return copy.deepcopy(obj)
    return _dc_to_dict(Config())
