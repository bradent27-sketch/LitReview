"""Converts between the web control panel's Settings form and config.yaml.
Keeps every form field name and the YAML shape it maps to in one place, so
the template and this module can't silently drift apart.

Fields not covered here (infra paths like db_path/cache_dir, embedding
model choice, ranker weight internals) are intentionally left out of the
web form — they're either dangerous to fat-finger or too in-the-weeds to
be worth surfacing — and are always carried over unchanged from whatever
`existing` config already has.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from litdesk.config import Config, config_to_dict


def _lines_to_list(text: str) -> list[str]:
    return [line.strip() for line in (text or "").splitlines() if line.strip()]


def _list_to_lines(values: list[str]) -> str:
    return "\n".join(values or [])


def config_to_form_dict(cfg: Config) -> dict:
    """Flat dict of form-field-name -> current value, used to pre-fill the
    settings page. Deliberately excludes the SMTP password — that field is
    session-only and never round-trips through config.yaml or this form."""
    return {
        "queries": _list_to_lines(cfg.queries),
        "project_description": cfg.project_description,
        "lookback_days": cfg.lookback_days,

        "seeds_dois": _list_to_lines(cfg.seeds.dois),
        "seeds_bibtex_path": cfg.seeds.bibtex_path or "",

        "sources_europepmc": cfg.sources.europepmc,
        "sources_biorxiv": cfg.sources.biorxiv,
        "sources_medrxiv": cfg.sources.medrxiv,

        "journals_always_include": _list_to_lines(cfg.journals_always_include),
        "watchlists_authors": _list_to_lines(cfg.watchlists.authors),

        "llm_enabled": cfg.llm.enabled,
        "llm_provider": cfg.llm.provider,
        "llm_tldr_top_n": cfg.llm.tldr_top_n,

        "email_enabled": cfg.email.enabled,
        "email_smtp_host": cfg.email.smtp_host,
        "email_smtp_port": cfg.email.smtp_port,
        "email_smtp_username": cfg.email.smtp_username,
        "email_use_tls": cfg.email.use_tls,
        "email_from_addr": cfg.email.from_addr,
        "email_to_addr": cfg.email.to_addr,

        "scoop_alarm_enabled": cfg.scoop_alarm.enabled,
        "scoop_alarm_threshold": cfg.scoop_alarm.threshold,

        # --- advanced / rarely touched ---
        "biorxiv_categories": _list_to_lines(cfg.biorxiv_categories),
        "digest_top_n": cfg.digest.top_n,
        "digest_catchup_gap_days": cfg.digest.catchup_gap_days,
        "digest_catchup_top_n": cfg.digest.catchup_top_n,
        "watchlists_labs": _list_to_lines(cfg.watchlists.labs),
        "llm_tldr_model": cfg.llm.tldr_model,
        "llm_synthesis_model": cfg.llm.synthesis_model,
        "llm_claude_code_timeout_seconds": cfg.llm.claude_code_timeout_seconds,
        "ranker_min_ratings_for_classifier": cfg.ranker.min_ratings_for_classifier,
        "rate_limits_europepmc_per_sec": cfg.rate_limits.europepmc_per_sec,
        "rate_limits_biorxiv_per_sec": cfg.rate_limits.biorxiv_per_sec,
        "server_port": cfg.server.port,
        "cache_ttl_hours": cfg.cache_ttl_hours,
    }


def form_to_config_dict(form: dict, existing: Config) -> dict:
    """Builds the nested, yaml.safe_dump-shaped dict for the new
    config.yaml: starts from `existing`'s current full state (so anything
    the form doesn't cover is preserved untouched) and overlays just the
    fields the settings page actually exposes."""
    d = config_to_dict(existing)
    _MISSING = object()

    def get(name: str, default: str = "") -> str:
        return form.get(name, default)

    def checked(name: str) -> bool:
        # HTML only includes a checkbox in the POST body when it's checked.
        return name in form

    def as_int(name: str, current: int) -> int:
        raw = get(name)
        return int(raw) if str(raw).strip() else current

    def as_float(name: str, current: float) -> float:
        raw = get(name)
        return float(raw) if str(raw).strip() else current

    def as_list(name: str, current: list[str]) -> list[str]:
        # A key entirely absent from the form (as opposed to present but
        # blank) means this submission never touched the field at all —
        # preserve whatever was already there rather than wiping it out.
        raw = form.get(name, _MISSING)
        return list(current) if raw is _MISSING else _lines_to_list(raw)

    d["queries"] = as_list("queries", existing.queries)
    d["project_description"] = get("project_description", "")
    d["lookback_days"] = as_int("lookback_days", existing.lookback_days)

    d["seeds"]["dois"] = as_list("seeds_dois", existing.seeds.dois)
    uploaded_bibtex_path = get("seeds_bibtex_path_uploaded", "").strip()
    if uploaded_bibtex_path:
        d["seeds"]["bibtex_path"] = uploaded_bibtex_path

    d["sources"]["europepmc"] = checked("sources_europepmc")
    d["sources"]["biorxiv"] = checked("sources_biorxiv")
    d["sources"]["medrxiv"] = checked("sources_medrxiv")

    d["journals_always_include"] = as_list("journals_always_include", existing.journals_always_include)
    d["watchlists"]["authors"] = as_list("watchlists_authors", existing.watchlists.authors)

    d["llm"]["enabled"] = checked("llm_enabled")
    d["llm"]["provider"] = get("llm_provider") or existing.llm.provider
    d["llm"]["tldr_top_n"] = as_int("llm_tldr_top_n", existing.llm.tldr_top_n)

    d["email"]["enabled"] = checked("email_enabled")
    d["email"]["smtp_host"] = get("email_smtp_host", "")
    d["email"]["smtp_port"] = as_int("email_smtp_port", existing.email.smtp_port)
    d["email"]["smtp_username"] = get("email_smtp_username", "")
    d["email"]["use_tls"] = checked("email_use_tls")
    d["email"]["from_addr"] = get("email_from_addr", "")
    d["email"]["to_addr"] = get("email_to_addr", "")

    d["scoop_alarm"]["enabled"] = checked("scoop_alarm_enabled")
    d["scoop_alarm"]["threshold"] = as_float("scoop_alarm_threshold", existing.scoop_alarm.threshold)

    d["biorxiv_categories"] = as_list("biorxiv_categories", existing.biorxiv_categories)
    d["digest"]["top_n"] = as_int("digest_top_n", existing.digest.top_n)
    d["digest"]["catchup_gap_days"] = as_int("digest_catchup_gap_days", existing.digest.catchup_gap_days)
    d["digest"]["catchup_top_n"] = as_int("digest_catchup_top_n", existing.digest.catchup_top_n)
    d["watchlists"]["labs"] = as_list("watchlists_labs", existing.watchlists.labs)
    d["llm"]["tldr_model"] = get("llm_tldr_model") or existing.llm.tldr_model
    d["llm"]["synthesis_model"] = get("llm_synthesis_model") or existing.llm.synthesis_model
    d["llm"]["claude_code_timeout_seconds"] = as_int(
        "llm_claude_code_timeout_seconds", existing.llm.claude_code_timeout_seconds
    )
    d["ranker"]["min_ratings_for_classifier"] = as_int(
        "ranker_min_ratings_for_classifier", existing.ranker.min_ratings_for_classifier
    )
    d["rate_limits"]["europepmc_per_sec"] = as_float(
        "rate_limits_europepmc_per_sec", existing.rate_limits.europepmc_per_sec
    )
    d["rate_limits"]["biorxiv_per_sec"] = as_float(
        "rate_limits_biorxiv_per_sec", existing.rate_limits.biorxiv_per_sec
    )
    d["server"]["port"] = as_int("server_port", existing.server.port)
    d["cache_ttl_hours"] = as_int("cache_ttl_hours", existing.cache_ttl_hours)

    return d


def save_config_yaml(config_path: str | Path, yaml_dict: dict) -> None:
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Written by the LitDesk web control panel's Settings page.\n"
        "# Hand-editing is still fine — it's reloaded fresh on every page load.\n\n"
    )
    config_path.write_text(header + yaml.safe_dump(yaml_dict, sort_keys=False))
