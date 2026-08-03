from __future__ import annotations

from litdesk import config_form
from litdesk.config import Config, load_config


def test_config_to_form_dict_flattens_lists_and_scalars():
    cfg = Config()
    cfg.queries = ["query one", "query two"]
    cfg.seeds.dois = ["10.1/a", "10.1/b"]
    cfg.llm.enabled = True
    cfg.llm.provider = "claude_code"
    cfg.email.enabled = True
    cfg.email.smtp_host = "smtp.gmail.com"

    values = config_form.config_to_form_dict(cfg)

    assert values["queries"] == "query one\nquery two"
    assert values["seeds_dois"] == "10.1/a\n10.1/b"
    assert values["llm_enabled"] is True
    assert values["llm_provider"] == "claude_code"
    assert values["email_enabled"] is True
    assert values["email_smtp_host"] == "smtp.gmail.com"
    assert "email_smtp_password" not in str(values.keys())  # never round-trips through the form


def test_form_to_config_dict_applies_exposed_fields():
    existing = Config()
    form = {
        "queries": "new query one\nnew query two\n",
        "project_description": "working on hexokinase",
        "lookback_days": "14",
        "seeds_dois": "10.1/x\n10.1/y",
        "sources_europepmc": "on",
        # sources_biorxiv omitted -> unchecked
        "journals_always_include": "Nature\nCell",
        "llm_enabled": "on",
        "llm_provider": "api",
        "llm_tldr_top_n": "5",
        "email_enabled": "on",
        "email_smtp_host": "smtp.gmail.com",
        "email_smtp_port": "587",
        "email_from_addr": "me@gmail.com",
        "email_to_addr": "me@gmail.com",
    }

    d = config_form.form_to_config_dict(form, existing)

    assert d["queries"] == ["new query one", "new query two"]
    assert d["project_description"] == "working on hexokinase"
    assert d["lookback_days"] == 14
    assert d["seeds"]["dois"] == ["10.1/x", "10.1/y"]
    assert d["sources"]["europepmc"] is True
    assert d["sources"]["biorxiv"] is False  # absent checkbox -> unchecked
    assert d["journals_always_include"] == ["Nature", "Cell"]
    assert d["llm"]["enabled"] is True
    assert d["llm"]["provider"] == "api"
    assert d["llm"]["tldr_top_n"] == 5
    assert d["email"]["enabled"] is True
    assert d["email"]["smtp_port"] == 587


def test_form_to_config_dict_blank_numeric_field_keeps_existing_value():
    existing = Config()
    existing.lookback_days = 21
    existing.digest.top_n = 40

    d = config_form.form_to_config_dict({}, existing)

    assert d["lookback_days"] == 21
    assert d["digest"]["top_n"] == 40


def test_form_to_config_dict_preserves_unexposed_fields():
    existing = Config()
    existing.db_path = "custom/path.db"
    existing.embeddings.model = "some/custom-model"
    existing.ranker.weight_positive = 2.5
    existing.digest.output_dir = "custom-digests"

    d = config_form.form_to_config_dict({}, existing)

    assert d["db_path"] == "custom/path.db"
    assert d["embeddings"]["model"] == "some/custom-model"
    assert d["ranker"]["weight_positive"] == 2.5
    assert d["digest"]["output_dir"] == "custom-digests"


def test_form_to_config_dict_uploaded_bibtex_path_overrides_seeds_config():
    existing = Config()
    existing.seeds.bibtex_path = "old/path.bib"

    d = config_form.form_to_config_dict({"seeds_bibtex_path_uploaded": "data/uploads/seeds.bib"}, existing)

    assert d["seeds"]["bibtex_path"] == "data/uploads/seeds.bib"


def test_form_to_config_dict_no_upload_keeps_existing_bibtex_path():
    existing = Config()
    existing.seeds.bibtex_path = "old/path.bib"

    d = config_form.form_to_config_dict({}, existing)

    assert d["seeds"]["bibtex_path"] == "old/path.bib"


def test_save_config_yaml_round_trips_through_load_config(tmp_path):
    existing = Config()
    existing.queries = ["a query"]

    # A checkbox only appears in real POST data when checked — "llm_enabled"
    # must be given explicitly here to simulate that, unlike the plain
    # text/list fields above which round-trip fine from an empty form.
    form = {"llm_enabled": "on", "llm_provider": "claude_code"}
    yaml_dict = config_form.form_to_config_dict(form, existing)
    config_path = tmp_path / "config.yaml"
    config_form.save_config_yaml(config_path, yaml_dict)

    reloaded = load_config(config_path)
    assert reloaded.queries == ["a query"]
    assert reloaded.llm.enabled is True
    assert reloaded.llm.provider == "claude_code"
