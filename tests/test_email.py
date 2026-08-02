import smtplib
from email.header import decode_header
from email import message_from_string

import pytest

from litdesk import email_digest
from litdesk.config import Config


class FakeSMTP:
    """Stands in for smtplib.SMTP — records calls instead of opening a
    real socket."""

    instances = []

    def __init__(self, host, port, timeout=None):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.starttls_called = False
        self.login_args = None
        self.sendmail_args = None
        self.quit_called = False
        FakeSMTP.instances.append(self)

    def starttls(self):
        self.starttls_called = True

    def login(self, username, password):
        self.login_args = (username, password)

    def sendmail(self, from_addr, to_addrs, msg):
        self.sendmail_args = (from_addr, to_addrs, msg)

    def quit(self):
        self.quit_called = True


@pytest.fixture(autouse=True)
def _reset_fake_smtp():
    FakeSMTP.instances = []
    yield
    FakeSMTP.instances = []


def _cfg(**overrides):
    cfg = Config()
    cfg.email.enabled = True
    cfg.email.smtp_host = "smtp.gmail.com"
    cfg.email.smtp_port = 587
    cfg.email.smtp_username = "me@gmail.com"
    cfg.email.use_tls = True
    cfg.email.from_addr = "me@gmail.com"
    cfg.email.to_addr = "me@gmail.com"
    for k, v in overrides.items():
        setattr(cfg.email, k, v)
    return cfg


def test_send_digest_email_noop_when_disabled(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("SMTP should not be constructed when email.enabled is false")
    monkeypatch.setattr(smtplib, "SMTP", boom)

    cfg = _cfg(enabled=False)
    assert email_digest.send_digest_email(cfg, "subject", "<p>hi</p>") is False


def test_send_digest_email_noop_when_host_missing(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("SMTP should not be constructed when required fields are missing")
    monkeypatch.setattr(smtplib, "SMTP", boom)

    cfg = _cfg(smtp_host="")
    assert email_digest.send_digest_email(cfg, "subject", "<p>hi</p>") is False


def test_send_digest_email_noop_when_password_env_var_missing(monkeypatch):
    monkeypatch.delenv(email_digest.SMTP_PASSWORD_ENV_VAR, raising=False)
    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)

    cfg = _cfg()
    assert email_digest.send_digest_email(cfg, "subject", "<p>hi</p>") is False
    assert FakeSMTP.instances == []


def test_send_digest_email_sends_successfully(monkeypatch):
    monkeypatch.setenv(email_digest.SMTP_PASSWORD_ENV_VAR, "app-password")
    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)

    cfg = _cfg()
    # A non-ASCII subject (matches the em-dash format cli.py actually sends)
    # exercises RFC 2047 header encoding, which is why the assertions below
    # decode it back rather than substring-matching the raw MIME text.
    result = email_digest.send_digest_email(cfg, "LitDesk digest — 2026-08-02", "<p>hi</p>")

    assert result is True
    assert len(FakeSMTP.instances) == 1
    smtp = FakeSMTP.instances[0]
    assert smtp.host == "smtp.gmail.com"
    assert smtp.starttls_called is True
    assert smtp.login_args == ("me@gmail.com", "app-password")
    from_addr, to_addrs, msg = smtp.sendmail_args
    assert from_addr == "me@gmail.com"
    assert to_addrs == ["me@gmail.com"]
    parsed = message_from_string(msg)
    subject, encoding = decode_header(parsed["Subject"])[0]
    subject = subject.decode(encoding) if encoding else subject
    assert subject == "LitDesk digest — 2026-08-02"
    assert "<p>hi</p>" in msg
    assert smtp.quit_called is True


def test_send_digest_email_skips_starttls_when_use_tls_false(monkeypatch):
    monkeypatch.setenv(email_digest.SMTP_PASSWORD_ENV_VAR, "app-password")
    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)

    cfg = _cfg(use_tls=False)
    email_digest.send_digest_email(cfg, "subject", "<p>hi</p>")

    assert FakeSMTP.instances[0].starttls_called is False


def test_send_digest_email_skips_login_when_no_username(monkeypatch):
    monkeypatch.delenv(email_digest.SMTP_PASSWORD_ENV_VAR, raising=False)
    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)

    cfg = _cfg(smtp_username="")
    result = email_digest.send_digest_email(cfg, "subject", "<p>hi</p>")

    assert result is True
    assert FakeSMTP.instances[0].login_args is None


def test_send_digest_email_returns_false_on_smtp_failure(monkeypatch):
    monkeypatch.setenv(email_digest.SMTP_PASSWORD_ENV_VAR, "app-password")

    def boom(*a, **k):
        raise OSError("connection refused")
    monkeypatch.setattr(smtplib, "SMTP", boom)

    cfg = _cfg()
    assert email_digest.send_digest_email(cfg, "subject", "<p>hi</p>") is False


def test_send_digest_email_returns_false_when_sendmail_raises(monkeypatch):
    monkeypatch.setenv(email_digest.SMTP_PASSWORD_ENV_VAR, "app-password")

    class FailingSMTP(FakeSMTP):
        def sendmail(self, *a, **k):
            raise smtplib.SMTPException("rejected")

    monkeypatch.setattr(smtplib, "SMTP", FailingSMTP)

    cfg = _cfg()
    assert email_digest.send_digest_email(cfg, "subject", "<p>hi</p>") is False
    assert FailingSMTP.instances[0].quit_called is True
