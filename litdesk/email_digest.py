"""Optional email delivery of the rendered digest (spec's local-only
delivery, extended on request). Gated on `email.enabled`; any failure — bad
credentials, unreachable host, timeout — is logged and skipped, matching
the LLM layer's "never break the digest run" contract.
"""

from __future__ import annotations

import logging
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from litdesk.config import Config

logger = logging.getLogger("litdesk.email")

SMTP_PASSWORD_ENV_VAR = "LITDESK_SMTP_PASSWORD"


def send_digest_email(cfg: Config, subject: str, html_body: str) -> bool:
    """Returns True if the email was sent, False if skipped or failed for
    any reason. Never raises."""
    if not cfg.email.enabled:
        return False
    if not cfg.email.smtp_host or not cfg.email.from_addr or not cfg.email.to_addr:
        logger.warning(
            "email.enabled is true but smtp_host/from_addr/to_addr aren't all set — skipping digest email"
        )
        return False

    password = os.environ.get(SMTP_PASSWORD_ENV_VAR, "")
    if cfg.email.smtp_username and not password:
        logger.warning("%s not set — skipping digest email", SMTP_PASSWORD_ENV_VAR)
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = cfg.email.from_addr
    msg["To"] = cfg.email.to_addr
    msg.attach(MIMEText(html_body, "html"))

    try:
        server = smtplib.SMTP(cfg.email.smtp_host, cfg.email.smtp_port, timeout=30)
        try:
            if cfg.email.use_tls:
                server.starttls()
            if cfg.email.smtp_username:
                server.login(cfg.email.smtp_username, password)
            server.sendmail(cfg.email.from_addr, [cfg.email.to_addr], msg.as_string())
        finally:
            server.quit()
    except Exception as exc:  # noqa: BLE001 - optional feature, must never break the digest run
        logger.warning("Failed to send digest email: %s", exc)
        return False

    logger.info("Digest emailed to %s", cfg.email.to_addr)
    return True
