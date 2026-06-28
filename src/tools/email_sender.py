# Sends the final report as HTML email via SendGrid
"""
src/tools/email_sender.py
=========================
Tool: Sends the final research report by email using SendGrid.

ROLE IN THE PIPELINE:
    Called by the Report Agent (src/agents/report_agent.py)
    after the research report is fully generated.
    The report is formatted into a clean HTML email and
    delivered to the configured recipient.

ONE FUNCTION:
    send_report_email(report_text, ticker, recipient_email) -> delivery status

USED BY:
    src/agents/report_agent.py -> calls send_report_email() as final step

GRACEFUL DEGRADATION:
    If SENDGRID_API_KEY is not set in .env, the function logs a warning
    and returns a skipped status instead of crashing the pipeline.
    The report is always saved locally regardless of email status.

HOW TO RUN:
    cd financial-research-agent
    python -m src.tools.email_sender

DEPENDENCIES:
    pip install sendgrid
"""

import sendgrid
from sendgrid.helpers.mail import Mail, Email, To, Content
from src.utils.config import config
from src.utils.logger import get_logger
from src.utils.retry import retry

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION — send_report_email
# Formats the research report as HTML and sends it via SendGrid.
# ─────────────────────────────────────────────────────────────────────────────

@retry(max_attempts=3, initial_wait=2)
def send_report_email(
    report_text:     str,
    ticker:          str,
    recipient_email: str = None,
) -> dict:
    """
    Send the final research report as an HTML email via SendGrid.

    HOW SENDGRID WORKS:
        1. You create a Mail object with: from, to, subject, content
        2. You call sg.client.mail.send.post(request_body=mail.get())
        3. SendGrid returns 202 Accepted — email queued for delivery
        4. SendGrid handles actual delivery, spam filtering, tracking

    WHY HTML EMAIL:
        Plain text emails look unprofessional.
        HTML lets us use headings, colors, spacing — a proper report layout.
        We build a simple HTML template here that wraps the report text.

    WHY 202 NOT 200:
        202 = "Accepted" — SendGrid received it, will deliver asynchronously.
        Email delivery is never instant — it goes through SMTP servers.
        200 = "OK" — would mean instant delivery, which email doesn't support.

    GRACEFUL DEGRADATION:
        If SENDGRID_API_KEY is not configured, we skip sending and return
        a "skipped" status. The pipeline continues normally.
        This prevents a missing optional key from crashing the whole pipeline.

    Args:
        report_text:     The full research report as plain text or markdown
        ticker:          Stock ticker e.g. "NVDA" — used in email subject
        recipient_email: Email address to send to.
                         Defaults to REPORT_RECIPIENT_EMAIL from .env

    Returns:
        dict with:
            success      -> True if sent, False if failed, None if skipped
            recipient    -> email address we sent to
            subject      -> the email subject line
            status_code  -> HTTP status from SendGrid (202 = success)
            message      -> human readable summary of what happened
    """

    # ── Step 1: Check if SendGrid is configured ───────────────────────────────
    # SendGrid key is optional — if missing, skip gracefully
    if not config.sendgrid_api_key:
        logger.warning(
            "SENDGRID_API_KEY not set — skipping email delivery. "
            "Report will be saved locally only."
        )
        return {
            "success":     None,
            "recipient":   recipient_email or config.report_recipient_email,
            "subject":     f"Financial Research Report — {ticker.upper()}",
            "status_code": None,
            "message":     "Skipped — SENDGRID_API_KEY not configured",
        }

    # ── Step 2: Resolve recipient email ───────────────────────────────────────
    # Use passed-in email if provided, otherwise fall back to .env setting
    to_email = recipient_email or config.report_recipient_email

    if not to_email:
        logger.error("No recipient email set — cannot send report")
        return {
            "success":     False,
            "recipient":   None,
            "subject":     f"Financial Research Report — {ticker.upper()}",
            "status_code": None,
            "message":     "Failed — no recipient email configured",
        }

    ticker_upper = ticker.upper()
    subject      = f"Financial Research Report — {ticker_upper}"

    logger.info(f"Sending research report for {ticker_upper} to {to_email}")

    # ── Step 3: Build HTML email body ─────────────────────────────────────────
    # We wrap the plain text report in a clean HTML template.
    # <pre> tag preserves whitespace and line breaks from the report text.
    # font-family: monospace keeps the report layout intact.
    html_body = _build_html_email(report_text, ticker_upper)

    # ── Step 4: Create the SendGrid Mail object ───────────────────────────────
    # Mail() is SendGrid's Python helper to build the request payload
    # Email()  → sender address (must be verified in SendGrid dashboard)
    # To()     → recipient address
    # Content() → the email body with MIME type
    message = Mail(
        from_email = Email(config.sendgrid_from_email),
        to_emails  = To(to_email),
        subject    = subject,
        html_content = Content("text/html", html_body),
    )

    # ── Step 5: Send via SendGrid API ─────────────────────────────────────────
    # SendGridAPIClient authenticates using the API key
    # .client.mail.send.post() makes the actual HTTP POST request
    # to https://api.sendgrid.com/v3/mail/send
    sg       = sendgrid.SendGridAPIClient(api_key=config.sendgrid_api_key)
    response = sg.client.mail.send.post(request_body=message.get())

    # ── Step 6: Check response ────────────────────────────────────────────────
    # 202 = Accepted (success) — SendGrid queued the email for delivery
    # Anything else = something went wrong
    if response.status_code == 202:
        logger.info(
            f"Report emailed successfully to {to_email} "
            f"(status={response.status_code})"
        )
        return {
            "success":     True,
            "recipient":   to_email,
            "subject":     subject,
            "status_code": response.status_code,
            "message":     f"Sent successfully to {to_email}",
        }
    else:
        logger.error(
            f"SendGrid returned unexpected status {response.status_code} "
            f"for recipient {to_email}"
        )
        return {
            "success":     False,
            "recipient":   to_email,
            "subject":     subject,
            "status_code": response.status_code,
            "message":     f"Unexpected status {response.status_code} from SendGrid",
        }


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — _build_html_email
# Wraps the plain text report in a clean HTML email template.
# Private function — only used inside this file.
# ─────────────────────────────────────────────────────────────────────────────

def _build_html_email(report_text: str, ticker: str) -> str:
    """
    Wrap the plain text report in a styled HTML email template.

    WHY A TEMPLATE:
        Raw text emails look like terminal output — unprofessional.
        A simple HTML wrapper with a header, divider, and clean font
        makes the report look like a real financial document.

    Args:
        report_text: The plain text research report
        ticker:      Stock ticker for the header e.g. "NVDA"

    Returns:
        HTML string ready to send as email body
    """
    from datetime import datetime
    generated_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    # We use inline CSS because many email clients (Gmail, Outlook) strip
    # external stylesheets — inline styles are the only reliable option.
    html = f"""
    <html>
    <body style="font-family: Arial, sans-serif; max-width: 800px;
                 margin: 0 auto; padding: 20px; color: #333;">

        <!-- Header -->
        <div style="background-color: #1a1a2e; color: white;
                    padding: 20px 30px; border-radius: 8px 8px 0 0;">
            <h1 style="margin: 0; font-size: 22px;">
                📊 Financial Research Report
            </h1>
            <p style="margin: 6px 0 0; font-size: 14px; color: #aaa;">
                Ticker: <strong style="color: white;">{ticker}</strong>
                &nbsp;|&nbsp; Generated: {generated_at}
            </p>
        </div>

        <!-- Divider -->
        <div style="height: 4px; background: linear-gradient(
                    to right, #667eea, #764ba2);"></div>

        <!-- Report Body -->
        <div style="background: #f9f9f9; padding: 30px;
                    border: 1px solid #ddd; border-top: none;
                    border-radius: 0 0 8px 8px;">
            <pre style="font-family: 'Courier New', monospace;
                        font-size: 13px; line-height: 1.7;
                        white-space: pre-wrap; word-wrap: break-word;
                        color: #222; margin: 0;">
{report_text}
            </pre>
        </div>

        <!-- Footer -->
        <div style="margin-top: 20px; padding: 15px;
                    background: #fff3cd; border-radius: 6px;
                    border-left: 4px solid #ffc107;">
            <p style="margin: 0; font-size: 12px; color: #856404;">
                <strong>Disclaimer:</strong> This report is generated by an AI
                system for informational purposes only and does not constitute
                financial advice. Always consult a qualified financial advisor
                before making investment decisions.
            </p>
        </div>

    </body>
    </html>
    """
    return html


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SANITY CHECK
# Run directly to test email sending.
# Requires SENDGRID_API_KEY and SENDGRID_FROM_EMAIL in your .env file.
#
# Usage:
#   cd financial-research-agent
#   python -m src.tools.email_sender
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print(f"\n{'='*60}")
    print(f"  Email Sender Tool — Sanity Check")
    print(f"{'='*60}")

    # ── Test 1: No SendGrid key — should skip gracefully ──────────────────────
    # Temporarily test the skip behavior by checking config
    print(f"\n── Test 1: SendGrid key status ──────────────────────────────\n")

    if config.sendgrid_api_key:
        print(f"  SendGrid API key : SET ✅")
        print(f"  From email       : {config.sendgrid_from_email or 'NOT SET ⚠️'}")
        print(f"  Recipient email  : {config.report_recipient_email or 'NOT SET ⚠️'}")
    else:
        print(f"  SendGrid API key : NOT SET")
        print(f"  Email sending will be skipped (graceful degradation)")

    # ── Test 2: HTML template generation (no API needed) ─────────────────────
    print(f"\n── Test 2: HTML email template generation ───────────────────\n")

    sample_report = """
NVIDIA Corporation (NVDA) — Research Report
============================================

Executive Summary:
NVIDIA continues to dominate the AI chip market with exceptional
revenue growth driven by data center demand.

Key Metrics:
  Current Price  : $321.50
  Market Cap     : $786.0B
  P/E Ratio      : 38.5x
  Revenue Growth : 122.4%
  Gross Margin   : 55.4%

Recommendation: HOLD
Strong fundamentals but current valuation leaves limited margin of safety.
    """.strip()

    html = _build_html_email(sample_report, "NVDA")
    print(f"  HTML generated   : {len(html):,} characters")
    print(f"  Contains header  : {'Financial Research Report' in html}")
    print(f"  Contains ticker  : {'NVDA' in html}")
    print(f"  Contains disclaimer: {'does not constitute' in html}")
    print(f"  ✅ HTML template working correctly")

    # ── Test 3: Send real email (only if SendGrid is configured) ─────────────
    print(f"\n── Test 3: Send real email ───────────────────────────────────\n")

    result = send_report_email(
        report_text=sample_report,
        ticker="NVDA",
    )

    print(f"  Success      : {result['success']}")
    print(f"  Recipient    : {result['recipient']}")
    print(f"  Subject      : {result['subject']}")
    print(f"  Status Code  : {result['status_code']}")
    print(f"  Message      : {result['message']}")
    print()