
from __future__ import annotations
from typing import Dict, Any, List
from .base import Tool, ToolError
import imaplib, email, ssl

HINTS = ["quote","premium","bindable","renewal","proposal"]

class EmailReaderTool(Tool):
    name = "email_read_quotes"
    description = "Scan IMAP inbox for quote-related messages."
    input_schema = {"type":"object","properties":{"imap_host":{"type":"string"},"imap_user":{"type":"string"},"imap_pass":{"type":"string"},"folder":{"type":"string","default":"INBOX"},"max_emails":{"type":"integer","default":50}},"required":["imap_host","imap_user","imap_pass"]}
    output_schema = {"type":"object","properties":{"matches":{"type":"array"}}}

    def run(self, **kwargs) -> Dict[str, Any]:
        host, user, pw = kwargs["imap_host"], kwargs["imap_user"], kwargs["imap_pass"]
        folder = kwargs.get("folder","INBOX"); max_emails = kwargs.get("max_emails",50)
        try:
            ctx = ssl.create_default_context()
            M = imaplib.IMAP4_SSL(host, ssl_context=ctx); M.login(user, pw); M.select(folder)
            typ, data = M.search(None, "ALL"); ids = data[0].split()[-max_emails:]
            matches: List[Dict[str, Any]] = []
            for eid in reversed(ids):
                typ, msg_data = M.fetch(eid, "(RFC822)")
                msg = email.message_from_bytes(msg_data[0][1])
                frm = email.utils.parseaddr(msg.get("From"))[1]; subj = msg.get("Subject","")
                # body
                body = ""
                if msg.is_multipart():
                    for part in msg.walk():
                        if part.get_content_type() == "text/plain":
                            body = (part.get_payload(decode=True) or b"").decode(errors="ignore"); break
                else:
                    payload = msg.get_payload(decode=True) or b""; body = payload.decode(errors="ignore")
                text = (subj + "\n\n" + body).lower()
                if any(h in text for h in HINTS):
                    matches.append({"from": frm, "subject": subj, "snippet": body[:500]})
            M.logout(); return {"matches": matches}
        except Exception as e:
            raise ToolError(f"IMAP error: {e}")
