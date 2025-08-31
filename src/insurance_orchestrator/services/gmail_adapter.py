
from __future__ import annotations
from typing import Dict, Any, List, Optional
import os, base64

try:
    from googleapiclient.discovery import build
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    import pickle
except Exception:
    build = None

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
HINTS = ["quote","premium","bindable","renewal","proposal"]

class GmailAdapter:
    def __init__(self, creds_path: str = "credentials.json", token_path: str = "token.pickle"):
        if build is None:
            raise RuntimeError("Missing google api deps")
        self.creds_path = creds_path; self.token_path = token_path
        self.creds = None; self.service = None

    def _auth(self):
        if os.path.exists(self.token_path):
            with open(self.token_path, "rb") as token:
                self.creds = pickle.load(token)
        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                self.creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(self.creds_path, SCOPES)
                self.creds = flow.run_local_server(port=0)
            with open(self.token_path, "wb") as token:
                pickle.dump(self.creds, token)
        self.service = build("gmail", "v1", credentials=self.creds)

    def ensure(self):
        if not self.service: self._auth()

    def list_quote_messages(self, max_results: int = 50, q: Optional[str] = None) -> List[Dict[str, Any]]:
        self.ensure()
        query = q or " OR ".join(HINTS)
        res = self.service.users().messages().list(userId="me", maxResults=max_results, q=query).execute()
        msgs = []
        for m in res.get("messages", []):
            full = self.service.users().messages().get(userId="me", id=m["id"], format="full").execute()
            headers = {h["name"].lower(): h["value"] for h in full.get("payload", {}).get("headers", [])}
            frm = headers.get("from",""); subj = headers.get("subject","")
            body = ""
            parts = full.get("payload", {}).get("parts", [])
            if parts:
                for p in parts:
                    if p.get("mimeType") == "text/plain" and p.get("body", {}).get("data"):
                        body = base64.urlsafe_b64decode(p["body"]["data"]).decode(errors="ignore"); break
            else:
                if full.get("payload", {}).get("body", {}).get("data"):
                    body = base64.urlsafe_b64decode(full["payload"]["body"]["data"]).decode(errors="ignore")
            text = (subj + "\n\n" + body).lower()
            if any(h in text for h in HINTS):
                msgs.append({"id": m["id"], "from": frm, "subject": subj, "snippet": body[:500]})
        return msgs
