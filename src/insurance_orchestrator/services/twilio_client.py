
from __future__ import annotations
import os
from typing import Optional, Dict, Any
try:
    from twilio.rest import Client
except Exception:
    Client = None

class TwilioClient:
    def __init__(self, account_sid: Optional[str] = None, auth_token: Optional[str] = None, from_number: Optional[str] = None):
        self.sid = account_sid or os.getenv("TWILIO_ACCOUNT_SID")
        self.token = auth_token or os.getenv("TWILIO_AUTH_TOKEN")
        self.from_number = from_number or os.getenv("TWILIO_FROM_NUMBER")
        if not all([self.sid, self.token, self.from_number]):
            raise RuntimeError("Missing Twilio env: TWILIO_ACCOUNT_SID/AUTH_TOKEN/FROM_NUMBER")
        if Client is None:
            raise RuntimeError("twilio not installed")
        self.client = Client(self.sid, self.token)

    def place_call(self, to: str, webhook_url: str, status_callback_url: Optional[str] = None) -> Dict[str, Any]:
        call = self.client.calls.create(to=to, from_=self.from_number, url=webhook_url,
                                        status_callback=status_callback_url,
                                        status_callback_event=["initiated","ringing","answered","completed"] if status_callback_url else None)
        return {"sid": call.sid, "status": call.status}

    def fetch_recordings(self, call_sid: str) -> Dict[str, Any]:
        recs = self.client.recordings.list(call_sid=call_sid)
        return {"count": len(recs), "recordings": [{"sid": r.sid, "duration": r.duration, "date_created": str(r.date_created)} for r in recs]}
