
from __future__ import annotations
from typing import Dict, Any
from .base import Tool, ToolError

class VoiceCallTool(Tool):
    name = "voice_call"
    description = "Prepare an outbound quote-request call; returns a call session JSON for the call server."
    input_schema = {"type":"object","properties":{"lead_name":{"type":"string"},"phone":{"type":"string"},"zip":{"type":"string"},"lines":{"type":"string"},"policy":{"type":"object"},"assets_summary":{"type":"string"},"callback_url":{"type":"string"}},"required":["lead_name","phone","zip","lines","callback_url"]}
    output_schema = {"type":"object","properties":{"session":{"type":"object"},"next":{"type":"string"}}}

    def run(self, **kwargs) -> Dict[str, Any]:
        phone = kwargs.get("phone"); cb = kwargs.get("callback_url")
        if not phone or not cb: raise ToolError("phone and callback_url are required")
        session = {
            "to": phone,
            "lead_name": kwargs.get("lead_name"),
            "script": {
                "opening": f"Hello, this is a brief call regarding {kwargs.get('lines','insurance')} renewal in ZIP {kwargs.get('zip','')}. Are you able to provide a quote or direct me to the right contact?",
                "disclosure": "This call may be recorded for note-taking and quote comparison.",
                "context": f"Current policy carrier {kwargs.get('policy',{}).get('carrier','unknown')}, expires {kwargs.get('policy',{}).get('expiry_date','unknown')}. Assets: {kwargs.get('assets_summary') or 'available on request'}.",
                "questions": ["Do you write this line in this ZIP?","What info do you need to generate a quote?","Best email for follow-up?"],
                "closing": "Thank you. I'll follow up by email with details."
            },
            "callbacks": {"events": cb.rstrip('/') + "/api/call/events", "recording": cb.rstrip('/') + "/api/call/recording"}
        }
        return {"session": session, "next": "POST this to your call server /api/call/start"}
