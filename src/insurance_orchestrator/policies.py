
from __future__ import annotations
from typing import Dict, Any

class PolicyDecision:
    def __init__(self, allow: bool, reason: str = "", require_approval: bool = False):
        self.allow = allow
        self.reason = reason
        self.require_approval = require_approval

def default_policy(tool_name: str, args: Dict[str, Any]) -> PolicyDecision:
    sensitive = { "voice_call": True, "form_submit": True, "email_send": True }
    if tool_name in sensitive:
        return PolicyDecision(True, f"{tool_name} requires explicit human approval", True)
    return PolicyDecision(True)
