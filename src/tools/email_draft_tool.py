
from __future__ import annotations
from typing import Dict, Any
from .base import Tool

TEMPLATE = '''Subject: Request for renewal quote

Hello {recipient_name},

I'm shopping for {lines} coverage in {zip}. Here's my current policy:
- Carrier: {carrier}
- Policy #: {policy_number}
- Expires: {expiry}

Assets:
{assets}

Please share an estimated quote range and next steps.
Thanks,
{sender_name}
'''

class EmailDraftTool(Tool):
    name = "email_draft"
    description = "Draft an email to request a renewal quote."
    input_schema = {"type":"object","properties":{"recipient_name":{"type":"string"},"sender_name":{"type":"string"},"zip":{"type":"string"},"lines":{"type":"string"},"policy":{"type":"object"},"vehicles":{"type":"array"},"properties":{"type":"array"}},"required":["recipient_name","sender_name","zip"]}
    output_schema = {"type":"object","properties":{"subject":{"type":"string"},"body":{"type":"string"}}}

    def run(self, **kwargs) -> Dict[str, Any]:
        recipient = kwargs.get("recipient_name","there")
        sender = kwargs.get("sender_name","")
        zip_code = kwargs.get("zip","")
        lines = kwargs.get("lines","insurance")
        policy = kwargs.get("policy",{})
        vehicles = kwargs.get("vehicles",[])
        properties = kwargs.get("properties",[])
        assets_lines = []
        for v in vehicles:
            parts = [v.get("year"), v.get("make"), v.get("model")]
            assets_lines.append("- Vehicle: " + " ".join([p for p in parts if p]))
        for h in properties:
            parts = [h.get("address"), h.get("sqft"), h.get("year_built")]
            assets_lines.append("- Property: " + ", ".join([p for p in parts if p]))
        assets_block = "\n".join(assets_lines) or "- (details available upon request)"
        body = TEMPLATE.format(recipient_name=recipient, lines=lines, zip=zip_code,
                               carrier=policy.get("carrier",""), policy_number=policy.get("policy_number",""),
                               expiry=policy.get("expiry_date",""), assets=assets_block, sender_name=sender)
        return {"subject":"Request for renewal quote","body":body}
