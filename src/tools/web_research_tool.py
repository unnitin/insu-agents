
from __future__ import annotations
from typing import Dict, Any
from .base import Tool

class WebResearchTool(Tool):
    name = "web_research"
    description = "Search for local insurance providers/agents."
    input_schema = {"type":"object","properties":{"zip":{"type":"string"},"asset_types":{"type":"array","items":{"type":"string"}},"top_k":{"type":"integer","default":5}},"required":["zip"]}
    output_schema = {"type":"object","properties":{"leads":{"type":"array"}}}

    def run(self, **kwargs) -> Dict[str, Any]:
        zip_code = kwargs.get("zip")
        top_k = kwargs.get("top_k",5)
        try:
            from tools.core.web_researcher import WebResearcher
            wr = WebResearcher(rate_limit_per_minute=10)
            queries = [f"insurance agents near {zip_code}"]
            leads = []
            for q in queries:
                for r in wr.search(q, max_results=top_k):
                    leads.append({"name": r.get("title") or r.get("name"), "url": r.get("url"), "snippet": r.get("snippet","")})
            return {"leads": leads[:top_k]}
        except Exception:
            return {"leads": []}
