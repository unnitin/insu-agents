
from __future__ import annotations
from typing import Dict, Any, List, Optional
from .base import Tool, ToolError

class AssetResearchTool(Tool):
    name = "asset_research"
    description = "Infer user's insured assets (vehicles, properties, personal items) from prompts, emails, or notes."
    input_schema = {
        "type":"object",
        "properties":{
            "prompt":{"type":"string","description":"Freeform context about the user, their household, and assets."},
            "sources":{"type":"array","items":{"type":"string"},"description":"Optional list of file paths (txt/eml/md) to read."},
            "max_results":{"type":"integer","default":5}
        },
        "required":[]
    }
    output_schema = {
        "type":"object",
        "properties":{
            "vehicles":{"type":"array"},
            "properties":{"type":"array"},
            "personals":{"type":"array"},
            "notes":{"type":"string"}
        }
    }

    def run(self, **kwargs) -> Dict[str, Any]:
        prompt: str = kwargs.get("prompt","")
        sources: List[str] = kwargs.get("sources",[]) or []
        max_results: int = kwargs.get("max_results",5)

        vehicles: List[Dict[str, Any]] = []
        homes: List[Dict[str, Any]] = []
        personals: List[Dict[str, Any]] = []
        notes: str = ""

        # Preferred path: leverage the project's asset researcher if available
        try:
            from insurance_tools.core.asset_researcher import PromptBasedAssetResearcher
            researcher = PromptBasedAssetResearcher(debug=False)
            res = researcher.research_assets(prompt=prompt, sources=sources, max_results=max_results)
            # Expecting res to possibly contain similar keys; normalize defensively
            vehicles = res.get("vehicles", []) if isinstance(res, dict) else []
            homes = res.get("properties", []) if isinstance(res, dict) else res.get("homes", [])
            personals = res.get("personals", [])
            notes = res.get("notes","")
        except Exception:
            # Fallback: naive keyword pulls from prompt
            p = (prompt or "").lower()
            if "tesla" in p or "model" in p:
                vehicles.append({"make":"Tesla"})
            if "home" in p or "condo" in p or "address" in p:
                homes.append({"address":"(inferred)"})
            notes = "Used fallback heuristics; integrate insurance_core.asset_researcher for better results."

        return {"vehicles": vehicles, "properties": homes, "personals": personals, "notes": notes}
