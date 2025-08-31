
from __future__ import annotations
from typing import Dict, Any, List
from dataclasses import asdict
from .schema import WorldState, Lead
from .tools.base import Tool
from .tools.pdf_reader_tool import PdfReaderTool
from .tools.card_ocr_tool import CardOCRTool
from .tools.web_research_tool import WebResearchTool
from .tools.form_filler_tool import FormFillerTool
from .tools.email_draft_tool import EmailDraftTool
from .tools.voice_call_tool import VoiceCallTool
from .tools.email_reader_tool import EmailReaderTool
from .tools.asset_research_tool import AssetResearchTool
from .validators import validate_args
from .policies import default_policy
from .planners.planner_llm import plan_with_llm

class ToolRegistry:
    def __init__(self, tools: List[Tool]):
        self._tools = {t.name: t for t in tools}
    def get(self, name: str) -> Tool:
        return self._tools[name]
    def list(self) -> List[str]:
        return list(self._tools.keys())
    def specs(self) -> List[Dict[str, Any]]:
        return [{"name": t.name, "description": getattr(t,"description",""), "input_schema": getattr(t,"input_schema",{}), "output_schema": getattr(t,"output_schema",{})} for t in self._tools.values()]

def plan_next(state: WorldState) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []
    if not (state.policy.policy_number or state.policy.carrier):
        actions.append({"tool":"pdf_reader","args":{"pdf_path": state.policy.raw_text_refs[0] if state.policy.raw_text_refs else ""}})
    if not state.vehicles or not state.properties:
        try:
            from insurance_core import resolve
            actions.append({"tool":"card_ocr","args":{"images_dir": str(resolve("input","images"))}})
        except Exception:
            actions.append({"tool":"card_ocr","args":{"images_dir":"./input/images"}})
    if state.user_zip and not state.leads:
        actions.append({"tool":"web_research","args":{"zip": state.user_zip, "asset_types":["auto","home"], "top_k":5}})
    if state.leads and not state.bid_results:
        actions.append({"tool":"voice_call","args":{
            "lead_name": state.leads[0].name,
            "phone": state.leads[0].phone or "+10000000000",
            "zip": state.user_zip or "",
            "lines": "auto + home",
            "policy": asdict(state.policy),
            "assets_summary": f"{len(state.vehicles)} vehicles, {len(state.properties)} properties",
            "callback_url": "http://localhost:8790"
        }})
    if state.leads and not state.bid_intents:
        actions.append({"tool":"form_filler","args":{"state": asdict(state)}})
        top = state.leads[0]
        actions.append({"tool":"email_draft","args":{
            "recipient_name": top.name, "sender_name":"(Your Name)",
            "zip": state.user_zip or "", "lines":"auto + home",
            "policy": asdict(state.policy),
            "vehicles": [asdict(v) for v in state.vehicles],
            "properties": [asdict(h) for h in state.properties]
        }})
    return actions

def execute_actions(registry: ToolRegistry, actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    observations = []
    for act in actions:
        tool = registry.get(act["tool"])
        args = act.get("args", {})
        ok, err = validate_args(args, getattr(tool, "input_schema", {}))
        if not ok:
            observations.append({"tool": tool.name, "error": f"Schema validation failed: {err}", "args": args})
            continue
        decision = default_policy(tool.name, args)
        if not decision.allow:
            observations.append({"tool": tool.name, "error": f"Blocked by policy: {decision.reason}", "args": args})
            continue
        if decision.require_approval:
            observations.append({"tool": tool.name, "require_approval": True, "reason": decision.reason, "args": args})
            continue
        try:
            result = tool.run(**args)
            observations.append({"tool": tool.name, "result": result})
        except Exception as e:
            observations.append({"tool": tool.name, "error": str(e), "args": args})
    return observations

def observe_and_update(state: WorldState, observations: List[Dict[str, Any]]) -> None:
    for obs in observations:
        if "error" in obs or "require_approval" in obs:
            continue
        name, res = obs["tool"], obs["result"]
        if name == "pdf_reader":
            pol = res.get("policy",{})
            state.policy.carrier = pol.get("carrier") or state.policy.carrier
            state.policy.policy_number = pol.get("policy_number") or state.policy.policy_number
            state.policy.effective_date = pol.get("effective_date") or state.policy.effective_date
            state.policy.expiry_date = pol.get("expiry_date") or state.policy.expiry_date
            state.policy.coverages.update(pol.get("coverages",{}))
        elif name == "card_ocr":
            for c in res.get("cards",[]):
                p = c.get("parsed",{})
                if p.get("policy_number"): state.policy.policy_number = state.policy.policy_number or p["policy_number"]
                if p.get("carrier"): state.policy.carrier = state.policy.carrier or p["carrier"]
        elif name == "web_research":
            for l in res.get("leads",[]):
                state.leads.append(Lead(name=l.get("name",""), url=l.get("url")))
        elif name == "form_filler":
            pass
        elif name == "email_draft":
            pass

def run_pipeline(state: WorldState, max_iters: int = 3, planner: str = 'heuristic', llm_call=None) -> WorldState:
    registry = ToolRegistry([PdfReaderTool(), CardOCRTool(), WebResearchTool(), AssetResearchTool(), FormFillerTool(), EmailDraftTool(), VoiceCallTool(), EmailReaderTool()])
    for _ in range(max_iters):
        if planner == 'llm' and llm_call is not None:
            tool_calls = plan_with_llm(asdict(state), registry.specs(), llm_call)
            actions = [{"tool": c.get("name"), "args": c.get("args", {})} for c in tool_calls]
        else:
            actions = plan_next(state)
        if not actions: break
        obs = execute_actions(registry, actions)
        observe_and_update(state, obs)
    return state
