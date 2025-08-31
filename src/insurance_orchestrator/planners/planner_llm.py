
from __future__ import annotations
from typing import Dict, Any, List

def summarize_state(state: Dict[str, Any]) -> str:
    pol = state.get("policy", {})
    return " | ".join([
        f"Policy carrier={pol.get('carrier')} number={pol.get('policy_number')} expires={pol.get('expiry_date')}",
        f"Vehicles={len(state.get('vehicles', []))} Properties={len(state.get('properties', []))}",
        f"Leads={len(state.get('leads', []))} ZIP={state.get('user_zip')}"
    ])

def plan_with_llm(state: Dict[str, Any], tools: List[Dict[str, Any]], llm_call) -> List[Dict[str, Any]]:
    prompt = f"""You are a planning agent for insurance renewals.
State summary: {summarize_state(state)}
Goal: obtain quotes. Choose tools and arguments conservatively.
Prefer: pdf/card parsing → research → outreach (email/call). Stop after 2–4 calls.
"""
    result = llm_call(prompt, tools)
    return result.get("tool_calls", [])
