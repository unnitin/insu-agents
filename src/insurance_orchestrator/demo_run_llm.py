
from __future__ import annotations
import argparse, json
from dataclasses import asdict
from insurance_orchestrator.schema import WorldState
from insurance_orchestrator.orchestrator import run_pipeline

def llm_call(prompt: str, tools: list[dict]) -> dict:
    # Stub: plan to research first
    return {"tool_calls":[{"name":"web_research","args":{"zip":"98109","asset_types":["auto","home"],"top_k":3}}]}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="98109")
    args = ap.parse_args()
    state = WorldState(user_zip=args.zip)
    state = run_pipeline(state, max_iters=2, planner='llm', llm_call=llm_call)
    print(json.dumps(asdict(state), indent=2))

if __name__ == "__main__":
    main()
