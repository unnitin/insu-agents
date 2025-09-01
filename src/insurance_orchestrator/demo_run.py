
from __future__ import annotations
import argparse, json
from dataclasses import asdict
from data_models import WorldState
from insurance_orchestrator.orchestrator import run_pipeline

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", dest="zip_code", default="98109")
    ap.add_argument("--pdf", dest="pdf_path", default=None)
    args = ap.parse_args()

    state = WorldState(user_zip=args.zip_code)
    if args.pdf:
        state.policy.raw_text_refs = [args.pdf]

    state = run_pipeline(state, max_iters=3, planner='heuristic')
    print(json.dumps(asdict(state), indent=2))

if __name__ == "__main__":
    main()
