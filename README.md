# Insurance Agentic Suite (src-layout)

Packages:
- insurance_core: original tools & data models
- insurance_orchestrator: orchestrator, tools, services, planners
- toolregistry_service: optional tool registry


### Consolidation
Common tool implementations are now in `src/insurance_tools/` and imported by both `insurance_core` and `insurance_orchestrator`.