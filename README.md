# Insurance Agentic Suite

A comprehensive AI-powered insurance processing system with automated tools for policy analysis, asset research, and quote orchestration.

## Quick Start

```bash
# Run the main demo
python main.py

# Run specific components
python main.py --tools-only          # Demo individual tools
python main.py --orchestrator-only   # Demo orchestration pipeline
python main.py --pdf policy.pdf      # Include PDF processing

# Run tests
python run_tests.py
```

## Architecture

**Core Packages:**
- `data_models/`: Comprehensive data models for insurance entities
- `tools/`: Specialized tools for insurance processing tasks  
- `insurance_orchestrator/`: Orchestrator, services, and planners
- `toolregistry/`: Local tool registry management

**Key Features:**
- PDF policy document processing
- Insurance card OCR
- Web research for provider discovery
- Asset research and analysis
- Email drafting and communication
- Automated orchestration pipeline