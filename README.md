# Insurance Agentic Suite

A comprehensive AI-powered insurance processing system with automated tools for policy analysis, asset research, and quote orchestration.

## Quick Start

```bash
# Run insurance quote processing (ZIP code required)
python main.py --zip 12345

# Process with specific PDF files
python main.py --zip 90210 --pdf policy1.pdf policy2.pdf

# Use custom input directory
python main.py --zip 98109 --input-dir /path/to/files

# Run tests
python run_tests.py
```

## Main Workflow

The system follows a 3-step process:

1. **Initialize System**: Import files and set up tool registry
2. **Analyze Requirements**: Scan PDFs and files to understand current insurance needs
3. **Research Options**: Use research agents to find and compile quote options

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