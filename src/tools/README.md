# Tools Module

The tools module provides a comprehensive suite of specialized tools for automating insurance-related tasks. It implements a plugin-based architecture with standardized interfaces for document processing, web research, communication, and data extraction operations.

## Overview

This module serves as the operational backbone of the insurance system, providing concrete implementations for all insurance-specific tasks. Each tool follows a consistent interface pattern and can operate independently or as part of orchestrated workflows.

## Architecture

### Base Framework

**Tool Base Class (`base.py`)**
- Standard interface definition for all insurance tools
- Common error handling and validation patterns
- Schema-based input/output validation
- Extensible architecture for custom tool development

### Core Processing Engines

**Insurance Base Agent (`core/insurance_base_agent.py`)**
- Foundation class for AI-powered insurance agents
- OpenAI and HuggingFace backend support with automatic selection
- Graceful degradation when AI services are unavailable
- Common text generation and model management capabilities

**PDF Reader (`core/pdf_reader.py`)**
- Advanced PDF text extraction and analysis
- Insurance policy structure recognition
- Multi-format PDF support with fallback mechanisms
- Structured data extraction from unstructured documents

**Insurance Card Processor (`core/insurance_card_processor.py`)**
- OCR-based insurance card image processing
- Image preprocessing and enhancement
- Structured information extraction from card images
- Support for multiple card formats and layouts

**Asset Researcher (`core/asset_researcher.py`)**
- AI-powered asset information research and enrichment
- Vehicle, property, and personal item analysis
- Market research and valuation capabilities
- Integration with external data sources

**Web Researcher (`core/web_researcher.py`)**
- Intelligent web research for insurance providers
- Lead generation and contact information discovery
- Geographic and demographic targeting
- Search result analysis and ranking

## Tool Implementations

### Document Processing Tools

**PDF Reader Tool (`pdf_reader_tool.py`)**
- Extracts text and policy information from insurance PDFs
- Handles various PDF formats with intelligent fallback
- Structures extracted data into policy models
- Supports both simple text extraction and AI-enhanced parsing

**Card OCR Tool (`card_ocr_tool.py`)**
- Processes insurance card images from directories
- Extracts structured information using OCR technology
- Handles multiple image formats and qualities
- Returns standardized insurance card data models

### Research and Discovery Tools

**Web Research Tool (`web_research_tool.py`)**
- Discovers insurance providers in specified geographic areas
- Generates leads with contact information
- Supports multiple asset types (auto, home, personal property)
- Configurable result limits and filtering

**Asset Research Tool (`asset_research_tool.py`)**
- Researches and enriches asset information
- Gathers detailed specifications and market data
- Provides comprehensive asset profiles
- Supports vehicles, properties, and personal items

### Communication Tools

**Email Draft Tool (`email_draft_tool.py`)**
- Generates professional insurance inquiry emails
- Customizes content based on recipient and assets
- Includes policy information and asset summaries
- Creates ready-to-send correspondence

**Email Reader Tool (`email_reader_tool.py`)**
- Processes incoming email responses
- Extracts quote information and contact details
- Identifies follow-up requirements
- Integrates with email service providers

**Voice Call Tool (`voice_call_tool.py`)**
- Initiates automated phone calls for quotes
- Integrates with Twilio for voice services
- Handles call session management
- Provides callback integration for responses

### Form Automation Tools

**Form Filler Tool (`form_filler_tool.py`)**
- Automates web form completion for quote requests
- Generates structured payloads from world state
- Handles various form formats and requirements
- Supports batch processing for multiple providers

## Key Features

### Standardized Interface
- Consistent tool interface across all implementations
- Schema-based validation for inputs and outputs
- Standardized error handling and reporting
- Plugin-compatible architecture

### AI Integration
- OpenAI GPT integration for intelligent processing
- HuggingFace transformer support for open-source models
- Automatic backend selection and fallback
- Configurable model parameters and behavior

### Robust Processing
- Multiple fallback mechanisms for reliability
- Graceful degradation when services are unavailable
- Comprehensive error handling and recovery
- Detailed logging and debugging support

### Flexible Configuration
- Environment-based configuration management
- Optional dependency handling
- Configurable timeouts and retry policies
- Development and production mode support

## Usage Examples

### PDF Processing
```python
from tools import PdfReaderTool

tool = PdfReaderTool()
result = tool.run(pdf_path="policy.pdf")

print(f"Extracted text length: {len(result['raw_text'])}")
print(f"Policy number: {result['policy'].get('policy_number', 'Not found')}")
```

### Insurance Card OCR
```python
from tools import CardOCRTool

tool = CardOCRTool()
result = tool.run(images_dir="input/images")

for card in result['cards']:
    print(f"Policy: {card.policy_number}")
    print(f"Carrier: {card.carrier_name}")
```

### Web Research
```python
from tools import WebResearchTool

tool = WebResearchTool()
result = tool.run(
    zip="12345",
    asset_types=["auto", "home"],
    top_k=5
)

print(f"Found {len(result['leads'])} potential providers")
```

### Asset Research
```python
from tools import AssetResearchTool

tool = AssetResearchTool()
result = tool.run(
    vehicles=[{"make": "Toyota", "model": "Camry", "year": "2020"}],
    properties=[{"address": "123 Main St", "type": "Single Family"}]
)

print(f"Researched {len(result['vehicles'])} vehicles")
print(f"Researched {len(result['properties'])} properties")
```

### Email Communication
```python
from tools import EmailDraftTool

tool = EmailDraftTool()
result = tool.run(
    recipient_name="State Farm Agent",
    sender_name="John Doe",
    zip="12345",
    lines="auto + home",
    policy={"carrier": "Current Insurer"},
    vehicles=[{"make": "Honda", "model": "Civic"}],
    properties=[{"type": "Condo", "address": "456 Oak Ave"}]
)

print("Generated email:")
print(result['email_content'])
```

## Configuration

### Environment Variables
- `OPENAI_API_KEY`: OpenAI API key for AI-powered tools
- `TWILIO_ACCOUNT_SID`: Twilio account for voice services
- `TWILIO_AUTH_TOKEN`: Twilio authentication token
- `GMAIL_CREDENTIALS`: Gmail API credentials file path

### Optional Dependencies
- **PIL/Pillow**: Image processing for OCR operations
- **pytesseract**: OCR engine for text extraction
- **OpenCV**: Advanced image preprocessing
- **OpenAI**: AI-powered text generation
- **Transformers**: Open-source AI model support

### Tool-Specific Configuration
- **OCR Settings**: Tesseract configuration and language packs
- **AI Models**: Model selection and parameters
- **Web Research**: Search engine preferences and limits
- **Communication**: Service provider settings and templates

## Error Handling

### Graceful Degradation
- Tools continue operating when optional dependencies are missing
- Fallback mechanisms for service unavailability
- Alternative processing methods for reliability

### Validation and Safety
- Input schema validation prevents invalid operations
- Output validation ensures data consistency
- Safety checks for potentially harmful operations

### Logging and Debugging
- Comprehensive logging for all tool operations
- Debug modes for development and troubleshooting
- Error tracking and reporting capabilities

## Integration Patterns

### Orchestrator Integration
```python
from toolregistry import ToolRegistry
from tools import *

# Register tools with orchestrator
tools = [
    PdfReaderTool(),
    CardOCRTool(), 
    WebResearchTool(),
    AssetResearchTool(),
    EmailDraftTool()
]

registry = ToolRegistry(tools)
```

### Custom Tool Development
```python
from tools.base import Tool, ToolError

class CustomTool(Tool):
    name = "custom_tool"
    description = "Custom insurance processing tool"
    input_schema = {"type": "object", "properties": {...}}
    output_schema = {"type": "object", "properties": {...}}
    
    def run(self, **kwargs):
        # Custom implementation
        return {"result": "processed"}
```

## Best Practices

1. **Schema Validation**: Always define clear input/output schemas
2. **Error Handling**: Implement comprehensive error handling with meaningful messages
3. **Fallback Strategies**: Provide alternative processing methods when primary methods fail
4. **Resource Management**: Properly manage external service connections and resources
5. **Testing**: Validate tools with representative data and edge cases
6. **Documentation**: Maintain clear documentation for tool capabilities and limitations

## Testing and Development

### Unit Testing
- Individual tool testing with mock data
- Schema validation testing
- Error condition testing
- Performance benchmarking

### Integration Testing
- End-to-end workflow testing
- Service integration validation
- Error recovery testing
- Load testing for production scenarios

### Development Tools
- Debug modes for detailed operation logging
- Mock services for offline development
- Test data generators for consistent testing
- Performance profiling utilities

## Dependencies

### Core Dependencies
- Python 3.9+
- Data Models module (for structured data)
- Standard library modules (pathlib, logging, etc.)

### Optional Dependencies
- OpenAI library (AI-powered features)
- Pillow (image processing)
- pytesseract (OCR capabilities)
- OpenCV (advanced image processing)
- Transformers (open-source AI models)
- Twilio SDK (voice services)
- Gmail API (email services)

## Performance Considerations

### Optimization Strategies
- Lazy loading of optional dependencies
- Caching for repeated operations
- Parallel processing where applicable
- Resource pooling for service connections

### Scalability
- Stateless tool design for horizontal scaling
- Configurable timeouts and limits
- Memory-efficient processing for large documents
- Batch processing capabilities

## Future Enhancements

- Enhanced AI model integration and fine-tuning
- Additional document format support (Word, Excel, etc.)
- Advanced image preprocessing and OCR accuracy
- Real-time web scraping and monitoring
- Integration with additional communication channels
- Machine learning-based result ranking and filtering
- Custom plugin framework for third-party extensions
- Enhanced security and privacy features
