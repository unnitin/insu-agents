# Tool Registry Module

The toolregistry module provides a centralized local registry for insurance tools. It implements an in-memory tool management system for direct tool execution and orchestration within a single process.

## Overview

This module serves as a local tool management hub, allowing insurance tools to be registered, discovered, and executed directly within the same process. It provides a unified interface for tool organization and access in monolithic application architectures.

## Architecture

### Core Components

**Tool Registry (`registry.py`)**
- In-memory tool management and organization
- Direct tool instance storage and retrieval
- Comprehensive tool lifecycle management
- Local tool execution and orchestration support

**Key Features**
- Tool registration and unregistration
- Tool discovery and filtering capabilities
- Schema validation and metadata management
- Type-safe tool access and execution

## Core API

### Tool Management Methods

**register(tool: Tool)**
- Register a new tool instance in the registry
- Validates tool name uniqueness
- Raises ValueError for duplicate names

**unregister(name: str) -> bool**
- Remove a tool from the registry
- Returns True if successful, False if not found

**get(name: str) -> Tool**
- Retrieve a tool instance by name
- Raises KeyError if tool not found
- Returns executable tool instance

**get_safe(name: str) -> Optional[Tool]**
- Safely retrieve a tool without exceptions
- Returns None if tool not found

### Tool Discovery Methods

**list() -> List[str]**
- Get list of all registered tool names

**list_tools() -> List[Tool]**
- Get list of all registered tool instances

**specs() -> List[Dict[str, Any]]**
- Get specifications for all tools including schemas

**filter_by_tag(tag: str) -> List[Tool]**
- Filter tools by specific tag
- Returns matching tool instances

## Tool Specifications

### Tool Instance Requirements

Tools must inherit from `tools.base.Tool` and implement:

```python
class ExampleTool(Tool):
    name = "example_tool"
    description = "Example tool description"
    input_schema = {
        "type": "object",
        "properties": {
            "input_param": {"type": "string"}
        },
        "required": ["input_param"]
    }
    output_schema = {
        "type": "object", 
        "properties": {
            "result": {"type": "string"}
        }
    }
    
    def run(self, **kwargs):
        # Tool implementation
        return {"result": "processed"}
```

## Usage Examples

### Basic Registry Usage
```python
from toolregistry import ToolRegistry
from tools.pdf_reader_tool import PdfReaderTool
from tools.card_ocr_tool import CardOCRTool

# Initialize registry with tools
registry = ToolRegistry([
    PdfReaderTool(),
    CardOCRTool()
])

# List available tools
tools = registry.list()
print(f"Available tools: {tools}")

# Get and execute a tool
pdf_tool = registry.get("pdf_reader")
result = pdf_tool.run(pdf_path="document.pdf")
```

### Dynamic Tool Management
```python
from toolregistry import ToolRegistry
from tools.web_research_tool import WebResearchTool

# Start with empty registry
registry = ToolRegistry()

# Register tools dynamically
registry.register(WebResearchTool())

# Check if tool exists
if registry.has_tool("web_research"):
    tool = registry.get("web_research")
    result = tool.run(zip="12345", asset_types=["auto"])

# Unregister tool
registry.unregister("web_research")
```

### Tool Discovery and Filtering
```python
# Get tool specifications
specs = registry.specs()
for spec in specs:
    print(f"Tool: {spec['name']} - {spec['description']}")

# Filter tools by tag (if tools have tags attribute)
document_tools = registry.filter_by_tag("document")

# Safe tool access
tool = registry.get_safe("unknown_tool")
if tool is not None:
    result = tool.run()
```

## Integration Patterns

### Orchestrator Integration
```python
from toolregistry import ToolRegistry
from insurance_orchestrator import execute_actions
from tools.pdf_reader_tool import PdfReaderTool
from tools.card_ocr_tool import CardOCRTool

# Create registry with all available tools
registry = ToolRegistry([
    PdfReaderTool(),
    CardOCRTool(),
    WebResearchTool(),
    AssetResearchTool(),
    FormFillerTool(),
    EmailDraftTool(),
    VoiceCallTool(),
    EmailReaderTool()
])

# Use with orchestrator pipeline
actions = [{"tool": "pdf_reader", "args": {"pdf_path": "policy.pdf"}}]
observations = execute_actions(registry, actions)
```

### Custom Tool Factory
```python
from toolregistry import ToolRegistry

def create_tool_registry() -> ToolRegistry:
    """Factory function to create a fully configured tool registry."""
    registry = ToolRegistry()
    
    # Register core tools
    registry.register(PdfReaderTool())
    registry.register(CardOCRTool())
    registry.register(WebResearchTool())
    
    # Add conditional tools based on configuration
    if os.getenv("ENABLE_VOICE_CALLS"):
        registry.register(VoiceCallTool())
    
    if os.getenv("ENABLE_EMAIL"):
        registry.register(EmailDraftTool())
        registry.register(EmailReaderTool())
    
    return registry

# Use factory
registry = create_tool_registry()
```

## Configuration

### Registry Configuration
- **Storage**: In-memory tool instance storage
- **Thread Safety**: Single-threaded access recommended
- **Tool Validation**: Automatic validation on registration
- **Error Handling**: Comprehensive exception handling for tool operations

### Best Practices
- Initialize registry once per application lifecycle
- Register tools during application startup
- Use factory functions for complex registry setup
- Implement proper error handling for tool execution

## Security Considerations

### Tool Validation
- Tool instances must inherit from base Tool class
- Schema validation for input/output specifications
- Name uniqueness validation prevents conflicts
- Type safety through proper inheritance

### Safe Tool Execution
- Tools are executed in same process context
- No network communication required
- Direct method invocation with exception handling
- Isolated tool execution through registry interface

## Error Handling

### Registry Errors
- `ValueError`: Raised when registering duplicate tool names
- `KeyError`: Raised when accessing non-existent tools
- Safe access methods return None instead of raising exceptions

### Tool Execution Errors
- Tools may raise custom exceptions during execution
- Registry passes through all tool exceptions
- Implement proper try/catch blocks around tool.run() calls

## Performance Considerations

### Memory Usage
- Tools are stored as instances in memory
- Registry scales with number of registered tools
- Consider tool lifecycle management for long-running applications

### Execution Speed
- Direct method calls with minimal overhead
- No serialization or network latency
- Optimal for single-process architectures

## Best Practices

1. **Tool Specifications**: Provide complete and accurate tool specifications
2. **Versioning**: Use semantic versioning for tool updates
3. **Validation**: Validate all inputs and outputs against schemas
4. **Error Handling**: Implement comprehensive error handling and reporting
5. **Security**: Secure tool endpoints and validate registrations
6. **Monitoring**: Monitor service health and tool availability

## Integration Points

The tool registry integrates with:

- **Insurance Orchestrator**: Dynamic tool discovery and loading
- **Insurance Tools**: Tool registration and metadata management
- **Microservices**: Service discovery and registration
- **Monitoring Systems**: Health checks and metrics collection

## Dependencies

### Core Dependencies
- FastAPI (web framework)
- Pydantic (data validation)
- Uvicorn (ASGI server)
- Requests (HTTP client)

### Optional Dependencies
- Database drivers (for persistent storage)
- Authentication libraries (for security)
- Monitoring libraries (for observability)
- Caching libraries (for performance)

## Future Enhancements

- Persistent storage backend (PostgreSQL, MongoDB)
- Authentication and authorization system
- Tool health monitoring and circuit breakers
- Advanced search and filtering capabilities
- Tool dependency management
- Automated tool testing and validation
- Performance monitoring and analytics
- Tool marketplace and discovery features
- Integration with CI/CD pipelines
- Multi-tenant support for enterprise deployments
