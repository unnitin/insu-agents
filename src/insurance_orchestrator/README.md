
# Agentic Insurance Orchestrator Module

The insurance_orchestrator module implements an intelligent agent-based system for automating insurance renewal processes. It follows a Plan-Act-Observe loop to systematically gather information, research options, and coordinate quote requests across multiple insurance providers.

## Overview

This module serves as the central coordination engine for insurance-related workflows. It manages the complete lifecycle from understanding current coverage through obtaining competitive quotes, using a combination of automated tools and intelligent decision-making.

## Architecture

### Core Components

**Orchestrator (`orchestrator.py`)**
- Main coordination engine implementing the Plan-Act-Observe pattern
- Tool registry management and execution pipeline
- State management and progress tracking
- Policy-based decision making with approval workflows

**Planners (`planners/`)**
- Strategic planning components for workflow orchestration
- LLM-based planning for complex decision scenarios
- Heuristic planning for standard workflows
- State analysis and action prioritization

**Services (`services/`)**
- External service integrations and adapters
- Communication service implementations
- API clients and protocol handlers

**Validators (`validators.py`)**
- Input validation and schema checking
- Tool argument verification
- Data integrity assurance

**Policies (`policies.py`)**
- Security and approval policy framework
- Tool execution authorization
- Risk assessment and mitigation

## Workflow Process

### 1. Understanding Current Insurance
- **PDF Analysis**: Extract policy information from insurance documents
- **Card OCR**: Process insurance card images for coverage details  
- **Policy Parsing**: Structure and validate insurance policy data

### 2. Asset Discovery
- **Vehicle Information**: Gather comprehensive vehicle details
- **Property Assessment**: Collect property characteristics and features
- **Personal Items**: Catalog valuable personal property requiring coverage

### 3. Provider Research
- **Local Surveys**: Research insurance providers in user's area
- **Lead Generation**: Identify potential quote sources
- **Contact Information**: Gather provider contact details and capabilities

### 4. Quote Coordination
- **Form Filling**: Automated completion of online quote forms
- **Email Drafting**: Professional correspondence with agents
- **Voice Calls**: Automated phone interactions for quotes
- **Bid Management**: Track and coordinate multiple quote requests

## Key Features

### Intelligent Planning
- **Adaptive Workflow**: Adjusts strategy based on available information
- **LLM Integration**: Uses language models for complex planning scenarios
- **State-Driven Decisions**: Makes informed choices based on current world state
- **Priority Management**: Optimizes action ordering for efficiency

### Tool Integration
- **Unified Registry**: Centralized tool management and discovery
- **Schema Validation**: Ensures tool inputs meet requirements
- **Error Handling**: Robust error recovery and reporting
- **Parallel Execution**: Concurrent tool execution where possible

### Policy Framework
- **Security Controls**: Prevents unauthorized or risky operations
- **Approval Workflows**: Human oversight for sensitive actions
- **Audit Trails**: Complete logging of all orchestrator actions
- **Risk Assessment**: Evaluates potential impact of planned actions

### State Management
- **Comprehensive Tracking**: Maintains complete system state
- **Progress Monitoring**: Tracks completion status and next steps
- **Data Persistence**: Preserves state across sessions
- **Rollback Capabilities**: Supports state recovery and undo operations

## Services Integration

### Communication Services
- **Gmail Adapter**: Email integration for correspondence
- **Twilio Client**: Voice and SMS communication capabilities
- **Call Server**: Automated phone interaction management

### External APIs
- **Web Research**: Insurance provider discovery and analysis
- **Form Automation**: Automated web form completion
- **Document Processing**: PDF and image analysis services

## Usage Examples

### Basic Orchestration
```python
from insurance_orchestrator import run_pipeline
from data_models import WorldState

# Initialize with user information
state = WorldState(user_zip="12345")

# Run automated pipeline
final_state = run_pipeline(state, max_iters=5)

# Check results
print(f"Found {len(final_state.leads)} potential providers")
print(f"Submitted {len(final_state.bid_results)} quote requests")
```

### Custom Planning
```python
from insurance_orchestrator.planners.planner_llm import plan_with_llm
from insurance_orchestrator import execute_actions
from toolregistry import ToolRegistry

# Setup tools and state
registry = ToolRegistry([...])  
state_dict = asdict(world_state)

# Plan with LLM
actions = plan_with_llm(state_dict, registry.specs(), llm_call)

# Execute planned actions
results = execute_actions(registry, actions)
```

### Policy Configuration
```python
from insurance_orchestrator.policies import PolicyDecision

def custom_policy(tool_name: str, args: dict) -> PolicyDecision:
    if tool_name == "voice_call":
        # Require approval for phone calls
        return PolicyDecision(True, require_approval=True, 
                            reason="Voice calls require human oversight")
    return PolicyDecision(True)
```

## Configuration

### Environment Variables
- `OPENAI_API_KEY`: OpenAI API key for LLM planning
- `TWILIO_ACCOUNT_SID`: Twilio account identifier
- `TWILIO_AUTH_TOKEN`: Twilio authentication token
- `GMAIL_CREDENTIALS`: Gmail API credentials path

### Planning Configuration
- **Planner Type**: Choose between 'heuristic' or 'llm' planning
- **Max Iterations**: Limit orchestration loop iterations
- **Tool Timeouts**: Configure tool execution timeouts
- **Retry Policies**: Set retry behavior for failed operations

## Tool Registry

The orchestrator manages a comprehensive tool registry including:

- **PDF Reader**: Extract text and policy information from documents
- **Card OCR**: Process insurance card images
- **Web Research**: Discover insurance providers and leads
- **Asset Research**: Gather detailed asset information
- **Form Filler**: Automate web form completion
- **Email Draft**: Generate professional correspondence
- **Voice Call**: Initiate automated phone conversations
- **Email Reader**: Process incoming email responses

## Error Handling

### Validation Errors
- Schema validation failures are logged and reported
- Invalid tool arguments prevent execution
- Data integrity checks ensure consistent state

### Tool Failures
- Individual tool failures don't stop the pipeline
- Error details are captured for debugging
- Fallback strategies maintain workflow continuity

### Policy Violations
- Security policy violations prevent risky operations
- Approval requirements pause execution for human review
- Audit logs track all policy decisions

## Best Practices

1. **State Validation**: Always validate world state before planning
2. **Tool Selection**: Choose appropriate tools based on available data
3. **Error Recovery**: Implement graceful degradation for tool failures
4. **Security First**: Configure appropriate policies for your environment
5. **Monitoring**: Track orchestrator performance and success rates
6. **Testing**: Validate workflows with representative test data

## Integration Points

The orchestrator integrates with:

- **Data Models**: Uses comprehensive data structures for state management
- **Insurance Tools**: Orchestrates all available insurance processing tools
- **External Services**: Integrates with communication and research services
- **User Interfaces**: Provides status updates and approval requests

## Dependencies

- Python 3.9+
- OpenAI library (optional, for LLM planning)
- Twilio SDK (optional, for voice services)
- Gmail API (optional, for email services)
- Insurance Tools module
- Data Models module

## Future Enhancements

- Machine learning-based planning optimization
- Advanced error recovery and retry strategies
- Real-time status monitoring and dashboards
- Integration with additional communication channels
- Enhanced policy framework with custom rule engines
- Workflow templates for common insurance scenarios
