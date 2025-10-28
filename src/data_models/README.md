# Data Models Module

The data_models module serves as the single source of truth for all data structures used throughout the insurance system. It provides comprehensive, type-safe data models for representing vehicles, properties, personal items, insurance policies, and orchestrator workflow states.

## Overview

This module consolidates all data models into a cohesive package with well-defined relationships and validation capabilities. The models support the entire insurance workflow from asset discovery through quote processing and policy management.

## Core Components

### Asset Models

**Vehicle (`vehicle_models.py`)**
- Comprehensive vehicle data class with basic information (make, model, year, VIN)
- Optional detailed fields for specifications, safety features, usage patterns
- Insurance coverage information and risk factors
- Support for modifications and aftermarket parts

**Property (`property_models.py`)**
- Complete property profile with basic information (type, address, year built)
- Construction details (roof type, foundation, exterior materials)
- Interior systems (heating, cooling, electrical, plumbing)
- Safety features and coverage information

**PersonalItem (`personal_models.py`)**
- Personal property items requiring separate coverage
- Category classification and valuation information
- Documentation and proof of ownership tracking

### Policy and Analysis Models

**PolicySummary (`policy_models.py`)**
- Comprehensive insurance policy information
- Policy identification, dates, premium details
- Coverage breakdown and carrier information
- Utility methods for premium calculations and status validation
- Date-based operations for policy lifecycle management

**AnalysisMetadata (`policy_models.py`)**
- Document analysis and processing operation metadata
- Performance metrics and data quality assessments
- Processing statistics and confidence scoring
- Efficiency calculations and quality grading

### Insurance Card Processing

**InsuranceCardInfo (`card_models.py`)**
- Structured information extracted from insurance card images
- Policy details, member information, coverage specifics
- Contact information and additional details dictionary
- Validation methods for card validity and expiration checking

### Web Research Models

**SearchResult (`web_models.py`)**
- Individual search result from web research operations
- Title, URL, snippet, and relevance scoring
- Source attribution and metadata

**WebContent (`web_models.py`)**
- Complete web content extraction results
- Full text content, structured data extraction
- Link analysis and content categorization

### Orchestrator Workflow Models

**Lead (`orchestrator_models.py`)**
- Potential insurance lead or quote source
- Contact information (URL, phone, email)
- Validation methods for contact availability
- Primary contact method determination

**BidIntent (`orchestrator_models.py`)**
- Intention to request quote from specific lead
- Communication channel specification
- Payload data for quote requests
- Validation and summary methods

**BidResult (`orchestrator_models.py`)**
- Result of quote request attempts
- Success status and reference information
- Notes and status reporting methods

**WorldState (`orchestrator_models.py`)**
- Central state container for orchestrator system
- User information, assets, leads, and bid tracking
- Progress monitoring and completion assessment
- Comprehensive state management utilities

### Collection Management

**AssetCollection (`collection_models.py`)**
- Container for managing multiple assets
- Grouping and organization capabilities
- Batch operations and validation

### Factory Functions

**Factory Functions (`factory_functions.py`)**
- Creation utilities for model instances from dictionaries
- Data validation and sanitization functions
- Type-safe conversion methods
- Backward compatibility support

## Key Features

### Data Validation
- Comprehensive input validation for all model fields
- Type safety through dataclass annotations
- Optional field support for flexible data collection
- Sanitization functions for data cleaning

### Backward Compatibility
- Compatibility aliases for legacy code migration
- Graceful handling of missing optional fields
- Version-safe data structure evolution

### Utility Methods
- Conversion methods (to_dict, from_dict)
- Validation utilities (is_valid, has_required_fields)
- Calculation methods (premium extraction, date operations)
- Summary and reporting functions

### Extensibility
- Optional field architecture for future expansion
- Modular design supporting new asset types
- Plugin-friendly structure for custom validators

## Usage Patterns

### Basic Asset Creation
```python
from data_models import Vehicle, Property, PersonalItem

# Create vehicle with basic information
vehicle = Vehicle(
    make="Toyota",
    model="Camry", 
    year="2020",
    vin="1234567890"
)

# Create property with detailed information
property = Property(
    property_type="Single Family",
    address="123 Main St",
    year_built="1995",
    square_footage="2000"
)
```

### Policy Management
```python
from data_models import PolicySummary

policy = PolicySummary(
    policy_number="POL123456",
    carrier="State Farm",
    premium="$1,200.00"
)

# Check policy status
is_active = policy.is_active()
premium_amount = policy.get_premium_numeric()
```

### Orchestrator Workflow
```python
from data_models import WorldState, Lead, BidIntent

# Initialize world state
state = WorldState(user_zip="12345")

# Add assets and leads
state.vehicles.append(vehicle)
state.properties.append(property)
state.leads.append(Lead(name="Geico", phone="1-800-GEICO"))

# Check readiness for quotes
ready = state.is_ready_for_quotes()
completion = state.get_completion_percentage()
```

### Factory Pattern Usage
```python
from data_models import create_vehicle_from_dict, validate_vehicle_data

# Create from dictionary data
vehicle_data = {"make": "Honda", "model": "Civic", "year": "2019"}
vehicle = create_vehicle_from_dict(vehicle_data)

# Validate data before processing
is_valid, errors = validate_vehicle_data(vehicle_data)
```

## Integration Points

The data_models module integrates with:

- **Insurance Tools**: Provides data structures for tool inputs/outputs
- **Orchestrator**: Supplies workflow state management models
- **PDF Processing**: Structures for policy extraction results
- **OCR Processing**: Models for insurance card data
- **Web Research**: Results containers for lead discovery
- **Form Filling**: Data sources for quote request payloads

## Best Practices

1. **Use Factory Functions**: Leverage factory functions for creating instances from external data
2. **Validate Early**: Use validation methods before processing or storing data
3. **Handle Optionals**: Check for None values on optional fields before use
4. **Leverage Utilities**: Use built-in calculation and conversion methods
5. **Maintain Compatibility**: Use compatibility aliases when migrating legacy code
6. **Document Extensions**: Add clear documentation when extending models with new fields

## Dependencies

- Python 3.9+
- dataclasses (standard library)
- typing (standard library)
- datetime (standard library)

## Future Enhancements

- Enhanced validation with custom validators
- Serialization support for additional formats (XML, YAML)
- Database integration capabilities
- Advanced relationship management between models
- Automated data quality scoring
- Machine learning integration for data completion
