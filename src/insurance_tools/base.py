
from __future__ import annotations
from typing import Dict, Any

class ToolError(Exception):
    pass

class Tool:
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]

    def run(self, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError
