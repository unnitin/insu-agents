
from __future__ import annotations
from typing import Dict, Any

def validate_args(args: Dict[str, Any], schema: Dict[str, Any]) -> tuple[bool, str]:
    if not schema:
        return True, ""
    try:
        import jsonschema
        jsonschema.validate(instance=args, schema=schema)
        return True, ""
    except ModuleNotFoundError:
        req = schema.get("required", [])
        missing = [k for k in req if k not in args]
        if missing:
            return False, f"Missing required fields: {missing}"
        return True, ""
    except Exception as e:
        return False, f"Schema validation error: {e}"
