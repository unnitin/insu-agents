
from __future__ import annotations
import requests
from typing import List, Dict, Any, Optional

class ToolRegistryClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
    def list_tools(self, enabled_only: bool = True) -> List[Dict[str, Any]]:
        return requests.get(f"{self.base_url}/tools", params={"enabled_only": str(enabled_only).lower()}).json()["tools"]
    def get_tool(self, name: str) -> Dict[str, Any]:
        return requests.get(f"{self.base_url}/tools/{name}").json()
    def register_tool(self, tool: Dict[str, Any]) -> Dict[str, Any]:
        return requests.post(f"{self.base_url}/tools", json=tool).json()
    def update_tool(self, name: str, patch: Dict[str, Any]) -> Dict[str, Any]:
        return requests.patch(f"{self.base_url}/tools/{name}", json=patch).json()
