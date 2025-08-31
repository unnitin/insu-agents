
from __future__ import annotations
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, HttpUrl

app = FastAPI(title="ToolRegistry", version="0.1.0")
DB: Dict[str, Dict[str, Any]] = {}

class JSONSchema(BaseModel):
    type: Optional[str] = None
    properties: Optional[Dict[str, Any]] = None
    required: Optional[List[str]] = None

class Tool(BaseModel):
    name: str
    version: str
    endpoint: HttpUrl
    method: str = Field("POST", pattern="^(GET|POST|PUT|PATCH)$")
    tags: List[str] = Field(default_factory=list)
    input_schema: Optional[JSONSchema] = None
    output_schema: Optional[JSONSchema] = None
    enabled: bool = True
    description: Optional[str] = None

class ToolUpdate(BaseModel):
    version: Optional[str] = None
    endpoint: Optional[HttpUrl] = None
    method: Optional[str] = Field(None, pattern="^(GET|POST|PUT|PATCH)$")
    tags: Optional[List[str]] = None
    input_schema: Optional[JSONSchema] = None
    output_schema: Optional[JSONSchema] = None
    enabled: Optional[bool] = None
    description: Optional[str] = None

@app.get("/tools")
def list_tools(enabled_only: bool = True, tag: Optional[str] = None):
    tools = [t for t in DB.values() if (t.get("enabled", True) or not enabled_only)]
    if tag: tools = [t for t in tools if tag in t.get("tags",[])]
    return {"tools": tools}

@app.get("/tools/{name}")
def get_tool(name: str):
    if name not in DB: raise HTTPException(404, "Tool not found")
    return DB[name]

@app.post("/tools")
def register_tool(tool: Tool):
    if tool.name in DB: raise HTTPException(409, "Tool already exists")
    DB[tool.name] = tool.dict()
    return {"ok": True, "tool": DB[tool.name]}

@app.patch("/tools/{name}")
def update_tool(name: str, patch: ToolUpdate):
    if name not in DB: raise HTTPException(404, "Tool not found")
    DB[name].update({k:v for k,v in patch.dict(exclude_unset=True).items()})
    return {"ok": True, "tool": DB[name]}
