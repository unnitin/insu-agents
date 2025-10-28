#!/usr/bin/env python3
"""
Tool Registry Implementation

Provides the core ToolRegistry class for managing and organizing insurance tools.
This implementation supports both local tool management and integration with
remote tool registry services.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional
from tools.base import Tool


class ToolRegistry:
    """
    Central registry for managing insurance tools.
    
    Provides a unified interface for tool discovery, access, and management.
    Supports both local tool storage and remote registry integration.
    
    Attributes:
        _tools: Dictionary mapping tool names to tool instances
    """
    
    def __init__(self, tools: Optional[List[Tool]] = None):
        """
        Initialize the tool registry.
        
        Args:
            tools: Optional list of tools to register initially
        """
        self._tools: Dict[str, Tool] = {}
        if tools:
            for tool in tools:
                self.register(tool)
    
    def register(self, tool: Tool) -> None:
        """
        Register a tool in the registry.
        
        Args:
            tool: Tool instance to register
            
        Raises:
            ValueError: If tool name already exists in registry
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool
    
    def unregister(self, name: str) -> bool:
        """
        Unregister a tool from the registry.
        
        Args:
            name: Name of the tool to unregister
            
        Returns:
            True if tool was removed, False if not found
        """
        if name in self._tools:
            del self._tools[name]
            return True
        return False
    
    def get(self, name: str) -> Tool:
        """
        Get a tool by name.
        
        Args:
            name: Name of the tool to retrieve
            
        Returns:
            Tool instance
            
        Raises:
            KeyError: If tool is not found
        """
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found in registry")
        return self._tools[name]
    
    def get_safe(self, name: str) -> Optional[Tool]:
        """
        Safely get a tool by name without raising exceptions.
        
        Args:
            name: Name of the tool to retrieve
            
        Returns:
            Tool instance if found, None otherwise
        """
        return self._tools.get(name)
    
    def list(self) -> List[str]:
        """
        List all registered tool names.
        
        Returns:
            List of tool names
        """
        return list(self._tools.keys())
    
    def list_tools(self) -> List[Tool]:
        """
        List all registered tool instances.
        
        Returns:
            List of tool instances
        """
        return list(self._tools.values())
    
    def specs(self) -> List[Dict[str, Any]]:
        """
        Get specifications for all registered tools.
        
        Returns:
            List of tool specifications including name, description, and schemas
        """
        return [
            {
                "name": tool.name,
                "description": getattr(tool, "description", ""),
                "input_schema": getattr(tool, "input_schema", {}),
                "output_schema": getattr(tool, "output_schema", {})
            }
            for tool in self._tools.values()
        ]
    
    def get_spec(self, name: str) -> Dict[str, Any]:
        """
        Get specification for a specific tool.
        
        Args:
            name: Name of the tool
            
        Returns:
            Tool specification dictionary
            
        Raises:
            KeyError: If tool is not found
        """
        tool = self.get(name)
        return {
            "name": tool.name,
            "description": getattr(tool, "description", ""),
            "input_schema": getattr(tool, "input_schema", {}),
            "output_schema": getattr(tool, "output_schema", {})
        }
    
    def has_tool(self, name: str) -> bool:
        """
        Check if a tool is registered.
        
        Args:
            name: Name of the tool to check
            
        Returns:
            True if tool exists, False otherwise
        """
        return name in self._tools
    
    def count(self) -> int:
        """
        Get the number of registered tools.
        
        Returns:
            Number of tools in registry
        """
        return len(self._tools)
    
    def clear(self) -> None:
        """Clear all tools from the registry."""
        self._tools.clear()
    
    def filter_by_tag(self, tag: str) -> List[Tool]:
        """
        Filter tools by tag.
        
        Args:
            tag: Tag to filter by
            
        Returns:
            List of tools that have the specified tag
        """
        filtered_tools = []
        for tool in self._tools.values():
            tags = getattr(tool, 'tags', [])
            if tag in tags:
                filtered_tools.append(tool)
        return filtered_tools
    
    def get_tools_with_capability(self, capability: str) -> List[Tool]:
        """
        Get tools that have a specific capability.
        
        Args:
            capability: Capability to search for
            
        Returns:
            List of tools with the specified capability
        """
        # This could be extended to check tool metadata, descriptions, etc.
        # For now, check if capability is in the tool name or description
        matching_tools = []
        for tool in self._tools.values():
            if (capability.lower() in tool.name.lower() or 
                capability.lower() in getattr(tool, 'description', '').lower()):
                matching_tools.append(tool)
        return matching_tools
    
    def update_from_specs(self, tool_specs: List[Dict[str, Any]]) -> None:
        """
        Update registry from tool specifications.
        
        This method is useful for loading tools from remote registries
        or configuration files.
        
        Args:
            tool_specs: List of tool specification dictionaries
            
        Note:
            This method assumes tools are already instantiated elsewhere
            and only updates metadata. For dynamic tool loading, additional
            factory methods would be needed.
        """
        # This is a placeholder for future implementation
        # Would require a tool factory to instantiate tools from specs
        pass
    
    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """
        Convert registry to dictionary representation.
        
        Returns:
            Dictionary mapping tool names to their specifications
        """
        return {name: self.get_spec(name) for name in self._tools.keys()}
    
    def __len__(self) -> int:
        """Return the number of tools in the registry."""
        return len(self._tools)
    
    def __contains__(self, name: str) -> bool:
        """Check if a tool name exists in the registry."""
        return name in self._tools
    
    def __iter__(self):
        """Iterate over tool names."""
        return iter(self._tools.keys())
    
    def __repr__(self) -> str:
        """String representation of the registry."""
        return f"ToolRegistry({len(self._tools)} tools: {list(self._tools.keys())})"
