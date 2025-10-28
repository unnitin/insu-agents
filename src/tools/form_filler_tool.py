
from __future__ import annotations
from typing import Dict, Any
from .base import Tool

COMMON = ["first_name","last_name","email","phone","address","zip","policy_number","current_carrier","expiry_date",
          "vehicle_year","vehicle_make","vehicle_model","vin","home_year_built","home_sqft","roof","construction"]

class FormFillerTool(Tool):
    name = "form_filler"
    description = "Generate normalized form payloads from world state."
    input_schema = {"type":"object","properties":{"state":{"type":"object"}},"required":["state"]}
    output_schema = {"type":"object","properties":{"payloads":{"type":"array"}}}

    def run(self, **kwargs) -> Dict[str, Any]:
        state = kwargs["state"]
        pol = state.get("policy",{})
        vehicles = state.get("vehicles",[])
        props = state.get("properties",[])
        payload = {k: None for k in COMMON}
        payload.update({
            "policy_number": pol.get("policy_number"),
            "current_carrier": pol.get("carrier"),
            "expiry_date": pol.get("expiry_date"),
        })
        if vehicles:
            v = vehicles[0]
            payload.update({"vehicle_year": v.get("year"), "vehicle_make": v.get("make"), "vehicle_model": v.get("model"), "vin": v.get("vin")})
        if props:
            h = props[0]
            payload.update({"home_year_built": h.get("year_built"), "home_sqft": h.get("sqft"), "roof": h.get("roof"), "construction": h.get("construction")})
        return {"payloads":[payload]}
