
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class PolicyInfo:
    carrier: Optional[str] = None
    policy_number: Optional[str] = None
    effective_date: Optional[str] = None
    expiry_date: Optional[str] = None
    coverages: Dict[str, Any] = field(default_factory=dict)
    raw_text_refs: List[str] = field(default_factory=list)

@dataclass
class AssetVehicle:
    year: Optional[str] = None
    make: Optional[str] = None
    model: Optional[str] = None
    vin: Optional[str] = None
    usage: Optional[str] = None

@dataclass
class AssetProperty:
    address: Optional[str] = None
    year_built: Optional[str] = None
    roof: Optional[str] = None
    sqft: Optional[str] = None
    construction: Optional[str] = None

@dataclass
class AssetPersonal:
    name: str = ""
    value: Optional[str] = None
    notes: Optional[str] = None

@dataclass
class Lead:
    name: str
    url: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    notes: Optional[str] = None

@dataclass
class BidIntent:
    lead_name: str
    channel: str
    payload: Dict[str, Any]

@dataclass
class BidResult:
    lead_name: str
    success: bool
    reference: Optional[str] = None
    notes: Optional[str] = None

@dataclass
class WorldState:
    user_zip: Optional[str] = None
    policy: PolicyInfo = field(default_factory=PolicyInfo)
    vehicles: List[AssetVehicle] = field(default_factory=list)
    properties: List[AssetProperty] = field(default_factory=list)
    personals: List[AssetPersonal] = field(default_factory=list)
    leads: List[Lead] = field(default_factory=list)
    bid_intents: List[BidIntent] = field(default_factory=list)
    bid_results: List[BidResult] = field(default_factory=list)
    missing_fields: List[str] = field(default_factory=list)

    def add_missing(self, field: str):
        if field not in self.missing_fields:
            self.missing_fields.append(field)
