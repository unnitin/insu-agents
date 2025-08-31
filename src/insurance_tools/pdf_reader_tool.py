
from __future__ import annotations
from typing import Dict, Any
from .base import Tool, ToolError
import os

class PdfReaderTool(Tool):
    name = "pdf_reader"
    description = "Extract text from insurance PDFs and infer policy fields."
    input_schema = {"type":"object","properties":{"pdf_path":{"type":"string"}},"required":["pdf_path"]}
    output_schema = {"type":"object","properties":{"raw_text":{"type":"string"},"policy":{"type":"object"}}}

    def run(self, **kwargs) -> Dict[str, Any]:
        pdf_path = kwargs.get("pdf_path")
        if not pdf_path or not os.path.exists(pdf_path):
            raise ToolError(f"PDF not found: {pdf_path}")
        try:
            from insurance_tools.core.pdf_reader import InsurancePDFReader
            reader = InsurancePDFReader()
            text = reader.extract_text(pdf_path)
            policy = reader.extract_policy_info(text)
            return {"raw_text": text, "policy": policy}
        except Exception:
            from PyPDF2 import PdfReader
            raw = ""
            with open(pdf_path, "rb") as f:
                r = PdfReader(f)
                for p in r.pages:
                    raw += p.extract_text() or ""
            return {"raw_text": raw[:20000], "policy": {}}
