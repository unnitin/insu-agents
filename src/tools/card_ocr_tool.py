
from __future__ import annotations
from typing import Dict, Any
from .base import Tool, ToolError
import os

class CardOCRTool(Tool):
    name = "card_ocr"
    description = "OCR insurance card images to extract key fields."
    input_schema = {"type":"object","properties":{"images_dir":{"type":"string"},"glob":{"type":"string","default":"*.png"}},"required":[]}
    output_schema = {"type":"object","properties":{"cards":{"type":"array"}}}

    def run(self, **kwargs) -> Dict[str, Any]:
        images_dir = kwargs.get("images_dir")
        if not images_dir:
            images_dir = "input/images"
        if not os.path.isdir(images_dir):
            raise ToolError(f"images_dir not found: {images_dir}")
        results = []
        try:
            from tools.core.insurance_card_processor import InsuranceCardProcessor
            proc = InsuranceCardProcessor(input_dir=images_dir)
            files = proc.find_image_files()
            for f in files:
                ocr = proc.ocr_image(f)
                parsed = proc.extract_insurance_info(ocr)
                results.append({"file": str(f), "raw": ocr, "parsed": parsed})
        except Exception:
            for fname in os.listdir(images_dir):
                if fname.lower().endswith((".png",".jpg",".jpeg",".webp",".tiff",".bmp")):
                    results.append({"file": os.path.join(images_dir,fname), "raw": "", "parsed": {}})
        return {"cards": results}
