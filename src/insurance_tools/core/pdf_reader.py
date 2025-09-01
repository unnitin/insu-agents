#!/usr/bin/env python3
"""
TODO: Generalize this to be able to read any pdf file, not just insurance coverage PDFs
Insurance Coverage PDF Reader

A simple Python module to read and extract text from insurance coverage PDFs.
Supports common PDF formats and provides structured output for insurance data analysis.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import logging

import PyPDF2
import pdfplumber

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class InsurancePDFReader:
    """
    A class to handle reading and processing insurance coverage PDF documents.
    """
    
    def __init__(self, input_dir: str = "input/pdf", output_dir: str = "output"):
        """
        Initialize the PDF reader with input and output directories.
        
        Args:
            input_dir (str): Directory containing input PDF files
            output_dir (str): Directory for saving processed output
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        # Create directories if they don't exist
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"PDF Reader initialized - Input: {self.input_dir}, Output: {self.output_dir}")
    
    def read_pdf_pypdf2(self, pdf_path: Path) -> str:
        """
        Read PDF using PyPDF2 library.
        
        Args:
            pdf_path (Path): Path to the PDF file
            
        Returns:
            str: Extracted text content
        """
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text_content = ""
                
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text_content += f"\n--- Page {page_num + 1} ---\n"
                    text_content += page.extract_text()
                
                return text_content
        except Exception as e:
            logger.error(f"Error reading PDF with PyPDF2: {e}")
            return ""
    
    def read_pdf_pdfplumber(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Read PDF using pdfplumber library for better text and table extraction.
        
        Args:
            pdf_path (Path): Path to the PDF file
            
        Returns:
            Dict: Extracted content including text and tables
        """
        try:
            extracted_data = {
                'text': "",
                'tables': [],
                'metadata': {}
            }
            
            with pdfplumber.open(pdf_path) as pdf:
                # Extract metadata
                extracted_data['metadata'] = {
                    'pages': len(pdf.pages),
                    'title': pdf.metadata.get('Title', ''),
                    'author': pdf.metadata.get('Author', ''),
                    'creator': pdf.metadata.get('Creator', ''),
                    'creation_date': pdf.metadata.get('CreationDate', '')
                }
                
                # Extract text and tables from each page
                for page_num, page in enumerate(pdf.pages):
                    extracted_data['text'] += f"\n--- Page {page_num + 1} ---\n"
                    extracted_data['text'] += page.extract_text() or ""
                    
                    # Extract tables if present
                    tables = page.extract_tables()
                    if tables:
                        for table_idx, table in enumerate(tables):
                            extracted_data['tables'].append({
                                'page': page_num + 1,
                                'table_index': table_idx + 1,
                                'data': table
                            })
            
            return extracted_data
        except Exception as e:
            logger.error(f"Error reading PDF with pdfplumber: {e}")
            return {'text': '', 'tables': [], 'metadata': {}}
    
    def process_pdf(self, filename: str, use_advanced: bool = True) -> Optional[Dict[str, Any]]:
        """
        Process a single PDF file and extract insurance coverage information.
        
        Args:
            filename (str): Name of the PDF file in the input directory
            use_advanced (bool): Whether to use advanced extraction (pdfplumber)
            
        Returns:
            Optional[Dict]: Extracted data or None if processing failed
        """
        pdf_path = self.input_dir / filename
        
        if not pdf_path.exists():
            logger.error(f"PDF file not found: {pdf_path}")
            return None
        
        logger.info(f"Processing PDF: {filename}")
        
        if use_advanced:
            extracted_data = self.read_pdf_pdfplumber(pdf_path)
        else:
            text_content = self.read_pdf_pypdf2(pdf_path)
            extracted_data = {'text': text_content, 'tables': [], 'metadata': {}}
        
        # Save extracted text to output directory
        output_filename = pdf_path.stem + "_extracted.txt"
        output_path = self.output_dir / output_filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"Insurance Coverage Document: {filename}\n")
            f.write("=" * 50 + "\n\n")
            
            # Write metadata if available
            if extracted_data.get('metadata'):
                f.write("Document Metadata:\n")
                for key, value in extracted_data['metadata'].items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
            
            # Write extracted text
            f.write("Extracted Text Content:\n")
            f.write("-" * 30 + "\n")
            f.write(extracted_data['text'])
            
            # Write table information if available
            if extracted_data.get('tables'):
                f.write("\n\nExtracted Tables:\n")
                f.write("-" * 20 + "\n")
                for table_info in extracted_data['tables']:
                    f.write(f"\nTable {table_info['table_index']} (Page {table_info['page']}):\n")
                    for row in table_info['data']:
                        f.write(f"  {row}\n")
        
        logger.info(f"Extracted content saved to: {output_path}")
        return extracted_data
    
    def process_all_pdfs(self) -> Dict[str, Any]:
        """
        Process all PDF files in the input directory.
        
        Returns:
            Dict: Summary of processing results
        """
        pdf_files = list(self.input_dir.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.input_dir}")
            return {'processed': 0, 'failed': 0, 'files': []}
        
        results = {'processed': 0, 'failed': 0, 'files': []}
        
        for pdf_file in pdf_files:
            try:
                extracted_data = self.process_pdf(pdf_file.name)
                if extracted_data:
                    results['processed'] += 1
                    results['files'].append({
                        'filename': pdf_file.name,
                        'status': 'success',
                        'pages': extracted_data.get('metadata', {}).get('pages', 0)
                    })
                else:
                    results['failed'] += 1
                    results['files'].append({
                        'filename': pdf_file.name,
                        'status': 'failed',
                        'error': 'Processing failed'
                    })
            except Exception as e:
                results['failed'] += 1
                results['files'].append({
                    'filename': pdf_file.name,
                    'status': 'failed',
                    'error': str(e)
                })
                logger.error(f"Failed to process {pdf_file.name}: {e}")
        
        logger.info(f"Processing complete - Processed: {results['processed']}, Failed: {results['failed']}")
        return results
    
    # --- Added adapter methods for agentic wrappers ---
    def extract_text(self, pdf_path) -> str:
        """Extract text from PDF file using available methods."""
        from pathlib import Path as _Path
        try:
            p = _Path(pdf_path)
            if p.exists():
                res = self.read_pdf_pdfplumber(p)
                if isinstance(res, dict) and res.get("text"):
                    return res.get("text")
                txt = self.read_pdf_pypdf2(p)
                return txt or ""
        except Exception:
            pass
        return ""

    def extract_policy_info(self, text: str) -> dict:
        """
        Extract policy information using agentic approach with regex fallback.
        
        Strategy:
        1. First attempt: Use LLM/AI to intelligently extract policy information
        2. Fallback: Use regex patterns if LLM fails or is unavailable
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # Try agentic extraction first
        try:
            agentic_result = self._extract_policy_info_agentic(text)
            if self._is_valid_policy_result(agentic_result):
                logger.info("Policy extraction: Agentic method successful")
                return agentic_result
            else:
                logger.warning("Policy extraction: Agentic method returned incomplete data, falling back to regex")
        except Exception as e:
            logger.warning(f"Policy extraction: Agentic method failed ({e}), falling back to regex")
        
        # Fallback to regex extraction
        regex_result = self._extract_policy_info_regex(text)
        logger.info("Policy extraction: Using regex fallback method")
        return regex_result
    
    def _extract_policy_info_agentic(self, text: str) -> dict:
        """Extract policy information using LLM/AI approach."""
        import json
        import os
        
        # Try to use OpenAI API if available
        try:
            import openai
            
            # Check if API key is available
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found")
            
            client = openai.OpenAI(api_key=api_key)
            
            prompt = f"""
Extract insurance policy information from the following text and return it as JSON.

Text to analyze:
{text[:2000]}  # Limit text to avoid token limits

Please extract the following information and return as valid JSON:
{{
    "carrier": "Insurance company name (e.g., 'ACME INSURANCE COMPANY')",
    "policy_number": "Policy number (e.g., 'HO-2024-123456')",
    "effective_date": "Policy effective date (e.g., 'January 1, 2024')",
    "expiry_date": "Policy expiration date (e.g., 'January 1, 2025')",
    "coverages": {{}}
}}

Rules:
- Return only valid JSON, no additional text
- Use null for missing information
- Extract exact text as it appears in the document
- For dates, preserve the original format
"""

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert at extracting insurance policy information. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                result = json.loads(result_text)
                return result
            except json.JSONDecodeError:
                # Try to extract JSON from response if wrapped in markdown
                import re
                json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', result_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group(1))
                    return result
                else:
                    raise ValueError("Could not parse JSON from LLM response")
                    
        except ImportError:
            raise ValueError("OpenAI library not available")
        except Exception as e:
            raise ValueError(f"OpenAI API call failed: {e}")
    
    def _extract_policy_info_regex(self, text: str) -> dict:
        """Extract policy information using regex patterns (fallback method)."""
        import re
        
        carrier = None
        # Look for carrier name at the beginning or with "COMPANY" suffix
        m = re.search(r"^([A-Z][A-Z0-9 &.-]+(?:INSURANCE|COMPANY))", text, re.I | re.M)
        if m: 
            carrier = m.group(1).strip()
        else:
            # Fallback to original carrier pattern
            m = re.search(r"Carrier[:\s]+([A-Za-z0-9 &.-]+)", text, re.I)
            if m: carrier = m.group(1).strip()
        
        policy_no = None
        m = re.search(r"Policy\s*(No\.|Number)[:\s]+([A-Z0-9-]+)", text, re.I)
        if m: policy_no = m.group(2).strip()
        
        eff = None; exp = None
        me = re.search(r"Effective\s*Date[:\s]+([0-9/\-,\s]+)", text, re.I)
        mx = re.search(r"Expir(?:y|ation)\s*Date[:\s]+([0-9/\-,\s]+)", text, re.I)
        if me: eff = me.group(1).strip()
        if mx: exp = mx.group(1).strip()
        
        return {"carrier": carrier, "policy_number": policy_no, "effective_date": eff, "expiry_date": exp, "coverages": {}}
    
    def _is_valid_policy_result(self, result: dict) -> bool:
        """Check if the policy extraction result has meaningful data."""
        if not isinstance(result, dict):
            return False
        
        # Check if we have at least one of the key fields
        key_fields = ['carrier', 'policy_number']
        return any(result.get(field) and str(result.get(field)).strip() for field in key_fields)


def main():
    """
    Main function to demonstrate the insurance PDF reader functionality.
    """
    print("Insurance Coverage PDF Reader")
    print("=" * 40)
    
    # Initialize the PDF reader
    reader = InsurancePDFReader()
    
    # Check if there are any PDF files to process
    pdf_files = list(reader.input_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"\nNo PDF files found in {reader.input_dir}")
        print("Please add some insurance coverage PDF files to the input directory and try again.")
        return
    
    print(f"\nFound {len(pdf_files)} PDF file(s) to process:")
    for pdf_file in pdf_files:
        print(f"  - {pdf_file.name}")
    
    # Process all PDFs
    print("\nProcessing PDFs...")
    results = reader.process_all_pdfs()
    
    # Display results
    print(f"\nProcessing Results:")
    print(f"  Successfully processed: {results['processed']}")
    print(f"  Failed to process: {results['failed']}")
    
    if results['files']:
        print("\nDetailed Results:")
        for file_info in results['files']:
            status_symbol = "✓" if file_info['status'] == 'success' else "✗"
            print(f"  {status_symbol} {file_info['filename']} - {file_info['status']}")
            if file_info['status'] == 'success' and 'pages' in file_info:
                print(f"    Pages: {file_info['pages']}")
            elif file_info['status'] == 'failed':
                print(f"    Error: {file_info.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()


