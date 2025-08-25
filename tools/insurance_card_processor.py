#!/usr/bin/env python3
"""
Insurance Card Image Processor

A tool for processing insurance card images using OCR and image preprocessing.
Extracts text and structured information from insurance card screenshots.
"""

import os
import sys
import re
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Optional image processing imports (graceful degradation)
try:
    from PIL import Image, ImageEnhance, ImageFilter
    HAS_PILLOW = True
except ImportError:
    HAS_PILLOW = False
    Image = None
    ImageEnhance = None
    ImageFilter = None
    logger.warning("Pillow library not installed. Image processing features will be disabled.")

try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False
    pytesseract = None
    logger.warning("pytesseract library not installed. OCR features will be disabled.")

try:
    import cv2
    import numpy as np
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    cv2 = None
    np = None
    logger.warning("OpenCV library not installed. Advanced image processing features will be disabled.")


@dataclass
class InsuranceCardInfo:
    """Data class for insurance card information"""
    policy_number: str = ""
    member_id: str = ""
    group_number: str = ""
    carrier_name: str = ""
    member_name: str = ""
    effective_date: str = ""
    expiration_date: str = ""
    
    # Coverage details
    copay_primary: str = ""
    copay_specialist: str = ""
    deductible: str = ""
    out_of_pocket_max: str = ""
    
    # Additional information
    phone_numbers: List[str] = None
    additional_details: Dict[str, str] = None
    
    def __post_init__(self):
        if self.phone_numbers is None:
            self.phone_numbers = []
        if self.additional_details is None:
            self.additional_details = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return asdict(self)
    
    def is_valid(self) -> bool:
        """Check if card has minimum required information"""
        return bool(self.policy_number or self.member_id or self.carrier_name)


class InsuranceCardProcessor:
    """
    A class to handle processing of insurance card images using OCR and image preprocessing.
    """
    
    def __init__(self, input_dir: str = "input/images", output_dir: str = "output/cards"):
        """
        Initialize the insurance card processor.
        
        Args:
            input_dir (str): Directory containing input image files
            output_dir (str): Directory for saving processed output
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        # Create directories if they don't exist
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check for required libraries
        self.has_image_processing = HAS_PILLOW and HAS_TESSERACT
        self.has_advanced_processing = HAS_OPENCV
        
        if not self.has_image_processing:
            logger.warning("Image processing capabilities limited. Install Pillow and pytesseract for full functionality.")
        
        logger.info(f"Insurance Card Processor initialized - Input: {self.input_dir}, Output: {self.output_dir}")
        logger.info(f"Image processing available: {self.has_image_processing}")
        logger.info(f"Advanced processing available: {self.has_advanced_processing}")
    
    def _load_and_preprocess_image(self, image_path: Path) -> Optional[Image.Image]:
        """
        Load and preprocess an image for better OCR results.
        
        Args:
            image_path (Path): Path to the image file
            
        Returns:
            Optional[Image.Image]: Preprocessed image or None if failed
        """
        if not HAS_PILLOW:
            raise ImportError("Pillow library not available. Install with: pip install Pillow")
        
        try:
            # Load image
            image = Image.open(image_path)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Enhance contrast and sharpness
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.5)
            
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.2)
            
            # Apply slight blur to reduce noise
            image = image.filter(ImageFilter.MedianFilter(size=3))
            
            logger.debug(f"Successfully preprocessed image: {image_path.name}")
            return image
            
        except Exception as e:
            logger.error(f"Failed to preprocess image {image_path.name}: {e}")
            return None
    
    def _enhance_image_opencv(self, image_path: Path) -> Optional[str]:
        """
        Enhance image using OpenCV for better OCR results.
        
        Args:
            image_path (Path): Path to the image file
            
        Returns:
            Optional[str]: Path to enhanced image or None if failed
        """
        if not HAS_OPENCV:
            logger.debug("OpenCV not available, skipping advanced enhancement")
            return None
        
        try:
            # Read image
            img = cv2.imread(str(image_path))
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply bilateral filter to reduce noise while preserving edges
            filtered = cv2.bilateralFilter(gray, 9, 75, 75)
            
            # Apply adaptive threshold to get better text contrast
            thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            
            # Apply morphological operations to clean up the image
            kernel = np.ones((2, 2), np.uint8)
            processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            # Save enhanced image
            enhanced_path = self.output_dir / f"enhanced_{image_path.name}"
            cv2.imwrite(str(enhanced_path), processed)
            
            logger.debug(f"Enhanced image saved: {enhanced_path}")
            return str(enhanced_path)
            
        except Exception as e:
            logger.warning(f"OpenCV enhancement failed for {image_path.name}: {e}")
            return None
    
    def _extract_text_with_ocr(self, image: Image.Image) -> str:
        """
        Extract text from image using Tesseract OCR.
        
        Args:
            image (Image.Image): Preprocessed image
            
        Returns:
            str: Extracted text
        """
        if not HAS_TESSERACT:
            logger.error("pytesseract not available. Install with: pip install pytesseract")
            return ""
        
        try:
            # Configure OCR for better accuracy with insurance cards
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz()/-:.,$ '
            
            # Extract text
            text = pytesseract.image_to_string(image, config=custom_config)
            
            # Clean up the text
            text = text.strip()
            text = re.sub(r'\n+', '\n', text)  # Remove multiple newlines
            text = re.sub(r' +', ' ', text)    # Remove multiple spaces
            
            logger.debug(f"Extracted {len(text)} characters of text")
            return text
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return ""
    
    def _parse_card_info_from_text(self, text: str) -> InsuranceCardInfo:
        """
        Parse insurance card information from extracted text.
        
        Args:
            text (str): Raw OCR text
            
        Returns:
            InsuranceCardInfo: Parsed insurance card information
        """
        card_info = InsuranceCardInfo()
        
        # Define patterns for common insurance card fields
        patterns = {
            'policy_number': [
                r'(?:Policy|Pol|ID)[\s#:]*([A-Z0-9\-]{6,20})',
                r'(?:Member ID|MID)[\s#:]*([A-Z0-9\-]{6,20})',
            ],
            'member_id': [
                r'(?:Member ID|MID|ID)[\s#:]*([A-Z0-9\-]{6,20})',
                r'(?:Subscriber ID)[\s#:]*([A-Z0-9\-]{6,20})',
            ],
            'group_number': [
                r'(?:Group|Grp|GRP)[\s#:]*([A-Z0-9\-]{4,15})',
                r'(?:Group Number|Group No)[\s#:]*([A-Z0-9\-]{4,15})',
            ],
            'member_name': [
                r'(?:Name|Member Name)[\s:]*([A-Z][a-z]+ [A-Z][a-z]+)',
                r'^([A-Z][a-z]+ [A-Z][a-z]+)',  # Name at beginning of line
            ],
            'effective_date': [
                r'(?:Effective|Eff)[\s:]*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})',
                r'(?:From|Start)[\s:]*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})',
            ],
            'expiration_date': [
                r'(?:Expires|Exp|Through|Thru)[\s:]*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})',
                r'(?:To|End)[\s:]*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})',
            ],
            'copay_primary': [
                r'(?:Primary|PCP)[\s:]*\$?(\d+)',
                r'(?:Office Visit|OV)[\s:]*\$?(\d+)',
            ],
            'copay_specialist': [
                r'(?:Specialist|Spec)[\s:]*\$?(\d+)',
                r'(?:Specialty|Spec Visit)[\s:]*\$?(\d+)',
            ],
            'deductible': [
                r'(?:Deductible|Ded)[\s:]*\$?(\d+)',
                r'(?:Annual Deductible)[\s:]*\$?(\d+)',
            ],
        }
        
        # Extract carrier name (often appears prominently)
        carrier_patterns = [
            r'(Blue Cross Blue Shield|BCBS)',
            r'(Aetna|AETNA)',
            r'(Cigna|CIGNA)',
            r'(UnitedHealthcare|United Healthcare|UHC)',
            r'(Anthem|ANTHEM)',
            r'(Kaiser Permanente|Kaiser)',
            r'(Humana|HUMANA)',
        ]
        
        for pattern in carrier_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                card_info.carrier_name = match.group(1)
                break
        
        # Extract phone numbers
        phone_pattern = r'(\d{3}[-.]?\d{3}[-.]?\d{4})'
        phone_matches = re.findall(phone_pattern, text)
        card_info.phone_numbers = phone_matches
        
        # Extract other fields using patterns
        for field, field_patterns in patterns.items():
            for pattern in field_patterns:
                match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                if match:
                    setattr(card_info, field, match.group(1).strip())
                    break
        
        # Store additional details
        card_info.additional_details['raw_text'] = text
        card_info.additional_details['text_length'] = str(len(text))
        
        return card_info
    
    def process_image(self, image_filename: str) -> Optional[InsuranceCardInfo]:
        """
        Process a single insurance card image.
        
        Args:
            image_filename (str): Name of the image file in the input directory
            
        Returns:
            Optional[InsuranceCardInfo]: Extracted card information or None if failed
        """
        image_path = self.input_dir / image_filename
        
        if not image_path.exists():
            logger.error(f"Image file not found: {image_path}")
            return None
        
        logger.info(f"Processing insurance card image: {image_filename}")
        
        if not self.has_image_processing:
            logger.error("Image processing not available. Install required libraries.")
            return None
        
        try:
            # Try OpenCV enhancement first
            enhanced_path = None
            if self.has_advanced_processing:
                enhanced_path = self._enhance_image_opencv(image_path)
            
            # Load and preprocess image
            if enhanced_path:
                # Use enhanced image
                image = Image.open(enhanced_path)
            else:
                # Use regular preprocessing
                image = self._load_and_preprocess_image(image_path)
            
            if not image:
                logger.error(f"Failed to load image: {image_filename}")
                return None
            
            # Extract text using OCR
            extracted_text = self._extract_text_with_ocr(image)
            
            if not extracted_text:
                logger.warning(f"No text extracted from image: {image_filename}")
                return None
            
            # Parse insurance card information
            card_info = self._parse_card_info_from_text(extracted_text)
            
            # Save raw text to file
            text_output_path = self.output_dir / f"{Path(image_filename).stem}_extracted_text.txt"
            with open(text_output_path, 'w', encoding='utf-8') as f:
                f.write(f"Insurance Card Image: {image_filename}\n")
                f.write("=" * 50 + "\n\n")
                f.write("Extracted Text:\n")
                f.write("-" * 20 + "\n")
                f.write(extracted_text)
            
            # Save parsed information to JSON
            json_output_path = self.output_dir / f"{Path(image_filename).stem}_parsed_info.json"
            with open(json_output_path, 'w', encoding='utf-8') as f:
                json.dump(card_info.to_dict(), f, indent=2, ensure_ascii=False)
            
            logger.info(f"Processed insurance card: {image_filename}")
            logger.info(f"  Text saved to: {text_output_path}")
            logger.info(f"  Parsed info saved to: {json_output_path}")
            
            return card_info
            
        except Exception as e:
            logger.error(f"Failed to process image {image_filename}: {e}")
            return None
    
    def process_directory(self) -> Dict[str, Any]:
        """
        Process all image files in the input directory.
        
        Returns:
            Dict: Summary of processing results
        """
        # Supported image extensions
        supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        
        # Find all image files
        image_files = []
        for ext in supported_extensions:
            image_files.extend(self.input_dir.glob(f"*{ext}"))
            image_files.extend(self.input_dir.glob(f"*{ext.upper()}"))
        
        if not image_files:
            logger.warning(f"No image files found in {self.input_dir}")
            return {'processed': 0, 'failed': 0, 'files': []}
        
        results = {'processed': 0, 'failed': 0, 'files': []}
        
        for image_file in image_files:
            try:
                card_info = self.process_image(image_file.name)
                
                if card_info and card_info.is_valid():
                    results['processed'] += 1
                    results['files'].append({
                        'filename': image_file.name,
                        'status': 'success',
                        'carrier': card_info.carrier_name,
                        'policy_number': card_info.policy_number,
                        'member_id': card_info.member_id
                    })
                else:
                    results['failed'] += 1
                    results['files'].append({
                        'filename': image_file.name,
                        'status': 'failed',
                        'error': 'No valid card information extracted'
                    })
                    
            except Exception as e:
                results['failed'] += 1
                results['files'].append({
                    'filename': image_file.name,
                    'status': 'failed',
                    'error': str(e)
                })
                logger.error(f"Failed to process {image_file.name}: {e}")
        
        logger.info(f"Directory processing complete - Processed: {results['processed']}, Failed: {results['failed']}")
        return results
    
    def generate_card_report(self, results: Dict[str, Any]) -> str:
        """
        Generate a summary report for processed insurance cards.
        
        Args:
            results (Dict): Results from process_directory()
            
        Returns:
            str: Formatted report
        """
        report = []
        report.append("INSURANCE CARD PROCESSING REPORT")
        report.append("=" * 50)
        report.append("")
        
        report.append(f"Total files processed: {results['processed'] + results['failed']}")
        report.append(f"Successfully processed: {results['processed']}")
        report.append(f"Failed to process: {results['failed']}")
        report.append("")
        
        if results['files']:
            report.append("DETAILED RESULTS:")
            report.append("-" * 30)
            
            for file_info in results['files']:
                status_symbol = "✓" if file_info['status'] == 'success' else "✗"
                report.append(f"{status_symbol} {file_info['filename']}")
                
                if file_info['status'] == 'success':
                    report.append(f"    Carrier: {file_info.get('carrier', 'Not detected')}")
                    report.append(f"    Policy: {file_info.get('policy_number', 'Not detected')}")
                    report.append(f"    Member ID: {file_info.get('member_id', 'Not detected')}")
                else:
                    report.append(f"    Error: {file_info.get('error', 'Unknown error')}")
                
                report.append("")
        
        return '\n'.join(report)


def main():
    """
    Main function to demonstrate the insurance card processor functionality.
    """
    print("Insurance Card Image Processor")
    print("=" * 40)
    
    # Initialize the card processor
    processor = InsuranceCardProcessor()
    
    if not processor.has_image_processing:
        print("⚠️  Image processing capabilities are limited.")
        print("Please install required libraries:")
        print("  pip install Pillow pytesseract")
        print("  And install Tesseract OCR system binaries")
        return
    
    # Check if there are any image files to process
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    image_files = []
    for ext in supported_extensions:
        image_files.extend(processor.input_dir.glob(f"*{ext}"))
        image_files.extend(processor.input_dir.glob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"\nNo image files found in {processor.input_dir}")
        print("Please add some insurance card images to the input directory and try again.")
        print(f"Supported formats: {', '.join(supported_extensions)}")
        return
    
    print(f"\nFound {len(image_files)} image file(s) to process:")
    for image_file in image_files:
        print(f"  - {image_file.name}")
    
    # Process all images
    print("\nProcessing insurance card images...")
    results = processor.process_directory()
    
    # Generate and display report
    report = processor.generate_card_report(results)
    print(f"\n{report}")
    
    # Save report to file
    report_path = processor.output_dir / "card_processing_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"📄 Full report saved to: {report_path}")


if __name__ == "__main__":
    main()
