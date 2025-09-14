import re
import pytesseract
from PIL import Image
import io
import os
from typing import Dict, Any, Optional, Tuple
import google.generativeai as genai
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

class PrescriptionParser:
    """
    A class to parse prescription text from images using OCR and process it with Gemini API.
    """
    
    def __init__(self, gemini_api_key: Optional[str] = None):
        """
        Initialize the parser with optional Gemini API key.
        
        Args:
            gemini_api_key: Optional API key for Gemini. If not provided, will try to load from environment.
        """
        self.gemini_api_key = gemini_api_key or os.getenv('GEMINI_API_KEY')
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)
        
        # Common medicine name mappings (brand to generic)
        self.medicine_mapping = {
            # Metformin brands
            'glucophage': 'metformin',
            'glucophage xr': 'metformin',
            'fortamet': 'metformin',
            'glumetza': 'metformin',
            'riomet': 'metformin',
            
            # Amlodipine brands
            'norvasc': 'amlodipine',
            'katerzia': 'amlodipine',
            'norliqva': 'amlodipine',
            
            # Salbutamol brands
            'ventolin': 'salbutamol',
            'proair': 'salbutamol',
            'proventil': 'salbutamol',
            'ventorlin': 'salbutamol',
            
            # Paracetamol brands
            'tylenol': 'paracetamol',
            'panadol': 'paracetamol',
            'calpol': 'paracetamol',
            'feverall': 'paracetamol',
            
            # Amoxicillin brands
            'amoxil': 'amoxicillin',
            'moxatag': 'amoxicillin',
            'trimox': 'amoxicillin',
        }
        
        # Common dosage units and their conversions to mg
        self.dosage_units = {
            'mg': 1,
            'mgm': 1,
            'mcg': 0.001,
            'microgram': 0.001,
            'g': 1000,
            'gram': 1000,
            'ml': 1,  # Assuming 1ml = 1mg for liquid medications
            'iu': 0.00067,  # Approximate conversion for some medications
        }
    
    def extract_text_from_image(self, image_path: str) -> str:
        """
        Extract text from a prescription image using Tesseract OCR.
        
        Args:
            image_path: Path to the prescription image
            
        Returns:
            Extracted text from the image
        """
        try:
            # Open the image file
            with Image.open(image_path) as img:
                # Convert to grayscale for better OCR
                img_gray = img.convert('L')
                # Use Tesseract to extract text
                text = pytesseract.image_to_string(img_gray)
                return text.strip()
        except Exception as e:
            raise Exception(f"Error extracting text from image: {str(e)}")
    
    def preprocess_text(self, text: str) -> str:
        """
        Clean and preprocess the extracted text.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace and newlines
        text = ' '.join(text.split())
        # Remove special characters except those used in dosages (e.g., /, -, .)
        text = re.sub(r'[^\w\s\d\.\-\/]', ' ', text, flags=re.UNICODE)
        return text.strip()
    
    def parse_with_gemini(self, text: str) -> Dict[str, Any]:
        """
        Parse prescription text using Gemini API to extract structured information.
        
        Args:
            text: Extracted prescription text
            
        Returns:
            Dictionary containing parsed prescription information
        """
        if not self.gemini_api_key:
            raise ValueError("Gemini API key is required for advanced parsing")
        
        # Initialize the Gemini model
        model = genai.GenerativeModel('gemini-pro')
        
        # Create a prompt for the model
        prompt = """
        Extract the following information from the prescription text below in JSON format:
        - patient_name (string): Name of the patient
        - medicines (list of objects): Each object should have:
          - name (string): Generic name of the medicine
          - dosage (string): Dosage information (e.g., "500mg", "1 tablet")
          - frequency (string): How often to take (e.g., "twice daily", "every 6 hours")
          - duration (string): How long to take the medicine (e.g., "7 days", "until finished")
          - instructions (string): Any special instructions
        - doctor_name (string): Name of the prescribing doctor
        - date (string): Date of the prescription
        
        Prescription text:
        """ + text + """
        
        Return ONLY the JSON object, with no additional text or markdown formatting.
        """
        
        try:
            # Call the Gemini API
            response = model.generate_content(prompt)
            
            # Extract JSON from the response
            json_str = response.text.strip()
            if '```json' in json_str:
                json_str = json_str.split('```json')[1].split('```')[0].strip()
            
            # Parse the JSON
            parsed_data = json.loads(json_str)
            return parsed_data
            
        except Exception as e:
            raise Exception(f"Error parsing with Gemini: {str(e)}")
    
    def parse_prescription(self, image_path: str, use_gemini: bool = True) -> Dict[str, Any]:
        """
        Parse a prescription image and return structured data.
        
        Args:
            image_path: Path to the prescription image
            use_gemini: Whether to use Gemini API for parsing (more accurate but requires API key)
            
        Returns:
            Dictionary containing parsed prescription information
        """
        # Extract text from image
        raw_text = self.extract_text_from_image(image_path)
        
        # Preprocess the text
        cleaned_text = self.preprocess_text(raw_text)
        
        # Parse using Gemini if available and requested
        if use_gemini and self.gemini_api_key:
            try:
                return self.parse_with_gemini(cleaned_text)
            except Exception as e:
                print(f"Warning: Gemini parsing failed, falling back to regex: {str(e)}")
        
        # Fall back to regex-based parsing
        return self.parse_with_regex(cleaned_text)
    
    def parse_with_regex(self, text: str) -> Dict[str, Any]:
        """
        Parse prescription text using regex patterns.
        This is a fallback method when Gemini is not available.
        
        Args:
            text: Preprocessed prescription text
            
        Returns:
            Dictionary containing parsed prescription information
        """
        result = {
            'patient_name': None,
            'medicines': [],
            'doctor_name': None,
            'date': None,
            'raw_text': text
        }
        
        # Try to extract patient name (simple pattern)
        name_match = re.search(r'(?:patient|name)[:\s]+([A-Za-z\s]+\w)', text, re.IGNORECASE)
        if name_match:
            result['patient_name'] = name_match.group(1).strip()
        
        # Try to extract doctor's name
        doctor_match = re.search(r'(?:doctor|dr|physician)[:\s.]+([A-Za-z\s]+\w)', text, re.IGNORECASE)
        if doctor_match:
            result['doctor_name'] = doctor_match.group(1).strip()
        
        # Try to extract date
        date_match = re.search(r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b', text)
        if date_match:
            result['date'] = date_match.group(1)
        
        # Extract medicine information
        medicine_patterns = [
            # Pattern for: DrugName 500mg 1-0-1 (morning-noon-night)
            r'([A-Z][a-zA-Z\s\-]+?)\s+(\d+\.?\d*\s*(?:mg|mcg|g|ml|IU|units?)?\b[^\n]*)',
            # Pattern for: Tab. DrugName 500mg BD
            r'(?:tab\.?|tablet|cap\.?|capsule|inj\.?|injection|syr\.?|syrup|susp\.?|suspension|cream|oint\.?|ointment)\s+([A-Z][a-zA-Z\s\-]+?)\s+(\d+\.?\d*\s*(?:mg|mcg|g|ml|IU|units?)?\b[^\n]*)',
        ]
        
        for pattern in medicine_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                medicine_name = match.group(1).strip()
                details = match.group(2).strip()
                
                # Extract dosage
                dosage_match = re.search(r'(\d+\.?\d*)\s*(mg|mcg|g|ml|IU|units?)?\b', details, re.IGNORECASE)
                dosage = None
                if dosage_match:
                    value = float(dosage_match.group(1))
                    unit = (dosage_match.group(2) or 'mg').lower()
                    # Convert to mg for consistency
                    conversion = self.dosage_units.get(unit, 1)
                    dosage = f"{value * conversion:.0f}mg"
                
                # Extract frequency
                frequency = None
                freq_patterns = [
                    (r'(\d+)\s*[xX]\s*(?:per\s*)?(?:day|daily)', lambda m: f"{m.group(1)} times daily"),
                    (r'\b(bd|bid|b\.i\.d\.?)\b', lambda _: "twice daily"),
                    (r'\b(tid|t\.i\.d\.?)\b', lambda _: "three times daily"),
                    (r'\b(qid|q\.i\.d\.?)\b', lambda _: "four times daily"),
                    (r'\b(q\.h\.?)\s*(\d+)', lambda m: f"every {m.group(2)} hours"),
                    (r'\b(nocte|hs|h\.s\.?)\b', lambda _: "at bedtime"),
                    (r'\b(mane|m\.m\.?)\b', lambda _: "in the morning"),
                ]
                
                for freq_pattern, freq_func in freq_patterns:
                    if re.search(freq_pattern, details, re.IGNORECASE):
                        frequency = re.sub(freq_pattern, freq_func, details, flags=re.IGNORECASE)
                        break
                
                if not frequency:
                    frequency = "as directed"
                
                # Map to generic name if possible
                generic_name = self.medicine_mapping.get(medicine_name.lower(), medicine_name)
                
                medicine_info = {
                    'name': generic_name,
                    'original_name': medicine_name,
                    'dosage': dosage or "Not specified",
                    'frequency': frequency,
                    'instructions': details
                }
                
                result['medicines'].append(medicine_info)
        
        return result
    
    def extract_dosage_info(self, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract dosage information from parsed prescription data.
        
        Args:
            parsed_data: Dictionary from parse_prescription()
            
        Returns:
            Dictionary with extracted dosage information in format expected by model
        """
        if not parsed_data.get('medicines'):
            return {
                'error': 'No medicines found in prescription',
                'parsed_data': parsed_data
            }
        
        # For simplicity, just return the first medicine's details
        # In a real application, you might want to handle multiple medicines
        medicine = parsed_data['medicines'][0]
        
        # Extract numeric dosage value
        dosage_value = None
        if 'dosage' in medicine and medicine['dosage'] != "Not specified":
            # Extract the numeric part of the dosage
            match = re.search(r'(\d+\.?\d*)', medicine['dosage'])
            if match:
                dosage_value = float(match.group(1))
        
        # Map medicine name to our standard list if possible
        standard_medicines = {
            'metformin': 'Metformin',
            'amlodipine': 'Amlodipine',
            'salbutamol': 'Salbutamol',
            'paracetamol': 'Paracetamol',
            'amoxicillin': 'Amoxicillin'
        }
        
        medicine_name = medicine['name'].lower()
        disease = None
        for med, std_med in standard_medicines.items():
            if med in medicine_name:
                disease = std_med
                break
        
        return {
            'disease': disease or 'Unknown',
            'medicine': medicine['name'],
            'dosage_mg': dosage_value,
            'frequency_per_day': self._parse_frequency(medicine.get('frequency', '')),
            'notes': medicine.get('instructions', '')
        }
    
    def _parse_frequency(self, frequency_str: str) -> int:
        """Convert frequency string to times per day."""
        if not frequency_str:
            return 1
        
        frequency_str = frequency_str.lower()
        
        # Check for exact matches first
        if 'once' in frequency_str or 'qd' in frequency_str or 'q24h' in frequency_str:
            return 1
        if 'twice' in frequency_str or 'bd' in frequency_str or 'bid' in frequency_str or 'q12h' in frequency_str:
            return 2
        if 'three' in frequency_str or 'tid' in frequency_str or '8' in frequency_str or 'q8h' in frequency_str:
            return 3
        if 'four' in frequency_str or 'qid' in frequency_str or '6' in frequency_str or 'q6h' in frequency_str:
            return 4
        
        # Look for patterns like "every X hours"
        every_match = re.search(r'every\s*(\d+)\s*h(?:ou)?r', frequency_str)
        if every_match:
            hours = int(every_match.group(1))
            if hours > 0:
                return min(24 // hours, 4)  # Cap at 4 times per day
        
        # Default to once daily if we can't determine
        return 1
