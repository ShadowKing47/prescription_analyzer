import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from xai_module import DosageExplainer, get_safety_notes
from ocr_parser import PrescriptionParser

@dataclass
class DosagePrediction:
    """Data class to hold dosage prediction results."""
    disease: str
    medicine: str
    predicted_dosage: float
    frequency_per_day: int
    confidence: float = 1.0
    explanation: str = ""
    safety_notes: str = ""
    raw_input: Optional[Dict[str, Any]] = None

class DrugDosageChatbot:
    """
    Main chatbot class for drug dosage explanation and prediction.
    """
    
    def __init__(self, model_path: str = 'dosage_predictor.joblib', 
                 gemini_api_key: Optional[str] = None):
        """
        Initialize the chatbot with the trained model and optional Gemini API key.
        
        Args:
            model_path: Path to the trained model file
            gemini_api_key: Optional API key for Gemini
        """
        try:
            # Load the trained model
            self.model = joblib.load(model_path)
        except FileNotFoundError:
            print(f"Model file not found at {model_path}. Training a new model...")
            from Model import DosagePredictionModel
            model = DosagePredictionModel()
            model.train("synthetic_disease_dosage.csv")
            model.save_model(model_path)
            self.model = model.model
            
        # Initialize Gemini if API key is provided
        self.gemini = None
        if gemini_api_key:
            from gemini_chat import GeminiHealthAssistant
            self.gemini = GeminiHealthAssistant(gemini_api_key)
        
        # Initialize the explainer
        self.explainer = DosageExplainer(model_path)
        
        # Initialize the prescription parser
        self.parser = PrescriptionParser(gemini_api_key=gemini_api_key)
        
        # Define valid diseases and their corresponding medicines
        self.valid_diseases = {
            'diabetes': 'metformin',
            'hypertension': 'amlodipine',
            'asthma': 'salbutamol',
            'fever': 'paracetamol',
            'infection': 'amoxicillin'
        }
    
    def predict_dosage(self, input_data: Dict[str, Any]) -> DosagePrediction:
        """
        Predict the recommended dosage based on input data.
        
        Args:
            input_data: Dictionary containing input features
            
        Returns:
            DosagePrediction object with prediction and explanation
        """
        # Store raw input for reference
        raw_input = input_data.copy()
        
        # Validate and preprocess input
        processed_data = self._preprocess_input(input_data)
        
        # Create a DataFrame for prediction
        X = pd.DataFrame([{
            'disease': processed_data['disease'],
            'medicine': processed_data['medicine'],
            'avg_age': processed_data.get('age'),  # Model expects avg_age
            'avg_weight': processed_data.get('weight'),  # Model expects avg_weight
            'frequency_per_day': processed_data.get('frequency_per_day', 1)
        }])
        
        # Make prediction
        predicted_dosage = self.model.predict(X)[0]
        
        # Get explanation
        explanation_data = self.explainer.explain_prediction(X.iloc[0].to_dict())
        
        # Get safety notes
        safety_notes = get_safety_notes(
            disease=processed_data['disease'],
            medicine=processed_data['medicine'],
            dosage=predicted_dosage
        )
        
        # Create and return prediction object
        return DosagePrediction(
            disease=processed_data['disease'],
            medicine=processed_data['medicine'],
            predicted_dosage=round(predicted_dosage, 2),
            frequency_per_day=processed_data.get('frequency_per_day', 1),
            explanation=explanation_data.get('explanation', ''),
            safety_notes=safety_notes,
            raw_input=raw_input
        )
    
    def process_prescription_image(self, image_path: str) -> DosagePrediction:
        """
        Process a prescription image and predict dosage.
        
        Args:
            image_path: Path to the prescription image
            
        Returns:
            DosagePrediction object with prediction and explanation
        """
        try:
            # Parse the prescription
            parsed_data = self.parser.parse_prescription(image_path)
            
            # Extract dosage information
            dosage_info = self.parser.extract_dosage_info(parsed_data)
            
            # If we couldn't extract enough information
            if 'error' in dosage_info:
                return DosagePrediction(
                    disease="Unknown",
                    medicine="Unknown",
                    predicted_dosage=0,
                    frequency_per_day=1,
                    explanation="Could not extract sufficient information from the prescription.",
                    safety_notes="Please consult your healthcare provider for dosage information.",
                    raw_input={"image_path": image_path, "parsed_data": parsed_data}
                )
            
            # Standardize the extracted features
            standardized_data = {
                'disease': str(dosage_info.get('disease', '')).lower(),
                'medicine': str(dosage_info.get('medicine', '')).lower(),
                'age': float(dosage_info.get('age', 0)) if dosage_info.get('age') else None,
                'weight': float(dosage_info.get('weight', 0)) if dosage_info.get('weight') else None,
                'frequency_per_day': int(dosage_info.get('frequency_per_day', 1))
            }
            
            # Make prediction with the standardized data
            return self.predict_dosage(standardized_data)
            
        except ValueError as e:
            # Handle specific validation errors
            return DosagePrediction(
                disease=str(dosage_info.get('disease', 'Unknown')),
                medicine=str(dosage_info.get('medicine', 'Unknown')),
                predicted_dosage=0,
                frequency_per_day=1,
                explanation=f"Error processing prescription: {str(e)}",
                safety_notes="Please consult your healthcare provider for accurate dosage information.",
                raw_input={"image_path": image_path, "parsed_data": parsed_data, "error": str(e)}
            )
        except Exception as e:
            # Handle unexpected errors
            return DosagePrediction(
                disease="Unknown",
                medicine="Unknown",
                predicted_dosage=0,
                frequency_per_day=1,
                explanation=f"Unexpected error processing prescription: {str(e)}",
                safety_notes="Please consult your healthcare provider for dosage information.",
                raw_input={"image_path": image_path, "error": str(e)}
            )
    
    def _preprocess_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and preprocess input data.
        
        Args:
            input_data: Raw input data
            
        Returns:
            Processed and validated input data
        """
        processed = {}
        
        # Process disease (required)
        disease = str(input_data.get('disease', '')).lower().strip()
        if not disease:
            raise ValueError("Disease is required")
        
        # Map to standard disease name if possible
        std_disease = None
        for d, m in self.valid_diseases.items():
            if d in disease:
                std_disease = d
                processed['medicine'] = m
                break
        
        if std_disease is None:
            # If disease not recognized, use the first one as default
            std_disease = next(iter(self.valid_diseases))
            processed['medicine'] = self.valid_diseases[std_disease]
        
        processed['disease'] = std_disease
        
        # Process medicine (optional, will be set based on disease if not provided)
        if 'medicine' in input_data and input_data['medicine']:
            medicine = str(input_data['medicine']).lower().strip()
            # Check if the provided medicine matches the expected one for the disease
            expected_medicine = self.valid_diseases.get(std_disease, '')
            if expected_medicine not in medicine:  # If the provided medicine doesn't match expected
                print(f"Warning: Expected medicine for {std_disease} is {expected_medicine}, "
                      f"but got {medicine}. Using {expected_medicine} instead.")
        
        # Process age (optional)
        if 'age' in input_data and input_data['age'] is not None:
            try:
                age = float(input_data['age'])
                if age <= 0 or age > 120:
                    print(f"Warning: Age {age} is outside the expected range (1-120). Using None.")
                else:
                    processed['age'] = age
            except (ValueError, TypeError):
                print(f"Warning: Could not parse age '{input_data['age']}'. Using None.")
        
        # Process weight (optional)
        if 'weight' in input_data and input_data['weight'] is not None:
            try:
                weight = float(input_data['weight'])
                if weight <= 0 or weight > 300:  # kg
                    print(f"Warning: Weight {weight}kg is outside the expected range (1-300kg). Using None.")
                else:
                    processed['weight'] = weight
            except (ValueError, TypeError):
                print(f"Warning: Could not parse weight '{input_data['weight']}'. Using None.")
        
        # Process frequency (optional)
        if 'frequency_per_day' in input_data and input_data['frequency_per_day'] is not None:
            try:
                freq = int(input_data['frequency_per_day'])
                if freq < 1 or freq > 4:
                    print(f"Warning: Frequency {freq} is outside the expected range (1-4). Using default.")
                else:
                    processed['frequency_per_day'] = freq
            except (ValueError, TypeError):
                print(f"Warning: Could not parse frequency '{input_data['frequency_per_day']}'. Using default.")
        
        return processed
    
    def generate_response(self, prediction: DosagePrediction) -> str:
        """
        Generate a natural language response from the prediction.
        
        Args:
            prediction: DosagePrediction object
            
        Returns:
            Formatted response string
        """
        response_parts = [
            f"Based on the provided information:",
            f"- Condition: {prediction.disease.title()}",
            f"- Medication: {prediction.medicine.title()}",
            f"- Recommended Dosage: {prediction.predicted_dosage:.0f}mg",
            f"- Frequency: {prediction.frequency_per_day} time(s) per day",
            "",
            f"Explanation: {prediction.explanation}",
            "",
            f"Safety Notes: {prediction.safety_notes}",
            "",
            "DISCLAIMER: This is for informational purposes only. "
            "Always consult with a healthcare professional before taking any medication."
        ]
        
        return "\n".join(response_parts)

# Example usage
if __name__ == "__main__":
    # Initialize the chatbot
    chatbot = DrugDosageChatbot()
    
    # Example 1: Predict with direct input
    print("Example 1: Direct Input")
    print("-" * 50)
    input_data = {
        'disease': 'diabetes',
        'age': 45,
        'weight': 70,
        'frequency_per_day': 2
    }
    
    prediction = chatbot.predict_dosage(input_data)
    print(chatbot.generate_response(prediction))
    
    print("\n" + "="*80 + "\n")
    
    # Example 2: Process a prescription image (uncomment to use)
    """
    print("Example 2: Prescription Image")
    print("-" * 50)
    image_path = "path/to/prescription.jpg"
    prediction = chatbot.process_prescription_image(image_path)
    print(chatbot.generate_response(prediction))
    """
