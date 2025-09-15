import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

# Try to import from the original module with SHAP
try:
    from xai_module import DosageExplainer, get_safety_notes
    print("Using SHAP-based XAI module")
except ImportError:
    try:
        # If SHAP is not available, try installing it
        import subprocess
        import sys
        print("SHAP not found. Attempting to install SHAP...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "shap"])
        from xai_module import DosageExplainer, get_safety_notes
        print("Successfully installed and imported SHAP-based XAI module")
    except Exception as e:
        # If installation fails, use simplified version
        print(f"Could not install SHAP: {str(e)}. Using simplified XAI module.")
        from xai_module_simple import DosageExplainer, get_safety_notes

from ocr_parser import PrescriptionParser

# Import our independent Gemini interface
from independent_gemini import MedicalAssistant

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
    
    def __init__(self, model_path: str = 'models/dosage_predictor.joblib', 
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
            
        # Initialize the medical assistant (which includes both Gemini interfaces)
        self.gemini = None
        if gemini_api_key:
            self.gemini = MedicalAssistant(gemini_api_key)
        
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
    
    def chat_about_prescription(self, user_query: str, patient_data: Optional[Dict[str, Any]] = None) -> str:
        """
        Get a response about a medical question, with optional prescription data.
        
        Args:
            user_query: The user's question
            patient_data: Optional dictionary with prescription data
            
        Returns:
            Response from the medical assistant
        """
        if self.gemini is None:
            return "The AI assistant is not available. Please check your Gemini API key configuration."
        
        return self.gemini.ask_question(user_query, patient_data)
    
    def process_prescription_image(self, image_path: str) -> DosagePrediction:
        """
        Process a prescription image and predict dosage.
        
        Args:
            image_path: Path to the prescription image
            
        Returns:
            DosagePrediction object with prediction and explanation
        """
        try:
            # Parse the prescription using OCR
            print("Starting prescription parsing...")
            parsed_data = self.parser.parse_prescription(image_path)
            
            # Always display what we found, even if incomplete
            st.write("📋 OCR Analysis Results:")
            
            # Display extracted text by lines
            if parsed_data.get('lines'):
                st.write("📝 Extracted Lines:")
                for line in parsed_data['lines']:
                    st.write(f"- {line}")
            
            # Display structured information
            extracted = parsed_data.get('extracted_items', {})
            
            # Medicines found
            if extracted.get('possible_medicines'):
                st.write("💊 Possible Medicines Detected:")
                for med in extracted['possible_medicines']:
                    st.write(f"- {med}")
            
            # Diseases/conditions found
            if extracted.get('possible_diseases'):
                st.write("🏥 Possible Conditions Detected:")
                for disease in extracted['possible_diseases']:
                    st.write(f"- {disease}")
            
            # Dosages found
            if extracted.get('possible_dosages'):
                st.write("⚖️ Possible Dosages Detected:")
                for dosage in extracted['possible_dosages']:
                    st.write(f"- {dosage}")
            
            # Dates found
            if extracted.get('dates'):
                st.write("📅 Dates Found:")
                for date in extracted['dates']:
                    st.write(f"- {date}")
            
            # Names found
            if extracted.get('possible_names'):
                st.write("👤 Possible Names Detected:")
                for name in extracted['possible_names']:
                    st.write(f"- {name}")
            
            st.write("---")
            
            # Check if we have enough information for prediction
            if not parsed_data.get('disease') and not parsed_data.get('medicines'):
                st.warning("⚠️ Could not identify condition or medicines clearly. Please verify the information above and enter details manually if needed.")
                
                # Try to get at least the disease from the raw text
                if 'raw_ocr_text' in parsed_data:
                    disease = self.parser.extract_disease_from_text(parsed_data['raw_ocr_text'])
                    if disease:
                        parsed_data['disease'] = disease
                        st.success(f"✅ Found condition: {disease}")
            
            # If we still don't have enough information
            if not parsed_data.get('disease'):
                raise ValueError("Could not identify the condition. Please check the extracted information above and enter details manually.")
            
            # Create prediction with whatever information we have
            prediction = self.predict_dosage(parsed_data)
            
            # Add the raw OCR text and extracted data to the prediction for reference
            prediction.ocr_data = parsed_data
            
            return prediction
            
        except Exception as e:
            st.error(f"⚠️ {str(e)}")
            st.info("Please ensure:")
            st.write("1. The image is clear and well-lit")
            st.write("2. Text is readable and not blurry")
            st.write("3. The prescription contains disease/condition information")
            
            # If we have any extracted data, show it even in case of error
            if 'parsed_data' in locals() and parsed_data.get('extracted_items'):
                st.write("Here's what we could extract from the image:")
                for category, items in parsed_data['extracted_items'].items():
                    if items:
                        st.write(f"{category.replace('_', ' ').title()}:")
                        for item in items:
                            st.write(f"- {item}")
            
            raise ValueError("Could not process the prescription image. Please enter details manually.")
            
        except Exception as e:
            print(f"Unexpected error processing prescription: {str(e)}")
            raise
    
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
            raise ValueError("Disease/condition is required for dosage prediction.")
        
        # Map to standard disease name if possible
        std_disease = None
        for d, m in self.valid_diseases.items():
            if d in disease or disease in d:
                std_disease = d
                break
        
        if std_disease is None:
            raise ValueError(f"Unsupported disease: {disease}. " +
                           f"Supported diseases are: {', '.join(self.valid_diseases.keys())}.")
        
        processed['disease'] = std_disease
        
        # Process medicine (optional, will be set based on disease if not provided)
        if 'medicine' in input_data and input_data['medicine']:
            medicine = str(input_data['medicine']).lower().strip()
            # Validate medicine
            if std_disease in self.valid_diseases and self.valid_diseases[std_disease] != medicine:
                print(f"Warning: Suggested medicine ({medicine}) is not standard for {std_disease}. " +
                      f"Using standard medicine: {self.valid_diseases[std_disease]}")
        
        # Use standard medicine for the disease
        processed['medicine'] = self.valid_diseases[std_disease]
        
        # Process age (optional)
        if 'age' in input_data and input_data['age'] is not None:
            try:
                age = float(input_data['age'])
                if age <= 0 or age > 120:
                    print(f"Warning: Age ({age}) is outside normal range. Using default.")
                else:
                    processed['age'] = age
            except (ValueError, TypeError):
                print(f"Warning: Invalid age value: {input_data['age']}. Using default.")
        
        # Process weight (optional)
        if 'weight' in input_data and input_data['weight'] is not None:
            try:
                weight = float(input_data['weight'])
                if weight <= 0 or weight > 500:
                    print(f"Warning: Weight ({weight}) is outside normal range. Using default.")
                else:
                    processed['weight'] = weight
            except (ValueError, TypeError):
                print(f"Warning: Invalid weight value: {input_data['weight']}. Using default.")
        
        # Process frequency (optional)
        if 'frequency_per_day' in input_data and input_data['frequency_per_day'] is not None:
            try:
                frequency = int(input_data['frequency_per_day'])
                if frequency <= 0 or frequency > 6:
                    print(f"Warning: Frequency ({frequency}) is outside normal range. Using default.")
                else:
                    processed['frequency_per_day'] = frequency
            except (ValueError, TypeError):
                print(f"Warning: Invalid frequency value: {input_data['frequency_per_day']}. Using default.")
        
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
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    chatbot = DrugDosageChatbot(gemini_api_key=gemini_api_key)
    
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
    
    # Example 2: Chat about the prescription
    print("\nExample 2: Chat about prescription")
    print("-" * 50)
    response = chatbot.chat_about_prescription(
        "What are the side effects of this medication?",
        prediction.raw_input
    )
    print(response)
    
    # Example 3: General medical question without prescription
    print("\nExample 3: General medical question")
    print("-" * 50)
    response = chatbot.chat_about_prescription(
        "What are the best ways to manage diabetes?"
    )
    print(response)