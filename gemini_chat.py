from typing import Dict, Any, Optional
import google.generativeai as genai

class GeminiHealthAssistant:
    """A class to handle medical queries using Google's Gemini AI."""
    
    def __init__(self, api_key: str):
        """Initialize the Gemini API with the provided key."""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        
    def get_medical_advice(self, query: Dict[str, Any]) -> str:
        """
        Get medical advice from Gemini based on prescription data.
        
        Args:
            query: Dictionary containing prescription and patient information
            
        Returns:
            Gemini's response as a string
        """
        # Construct a detailed prompt
        prompt = self._construct_medical_prompt(query)
        
        try:
            # Get response from Gemini
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Sorry, I couldn't get a response from the AI assistant. Error: {str(e)}"
    
    def _construct_medical_prompt(self, query: Dict[str, Any]) -> str:
        """Construct a medical prompt from the query data."""
        disease = query.get('disease', 'Unknown')
        medicine = query.get('medicine', 'Unknown')
        age = query.get('age', 'Not provided')
        weight = query.get('weight', 'Not provided')
        
        prompt = f"""As a medical AI assistant, please provide detailed information about:

1. The condition: {disease}
2. The medication: {medicine}
3. Patient details:
   - Age: {age}
   - Weight: {weight}

Please include:
- General information about the condition
- How this medication typically works for this condition
- Common dosage ranges and factors affecting dosage
- Important considerations for the patient
- When to seek immediate medical attention
- Alternative treatment options to discuss with a doctor

Note: This is for informational purposes only. The patient should always consult with their healthcare provider.
"""
        return prompt
    
    def chat_about_prescription(self, patient_data: Dict[str, Any], user_query: str) -> str:
        """
        Have an interactive chat about the prescription.
        
        Args:
            patient_data: Dictionary containing patient and prescription info
            user_query: User's specific question
            
        Returns:
            Gemini's response
        """
        context = self._construct_medical_prompt(patient_data)
        
        # Construct the chat prompt
        chat_prompt = f"""Context about the patient and prescription:
{context}

User's question: {user_query}

Please provide a helpful and informative response, keeping in mind that this is for informational purposes only."""

        try:
            response = self.model.generate_content(chat_prompt)
            return response.text
        except Exception as e:
            return f"Sorry, I couldn't process your question. Error: {str(e)}"