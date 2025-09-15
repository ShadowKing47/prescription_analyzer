from typing import Dict, Any, Optional, List
import google.generativeai as genai

class IndependentGeminiChat:
    """
    A completely independent Gemini chat interface that can answer medical questions
    without requiring any prediction data or dependencies on other components.
    """
    
    def __init__(self, api_key: str):
        """Initialize the Gemini API with the provided key."""
        genai.configure(api_key=api_key)
        # Use the latest Gemini model for best results
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    def ask_medical_question(self, query: str) -> str:
        """
        Ask a general medical question without needing any prescription data.
        
        Args:
            query: User's medical question
            
        Returns:
            Gemini's response as a string
        """
        # Create a prompt that instructs Gemini to act as a medical assistant
        prompt = f"""
        You are a helpful medical assistant AI that provides information on medications, conditions, 
        and general health advice. Please answer the following question, explaining any medical terms 
        in simple language.
        
        User's Question: {query}
        
        Important: Always emphasize that your responses are for informational purposes only and not 
        a substitute for professional medical advice. When discussing medications, include standard 
        dosage information, common side effects, and when to consult a doctor.
        """
        
        try:
            # Generate response from Gemini
            response = self.model.generate_content(prompt)
            return response.text if response and hasattr(response, 'text') else "No response generated."
        except Exception as e:
            return f"Sorry, I couldn't process your question. Error: {str(e)}"

class MedicalAssistant:
    """
    A unified interface for medical assistance, with support for both
    independent queries and prescription-enhanced queries.
    """
    
    def __init__(self, api_key: str):
        """Initialize the medical assistant with the Gemini API key."""
        self.api_key = api_key
        self.independent_chat = IndependentGeminiChat(api_key)
        
        # Previously we tried to import from gemini_chat, but now we use our own implementation
        # Define a simple class to handle prescription data directly
        class PrescriptionChat:
            def __init__(self, api_key):
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel('gemini-1.5-flash')
                
            def generate_response(self, prescription_data, query):
                # Format prescription data
                formatted = self._format_prescription_data(prescription_data or {})
                
                # Construct the prompt
                prompt = f"""
                You are a medical assistant AI specialized in prescription information.
                
                Here is the prescription data (already preprocessed):
                {formatted}
                
                Patient's Question: {query}
                
                Please provide a helpful and clear answer, explaining any medical terms in simple language.
                Remember to emphasize that this is for informational purposes only and not medical advice.
                """
                
                try:
                    # Generate response
                    response = self.model.generate_content(prompt)
                    return response.text if response and hasattr(response, 'text') else "No response generated."
                except Exception as e:
                    return f"Sorry, I couldn't process your question. Error: {str(e)}"
            
            def _format_prescription_data(self, data):
                """Format prescription data into a readable string."""
                if not data:
                    return "No prescription data available."
                    
                formatted = []
                
                # Basic patient info
                if 'disease' in data:
                    formatted.append(f"Condition: {data.get('disease', 'Unknown')}")
                if 'medicine' in data:
                    formatted.append(f"Medication: {data.get('medicine', 'Unknown')}")
                if 'age' in data:
                    formatted.append(f"Patient Age: {data.get('age', 'Not provided')}")
                if 'weight' in data:
                    formatted.append(f"Patient Weight: {data.get('weight', 'Not provided')} kg")
                if 'frequency_per_day' in data:
                    formatted.append(f"Dosage Frequency: {data.get('frequency_per_day', 1)} times per day")
                if 'dosage_mg' in data:
                    formatted.append(f"Prescribed Dosage: {data.get('dosage_mg', 'Not specified')} mg")
                
                # Add any additional data
                for key, value in data.items():
                    if key not in ['disease', 'medicine', 'age', 'weight', 'frequency_per_day', 'dosage_mg']:
                        formatted.append(f"{key.replace('_', ' ').title()}: {value}")
                
                if not formatted:
                    return "No specific prescription details available."
                    
                return "\n".join(formatted)
                
        # Create a compatibility class that matches the GeminiHealthAssistant interface
        class PrescriptionAssistant:
            def __init__(self, api_key):
                self.chat = PrescriptionChat(api_key)
                
            def get_medical_advice(self, query):
                return self.chat.generate_response(query or {}, "Can you provide general information about this condition and medication?")
                
            def chat_about_prescription(self, patient_data, user_query):
                return self.chat.generate_response(patient_data or {}, user_query)
        
        # Use our internal implementation
        self.prescription_chat = PrescriptionAssistant(api_key)
    
    def ask_question(self, query: str, prescription_data: Optional[Dict[str, Any]] = None) -> str:
        """
        Ask a medical question, with optional prescription data for enhanced responses.
        
        Args:
            query: User's medical question
            prescription_data: Optional dictionary with prescription information
            
        Returns:
            Response from Gemini
        """
        # The prescription_chat is always available now
        if prescription_data is None:
            return self.independent_chat.ask_medical_question(query)
        
        # Otherwise, use the prescription-aware chat
        return self.prescription_chat.chat_about_prescription(prescription_data, query)
    
    def get_prescription_info(self, prescription_data: Dict[str, Any]) -> str:
        """
        Get information about a specific prescription.
        
        Args:
            prescription_data: Dictionary with prescription information
            
        Returns:
            Information about the prescription
        """
        # The prescription_chat is always available now
        return self.prescription_chat.get_medical_advice(prescription_data)

# Example usage
if __name__ == "__main__":
    import os
    
    # Get API key from environment variable
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("Please set the GEMINI_API_KEY environment variable.")
        exit(1)
    
    # Create the medical assistant
    assistant = MedicalAssistant(api_key)
    
    # Example 1: Ask a general medical question without prescription data
    question = "What are the common side effects of ibuprofen?"
    print(f"Question: {question}")
    print(f"Answer: {assistant.ask_question(question)}")
    
    # Example 2: Ask a question with prescription data
    prescription = {
        "disease": "hypertension",
        "medicine": "amlodipine",
        "age": 65,
        "weight": 75,
        "frequency_per_day": 1
    }
    question = "How should I take this medication?"
    print(f"\nQuestion with prescription: {question}")
    print(f"Answer: {assistant.ask_question(question, prescription)}")
