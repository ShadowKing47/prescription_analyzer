"""
Simplified XAI Module with SHAP Support

This module provides explanations for medication dosage predictions. It will attempt
to use SHAP (SHapley Additive exPlanations) for detailed, feature-level explanations
when available, but falls back to simpler explanation methods when SHAP is not installed.

This hybrid approach ensures the application can work even without SHAP while
still providing enhanced explanations when possible.
"""

import numpy as np
import pandas as pd
import joblib
from typing import Dict, Any, List
import warnings

# Try to import SHAP, but don't fail if it's not available
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. Using simplified explanations instead.")

class DosageExplainer:
    """
    A hybrid class for explaining dosage predictions.
    
    This class provides explanations for medication dosage predictions using
    SHAP when available, but falls back to simpler rule-based explanations when
    SHAP is not installed or fails to initialize.
    
    Key capabilities:
    1. Auto-detection of SHAP availability
    2. Graceful fallback to simpler explanations
    3. Consistent interface regardless of explanation method
    4. Compatibility with the original xai_module.py
    
    Usage example:
    ```python
    explainer = DosageExplainer()
    input_data = {'disease': 'diabetes', 'medicine': 'metformin', 'age': 45, 'weight': 70}
    explanation = explainer.explain_prediction(input_data)
    print(explanation['explanation'])
    ```
    """
    
    def __init__(self, model_path: str = 'models/dosage_predictor.joblib'):
        """
        Initialize the explainer with a trained model.
        
        Args:
            model_path: Path to the trained model file
        """
        try:
            self.model = joblib.load(model_path)
        except FileNotFoundError:
            print(f"Model file not found at {model_path}. Please train the model first.")
            raise
            
        self.feature_names = None
        self.explainer = None
        self.using_shap = False
        
        # Initialize explainer
        self._initialize_explainer()
    
    def _initialize_explainer(self):
        """Initialize the explainer with the model's prediction function."""
        # Get the preprocessor and model from the pipeline
        preprocessor = self.model.named_steps['preprocessor']
        
        # Get feature names after preprocessing
        self.feature_names = self._get_feature_names(preprocessor)
        
        # Try to initialize SHAP explainer if available
        if SHAP_AVAILABLE:
            try:
                # Create a SHAP explainer with the model's prediction function
                def predict_fn(X):
                    return self.model.predict(X)
                
                # Initialize the explainer with a sample of the training data
                self.explainer = shap.Explainer(
                    predict_fn,
                    masker=shap.maskers.Independent(np.zeros((1, len(self.feature_names)))),
                    feature_names=self.feature_names
                )
                self.using_shap = True
                print("Using SHAP for explanations")
            except Exception as e:
                warnings.warn(f"Could not initialize SHAP explainer: {str(e)}. Falling back to simple explanations.")
                self.using_shap = False
        else:
            self.using_shap = False
    
    def _get_feature_names(self, preprocessor) -> List[str]:
        """Extract feature names after preprocessing."""
        feature_names = []
        
        # Get numeric feature names
        numeric_features = preprocessor.named_transformers_['num'].feature_names_in_
        feature_names.extend(numeric_features)
        
        # Get categorical feature names after one-hot encoding
        ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_features = ohe.get_feature_names_out()
        feature_names.extend(cat_features)
        
        return feature_names
    
    def explain_prediction(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Provide an explanation for dosage prediction, using SHAP if available.
        
        Args:
            input_data: Dictionary containing input features
            
        Returns:
            Dictionary containing explanation and feature importances
        """
        # Convert input to DataFrame for preprocessing
        X = pd.DataFrame([input_data])
        
        # Get the preprocessed data
        preprocessed_data = self.model.named_steps['preprocessor'].transform(X)
        
        # Get the prediction
        prediction = self.model.predict(X)[0]
        
        # Generate explanation based on availability of SHAP
        if self.using_shap and self.explainer is not None:
            try:
                # Get SHAP values
                shap_values = self.explainer.shap_values(preprocessed_data)
                
                # Format explanation using SHAP values
                explanation = self._format_shap_explanation(
                    shap_values[0], 
                    preprocessed_data[0], 
                    input_data
                )
                
                # Get feature importances from SHAP values
                feature_importances = self._get_feature_importances(shap_values[0])
                
                return {
                    'predicted_dosage': round(prediction, 2),
                    'explanation': explanation,
                    'feature_importances': feature_importances,
                    'method': 'SHAP'
                }
            except Exception as e:
                print(f"Error using SHAP for explanation: {str(e)}. Falling back to simple explanation.")
                # Fall back to simple explanation if SHAP fails
        
        # Use simple explanation if SHAP is not available or failed
        explanation = self._generate_simple_explanation(input_data)
        
        return {
            'predicted_dosage': round(prediction, 2),
            'explanation': explanation,
            'feature_importances': {},  # Empty dict since we don't have SHAP values
            'method': 'Simple'
        }
    
    def _generate_simple_explanation(self, input_data: Dict[str, Any]) -> str:
        """Generate a simple explanation based on input features without SHAP."""
        disease = input_data.get('disease', 'unknown condition')
        medicine = input_data.get('medicine', 'unknown medication')
        age = input_data.get('age', None)
        weight = input_data.get('weight', None)
        
        explanation_parts = ["The predicted dosage is based on:"]
        
        # Add disease and medicine explanation
        explanation_parts.append(f"1. Your condition ({disease})")
        explanation_parts.append(f"2. The prescribed medication ({medicine})")
        
        # Add age and weight if available
        if age is not None:
            explanation_parts.append(f"3. Your age ({age} years)")
        if weight is not None:
            explanation_parts.append(f"4. Your weight ({weight} kg)")
        
        # Add general note
        explanation_parts.append("\nDosage recommendations are calculated based on standard clinical guidelines and may be adjusted by your healthcare provider based on your specific needs.")
        
        return "\n".join(explanation_parts)
        
    def _format_shap_explanation(self, shap_values: np.ndarray, 
                               feature_values: np.ndarray,
                               original_input: Dict[str, Any]) -> str:
        """Convert SHAP values to a human-readable explanation."""
        
        # Pair feature names with their SHAP values and values
        feature_impacts = list(zip(self.feature_names, shap_values, feature_values))
        
        # Sort by absolute SHAP value (most impactful first)
        feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Generate explanation text
        explanation_parts = ["The predicted dosage is primarily influenced by:"]
        
        for i, (name, shap_val, val) in enumerate(feature_impacts[:3], 1):
            direction = "increases" if shap_val > 0 else "decreases"
            
            # Make the explanation more natural
            if 'disease_' in name:
                disease = name.split('_', 1)[1].replace('_', ' ')
                if val > 0:  # This category was selected
                    explanation = f"{i}. Having {disease} "
                else:
                    continue  # Skip non-selected one-hot encoded features
            elif 'medicine_' in name:
                medicine = name.split('_', 1)[1].replace('_', ' ')
                if val > 0:  # This category was selected
                    explanation = f"{i}. Being prescribed {medicine} "
                else:
                    continue  # Skip non-selected one-hot encoded features
            else:
                explanation = f"{i}. {name.replace('_', ' ').title()} "
                if name in ['age', 'weight']:
                    explanation += f"of {val:.1f} "
                else:
                    explanation += f"({val}) "
            
            explanation += f"{direction} the predicted dosage"
            explanation_parts.append(explanation)
        
        # Add a note about the most important factor
        if feature_impacts:
            main_factor = feature_impacts[0][0]
            if 'disease_' in main_factor:
                factor_name = main_factor.split('_', 1)[1].replace('_', ' ')
                explanation_parts.append(f"\nThe most significant factor is the disease type ({factor_name}).")
            elif 'medicine_' in main_factor:
                factor_name = main_factor.split('_', 1)[1].replace('_', ' ')
                explanation_parts.append(f"\nThe most significant factor is the prescribed medication ({factor_name}).")
            else:
                explanation_parts.append(f"\nThe most significant factor is {main_factor.replace('_', ' ')}.")
        
        return "\n".join(explanation_parts)
    
    def _get_feature_importances(self, shap_values: np.ndarray) -> Dict[str, float]:
        """Get feature importances from SHAP values."""
        # Calculate mean absolute SHAP values as importance scores
        if len(shap_values.shape) > 1:  # If we have multiple outputs
            importances = np.mean(np.abs(shap_values), axis=0)
        else:
            importances = np.abs(shap_values)
        
        # Create a dictionary of feature names and their importance scores
        return dict(zip(self.feature_names, importances))

def get_safety_notes(disease: str, medicine: str, dosage: float) -> str:
    """Generate safety notes based on the predicted dosage."""
    notes = []
    
    # Define safety thresholds for each medicine (in mg)
    safety_limits = {
        'metformin': {
            'max_daily': 2000,  # mg/day
            'warning': "Take with meals to reduce gastrointestinal side effects."
        },
        'amlodipine': {
            'max_daily': 10,  # mg/day
            'warning': "May cause swelling in the ankles. Notify your doctor if this occurs."
        },
        'salbutamol': {
            'max_daily': 800,  # mcg/day (inhaler)
            'warning': "Use only as needed for symptoms. Overuse may worsen asthma."
        },
        'paracetamol': {
            'max_daily': 4000,  # mg/day
            'warning': "Do not exceed the recommended dosage to avoid liver damage."
        },
        'amoxicillin': {
            'max_daily': 1750,  # mg/day
            'warning': "Complete the full course even if you feel better."
        }
    }
    
    # Add medicine-specific notes
    if medicine in safety_limits:
        max_daily = safety_limits[medicine]['max_daily']
        notes.append(safety_limits[medicine]['warning'])
        
        # Check for potential overdose
        if dosage > max_daily:
            notes.append(f"WARNING: Predicted dosage ({dosage:.0f}mg) exceeds the maximum recommended daily dose of {max_daily}mg for {medicine}.")
    
    # Add general notes
    notes.extend([
        "Always follow your healthcare provider's instructions.",
        "Do not adjust your medication without consulting a healthcare professional.",
        "Report any unusual symptoms or side effects to your doctor immediately."
    ])
    
    return " ".join(notes)