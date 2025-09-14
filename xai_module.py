import shap
import numpy as np
import pandas as pd
import joblib
from typing import Dict, Any, List, Tuple

class DosageExplainer:
    """
    A class to explain dosage predictions using SHAP values.
    """
    
    def __init__(self, model_path: str = 'dosage_predictor.joblib'):
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
            
        self.explainer = None
        self.feature_names = None
        
        # Initialize SHAP explainer
        self._initialize_explainer()
    
    def _initialize_explainer(self):
        """Initialize the SHAP explainer with the model's prediction function."""
        # Get the preprocessor and model from the pipeline
        preprocessor = self.model.named_steps['preprocessor']
        
        # Get feature names after preprocessing
        self.feature_names = self._get_feature_names(preprocessor)
        
        # Create a SHAP explainer with the model's prediction function
        def predict_fn(X):
            return self.model.predict(X)
        
        # Initialize the explainer with a sample of the training data
        # (in a real application, you would pass actual training data here)
        self.explainer = shap.Explainer(
            predict_fn,
            masker=shap.maskers.Independent(np.zeros((1, len(self.feature_names)))),
            feature_names=self.feature_names
        )
    
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
        Explain a dosage prediction using SHAP values.
        
        Args:
            input_data: Dictionary containing input features
            
        Returns:
            Dictionary containing explanation and feature importances
        """
        # Convert input to DataFrame for preprocessing
        X = pd.DataFrame([input_data])
        
        # Get the preprocessed data
        preprocessed_data = self.model.named_steps['preprocessor'].transform(X)
        
        # Get SHAP values
        shap_values = self.explainer.shap_values(preprocessed_data)
        
        # Get the prediction
        prediction = self.model.predict(X)[0]
        
        # Convert SHAP values to a more interpretable format
        explanation = self._format_shap_explanation(
            shap_values[0], 
            preprocessed_data[0], 
            input_data
        )
        
        return {
            'predicted_dosage': round(prediction, 2),
            'explanation': explanation,
            'feature_importances': self._get_feature_importances(shap_values[0])
        }
    
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
            abs_impact = abs(shap_val)
            
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
