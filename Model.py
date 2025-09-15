import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

class DosagePredictionModel:
    """Main model class for drug dosage prediction."""
    
    def __init__(self, model_path: str = None):
        """
        Initialize the model.
        
        Args:
            model_path: Optional path to load a pre-trained model
        """
        self.model = None
        self.preprocessor = None
        self.feature_names = ['disease', 'medicine', 'avg_age', 'avg_weight', 'frequency_per_day']
        
        # Initialize the model pipeline
        self._initialize_pipeline()
        
        # Load pre-trained model if provided
        if model_path:
            self.load_model(model_path)
    
    def _initialize_pipeline(self):
        """Initialize the preprocessing and model pipeline."""
        # Define feature groups
        numeric_features = ['avg_age', 'avg_weight', 'frequency_per_day']
        categorical_features = ['disease', 'medicine']
        
        # Create preprocessing steps
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Create preprocessor
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        # Create full pipeline
        self.model = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('regressor', RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            ))
        ])
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess input data before training or prediction.
        
        Args:
            df: Input DataFrame with any number of features
            
        Returns:
            Preprocessed DataFrame with standardized features
        """
        # Create a copy to avoid modifying the original
        processed_df = pd.DataFrame()
        
        # Define required features and their handlers
        feature_handlers = {
            'disease': self._process_disease,
            'medicine': self._process_medicine,
            'avg_age': self._process_age,
            'avg_weight': self._process_weight,
            'frequency_per_day': self._process_frequency
        }
        
        # Process each feature using its handler
        for feature, handler in feature_handlers.items():
            processed_df[feature] = handler(df)
            
        return processed_df
        
    def _process_disease(self, df: pd.DataFrame) -> pd.Series:
        """Extract and process disease information."""
        if 'disease' in df.columns:
            return df['disease'].astype(str).str.lower()
        elif 'condition' in df.columns:
            return df['condition'].astype(str).str.lower()
        elif 'diagnosis' in df.columns:
            return df['diagnosis'].astype(str).str.lower()
        else:
            # Default to 'unknown' - will be handled by model's imputer
            return pd.Series(['unknown'] * len(df))
            
    def _process_medicine(self, df: pd.DataFrame) -> pd.Series:
        """Extract and process medicine information."""
        medicine_columns = ['medicine', 'medication', 'drug', 'prescription']
        
        for col in medicine_columns:
            if col in df.columns:
                return df[col].astype(str).str.lower()
        
        return pd.Series(['unknown'] * len(df))
            
    def _process_age(self, df: pd.DataFrame) -> pd.Series:
        """Extract and process age information."""
        if all(col in df.columns for col in ['min_age', 'max_age']):
            return (df['min_age'] + df['max_age']) / 2
        elif 'age' in df.columns:
            return pd.to_numeric(df['age'], errors='coerce')
        elif 'patient_age' in df.columns:
            return pd.to_numeric(df['patient_age'], errors='coerce')
        elif 'years_old' in df.columns:
            return pd.to_numeric(df['years_old'], errors='coerce')
        else:
            return pd.Series([None] * len(df))  # Will be imputed
            
    def _process_weight(self, df: pd.DataFrame) -> pd.Series:
        """Extract and process weight information."""
        if all(col in df.columns for col in ['min_weight', 'max_weight']):
            return (df['min_weight'] + df['max_weight']) / 2
        elif 'weight' in df.columns:
            return pd.to_numeric(df['weight'], errors='coerce')
        elif 'patient_weight' in df.columns:
            return pd.to_numeric(df['patient_weight'], errors='coerce')
        elif 'mass_kg' in df.columns:
            return pd.to_numeric(df['mass_kg'], errors='coerce')
        else:
            return pd.Series([None] * len(df))  # Will be imputed
            
    def _process_frequency(self, df: pd.DataFrame) -> pd.Series:
        """Extract and process frequency information."""
        frequency_columns = ['frequency_per_day', 'daily_frequency', 'doses_per_day', 'times_per_day']
        
        for col in frequency_columns:
            if col in df.columns:
                return pd.to_numeric(df[col], errors='coerce').fillna(1).astype(int)
            
        # Handle weight calculations
        if all(col in df.columns for col in ['min_weight', 'max_weight']):
            processed_df['avg_weight'] = (df['min_weight'] + df['max_weight']) / 2
        elif 'weight' in df.columns:
            processed_df['avg_weight'] = df['weight']
        else:
            processed_df['avg_weight'] = None  # Will be imputed by preprocessor
        
        # Handle frequency
        if 'frequency_per_day' not in processed_df.columns:
            processed_df['frequency_per_day'] = 1  # Default to once per day
        
        # Ensure disease and medicine are present
        for feature in ['disease', 'medicine']:
            if feature not in processed_df.columns:
                raise ValueError(f"Missing required feature: {feature}")
            
        # Convert types and clean data
        processed_df['disease'] = processed_df['disease'].astype(str).str.lower()
        processed_df['medicine'] = processed_df['medicine'].astype(str).str.lower()
        processed_df['frequency_per_day'] = pd.to_numeric(processed_df['frequency_per_day'], errors='coerce').fillna(1).astype(int)
        
        # Clean dosage values if present
        if 'dosage_mg' in processed_df.columns:
            processed_df['dosage_mg'] = processed_df['dosage_mg'].apply(self._clean_dosage)
        
        # Select only required features in the correct order
        final_features = ['disease', 'medicine', 'avg_age', 'avg_weight', 'frequency_per_day']
        return processed_df[final_features]
    
    def _clean_dosage(self, dosage):
        """Clean and standardize dosage values."""
        if isinstance(dosage, str):
            if 'puffs' in dosage:
                return float(dosage.split()[0])
            elif 'mg/kg' in dosage:
                mg_value = dosage.split('(')[1].split('mg')[0]
                return float(mg_value)
        return float(dosage)
    
    def train(self, data_path: str) -> float:
        """
        Train the model using data from CSV file.
        
        Args:
            data_path: Path to the training data CSV
            
        Returns:
            R² score on test set
        """
        try:
            # Load and preprocess data
            df = pd.read_csv(data_path)
        except FileNotFoundError:
            print(f"Dataset not found at {data_path}. Generating synthetic data first...")
            from data_generator import generate_synthetic_data, save_dataset
            df = generate_synthetic_data(num_samples=200)
            save_dataset(df, data_path)
            print(f"Generated and saved synthetic dataset to {data_path}")
        
        df = self._preprocess_data(df)
        
        # Prepare features and target
        X = df[['disease', 'medicine', 'age', 'weight', 'frequency_per_day']]
        X.columns = ['disease', 'medicine', 'avg_age', 'avg_weight', 'frequency_per_day']  # Rename for consistency
        y = df['dosage_mg']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        print(f"Model Performance:")
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"R² Score: {r2:.2f}")
        
        return r2
    
    def predict(self, input_data: dict) -> float:
        """
        Make a dosage prediction with flexible feature input.
        
        Args:
            input_data: Dictionary with patient information. Can contain various field names
                       that will be mapped to the required features.
            
        Returns:
            Predicted dosage in mg
            
        Raises:
            ValueError: If critical features cannot be extracted or mapped
        """
        try:
            # Convert input to DataFrame and preprocess
            df = pd.DataFrame([input_data])
            
            # Clean and standardize the features
            try:
                processed_df = self._preprocess_data(df)
                prediction = self.model.predict(processed_df)
                return float(prediction[0])
            except Exception as e:
                # Log the available features for debugging
                available_features = ", ".join(df.columns)
                mapped_features = ", ".join(processed_df.columns if 'processed_df' in locals() else [])
                
                error_msg = (
                    f"Error processing features. Available features: [{available_features}]. "
                    f"Mapped features: [{mapped_features}]. Original error: {str(e)}"
                )
                raise ValueError(error_msg)
                
        except Exception as e:
            error_msg = str(e)
            if "ColumnTransformer" in error_msg or "features" in error_msg:
                raise ValueError(
                    "Could not map input features to required model features. Please ensure "
                    "the input contains information about: disease/condition, "
                    "medicine/medication, age, weight, and frequency (optional). "
                    f"Error details: {error_msg}"
                )
            raise
        
        # Make prediction
        return self.model.predict(X)[0]
    
    def save_model(self, path: str):
        """Save the model to disk."""
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load the model from disk."""
        self.model = joblib.load(path)
        self.preprocessor = self.model.named_steps['preprocessor']
        print(f"Model loaded from {path}")

# Example usage
if __name__ == "__main__":
    # Initialize model
    model = DosagePredictionModel()
    
    # Train model
    model.train("synthetic_disease_dosage.csv")
    
    # Save model
    model.save_model("models/dosage_predictor.joblib")
    
    # Example prediction
    sample_input = {
        'disease': 'Diabetes',
        'medicine': 'Metformin',
        'avg_age': 45,
        'avg_weight': 70,
        'frequency_per_day': 2
    }
    
    prediction = model.predict(sample_input)
    print(f"\nSample prediction for diabetic patient: {prediction:.2f} mg")
