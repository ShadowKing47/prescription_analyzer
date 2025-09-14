import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

def load_data(filepath='synthetic_disease_dosage.csv'):
    """Load and preprocess the dataset."""
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Dataset not found at {filepath}. Generating synthetic data first...")
        from data_generator import generate_synthetic_data, save_dataset
        df = generate_synthetic_data(num_samples=200)
        save_dataset(df, filepath)
        print(f"Generated and saved synthetic dataset to {filepath}")
    
    # Ensure numeric columns are properly typed
    numeric_cols = ['age', 'weight', 'dosage_mg', 'frequency_per_day']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def preprocess_data(df):
    """
    Preprocess the data for model training.
    Returns X (features) and y (target).
    """
    # Define features and target
    categorical_features = ['disease', 'medicine']
    numeric_features = ['age', 'weight', 'frequency_per_day']
    
    # Create transformers for numeric and categorical features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Prepare features and target
    X = df[['disease', 'medicine', 'age', 'weight', 'frequency_per_day']]
    y = df['dosage_mg']
    
    return X, y, preprocessor

def train_model(X, y, preprocessor):
    """Train the Random Forest model with cross-validation."""
    # Create the model pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train the model
    print("Training the model...")
    model.fit(X_train, y_train)
    
    # Evaluate the model
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"Training R² score: {train_score:.4f}")
    print(f"Test R² score: {test_score:.4f}")
    
    # Calculate MAE
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Mean Absolute Error: {mae:.2f} mg")
    
    return model, X_test, y_test

def save_model(model, filename='models/dosage_predictor.joblib'):
    """Save the trained model to disk."""
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Save the model
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

def main():
    # Load and preprocess the data
    print("Loading data...")
    df = load_data()
    
    print("Preprocessing data...")
    X, y, preprocessor = preprocess_data(df)
    
    # Train the model
    model, X_test, y_test = train_model(X, y, preprocessor)
    
    # Save the model
    save_model(model)
    
    # Print feature importances if available
    if hasattr(model.named_steps['regressor'], 'feature_importances_'):
        print("\nFeature importances:")
        # Get feature names after one-hot encoding
        cat_columns = list(model.named_steps['preprocessor'].named_transformers_['cat']
                          .named_steps['onehot'].get_feature_names_out(['disease', 'medicine']))
        feature_names = ['age', 'weight', 'frequency_per_day'] + cat_columns.tolist()
        
        # Get feature importances
        importances = model.named_steps['regressor'].feature_importances_
        
        # Print feature importances
        for name, importance in zip(feature_names, importances):
            print(f"{name}: {importance:.4f}")

if __name__ == "__main__":
    main()
