# Model Directory

This directory contains trained machine learning models for the Healthcare Prescription Assistant application.

## Models

- `dosage_predictor.joblib`: Random Forest model for predicting medication dosages based on patient information and medical conditions.

## Model Training

Models are trained using the script `train_model.py` in the root directory. If no model exists when you run the application, the system will automatically train a new model based on the synthetic data.

## Manual Training

To manually retrain the model:

```bash
python train_model.py
```

This will create a new model file and save it in this directory.

## Model Details

The dosage prediction model:

- **Type**: Random Forest Regressor
- **Features**: Disease, medicine, age, weight, frequency
- **Target**: Recommended dosage in mg
- **Preprocessing**: One-hot encoding for categorical features, standard scaling for numerical features