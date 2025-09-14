import random
import pandas as pd
from faker import Faker
from datetime import datetime, timedelta

# Initialize faker
fake = Faker()

def generate_synthetic_data(num_samples=200):
    """
    Generate synthetic prescription data for different diseases.
    
    Args:
        num_samples (int): Number of samples to generate
        
    Returns:
        pd.DataFrame: Generated synthetic data
    """
    # Define disease and medicine mappings
    disease_meds = {
        'Diabetes': 'Metformin',
        'Hypertension': 'Amlodipine',
        'Asthma': 'Salbutamol',
        'Fever': 'Paracetamol',
        'Infection': 'Amoxicillin'
    }
    
    # Define dosage ranges (mg) for each medicine
    dosage_ranges = {
        'Metformin': (250, 1000),  # mg
        'Amlodipine': (2.5, 10),   # mg
        'Salbutamol': (100, 400),   # mcg per dose (inhaler)
        'Paracetamol': (325, 1000), # mg
        'Amoxicillin': (250, 875)   # mg
    }
    
    # Define frequency options
    frequencies = [1, 2, 3, 4]  # times per day
    
    data = []
    
    for _ in range(num_samples):
        # Randomly select a disease and corresponding medicine
        disease = random.choice(list(disease_meds.keys()))
        medicine = disease_meds[disease]
        
        # Generate patient details
        age = random.randint(1, 90)
        weight = random.uniform(2.5, 120)  # kg (from infant to adult)
        
        # Adjust weight if patient is a child
        if age < 18:
            weight = min(weight, 100)  # Cap weight for children
        
        # Generate dosage based on disease and weight
        min_dose, max_dose = dosage_ranges[medicine]
        
        # Adjust dosage based on weight (for children) and disease severity
        if age < 12:  # Pediatric dosing
            base_dose = min_dose + (max_dose - min_dose) * (weight / 70)  # 70kg as adult reference
            dosage = round(base_dose / 25) * 25  # Round to nearest 25mg
        else:  # Adult dosing
            base_dose = min_dose + (max_dose - min_dose) * random.uniform(0.7, 1.3)
            dosage = round(base_dose / 25) * 25  # Round to nearest 25mg
        
        # Ensure dosage is within safe bounds
        dosage = max(min_dose, min(dosage, max_dose))
        
        # Determine frequency
        if disease == 'Diabetes':
            frequency = 2  # Typically BID for Metformin
        elif disease == 'Hypertension':
            frequency = 1  # Typically once daily for Amlodipine
        elif disease == 'Asthma':
            frequency = random.choices([2, 3, 4], weights=[0.5, 0.3, 0.2])[0]  # More likely BID
        else:
            frequency = random.choice(frequencies)
        
        # Generate notes based on the prescription
        notes = []
        if disease == 'Diabetes':
            notes.append("Take with meals")
        if disease == 'Asthma':
            notes.append("Use inhaler as needed for symptoms")
        if disease in ['Fever', 'Infection'] and age < 12:
            notes.append("Use pediatric formulation")
        
        # Add some random missing values (10% chance for each field)
        if random.random() < 0.1:
            age = None
        if random.random() < 0.1:
            weight = None
        
        data.append({
            'disease': disease,
            'medicine': medicine,
            'age': age,
            'weight': round(weight, 1) if weight is not None else None,
            'dosage_mg': round(dosage),
            'frequency_per_day': frequency,
            'notes': '; '.join(notes) if notes else 'Take as directed by physician'
        })
    
    return pd.DataFrame(data)

def save_dataset(df, filename='synthetic_disease_dosage.csv'):
    """Save the generated dataset to a CSV file."""
    df.to_csv(filename, index=False)
    print(f"Dataset saved to {filename}")

def main():
    # Generate and save the dataset
    print("Generating synthetic prescription data...")
    df = generate_synthetic_data(num_samples=200)
    save_dataset(df, 'synthetic_disease_dosage.csv')
    
    # Print dataset summary
    print("\nDataset Summary:")
    print(f"Total samples: {len(df)}")
    print("\nSample data:")
    print(df.head())
    print("\nMissing values per column:")
    print(df.isnull().sum())

if __name__ == "__main__":
    main()
