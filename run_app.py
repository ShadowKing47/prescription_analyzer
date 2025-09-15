import os
import streamlit as st
import subprocess
import sys

def main():
    """
    Launcher script that sets up the environment and runs the healthcare application
    with the enhanced chatbot when available.
    """
    # Check if the GEMINI_API_KEY is set
    if not os.getenv('GEMINI_API_KEY'):
        print("Warning: GEMINI_API_KEY environment variable is not set.")
        print("The chat functionality will not work without a valid API key.")
        
        # Ask if the user wants to set it now
        set_key = input("Do you want to set your Gemini API key now? (y/n): ")
        if set_key.lower() == 'y':
            api_key = input("Enter your Gemini API key: ")
            os.environ['GEMINI_API_KEY'] = api_key
            print("API key set for this session.")
    
    # Install required packages if missing
    packages_to_check = ['streamlit', 'google-generativeai', 'pillow', 'scikit-learn', 'pandas', 'numpy', 'joblib']
    
    for package in packages_to_check:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            print(f"{package} is not installed. Installing now...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    # Try to install SHAP, but don't fail if it can't be installed
    try:
        __import__('shap')
    except ImportError:
        print("SHAP is not installed. Attempting to install...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "shap"])
            print("SHAP installed successfully.")
        except Exception as e:
            print(f"Could not install SHAP: {e}")
            print("The application will use a simplified explainer instead.")
    
    # Run the Streamlit app
    print("Starting the healthcare application...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])

if __name__ == "__main__":
    main()