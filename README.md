# Drug Dosage Explanation Chatbot

An intelligent chatbot that explains medication dosages based on prescription information, powered by machine learning and natural language processing.

## Features

- **Prescription Image Processing**: Extract text from prescription images using OCR
- **Dosage Prediction**: Machine learning model to predict appropriate dosages
- **Explainable AI**: Understand why a particular dosage was recommended
- **User-friendly Interface**: Simple web interface built with Streamlit
- **Multiple Input Methods**: Upload prescription images or enter details manually

## Prerequisites

- Python 3.8+
- Tesseract OCR (for image processing)
- Google Gemini API key (optional, for enhanced text processing)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd drug-dosage-chatbot
   ```

2. Install Tesseract OCR:
   - **Windows**: Download and install from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
   - **macOS**: `brew install tesseract`
   - **Linux**: `sudo apt-get install tesseract-ocr`

3. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

5. (Optional) Set up Google Gemini API:
   - Get an API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a `.env` file in the project root and add:
     ```
     GEMINI_API_KEY=your_api_key_here
     ```

## Usage

### 1. Generate Synthetic Data (Optional)

If you want to generate a new synthetic dataset:

```bash
python data_generator.py
```

This will create a `synthetic_disease_dosage.csv` file.

### 2. Train the Model

Train the dosage prediction model:

```bash
python train_model.py
```

This will train a Random Forest model and save it as `models/dosage_predictor.joblib`.

### 3. Run the Web Application

Start the Streamlit web application:

```bash
streamlit run app.py
```

Then open your web browser and navigate to `http://localhost:8501`.

## How to Use

1. **Upload Prescription**:
   - Click on the "Upload Prescription" tab
   - Upload an image of your prescription
   - Click "Analyze Prescription" to process the image

2. **Or Enter Details Manually**:
   - Click on the "Enter Details Manually" tab
   - Fill in the required information
   - Click "Get Dosage Recommendation"

3. **View Results**:
   - See the recommended dosage and frequency
   - Expand the explanation to understand how the recommendation was made
   - Review important safety information

## Project Structure

- `app.py`: Streamlit web application
- `chatbot.py`: Main chatbot logic and prediction pipeline
- `data_generator.py`: Script to generate synthetic prescription data
- `train_model.py`: Script to train the dosage prediction model
- `xai_module.py`: Explainable AI functionality using SHAP
- `ocr_parser.py`: OCR and text processing for prescription images
- `requirements.txt`: Python dependencies
- `models/`: Directory for trained models
- `data/`: Directory for datasets (not included in version control)

## Customization

### Adding New Medications

To add support for additional medications:

1. Update the `medicine_mapping` in `ocr_parser.py`
2. Add the medication to the `valid_diseases` dictionary in `chatbot.py`
3. Add appropriate safety notes in `xai_module.py`
4. Retrain the model with updated data

### Modifying the Model

To use a different machine learning model:

1. Update the model pipeline in `train_model.py`
2. Modify the prediction logic in `chatbot.py`
3. Update the explanation code in `xai_module.py` if needed

## Limitations

- This is a prototype and not intended for clinical use
- The model is trained on synthetic data and may not be accurate for all cases
- Always consult with a healthcare professional for medical advice

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with Streamlit, scikit-learn, and Google's Gemini API
- Icons by [Font Awesome](https://fontawesome.com/)
- Color scheme inspired by Material Design
