# Drug Dosage Explanation and Medical Assistant

An intelligent healthcare application that explains medication dosages based on prescription information and provides medical assistance through a dual-mode chat interface.

## Features

- **Prescription Image Processing**: Extract text from prescription images using OCR
- **Dosage Prediction**: Machine learning model to predict appropriate dosages
- **Explainable AI**: Understand why a particular dosage was recommended
- **User-friendly Interface**: Simple web interface built with Streamlit
- **Multiple Input Methods**: Upload prescription images or enter details manually
- **Dual-Mode Medical Assistant**: 
  - **Independent Mode**: Ask general medical questions without requiring a prescription
  - **Enhanced Mode**: Get personalized answers based on your prescription data

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

### Quick Start

The easiest way to run the application is using the launcher script:

```bash
python run_app.py
```

This script will:
1. Check if your Gemini API key is set
2. Install required dependencies
3. Start the Streamlit application

### Manual Setup

1. **Set up Google Gemini API**:
   - Get an API key from [Google AI Studio](https://aistudio.google.com/)
   - Set it as an environment variable:
     ```bash
     # Windows (PowerShell)
     $env:GEMINI_API_KEY="your-api-key-here"
     
     # Windows (Command Prompt)
     set GEMINI_API_KEY=your-api-key-here
     
     # Linux/Mac
     export GEMINI_API_KEY=your-api-key-here
     ```

2. **Generate Synthetic Data** (Optional):
   ```bash
   python data_generator.py
   ```

3. **Train the Model** (Optional):
   ```bash
   python train_model.py
   ```

4. **Run the Web Application**:
   ```bash
   streamlit run app.py
   ```

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

4. **Chat with Medical Assistant**:
   - Click on "Chat with Medical Assistant" 
   - Ask questions about your prescription if one is loaded
   - Ask general medical questions even without a prescription

## Project Structure

- `app.py`: Streamlit web application
- `enhanced_chatbot.py`: Improved chatbot with independent Gemini interface
- `independent_gemini.py`: Standalone Gemini interface for general medical queries
- `chatbot.py`: Original chatbot logic and prediction pipeline
- `gemini_chat.py`: Original Gemini chat implementation
- `data_generator.py`: Script to generate synthetic prescription data
- `train_model.py`: Script to train the dosage prediction model
- `xai_module.py`: Explainable AI functionality using SHAP
- `xai_module_simple.py`: Simplified XAI when SHAP is not available
- `ocr_parser.py`: OCR and text processing for prescription images
- `pages/chat.py`: Streamlit chat interface page
- `run_app.py`: Launcher script with dependency checks
- `models/`: Directory for trained models
- `data/`: Directory for datasets (not included in version control)

## Architecture

### Dual-Mode Medical Assistant

The application provides two modes of operation:

1. **Independent Mode**:
   - Can answer general medical questions without requiring a prescription
   - Uses Google's Gemini API to generate responses
   - Works even when no prescription data is available

2. **Enhanced Mode**:
   - Provides personalized answers based on prescription data
   - Incorporates patient-specific information in responses
   - Leverages the dosage prediction model for more accurate advice

### Graceful Degradation

The system is designed to handle missing components gracefully:

- If SHAP is not available, it falls back to a simplified explainer
- If no prescription data is available, it still provides general medical advice
- If OCR fails on a prescription image, it allows manual entry of data

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
