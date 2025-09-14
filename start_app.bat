@echo off
REM Start the Drug Dosage Explanation Chatbot

echo =======================================
echo  Starting Drug Dosage Explanation Chatbot
echo =======================================

echo [1/4] Activating virtual environment...
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
    if %ERRORLEVEL% NEQ 0 (
        echo Failed to create virtual environment
        pause
        exit /b
    )
) else (
    echo Virtual environment already exists
)

call venv\Scripts\activate
if %ERRORLEVEL% NEQ 0 (
    echo Failed to activate virtual environment
    pause
    exit /b
)

echo [2/4] Installing required packages...
pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo Failed to install required packages
    pause
    exit /b
)

echo [3/4] Checking for data and model...
if not exist "synthetic_disease_dosage.csv" (
    echo Generating synthetic data...
    python data_generator.py
    if %ERRORLEVEL% NEQ 0 (
        echo Failed to generate synthetic data
        pause
        exit /b
    )
)

if not exist "models\dosage_predictor.joblib" (
    echo Training the model...
    python train_model.py
    if %ERRORLEVEL% NEQ 0 (
        echo Failed to train the model
        pause
        exit /b
    )
)

echo [4/4] Starting the application...
REM Let Streamlit handle opening the browser
streamlit run app.py

pause
