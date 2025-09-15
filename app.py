import streamlit as st
import os
from PIL import Image
import tempfile
from datetime import datetime
from pathlib import Path

# Try to import the chatbot
from chatbot import DrugDosageChatbot
USING_ENHANCED = True
print("Using enhanced chatbot with independent Gemini interface")

# Set page config
st.set_page_config(
    page_title="Drug Dosage Explanation",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add pages directory
PAGES_DIR = Path(__file__).parent / "pages"
PAGES_DIR.mkdir(exist_ok=True)

# Custom CSS for better styling
st.markdown("""
    <style>
        .main {
            max-width: 1000px;
            padding: 2rem;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            padding: 0.5rem 1rem;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .stTextInput>div>div>input {
            border: 1px solid #4CAF50;
            border-radius: 5px;
        }
        .stSelectbox>div>div>div {
            border: 1px solid #4CAF50;
            border-radius: 5px;
        }
        .stNumberInput>div>div>input {
            border: 1px solid #4CAF50;
            border-radius: 5px;
        }
        .stFileUploader>div>div>button {
            border: 1px solid #4CAF50;
            color: #4CAF50;
        }
        .stMarkdown h1 {
            color: #4CAF50;
        }
        .stMarkdown h2 {
            color: #2E7D32;
        }
        .info-box {
            background-color: #E8F5E9;
            border-left: 5px solid #4CAF50;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 0 5px 5px 0;
        }
    </style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize all session state variables needed for the app."""
    # Initialize chat-related states
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'current_prediction' not in st.session_state:
        st.session_state.current_prediction = None
    if 'page' not in st.session_state:
        st.session_state.page = "home"
    if 'chatbot' not in st.session_state:
        # Initialize the chatbot with hardcoded API key
        gemini_api_key = "AIzaSyCxhduD3UiOU1gFSqwMonmW6ItTBcSMEIw"  # Temporary hardcoded key for testing
        st.session_state.chatbot = DrugDosageChatbot(gemini_api_key=gemini_api_key)

def add_message(question, answer):
    """Add a message to the chat history."""
    st.session_state.chat_history.append({
        "question": question,
        "answer": answer,
        "timestamp": datetime.now().strftime("%H:%M:%S")
    })

def clear_chat_history():
    """Clear the chat history."""
    st.session_state.chat_history = []

def show_chat_button():
    """Display a button to navigate to the chat page."""
    if st.button("💬 Chat with Prescription Assistant", use_container_width=True, type="primary"):
        # Store the current page to return to
        st.session_state.previous_page = "home"
        # Switch to chat page
        st.session_state.page = "chat"
        st.rerun()

def initialize_chatbot():
    """Initialize the chatbot with hardcoded API key temporarily."""
    gemini_api_key = "AIzaSyCxhduD3UiOU1gFSqwMonmW6ItTBcSMEIw"  # Temporary hardcoded key for testing
    chatbot = DrugDosageChatbot(gemini_api_key=gemini_api_key)
    
    if USING_ENHANCED:
        st.sidebar.success("✅ Using enhanced chatbot with independent Gemini interface")
    else:
        st.sidebar.info("ℹ️ Using original chatbot implementation")
        
    return chatbot

def display_sidebar():
    """Display the sidebar with app information and navigation."""
    st.sidebar.title("💊 Navigation")
    
    # Navigation menu
    page = st.sidebar.radio(
        "Go to",
        ["🏠 Home", "💬 Chat with Assistant"],
        index=0 if st.session_state.get('page') == "home" else 1
    )
    
    # Update page state based on selection
    if page == "🏠 Home" and st.session_state.page != "home":
        st.session_state.page = "home"
        st.rerun()
    elif page == "💬 Chat with Assistant" and st.session_state.page != "chat":
        st.session_state.page = "chat"
        st.rerun()
    
    st.sidebar.markdown("---")
    st.sidebar.title("About")
    st.sidebar.markdown(
        """
        **Drug Dosage Explanation** helps you understand your medication dosage 
        based on your prescription and health information.
        
        ### How it works:
        1. Upload a prescription image or enter details manually
        2. The system processes the information
        3. Get dosage recommendations with explanations
        4. Chat with the assistant for any questions
        
        ### Supported Medications:
        - Metformin (Diabetes)
        - Amlodipine (Hypertension)
        - Salbutamol (Asthma)
        - Paracetamol (Fever)
        - Amoxicillin (Infection)
        
        *Note: This is for informational purposes only. Always consult with a healthcare professional.*
        """
    )

def process_prescription_image(chatbot, uploaded_file):
    """Process an uploaded prescription image."""
    with st.spinner("Processing your prescription..."):
        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            # Process the image
            prediction = chatbot.process_prescription_image(tmp_path)
            
            # Display the prediction
            display_prediction(prediction, chatbot)
            
        except Exception as e:
            st.error(f"Error processing the image: {str(e)}")
            st.info("Please try again or enter the details manually.")
        finally:
            # Clean up the temporary file
            try:
                os.unlink(tmp_path)
            except:
                pass

def display_prediction(prediction, chatbot):
    """Display the prediction results in a user-friendly format."""
    st.subheader("📋 Dosage Recommendation")
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Condition", prediction.disease.title())
        st.metric("Medication", prediction.medicine.title())
    
    with col2:
        st.metric("Recommended Dosage", f"{prediction.predicted_dosage:.0f}mg")
        st.metric("Frequency", f"{prediction.frequency_per_day} time(s) per day")
    
    # Display model confidence and prediction details
    if hasattr(prediction, 'confidence'):
        confidence = prediction.confidence
        st.progress(confidence, text=f"Model Confidence: {confidence*100:.1f}%")
    
    # Display explanation in an expandable section
    with st.expander("ℹ️ How was this dosage determined?"):
        st.markdown("""
        The recommended dosage is based on:
        
        - Your specific condition and medication
        - Standard dosing guidelines
        - Your provided health information (if any)
        
        **Explanation:**
        {}
        """.format(prediction.explanation))
    
    # Display safety notes
    st.markdown("### ⚠️ Important Safety Information")
    st.warning(prediction.safety_notes)
    
    # Store the current prediction in session state for chat
    st.session_state.current_prediction = prediction
    
    # Show chat button
    show_chat_button()
    
    # Disclaimer
    st.markdown("---")
    st.markdown("""
    *This information is provided for educational purposes only and is not 
    intended as medical advice. Always consult with a healthcare professional 
    before starting or changing any medication regimen.*
    """)

def show_home_page():
    """Show the home page with prescription upload and manual entry options."""
    st.title("💊 Drug Dosage Explanation Chatbot")
    st.markdown("Upload a prescription image or enter details below to get dosage information.")
    
    # Initialize the chatbot
    chatbot = st.session_state.chatbot
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["📷 Upload Prescription", "✍️ Enter Details Manually"])
    
    with tab1:
        st.subheader("Upload Your Prescription")
        uploaded_file = st.file_uploader(
            "Choose an image of your prescription", 
            type=["jpg", "jpeg", "png"],
            help="Upload a clear photo of your prescription"
        )
        
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Prescription", use_container_width=True)
            
            # Process the image when the user clicks the button
            if st.button("Analyze Prescription", key="analyze_btn"):
                process_prescription_image(chatbot, uploaded_file)
    
    with tab2:
        st.subheader("Enter Prescription Details")
        
        # Create a form for manual input
        with st.form("prescription_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                # Disease selection
                disease = st.selectbox(
                    "Medical Condition",
                    ["", "Diabetes", "Hypertension", "Asthma", "Fever", "Infection"],
                    help="Select your medical condition"
                )
                
                # Age input
                age = st.number_input(
                    "Age (years)", 
                    min_value=1, 
                    max_value=120, 
                    value=None,
                    step=1,
                    help="Your age in years"
                )
            
            with col2:
                # Medicine input (auto-filled based on disease)
                medicine_map = {
                    "Diabetes": "Metformin",
                    "Hypertension": "Amlodipine",
                    "Asthma": "Salbutamol",
                    "Fever": "Paracetamol",
                    "Infection": "Amoxicillin"
                }
                
                medicine = st.text_input(
                    "Medication",
                    value=medicine_map.get(disease, ""),
                )
                
                weight = st.number_input("Weight (kg)", min_value=10, max_value=300, value=70)
            
            frequency = st.number_input("Frequency per day", min_value=1, max_value=4, value=1)
            
            submitted = st.form_submit_button("Get Dosage Recommendation")
            
            if submitted:
                if not disease or not medicine:
                    st.error("Please select both disease and medicine.")
                else:
                    try:
                        # Create input data dictionary
                        input_data = {
                            'disease': disease.lower(),
                            'medicine': medicine.lower(),
                            'age': age,
                            'weight': weight,
                            'frequency_per_day': frequency
                        }
                        
                        # Make prediction
                        prediction = st.session_state.chatbot.predict_dosage(input_data)
                        st.session_state.current_prediction = prediction
                        
                        # Display the prediction
                        display_prediction(prediction, st.session_state.chatbot)
                        
                        # Show chat button after successful prediction
                        show_chat_button()
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        st.rerun()

def main():
    """Main function to run the Streamlit app."""
    # Initialize session state
    initialize_session_state()
    
    # Display the sidebar
    display_sidebar()
    
    # Show the appropriate page based on state
    if st.session_state.page == "chat":
        # Import and run the chat page
        from pages.chat import main as chat_main
        chat_main()
    else:
        # Show home page
        show_home_page()

if __name__ == "__main__":
    main()
