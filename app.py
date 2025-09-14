import streamlit as st
import os
from PIL import Image
import tempfile
from datetime import datetime
from chatbot import DrugDosageChatbot

# Set page config
st.set_page_config(
    page_title="Drug Dosage Explanation Chatbot",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

def toggle_chat():
    """Toggle the chat visibility state."""
    st.session_state.chat_visible = not st.session_state.chat_visible

def initialize_session_state():
    """Initialize all session state variables needed for the app."""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'current_prediction' not in st.session_state:
        st.session_state.current_prediction = None
    if 'chat_visible' not in st.session_state:
        st.session_state.chat_visible = False
    if 'last_question' not in st.session_state:
        st.session_state.last_question = ""

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

def show_chat_interface(prediction, chatbot):
    """Display the chat interface and history."""
    # Chat interface container
    chat_container = st.container()
    
    with chat_container:
        # Display chat history
        if st.session_state.chat_history:
            for chat in st.session_state.chat_history:
                # User message
                with st.container():
                    st.markdown(f"<div style='background-color: #E8F0FE; padding: 10px; border-radius: 10px; margin-bottom: 10px;'>"
                                f"<strong>You:</strong> {chat['question']}"
                                f"</div>", unsafe_allow_html=True)
                
                # Assistant message
                with st.container():
                    st.markdown(f"<div style='background-color: #F0F2F6; padding: 10px; border-radius: 10px; margin-bottom: 20px;'>"
                                f"<strong>Assistant:</strong> {chat['answer']}"
                                f"</div>", unsafe_allow_html=True)
        
        # Clear chat button
        if st.session_state.chat_history:
            if st.button("Clear Chat History", key="clear_chat"):
                st.session_state.chat_history = []
                st.experimental_rerun()
    
    # Chat input
    with st.form(key="chat_message_form"):
        user_question = st.text_area(
            "Ask a question about your prescription:",
            placeholder="E.g., What are typical dosage ranges for my condition?",
            key="chat_input_field",
            height=100
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            submitted = st.form_submit_button("Send", use_container_width=True)
        
        if submitted and user_question:
            # Process the question
            with st.spinner("Getting information from AI assistant..."):
                try:
                    # Check if Gemini is available
                    if chatbot.gemini is None:
                        answer = "The AI assistant is not available. Please check your Gemini API key configuration."
                    else:
                        # Get response from Gemini
                        answer = chatbot.gemini.chat_about_prescription(
                            patient_data=st.session_state.current_prediction.raw_input,
                            user_query=user_question
                        )
                    
                    # Add the Q&A to chat history
                    st.session_state.chat_history.append({
                        "question": user_question,
                        "answer": answer,
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })
                    
                except Exception as e:
                    error_msg = str(e)
                    # Add error message to chat history
                    st.session_state.chat_history.append({
                        "question": user_question,
                        "answer": f"⚠️ I encountered an error while processing your question: {error_msg}\n\nPlease try rephrasing your question or asking something else.",
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })
                
                # Rerun to display the new message
                st.experimental_rerun()

def initialize_chatbot():
    """Initialize the chatbot with Gemini API key from environment variables."""
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    return DrugDosageChatbot(gemini_api_key=gemini_api_key)

def display_sidebar():
    """Display the sidebar with app information."""
    st.sidebar.title("💊 About")
    st.sidebar.markdown(
        """
        **Drug Dosage Explanation Chatbot** helps you understand your medication dosage 
        based on your prescription and health information.
        
        ### How it works:
        1. Upload a prescription image or enter details manually
        2. The system processes the information
        3. Get dosage recommendations with explanations
        
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
    
    # Determine if chat should be offered/visible
    show_chat_offer = (prediction.predicted_dosage == 0 or 
                       (hasattr(prediction, 'confidence') and prediction.confidence < 0.5))
    
    # Store the current prediction in session state for chat
    st.session_state.current_prediction = prediction
    
    # Chat section
    st.markdown("### 💬 Medical Information")
    
    # Chat toggle button - always visible
    if not st.session_state.chat_visible:
        message = "Chat with Medical Assistant"
        if show_chat_offer:
            st.info("You can ask our AI assistant for more information about your condition and medication.")
    else:
        message = "Close Chat"
        
    # Button to toggle chat visibility
    st.button(message, key="toggle_chat", on_click=toggle_chat)
    
    # Show chat interface if visible
    if st.session_state.chat_visible:
        show_chat_interface(prediction, chatbot)
    
    # Disclaimer
    st.markdown("---")
    st.markdown("""
    *This information is provided for educational purposes only and is not 
    intended as medical advice. Always consult with a healthcare professional 
    before starting or changing any medication regimen.*
    """)

def main():
    """Main function to run the Streamlit app."""
    # Initialize session state
    initialize_session_state()
    
    # Initialize the chatbot
    chatbot = initialize_chatbot()
    
    # Display the sidebar
    display_sidebar()
    
    # Main content
    st.title("💊 Drug Dosage Explanation Chatbot")
    st.markdown("Upload a prescription image or enter details below to get dosage information.")
    
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
                    disabled=bool(disease),
                    help="Medication name (auto-filled based on condition)"
                )
                
                # Weight input
                weight = st.number_input(
                    "Weight (kg)", 
                    min_value=1.0, 
                    max_value=300.0, 
                    value=None,
                    step=0.1,
                    help="Your weight in kilograms"
                )
            
            # Frequency input
            frequency = st.selectbox(
                "Frequency per day",
                ["", "Once daily", "Twice daily", "Three times daily", "Four times daily"],
                help="How often you take the medication"
            )
            
            # Additional notes
            notes = st.text_area(
                "Additional Notes",
                placeholder="Any other relevant information about your prescription...",
                help="Optional: Add any other details about your prescription"
            )
            
            # Submit button
            submitted = st.form_submit_button("Get Dosage Recommendation")
            
            if submitted:
                if not disease:
                    st.error("Please select a medical condition.")
                else:
                    # Process the form data
                    frequency_map = {
                        "Once daily": 1,
                        "Twice daily": 2,
                        "Three times daily": 3,
                        "Four times daily": 4
                    }
                    
                    input_data = {
                        'disease': disease.lower(),
                        'medicine': medicine.lower(),
                        'age': age,
                        'weight': weight,
                        'frequency_per_day': frequency_map.get(frequency, 1),
                        'notes': notes
                    }
                    
                    try:
                        # Get prediction
                        with st.spinner("Analyzing your information..."):
                            prediction = chatbot.predict_dosage(input_data)
                        
                        # Display the prediction
                        display_prediction(prediction, chatbot)
                        
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
                        st.info("Please check your input and try again.")
    
    # Add a footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: gray; font-size: 0.9em;">
            <p>This tool is for informational purposes only and does not provide medical advice.</p>
            <p>Always consult with a qualified healthcare provider for medical advice.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
