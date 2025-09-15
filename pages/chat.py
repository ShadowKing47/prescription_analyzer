import streamlit as st
from datetime import datetime
from typing import Dict, Any, Optional

def safe_extract_patient_data(prediction) -> Dict[str, Any]:
    """Safely extract raw_input dict from prediction objects of various types."""
    if not prediction:  # Covers None, empty dict, empty object
        return {}

    # Case 1: plain dict
    if isinstance(prediction, dict):
        raw_input = prediction.get("raw_input")
        return raw_input if isinstance(raw_input, dict) else {}

    # Case 2: object with attribute
    if hasattr(prediction, "raw_input"):
        raw_input = getattr(prediction, "raw_input", None)
        return raw_input if isinstance(raw_input, dict) else {}

    # Case 3: Pydantic model
    if hasattr(prediction, "dict"):
        try:
            raw_input = prediction.dict().get("raw_input")
            return raw_input if isinstance(raw_input, dict) else {}
        except Exception:
            return {}

    # Fallback
    return {}

def show_chat_interface(chatbot):
    """Display the chat interface on its own page."""
    st.title("💬 Medical Assistant")
    st.write("Ask me anything about your prescription, medication, or general health questions.")
    
    # Initialize chat history if it doesn't exist
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            # User message
            st.markdown(
                f"""
                <div style='background-color:#1E88E5; padding:12px; border-radius:8px; 
                            margin-bottom:8px; border-left:4px solid #004D40;'>
                    <div style='font-weight:bold; color:#FFFFFF;'>You</div>
                    <div style='color:#FFFFFF;'>{message['question']}</div>
                    <div style='font-size:0.8em; color:#E1F5FE; text-align:right;'>{message['timestamp']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Assistant message
            st.markdown(
                f"""
                <div style='background-color:#004D40; padding:12px; border-radius:8px; 
                            margin-bottom:16px; border-left:4px solid #1E88E5;'>
                    <div style='font-weight:bold; color:#E0F2F1;'>Medical Assistant</div>
                    <div style='color:#FFFFFF;'>{message['answer']}</div>
                    <div style='font-size:0.8em; color:#B2DFDB; text-align:right;'>{message['timestamp']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    # Chat input area at the bottom
    st.markdown("---")
    
    # Show info about current prescription
    current_prediction = st.session_state.get('current_prediction')
    if current_prediction:
        try:
            # Extract information from the prediction
            disease = getattr(current_prediction, 'disease', None) or current_prediction.get('disease', 'Unknown')
            medicine = getattr(current_prediction, 'medicine', None) or current_prediction.get('medicine', 'Unknown')
            
            st.info(f"💊 Current prescription: {medicine.title()} for {disease.title()}")
        except Exception:
            st.info("💊 Prescription loaded")
    else:
        st.info("ℹ️ No prescription loaded. You can still ask general medical questions.")
    
    # Clear chat button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    
    # Text input with send button
    with st.form("chat_form", clear_on_submit=True):
        user_question = st.text_area(
            "Your message",
            placeholder="Type your question here...",
            key="user_question",
            height=100
        )
        
        col1, _ = st.columns([1, 5])
        with col1:
            send_button = st.form_submit_button("Send", use_container_width=True)
    
    # Process the question when form is submitted
    if send_button and user_question:
        with st.spinner("Getting response..."):
            try:
                # Check if we have a chatbot with Gemini capabilities
                if not hasattr(chatbot, 'chat_about_prescription') and not hasattr(chatbot, 'gemini'):
                    answer = "The AI assistant is not available. Please check your configuration."
                else:
                    try:
                        # Get patient data safely from the current prediction
                        patient_data = safe_extract_patient_data(st.session_state.get('current_prediction'))
                        
                        # Use the new interface if available, otherwise fall back to old interface
                        if hasattr(chatbot, 'chat_about_prescription'):
                            # Use the new unified interface
                            answer = chatbot.chat_about_prescription(user_question, patient_data)
                        elif hasattr(chatbot, 'gemini'):
                            # Use the old interface via the chatbot.gemini attribute
                            answer = chatbot.gemini.ask_question(user_question, patient_data)
                        else:
                            answer = "Medical assistant not properly configured."
                    except Exception as e:
                        st.error(f"Error getting response: {str(e)}")
                        answer = f"I'm sorry, I encountered an error: {str(e)}. Please try again."
                
                # Add to chat history
                st.session_state.chat_history.append({
                    "question": user_question,
                    "answer": answer,
                    "timestamp": datetime.now().strftime("%H:%M")
                })
                
                # Rerun to display the new message
                st.rerun()
                
            except Exception as e:
                st.error(f"Error processing your question: {str(e)}")
                st.session_state.chat_history.append({
                    "question": user_question,
                    "answer": f"I'm sorry, I encountered an error: {str(e)}. Please try again.",
                    "timestamp": datetime.now().strftime("%H:%M")
                })
                st.rerun()

def main():
    """Main function to run the chat page."""
    # Check if chatbot is initialized in session state
    if 'chatbot' not in st.session_state:
        st.error("Chatbot not initialized. Please go back to the main page.")
        return
    
    # Show the chat interface
    show_chat_interface(st.session_state.chatbot)

if __name__ == "__main__":
    main()
