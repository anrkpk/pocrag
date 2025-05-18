import streamlit as st
import os
import sys
import pages.upload as upload
import pages.chatbot as chatbot




from dotenv import load_dotenv
load_dotenv()

#add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



def main():
    """Main function to run the Streamlit app."""
    
    # Set the page configuration
    st.set_page_config(
        page_title="Customer Care AI Assistant",
        page_icon="🤖",
        layout="wide"
    )
    
    # Title and description
    st.title("Customer Care Assistant")
    st.markdown("Use the sidebar to navigate between uploading a file and chatting with the assistant.")
    
    #st.sidebar.title("Navigation")
    #st.sidebar.markdown("Use the sidebar to navigate through the app.") 
    page = st.sidebar.radio("Select a page", ["upload", "Chatbot"])
    
    if page == "upload":
        upload.render()

    elif page == "Chatbot":
        chatbot.render()
               
if __name__ == "__main__":
    main()
    # Run the main function
       