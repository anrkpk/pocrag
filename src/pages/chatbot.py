import streamlit as st

def render():
    """Render the chatbot page"""
    
    # Initialize chat history if not exists
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Title
    st.header("Ask Questions")
    
    # Check if a document has been processed
    if not st.session_state.get('initialized', False):
        st.warning("⚠️ Please upload and process a document first in the File Upload tab.")
        return

    # Show current document info
    st.info(f"📄 Using document: **{st.session_state.get('pdf_name', 'Unknown')}**")

    # Optional: Status of generator
    st.info(f"🔌 Generator ready: `{st.session_state.get('generator') is not None}`")

    # Query input
    query = st.text_input("Enter your question:")

    col1, col2 = st.columns([1, 5])
    with col1:
        submit_button = st.button("Submit")
    with col2:
        clear_button = st.button("Clear Results")

    # Process query
    if submit_button and query:
        with st.spinner("Generating answer..."):
            generator = st.session_state.get("generator")
            if not generator:
                st.error("❌ Generator is not initialized. Please process your documents first.")
                return

            try:
                response = generator.invoke({"input": query})
            except Exception as e:
                st.error(f"❌ Failed to generate response: {e}")
                return

            # Add to chat history
            st.session_state.chat_history.append({
                "query": query,
                "answer": response.get("answer", "No answer returned."),
                "context": response.get("context", [])
            })

    # Display the answer
    if st.session_state.chat_history:
        st.subheader("🧠 Conversation History")
        
        for i, exchange in enumerate(reversed(st.session_state.chat_history)):
            st.markdown(f"### ❓ Question {len(st.session_state.chat_history) - i}")
            st.markdown(f"> {exchange['query']}")
            
            st.markdown("### 💬 Answer")
            st.markdown(exchange['answer'])
            
            # Display retrieved documents
            with st.expander("📚 View Source Documents"):
                for j, doc in enumerate(exchange.get('context', [])):
                    st.markdown(f"**Document {j+1}**")
                    st.markdown(f"```\n{doc.page_content}\n```")
                    st.markdown(f"*Source: Page {doc.metadata.get('page', 'unknown')}*")
                    st.markdown("---")
            
            st.markdown("---")

    # Clear results
    if clear_button:
        st.session_state.chat_history = []
        st.rerun()
