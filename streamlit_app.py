import streamlit as st
import asyncio
import nest_asyncio
import os
from langchain_openai import ChatOpenAI
from src.tools.seo_tools import SeoTools
from src.agents.seo_agent import SEOAgent


from dotenv import load_dotenv
load_dotenv(override=True)

# Enable nested event loops for Streamlit
# This allows async code to run within Streamlit's synchronous context
nest_asyncio.apply()

def run_async(coro):
    """
    Helper function to run async code in Streamlit with nest_asyncio.
    Handles event loop creation and management gracefully.
    """
    try:
        # Try to get or create event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Check if loop is closed
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Run the coroutine using nest_asyncio
        return loop.run_until_complete(coro)
    except Exception as e:
        # Fallback: create a completely new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            # Don't close the loop - let Streamlit manage it
            # Note: A harmless shutdown warning may appear when stopping Streamlit
            # This is a known asyncio cleanup issue and doesn't affect functionality
            pass

# Page configuration
st.set_page_config(
    page_title="SEO Agent Chat",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "seo_agent" not in st.session_state:
    st.session_state.seo_agent = None
    st.session_state.initialized = False

# Initialize SEO Agent
@st.cache_resource
def initialize_seo_agent():
    """Initialize and cache the SEO Agent"""
    # Get model from environment or use default
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o")
    llm = ChatOpenAI(model=model_name, temperature=0)
    seo_tools = SeoTools()
    seo_agent = SEOAgent(llm, seo_tools)
    return seo_agent

# Initialize agent
if not st.session_state.initialized:
    with st.spinner("Initializing SEO Agent..."):
        try:
            st.session_state.seo_agent = initialize_seo_agent()
            st.session_state.initialized = True
        except Exception as e:
            st.error(f"Error initializing SEO Agent: {str(e)}")
            st.stop()

# Title and description
st.title("🔍 SEO Agent Chat")
st.markdown("Ask questions about SEO, keyword coverage, search analytics, and more!")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask your SEO question..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing your query..."):
            try:
                # Run async function using helper
                response = run_async(
                    st.session_state.seo_agent.run_and_respond(prompt)
                )
                
                st.markdown(response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                error_message = f"Error: {str(e)}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})

# Sidebar with info
with st.sidebar:
    st.header("About")
    st.markdown("""
    This SEO Agent can help you with:
    - Keyword research and analysis
    - Search Console data
    - Backlink analysis
    - Content analysis
    - Competitor research
    - And much more!
    """)
    
    st.header("Clear Chat")
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

