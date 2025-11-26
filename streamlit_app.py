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
                # Run the full pipeline to get all intermediate outputs
                pipeline_result = run_async(
                    st.session_state.seo_agent.run_query_pipeline(prompt)
                )
                
                # Get the final summarized response
                execution = pipeline_result["execution"]
                tool_logs = execution.get("tool_logs", [])
                
                # Display tool execution logs and errors
                if tool_logs:
                    with st.expander("🔧 Tool Execution Logs", expanded=True):
                        # Summary
                        total_tools = len(tool_logs)
                        successful = sum(1 for log in tool_logs if log.get("status") == "success")
                        failed = sum(1 for log in tool_logs if log.get("status") == "error")
                        total_duration = sum(log.get("duration", 0) for log in tool_logs)
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Tools", total_tools)
                        with col2:
                            st.metric("✅ Successful", successful)
                        with col3:
                            st.metric("❌ Failed", failed)
                        with col4:
                            st.metric("⏱️ Total Time", f"{total_duration:.2f}s")
                        
                        st.divider()
                        st.subheader("Tool Calls")
                        for i, log in enumerate(tool_logs, 1):
                            tool_name = log.get("tool_name", "Unknown")
                            status = log.get("status", "unknown")
                            duration = log.get("duration", 0)
                            
                            # Color code based on status
                            if status == "success":
                                status_icon = "✅"
                                status_color = "green"
                            elif status == "error":
                                status_icon = "❌"
                                status_color = "red"
                            else:
                                status_icon = "⏳"
                                status_color = "orange"
                            
                            with st.expander(
                                f"{status_icon} {tool_name} ({duration:.2f}s)" if duration else f"{status_icon} {tool_name}",
                                expanded=(status == "error")  # Auto-expand errors
                            ):
                                # Tool arguments
                                st.write("**Arguments:**")
                                st.json(log.get("args", {}))
                                
                                if status == "success":
                                    st.write("**Result:**")
                                    result = log.get("result")
                                    # Truncate very large results for display
                                    if isinstance(result, (dict, list)):
                                        # Check size and truncate if too large
                                        result_str = str(result)
                                        if len(result_str) > 10000:  # 10KB limit
                                            st.warning("⚠️ Result is very large. Showing truncated version.")
                                            if isinstance(result, dict):
                                                # Show first 5 keys
                                                truncated = dict(list(result.items())[:5])
                                                truncated["_truncated"] = f"... ({len(result)} total items, showing first 5)"
                                                st.json(truncated)
                                            else:
                                                # Show first 5 items
                                                truncated = result[:5] if len(result) > 5 else result
                                                st.json(truncated)
                                                st.caption(f"... ({len(result)} total items, showing first {len(truncated)})")
                                        else:
                                            st.json(result)
                                    else:
                                        result_str = str(result)
                                        if len(result_str) > 5000:
                                            st.warning("⚠️ Result is very large. Showing truncated version.")
                                            st.text(result_str[:5000] + "\n... (truncated)")
                                        else:
                                            st.write(result)
                                elif status == "error":
                                    st.error(f"**Error:** {log.get('error', 'Unknown error')}")
                                    st.write(f"**Error Type:** {log.get('error_type', 'Unknown')}")
                                    if log.get("traceback"):
                                        with st.expander("Full Traceback", expanded=False):
                                            st.code(log.get("traceback"), language="python")
                                
                                if duration:
                                    st.caption(f"Duration: {duration:.2f} seconds")
                
                if not execution.get("ok"):
                    err = execution.get("error", "Unknown error")
                    tb = execution.get("traceback", "")
                    error_message = f"Sorry, I ran into an error while executing the analysis: {err}\n\n{tb}"
                    st.error("### ❌ Execution Error")
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
                else:
                    result = execution.get("result", {}) or {}
                    final_response = run_async(
                        st.session_state.seo_agent._summarizer.summarize(
                            user_query=prompt,
                            plan_dict=pipeline_result["plan"],
                            execution_result=result,
                        )
                    )
                    
                    # Display all function outputs in a dropdown/expander
                    with st.expander("📊 View All Function Outputs", expanded=False):
                        # Display execution steps
                        if result.get("steps"):
                            st.subheader("Execution Steps")
                            for i, step in enumerate(result.get("steps", []), 1):
                                step_title = f"Step {step.get('step_id', i)}: {step.get('description', 'N/A')}"
                                with st.expander(step_title, expanded=False):
                                    # Key Insights
                                    insights = step.get("key_insights", "No insights available")
                                    if insights and insights != "No insights available":
                                        st.write("**Key Insights:**")
                                        st.write(insights)
                                    
                                    # Raw Results
                                    st.write("**Raw Results:**")
                                    raw_results = step.get("raw_results", {})
                                    if isinstance(raw_results, (dict, list)):
                                        st.json(raw_results)
                                    elif raw_results:
                                        st.write(raw_results)
                                    else:
                                        st.write("No raw results available")
                        else:
                            # If no steps, show the overall summary
                            if result.get("summary"):
                                st.subheader("Execution Summary")
                                st.write(result.get("summary"))
                        
                        # Display plan
                        if pipeline_result.get("plan"):
                            st.subheader("Query Plan")
                            st.json(pipeline_result.get("plan", {}))
                        
                        # Display tool selection
                        if pipeline_result.get("tool_selection"):
                            st.subheader("Tool Selection")
                            st.json(pipeline_result.get("tool_selection", {}))
                        
                        # Display generated code (optional, can be collapsed)
                        if pipeline_result.get("code"):
                            with st.expander("Generated Code", expanded=False):
                                st.code(pipeline_result.get("code", ""), language="python")
                    
                    # Display the final summarized response
                    st.markdown("### Final Answer")
                    st.markdown(final_response)
                
                # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": final_response})
                
            except Exception as e:
                import traceback
                error_message = f"Error: {str(e)}"
                tb = traceback.format_exc()
                
                st.error("### ❌ Pipeline Error")
                st.error(error_message)
                with st.expander("Full Error Details", expanded=False):
                    st.code(tb, language="python")
                
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

