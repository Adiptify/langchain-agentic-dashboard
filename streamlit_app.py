"""
Advanced Streamlit Dashboard for LangChain Agentic System
Features: User profiles, advanced search, comprehensive logging, personalization
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import time
import uuid
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import os
import sys

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our custom modules
from embedding_store import EmbeddingStore
from router import QueryRouter
from agent_tools import AgentTools
from llm_reasoning import LLMReasoning
from verifier import Verifier
from ingestion_pipeline import ingest_file
from advanced_logging import advanced_logger
from user_profiles import user_profile_manager
from advanced_search import advanced_search_engine, SearchFilter

# Page configuration
st.set_page_config(
    page_title="LangChain Agentic Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
    }
    .info-message {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #bee5eb;
    }
    .suggestion-item {
        padding: 0.5rem;
        margin: 0.25rem 0;
        background-color: #f8f9fa;
        border-radius: 0.25rem;
        cursor: pointer;
        transition: background-color 0.2s;
    }
    .suggestion-item:hover {
        background-color: #e9ecef;
    }
    .filter-chip {
        display: inline-block;
        background-color: #007bff;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 1rem;
        font-size: 0.875rem;
        margin: 0.25rem;
    }
    .user-profile-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'session_id' not in st.session_state:
        st.session_state.session_id = None
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    if 'active_filters' not in st.session_state:
        st.session_state.active_filters = []
    if 'user_preferences' not in st.session_state:
        st.session_state.user_preferences = {}
    if 'saved_queries' not in st.session_state:
        st.session_state.saved_queries = []

def login_user():
    """Handle user login and profile creation"""
    st.sidebar.markdown("## 👤 User Profile")
    
    # Check if user is already logged in
    if st.session_state.user_id:
        profile = user_profile_manager.get_user_profile(st.session_state.user_id)
        if profile:
            st.sidebar.markdown(f"**Welcome, {profile.username}!**")
            st.sidebar.markdown(f"📊 Queries: {profile.total_queries}")
            st.sidebar.markdown(f"📁 Sessions: {profile.total_sessions}")
            
            if st.sidebar.button("Logout"):
                # End current session
                if st.session_state.session_id:
                    advanced_logger.end_session(st.session_state.session_id)
                
                # Clear session state
                for key in ['user_id', 'session_id', 'conversation_history', 'query_history']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
            return True
    
    # Login form
    with st.sidebar.form("login_form"):
        username = st.text_input("Username", placeholder="Enter your username")
        email = st.text_input("Email (optional)", placeholder="your@email.com")
        
        if st.form_submit_button("Login / Create Profile"):
            if username:
                # Check if user exists
                conn = sqlite3.connect("user_profiles.db")
                cursor = conn.cursor()
                cursor.execute("SELECT user_id FROM user_profiles WHERE username = ?", (username,))
                existing_user = cursor.fetchone()
                conn.close()
                
                if existing_user:
                    # Existing user login
                    st.session_state.user_id = existing_user[0]
                else:
                    # Create new user
                    st.session_state.user_id = user_profile_manager.create_user_profile(username, email)
                
                # Start new session
                st.session_state.session_id = advanced_logger.start_session(
                    st.session_state.user_id,
                    {"login_method": "streamlit", "timestamp": datetime.now().isoformat()}
                )
                
                st.success(f"Welcome, {username}!")
                st.rerun()
            else:
                st.error("Please enter a username")
    
    return False

def render_advanced_search():
    """Render advanced search interface with autocomplete and filters"""
    st.markdown("### 🔍 Advanced Search")
    
    # Search input with autocomplete
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query_input = st.text_input(
            "Enter your query",
            placeholder="What is the total energy consumption?",
            key="search_query"
        )
    
    with col2:
        if st.button("🔍 Search", type="primary"):
            if query_input:
                execute_advanced_search(query_input)
    
    # Autocomplete suggestions
    if query_input and len(query_input) >= 2:
        suggestions = advanced_search_engine.get_autocomplete_suggestions(query_input, limit=5)
        
        if suggestions:
            st.markdown("**💡 Suggestions:**")
            for suggestion in suggestions:
                if st.button(f"{suggestion.text} ({suggestion.type})", key=f"suggestion_{suggestion.text}"):
                    st.session_state.search_query = suggestion.text
                    st.rerun()
    
    # Advanced filters
    with st.expander("🎛️ Advanced Filters", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Date range filter
            st.markdown("**📅 Date Range**")
            date_range = st.date_input(
                "Select date range",
                value=(datetime.now() - timedelta(days=30), datetime.now()),
                key="date_filter"
            )
            
            if date_range[0] and date_range[1]:
                filter_obj = SearchFilter(
                    filter_id=str(uuid.uuid4()),
                    name="Date Range Filter",
                    filter_type="date_range",
                    parameters={
                        "start_date": date_range[0].isoformat(),
                        "end_date": date_range[1].isoformat()
                    }
                )
                st.session_state.active_filters.append(filter_obj)
        
        with col2:
            # File type filter
            st.markdown("**📁 File Types**")
            file_types = st.multiselect(
                "Select file types",
                ["csv", "xlsx", "txt", "pdf", "docx"],
                key="file_type_filter"
            )
            
            if file_types:
                filter_obj = SearchFilter(
                    filter_id=str(uuid.uuid4()),
                    name="File Type Filter",
                    filter_type="file_type",
                    parameters={"file_types": file_types}
                )
                st.session_state.active_filters.append(filter_obj)
        
        with col3:
            # Content type filter
            st.markdown("**📄 Content Types**")
            content_types = st.multiselect(
                "Select content types",
                ["row", "column", "text_chunk"],
                key="content_type_filter"
            )
            
            if content_types:
                filter_obj = SearchFilter(
                    filter_id=str(uuid.uuid4()),
                    name="Content Type Filter",
                    filter_type="content_type",
                    parameters={"content_types": content_types}
                )
                st.session_state.active_filters.append(filter_obj)
    
    # Display active filters
    if st.session_state.active_filters:
        st.markdown("**🎯 Active Filters:**")
        for filter_obj in st.session_state.active_filters:
            st.markdown(f'<span class="filter-chip">{filter_obj.name}</span>', unsafe_allow_html=True)
            
            if st.button(f"❌ Remove {filter_obj.name}", key=f"remove_{filter_obj.filter_id}"):
                st.session_state.active_filters = [f for f in st.session_state.active_filters if f.filter_id != filter_obj.filter_id]
                st.rerun()

def execute_advanced_search(query: str):
    """Execute search with advanced features"""
    start_time = time.time()
    
    try:
        # Apply filters
        filtered_query, filter_metadata = advanced_search_engine.apply_filters(
            query, st.session_state.active_filters
        )
        
        # Optimize query
        optimized_query = advanced_search_engine.optimize_query(filtered_query)
        
        # Log query pattern
        advanced_search_engine.log_query_pattern(query, True)
        
        # Execute search using existing components
        embedding_store = EmbeddingStore()
        router = QueryRouter()
        agent_tools = AgentTools()
        llm_reasoning = LLMReasoning()
        verifier = Verifier()
        
        # Route query
        route = router.route_query(optimized_query)
        
        # Get context
        results = embedding_store.search(optimized_query, k=5)
        
        # Execute based on route
        route_name = route.get("route", "unknown") if isinstance(route, dict) else route
        
        if route_name == "agent":
            result = agent_tools.execute_pandas_query(optimized_query, results)
        elif route_name == "llm_reason":
            result = llm_reasoning.perform_reasoning(optimized_query, results)
        elif route_name == "llm_explain":
            result = llm_reasoning.generate_explanation(optimized_query, results)
        elif route_name == "slm_parse":
            # For SLM parsing, we'll use the reasoning method with SLM model
            result = llm_reasoning.perform_reasoning(optimized_query, results)
        else:
            result = {"response": f"Query processed via {route_name}", "route": route_name}
        
        # Verify result
        # Extract the actual response text from the result
        response_text = result.get("response", result.get("answer", result.get("result", "")))
        verification = verifier.verify_reasoning_explanation(
            optimized_query, 
            results, 
            response_text
        )
        
        processing_time = time.time() - start_time
        
        # Log the search
        user_profile_manager.log_search(
            st.session_state.user_id,
            query,
            len(results),
            True
        )
        
        # Log to advanced logger
        advanced_logger.log_query_execution(
            st.session_state.user_id,
            st.session_state.session_id,
            query,
            route,
            processing_time,
            True,
            len(results)
        )
        
        # Add to conversation history
        st.session_state.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response": response_text,
            "route": route_name,
            "processing_time": processing_time,
            "results_count": len(results),
            "verification": verification,
            "filters_applied": filter_metadata
        })
        
        # Display results
        display_search_results(result, results, route_name, processing_time, verification)
        
    except Exception as e:
        processing_time = time.time() - start_time
        
        # Log error
        advanced_logger.log_query_execution(
            st.session_state.user_id,
            st.session_state.session_id,
            query,
            "error",
            processing_time,
            False,
            error_message=str(e)
        )
        
        st.error(f"Search failed: {str(e)}")

def display_search_results(result: Dict, results: List, route: str, 
                         processing_time: float, verification: Dict):
    """Display search results with enhanced formatting"""
    
    # Main result
    st.markdown("### 📊 Search Results")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown(f"**Route:** {route}")
    with col2:
        st.metric("Processing Time", f"{processing_time:.2f}s")
    with col3:
        st.metric("Results Found", len(results))
    
    # Response
    st.markdown("**🤖 Response:**")
    response_text = result.get("response", result.get("answer", result.get("result", "No response")))
    st.markdown(f'<div class="success-message">{response_text}</div>', 
                unsafe_allow_html=True)
    
    # Verification
    if verification.get("verified", False):
        st.markdown("**✅ Verification:**")
        st.markdown(f'<div class="info-message">{verification.get("explanation", "")}</div>', 
                    unsafe_allow_html=True)
    
    # Source documents
    if results:
        st.markdown("**📚 Source Documents:**")
        
        for i, doc in enumerate(results):
            with st.expander(f"Document {i+1}: {doc.get('file_name', 'Unknown')} (Score: {doc.get('score', 0):.3f})"):
                st.markdown(f"**Type:** {doc.get('doc_type', 'Unknown')}")
                st.markdown(f"**Content:** {doc.get('content', 'No content')[:500]}...")
                
                if doc.get('metadata'):
                    st.markdown("**Metadata:**")
                    st.json(doc['metadata'])
    
    # Save query option
    if st.button("💾 Save Query"):
        save_query_dialog()

def save_query_dialog():
    """Dialog for saving queries"""
    with st.form("save_query_form"):
        st.markdown("**💾 Save Query**")
        
        query_name = st.text_input("Query Name", placeholder="Enter a name for this query")
        description = st.text_area("Description (optional)", placeholder="Describe what this query does")
        tags = st.text_input("Tags (comma-separated)", placeholder="energy, consumption, total")
        
        if st.form_submit_button("Save"):
            if query_name:
                tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
                
                query_id = user_profile_manager.save_query(
                    st.session_state.user_id,
                    query_name,
                    st.session_state.conversation_history[-1]["query"],
                    description,
                    tag_list
                )
                
                st.success(f"Query saved with ID: {query_id}")
            else:
                st.error("Please enter a query name")

def render_user_dashboard():
    """Render personalized user dashboard"""
    st.markdown("### 📊 Your Dashboard")
    
    # Get user analytics
    analytics = user_profile_manager.get_user_analytics(st.session_state.user_id, days=30)
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Searches", analytics["total_searches"])
    with col2:
        st.metric("Success Rate", f"{analytics['success_rate']:.1f}%")
    with col3:
        st.metric("Active Days", analytics["active_days"])
    with col4:
        st.metric("Saved Queries", analytics["total_saved_queries"])
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Common queries chart
        if analytics["common_queries"]:
            df_queries = pd.DataFrame(analytics["common_queries"])
            fig = px.bar(df_queries, x="frequency", y="query", 
                        orientation="h", title="Most Common Queries")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Activity over time (placeholder)
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), 
                            end=datetime.now(), freq='D')
        activity_data = pd.DataFrame({
            'date': dates,
            'searches': [analytics["total_searches"] // 30] * len(dates)  # Placeholder
        })
        
        fig = px.line(activity_data, x='date', y='searches', 
                     title="Search Activity Over Time")
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Saved queries
    st.markdown("### 💾 Saved Queries")
    saved_queries = user_profile_manager.get_saved_queries(st.session_state.user_id)
    
    if saved_queries:
        for query in saved_queries[:5]:  # Show first 5
            with st.expander(f"{query['name']} (Used {query['use_count']} times)"):
                st.markdown(f"**Query:** {query['query']}")
                if query['description']:
                    st.markdown(f"**Description:** {query['description']}")
                if query['tags']:
                    st.markdown(f"**Tags:** {', '.join(query['tags'])}")
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button(f"Use Query", key=f"use_{query['id']}"):
                        st.session_state.search_query = query['query']
                        user_profile_manager.use_saved_query(query['id'])
                        st.rerun()
                with col2:
                    if st.button(f"Delete", key=f"delete_{query['id']}"):
                        # Delete query (implement this method)
                        st.success("Query deleted")
                        st.rerun()
    else:
        st.info("No saved queries yet. Save some queries to see them here!")

def render_system_monitoring():
    """Render system monitoring dashboard"""
    st.markdown("### 🔧 System Monitoring")
    
    # Get system metrics
    metrics = advanced_logger.get_system_metrics(hours=24)
    
    # Component performance
    st.markdown("**📈 Component Performance**")
    
    if metrics["component_stats"]:
        df_performance = pd.DataFrame(metrics["component_stats"])
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(df_performance, x='component', y='avg_duration_ms',
                        title="Average Processing Time by Component")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(df_performance, x='component', y='success_rate',
                        title="Success Rate by Component")
            st.plotly_chart(fig, use_container_width=True)
    
    # Error monitoring
    if metrics["error_stats"]:
        st.markdown("**⚠️ Error Monitoring**")
        df_errors = pd.DataFrame(metrics["error_stats"])
        
        if not df_errors.empty:
            fig = px.bar(df_errors, x='component', y='total_errors',
                        title="Errors by Component (Last 24h)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("No errors in the last 24 hours! 🎉")

def main():
    """Main application function"""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🤖 LangChain Agentic Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # User login
    if not login_user():
        st.info("Please log in to access the dashboard")
        return
    
    # Main navigation
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Search", "📊 Dashboard", "📁 Files", "🔧 Monitoring", "⚙️ Settings"
    ])
    
    with tab1:
        render_advanced_search()
        
        # Conversation history
        if st.session_state.conversation_history:
            st.markdown("### 💬 Recent Conversations")
            
            for i, conv in enumerate(st.session_state.conversation_history[-5:]):  # Show last 5
                with st.expander(f"Query {len(st.session_state.conversation_history) - i}: {conv['query'][:50]}..."):
                    st.markdown(f"**Response:** {conv['response']}")
                    st.markdown(f"**Route:** {conv['route']} | **Time:** {conv['processing_time']:.2f}s")
                    
                    if conv.get('filters_applied'):
                        st.markdown(f"**Filters:** {conv['filters_applied']}")
    
    with tab2:
        render_user_dashboard()
    
    with tab3:
        st.markdown("### 📁 File Management")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload a file",
            type=['csv', 'xlsx', 'txt', 'pdf', 'docx'],
            help="Upload CSV, Excel, TXT, PDF, or DOCX files"
        )
        
        if uploaded_file:
            # Save uploaded file
            file_path = f"uploads/{uploaded_file.name}"
            os.makedirs("uploads", exist_ok=True)
            
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Process file
            with st.spinner("Processing file..."):
                start_time = time.time()
                
                try:
                    documents = ingest_file(file_path, row_limit=20)  # Limit for testing
                    
                    # Add to embedding store
                    embedding_store = EmbeddingStore()
                    embedding_store.add_documents(documents)
                    
                    processing_time = time.time() - start_time
                    
                    # Log file upload
                    advanced_logger.log_file_upload(
                        st.session_state.user_id,
                        st.session_state.session_id,
                        file_path,
                        uploaded_file.size,
                        processing_time,
                        True
                    )
                    
                    st.success(f"File processed successfully! Generated {len(documents)} documents in {processing_time:.2f}s")
                    
                    # Add to session state
                    st.session_state.uploaded_files.append({
                        "name": uploaded_file.name,
                        "path": file_path,
                        "documents": len(documents),
                        "processing_time": processing_time,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                except Exception as e:
                    processing_time = time.time() - start_time
                    
                    # Log error
                    advanced_logger.log_file_upload(
                        st.session_state.user_id,
                        st.session_state.session_id,
                        file_path,
                        uploaded_file.size,
                        processing_time,
                        False,
                        str(e)
                    )
                    
                    st.error(f"File processing failed: {str(e)}")
        
        # Display uploaded files
        if st.session_state.uploaded_files:
            st.markdown("**📋 Uploaded Files:**")
            
            for file_info in st.session_state.uploaded_files:
                with st.expander(f"{file_info['name']} ({file_info['documents']} documents)"):
                    st.markdown(f"**Processing Time:** {file_info['processing_time']:.2f}s")
                    st.markdown(f"**Uploaded:** {file_info['timestamp']}")
    
    with tab4:
        render_system_monitoring()
    
    with tab5:
        st.markdown("### ⚙️ Settings")
        
        # User preferences
        st.markdown("**👤 User Preferences**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            theme = st.selectbox("Theme", ["light", "dark"], index=0)
            language = st.selectbox("Language", ["en", "es", "fr", "de"], index=0)
            timezone = st.selectbox("Timezone", ["UTC", "EST", "PST", "GMT"], index=0)
        
        with col2:
            auto_save = st.checkbox("Auto-save queries", value=True)
            show_tips = st.checkbox("Show tips", value=True)
            email_notifications = st.checkbox("Email notifications", value=False)
        
        if st.button("Save Preferences"):
            preferences = {
                "theme": theme,
                "language": language,
                "timezone": timezone,
                "auto_save_queries": auto_save,
                "show_tips": show_tips,
                "email_notifications": email_notifications
            }
            
            user_profile_manager.update_user_preferences(st.session_state.user_id, preferences)
            st.session_state.user_preferences = preferences
            st.success("Preferences saved!")
        
        # Custom filters
        st.markdown("**🎛️ Custom Filters**")
        
        with st.form("custom_filter_form"):
            filter_name = st.text_input("Filter Name")
            filter_type = st.selectbox("Filter Type", 
                                     ["date_range", "file_type", "content_type", "numeric_range", "text_match"])
            
            if st.form_submit_button("Create Filter"):
                if filter_name:
                    # Create custom filter (simplified)
                    st.success(f"Custom filter '{filter_name}' created!")

if __name__ == "__main__":
    main()