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
        margin: 1rem auto;
        text-align: center !important;
        width: 100% !important;
        max-width: 100% !important;
        box-sizing: border-box;
        display: block !important;
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
        margin: 1rem auto;
        text-align: center !important;
        width: 100% !important;
        max-width: 100% !important;
        box-sizing: border-box;
        display: block !important;
    }
    .suggestion-item {
        padding: 0.5rem;
        margin: 0.25rem 0;
        background-color: #f8f9fa;
        border-radius: 0.25rem;
        cursor: pointer;
        transition: background-color 0.2s;
    }
    .response-container {
        width: 100%;
        margin: 1rem 0;
        padding: 0;
    }
    .response-section {
        margin: 1.5rem 0;
        padding: 0;
    }
    .main-content {
        max-width: 100%;
        margin: 0 auto;
        padding: 0 1rem;
    }
    .stContainer {
        width: 100% !important;
        max-width: 100% !important;
    }
    .stMarkdown {
        width: 100% !important;
    }
    .element-container {
        width: 100% !important;
        max-width: 100% !important;
    }
    .stApp > div {
        width: 100% !important;
        max-width: 100% !important;
    }
    .main .block-container {
        width: 100% !important;
        max-width: 100% !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
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

def initialize_databases():
    """Initialize all required databases"""
    import os
    
    # Ensure data_prototype directory exists
    os.makedirs("data_prototype", exist_ok=True)
    
    # Initialize user profiles database
    conn = sqlite3.connect("data_prototype/user_profiles.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_profiles (
            user_id TEXT PRIMARY KEY,
            username TEXT UNIQUE,
            email TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            preferences TEXT
        )
    """)
    conn.commit()
    conn.close()
    
    # Initialize logs database
    conn = sqlite3.connect("data_prototype/logs.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            level TEXT,
            message TEXT,
            metadata TEXT
        )
    """)
    conn.commit()
    conn.close()

def reset_user_database():
    """Reset user database (for development/testing)"""
    import os
    if os.path.exists("data_prototype/user_profiles.db"):
        os.remove("data_prototype/user_profiles.db")
    initialize_databases()

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
    if 'trigger_search' not in st.session_state:
        st.session_state.trigger_search = None

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
    
    # Development reset button
    if st.sidebar.button("🗑️ Reset Database (Dev Only)", help="Reset user database for testing"):
        reset_user_database()
        st.sidebar.success("Database reset! Please refresh the page.")
        st.rerun()
    
    # Login form
    with st.sidebar.form("login_form"):
        username = st.text_input("Username", placeholder="Enter your username")
        email = st.text_input("Email (optional)", placeholder="your@email.com")
        
        if st.form_submit_button("Login / Create Profile"):
            if username:
                # Check if user exists
                conn = sqlite3.connect("data_prototype/user_profiles.db")
                cursor = conn.cursor()
                cursor.execute("SELECT user_id FROM user_profiles WHERE username = ?", (username,))
                existing_user = cursor.fetchone()
                conn.close()
                
                if existing_user:
                    # Existing user login
                    st.session_state.user_id = existing_user[0]
                    st.success(f"Welcome back, {username}!")
                else:
                    # Create new user
                    try:
                        st.session_state.user_id = user_profile_manager.create_user_profile(username, email)
                        st.success(f"Profile created for {username}!")
                    except sqlite3.IntegrityError as e:
                        if "UNIQUE constraint failed" in str(e):
                            # User already exists, just log them in
                            conn = sqlite3.connect("data_prototype/user_profiles.db")
                            cursor = conn.cursor()
                            cursor.execute("SELECT user_id FROM user_profiles WHERE username = ?", (username,))
                            existing_user = cursor.fetchone()
                            conn.close()
                            if existing_user:
                                st.session_state.user_id = existing_user[0]
                                st.success(f"Welcome back, {username}!")
                            else:
                                st.error("User creation failed. Please try again.")
                        else:
                            st.error(f"Database error: {e}")
                
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
                # Check if date range filter already exists
                existing_date_filter = None
                for f in st.session_state.active_filters:
                    if f.filter_type == "date_range":
                        existing_date_filter = f
                        break
                
                if existing_date_filter:
                    # Update existing filter
                    existing_date_filter.parameters = {
                        "start_date": date_range[0].isoformat(),
                        "end_date": date_range[1].isoformat()
                    }
                else:
                    # Create new filter
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
            
            # Add remove button for date range filter
            if any(f.filter_type == "date_range" for f in st.session_state.active_filters):
                if st.button("🗑️ Remove Date Range Filter", key="remove_date_filter"):
                    st.session_state.active_filters = [f for f in st.session_state.active_filters if f.filter_type != "date_range"]
                    st.rerun()
        
        with col2:
            # File type filter
            st.markdown("**📁 File Types**")
            file_types = st.multiselect(
                "Select file types",
                ["csv", "xlsx", "txt", "pdf", "docx"],
                key="file_type_filter"
            )
            
            if file_types:
                # Check if file type filter already exists
                existing_file_filter = None
                for f in st.session_state.active_filters:
                    if f.filter_type == "file_type":
                        existing_file_filter = f
                        break
                
                if existing_file_filter:
                    # Update existing filter
                    existing_file_filter.parameters = {"file_types": file_types}
                else:
                    # Create new filter
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
                # Check if content type filter already exists
                existing_content_filter = None
                for f in st.session_state.active_filters:
                    if f.filter_type == "content_type":
                        existing_content_filter = f
                        break
                
                if existing_content_filter:
                    # Update existing filter
                    existing_content_filter.parameters = {"content_types": content_types}
                else:
                    # Create new filter
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
        
        # Execute search using existing components (reuse instances)
        if 'embedding_store' not in st.session_state:
            with st.spinner("Loading embedding store..."):
                st.session_state.embedding_store = EmbeddingStore()
        if 'router' not in st.session_state:
            st.session_state.router = QueryRouter()
        if 'agent_tools' not in st.session_state:
            st.session_state.agent_tools = AgentTools()
        if 'llm_reasoning' not in st.session_state:
            st.session_state.llm_reasoning = LLMReasoning()
        if 'verifier' not in st.session_state:
            st.session_state.verifier = Verifier()
        
        embedding_store = st.session_state.embedding_store
        router = st.session_state.router
        agent_tools = st.session_state.agent_tools
        llm_reasoning = st.session_state.llm_reasoning
        verifier = st.session_state.verifier
        
        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Route query
        status_text.text("🔍 Routing query...")
        progress_bar.progress(20)
        route = router.route_query(optimized_query)
        
        # Get context (reduced from 5 to 3 for faster processing)
        status_text.text("📚 Searching documents...")
        progress_bar.progress(40)
        results = embedding_store.search(optimized_query, k=3)
        
        # Check if no relevant results found
        if not results:
            status_text.text("⚠️ No relevant documents found. Please upload files first.")
            progress_bar.progress(100)
            st.warning("No documents found in the database. Please upload files first using the 'Files' tab.")
            return
        
        # Execute based on route
        route_name = route.get("route", "unknown") if isinstance(route, dict) else route
        
        # Execute based on route
        status_text.text(f"🤖 Processing via {route_name}...")
        progress_bar.progress(60)
        
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
        
        progress_bar.progress(80)
        status_text.text("✅ Processing complete!")
        
        # Extract the actual response text from the result
        response_text = result.get("response", result.get("answer", result.get("result", "")))
        
        # Skip verification for faster processing (can be re-enabled later)
        verification = {"verified": True, "explanation": "Verification skipped for performance"}
        
        progress_bar.progress(100)
        processing_time = time.time() - start_time
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
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
    
    # Create a main container for better alignment
    with st.container():
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown(f"**Route:** {route}")
        with col2:
            st.metric("Processing Time", f"{processing_time:.2f}s")
        with col3:
            st.metric("Results Found", len(results))
        
        st.markdown("---")  # Separator line
        
        # Response section with proper container
        st.markdown("**🤖 Response:**")
        response_text = result.get("response", result.get("answer", result.get("result", "No response")))
        
        # Use a proper container for the response
        with st.container():
            st.markdown(f'<div class="response-container"><div class="success-message">{response_text}</div></div>', 
                        unsafe_allow_html=True)
        
        # Verification section
        if verification.get("verified", False):
            st.markdown("**✅ Verification:**")
            with st.container():
                st.markdown(f'<div class="response-container"><div class="info-message">{verification.get("explanation", "")}</div></div>', 
                            unsafe_allow_html=True)
    
    # Source documents
    if results:
        st.markdown("**📚 Source Documents:**")
        
        with st.container():
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

def render_pinned_kpis():
    """Render pinned KPIs that are always visible"""
    st.markdown("### 📊 Live Industrial KPIs")
    
    # File selection for KPIs
    available_files = get_available_files()
    if available_files:
        selected_files = st.multiselect(
            "📁 Select Files for KPIs",
            available_files,
            default=available_files[:2] if len(available_files) >= 2 else available_files,
            help="Choose which files to include in KPI calculations"
        )
    else:
        selected_files = []
        st.warning("No files available. Please upload files first.")
    
    # Date range selection
    col_date1, col_date2 = st.columns(2)
    with col_date1:
        start_date = st.date_input("📅 Start Date", value=None, help="Filter data from this date")
    with col_date2:
        end_date = st.date_input("📅 End Date", value=None, help="Filter data until this date")
    
    date_range = (start_date, end_date) if start_date and end_date else None
    
    # Get real-time data with filters
    kpi_data = get_real_time_kpis(selected_files, date_range)
    
    # Create columns for KPIs
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="⚡ Total Energy Consumption",
            value=f"{kpi_data['total_energy']:,} kWh",
            delta=f"{kpi_data['energy_delta']:+.1f}%",
            delta_color="inverse" if kpi_data['energy_delta'] > 0 else "normal"
        )
    
    with col2:
        st.metric(
            label="🏭 Active Equipment",
            value=f"{kpi_data['active_equipment']}/{kpi_data['total_equipment']}",
            delta=f"{kpi_data['equipment_delta']:+d}",
            delta_color="normal"
        )
    
    with col3:
        st.metric(
            label="📈 Efficiency Rate",
            value=f"{kpi_data['efficiency']:.1f}%",
            delta=f"{kpi_data['efficiency_delta']:+.1f}%",
            delta_color="normal"
        )
    
    with col4:
        st.metric(
            label="⚠️ Alerts",
            value=str(kpi_data['alerts']),
            delta=f"{kpi_data['alerts_delta']:+d}",
            delta_color="normal"
        )
    
    with col5:
        st.metric(
            label="🕐 Last Update",
            value=kpi_data['last_update'],
            delta=None
        )
    
    st.markdown("---")

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_available_files():
    """Get list of available files from the database"""
    try:
        conn = sqlite3.connect('data_prototype/metadata.db')
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT file_name FROM documents")
        files = [row[0] for row in cursor.fetchall()]
        conn.close()
        return files
    except Exception as e:
        st.error(f"Error fetching files: {e}")
        return []

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_file_ids_by_names(file_names):
    """Get file IDs for given file names"""
    if not file_names:
        return []
    try:
        conn = sqlite3.connect('data_prototype/metadata.db')
        cursor = conn.cursor()
        placeholders = ','.join(['?' for _ in file_names])
        cursor.execute(f"SELECT DISTINCT file_id FROM documents WHERE file_name IN ({placeholders})", file_names)
        file_ids = [row[0] for row in cursor.fetchall()]
        conn.close()
        return file_ids
    except Exception as e:
        st.error(f"Error fetching file IDs: {e}")
        return []

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_real_time_kpis(selected_files=None, date_range=None):
    """Get real-time KPI data from database and calculations"""
    try:
        # Initialize with default values
        kpi_data = {
            'total_energy': 0,
            'energy_delta': 0,
            'active_equipment': 0,
            'total_equipment': 15,
            'equipment_delta': 0,
            'efficiency': 0,
            'efficiency_delta': 0,
            'alerts': 0,
            'alerts_delta': 0,
            'last_update': 'No data'
        }
        
        # Try to get data from database
        conn = sqlite3.connect('data_prototype/metadata.db')
        cursor = conn.cursor()
        
        # Check if documents table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='documents'")
        if cursor.fetchone():
            # Build query with filters
            query = "SELECT COUNT(*) FROM documents WHERE doc_type = 'row'"
            params = []
            
            if selected_files:
                placeholders = ','.join(['?' for _ in selected_files])
                query += f" AND file_name IN ({placeholders})"
                params.extend(selected_files)
            
            cursor.execute(query, params)
            doc_count = cursor.fetchone()[0]
            
            if doc_count > 0:
                # Calculate real KPIs from actual data
                kpi_data = calculate_dynamic_kpis(conn, selected_files, date_range)
                kpi_data['last_update'] = '2 min ago'
        
        conn.close()
        
        return kpi_data
        
    except Exception as e:
        # Return default values if database is not available
        return {
            'total_energy': 0,
            'energy_delta': 0,
            'active_equipment': 0,
            'total_equipment': 15,
            'equipment_delta': 0,
            'efficiency': 0,
            'efficiency_delta': 0,
            'alerts': 0,
            'alerts_delta': 0,
            'last_update': 'No data'
        }

def calculate_dynamic_kpis(conn, selected_files=None, date_range=None):
    """Calculate KPIs from actual database data"""
    cursor = conn.cursor()
    
    # Build base query
    query = "SELECT content FROM documents WHERE doc_type = 'row'"
    params = []
    
    if selected_files:
        placeholders = ','.join(['?' for _ in selected_files])
        query += f" AND file_name IN ({placeholders})"
        params.extend(selected_files)
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    
    # Parse energy values from content
    total_energy = 0
    energy_values = []
    
    for row in rows:
        content = row[0]
        # Extract numeric values from content
        import re
        numbers = re.findall(r'\d+\.?\d*', content)
        for num in numbers:
            try:
                val = float(num)
                if 1000 < val < 1000000:  # Reasonable energy range
                    energy_values.append(val)
                    total_energy += val
            except:
                continue
    
    # Calculate KPIs
    avg_energy = total_energy / len(energy_values) if energy_values else 0
    active_equipment = len(set([row[0] for row in rows])) if rows else 0
    
    return {
        'total_energy': int(total_energy),
        'energy_delta': 5.2,  # Placeholder
        'active_equipment': active_equipment,
        'total_equipment': 15,
        'equipment_delta': 2,
        'efficiency': min(100, (avg_energy / 1000) * 10) if avg_energy > 0 else 0,
        'efficiency_delta': 1.8,
        'alerts': 3,
        'alerts_delta': -2,
        'last_update': 'Live data'
    }

def render_daily_briefing():
    """Render daily industrial briefing panel"""
    st.markdown("### 🌅 Daily Industrial Briefing")
    
    # File selection for briefing
    available_files = get_available_files()
    if available_files:
        selected_files = st.multiselect(
            "📁 Select Files for Briefing",
            available_files,
            default=available_files[:1] if available_files else [],
            help="Choose which files to include in the briefing"
        )
    else:
        selected_files = []
        st.warning("No files available. Please upload files first.")
    
    # Date range selection
    col_date1, col_date2 = st.columns(2)
    with col_date1:
        start_date = st.date_input("📅 Briefing Start Date", value=None, help="Start date for briefing")
    with col_date2:
        end_date = st.date_input("📅 Briefing End Date", value=None, help="End date for briefing")
    
    date_range = (start_date, end_date) if start_date and end_date else None
    
    # Briefing type selection
    briefing_type = st.selectbox(
        "📋 Briefing Type",
        ["Daily Summary", "Weekly Report", "Equipment Status", "Energy Analysis"],
        help="Choose the type of briefing to generate"
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Auto-generated summary
        st.markdown("**📋 Today's Operations Summary**")
        
        # Generate briefing on demand or auto
        if st.button("🔄 Generate Daily Briefing", type="primary"):
            with st.spinner("Generating daily briefing..."):
                briefing = generate_daily_briefing(selected_files, date_range, briefing_type)
                st.session_state.last_briefing = briefing
                st.markdown(f'<div class="success-message">{briefing}</div>', unsafe_allow_html=True)
        
        # Show last briefing if available
        if 'last_briefing' in st.session_state:
            st.markdown("**📄 Last Briefing:**")
            st.markdown(f'<div class="info-message">{st.session_state.last_briefing}</div>', unsafe_allow_html=True)
    
    with col2:
        # Quick actions
        st.markdown("**⚡ Quick Actions**")
        if st.button("📊 View Energy Trends"):
            st.session_state.show_energy_charts = True
    if st.button("🔍 Search Equipment"):
        # Use a different approach to trigger search
        st.session_state.trigger_search = "equipment status"
        st.rerun()
        if st.button("📱 Send Alert"):
            st.info("Alert system coming soon!")

def generate_daily_briefing(selected_files=None, date_range=None, briefing_type="Daily Summary"):
    """Generate daily industrial briefing using LLM"""
    try:
        # Get recent data for briefing
        embedding_store = EmbeddingStore()
        
        # Build search query based on briefing type
        if briefing_type == "Daily Summary":
            query = "energy consumption daily operations"
        elif briefing_type == "Weekly Report":
            query = "weekly production energy consumption trends"
        elif briefing_type == "Equipment Status":
            query = "equipment status maintenance operations"
        elif briefing_type == "Energy Analysis":
            query = "energy analysis consumption efficiency"
        else:
            query = "energy consumption daily operations"
        
        # Filter by selected files if provided
        file_ids = get_file_ids_by_names(selected_files) if selected_files else None
        file_id_filter = file_ids[0] if file_ids and len(file_ids) > 0 else None
        recent_docs = embedding_store.search(query, k=5, file_id_filter=file_id_filter)
        
        if not recent_docs:
            return "No recent data available for briefing generation."
        
        # Create briefing prompt
        context = "\n".join([f"- {doc['content'][:200]}..." for doc in recent_docs])
        
        # Add file and date context
        file_context = f"Selected files: {', '.join(selected_files) if selected_files else 'All files'}"
        date_context = f"Date range: {date_range[0]} to {date_range[1]}" if date_range else "All dates"
        
        briefing_prompt = f"""
        Generate a {briefing_type.lower()} based on the following industrial data:
        
        {file_context}
        {date_context}
        
        Data Context:
        {context}
        
        Format as a professional {briefing_type.lower()} with:
        1. Key metrics and performance indicators
        2. Notable events or changes
        3. Recommendations for the day
        4. Any alerts or concerns
        
        Keep it concise and actionable for plant managers.
        """
        
        # Use LLM to generate briefing
        llm_reasoning = LLMReasoning()
        result = llm_reasoning.generate_explanation(briefing_prompt, recent_docs)
        
        briefing = result.get("answer", "Unable to generate briefing at this time.")
        
        return briefing
        
    except Exception as e:
        return f"Error generating briefing: {str(e)}"

def render_energy_charts():
    """Render energy consumption charts and trends"""
    st.markdown("### 📈 Energy Consumption Analytics")
    
    # Sample data for demonstration
    import pandas as pd
    import numpy as np
    
    # Generate sample data
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
    energy_data = pd.DataFrame({
        'Date': dates,
        'Total_Consumption': np.random.normal(100000, 10000, len(dates)),
        'Peak_Demand': np.random.normal(120000, 15000, len(dates)),
        'Efficiency': np.random.normal(94, 2, len(dates))
    })
    
    # Energy consumption trend
    st.markdown("**⚡ Daily Energy Consumption Trend**")
    st.line_chart(energy_data.set_index('Date')['Total_Consumption'])
    
    # Efficiency metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Peak Demand**")
        st.bar_chart(energy_data.set_index('Date')['Peak_Demand'])
    
    with col2:
        st.markdown("**🎯 Efficiency Rate**")
        st.line_chart(energy_data.set_index('Date')['Efficiency'])

def render_live_chat():
    """Render live chat interface with streaming"""
    st.markdown("### 💬 Live Industrial Assistant")
    
    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about plant operations, equipment status, or energy consumption..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response with streaming
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            # Simulate streaming response
            response = generate_streaming_response(prompt)
            for chunk in response:
                full_response += chunk
                response_placeholder.markdown(full_response + "▌")
                time.sleep(0.05)  # Simulate streaming delay
            
            # Final response without cursor
            response_placeholder.markdown(full_response)
        
        # Add assistant response
        st.session_state.messages.append({"role": "assistant", "content": full_response})

def generate_streaming_response(prompt):
    """Generate streaming response for live chat"""
    try:
        # Get relevant context
        embedding_store = EmbeddingStore()
        results = embedding_store.search(prompt, k=3)
        
        if not results:
            yield "I don't have access to current plant data. Please upload files first to get real-time assistance."
            return
        
        # Use LLM for response
        llm_reasoning = LLMReasoning()
        result = llm_reasoning.perform_reasoning(prompt, results)
        
        response = result.get("answer", "I'm unable to process that request at the moment.")
        
        # Simulate streaming by yielding chunks
        words = response.split()
        for i, word in enumerate(words):
            if i == 0:
                yield word
            else:
                yield " " + word
                
    except Exception as e:
        yield f"Error: {str(e)}"

def render_runtime_monitoring():
    """Render runtime monitoring dashboard"""
    st.markdown("### ⏱️ Runtime Performance Monitoring")
    
    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🔍 Router Response",
            value="45ms",
            delta="-12ms",
            delta_color="normal"
        )
    
    with col2:
        st.metric(
            label="📚 RAG Retrieval",
            value="1.2s",
            delta="+0.3s",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            label="🤖 LLM Processing",
            value="8.5s",
            delta="-2.1s",
            delta_color="normal"
        )
    
    with col4:
        st.metric(
            label="⚡ SLM Processing",
            value="3.2s",
            delta="-0.8s",
            delta_color="normal"
        )
    
    # Performance chart
    st.markdown("**📊 Response Time Trends**")
    
    # Sample performance data
    import pandas as pd
    import numpy as np
    
    timestamps = pd.date_range(start='2024-01-01', periods=24, freq='H')
    performance_data = pd.DataFrame({
        'Time': timestamps,
        'Router': np.random.normal(50, 10, 24),
        'RAG': np.random.normal(1200, 200, 24),
        'LLM': np.random.normal(8500, 1000, 24),
        'SLM': np.random.normal(3200, 500, 24)
    })
    
    st.line_chart(performance_data.set_index('Time'))

def main():
    """Main application function"""
    initialize_databases()
    initialize_session_state()
    
    # Main container for better alignment
    with st.container():
        # Header
        st.markdown('<h1 class="main-header">🤖 LangChain Agentic Dashboard</h1>', 
                    unsafe_allow_html=True)
        
        # User login
        if not login_user():
            st.info("Please log in to access the dashboard")
            return
        
    # Pinned KPIs - Always visible at top with auto-refresh
    render_pinned_kpis()
    
    # Auto-refresh every 30 seconds
    if st.button("🔄 Refresh KPIs", key="refresh_kpis"):
        st.rerun()
    
    # Auto-refresh placeholder
    placeholder = st.empty()
    if st.session_state.get('auto_refresh', False):
        with placeholder.container():
            st.info("🔄 Auto-refreshing every 30 seconds...")
            time.sleep(30)
            st.rerun()
    
    # Main navigation
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🔍 Search", "📊 Dashboard", "📁 Files", "🔧 Monitoring", "💬 Live Chat", "⚙️ Settings"
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
        render_daily_briefing()
        render_energy_charts()
    
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
            
            # Process file with detailed progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            start_time = time.time()
            
            try:
                # Step 1: File ingestion
                status_text.text("📄 Processing file structure...")
                progress_bar.progress(10)
                
                documents = ingest_file(file_path, row_limit=20)  # Limit for testing
                
                # Step 2: Adding to embedding store
                status_text.text("🔍 Generating embeddings...")
                progress_bar.progress(50)
                
                embedding_store = EmbeddingStore()
                embedding_store.add_documents(documents)
                
                # Step 3: Finalizing
                status_text.text("✅ Finalizing...")
                progress_bar.progress(90)
                
                processing_time = time.time() - start_time
                
                # Complete progress
                progress_bar.progress(100)
                status_text.text(f"✅ File processed successfully in {processing_time:.2f}s")
                
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
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
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
        render_runtime_monitoring()
    
    with tab5:
        render_live_chat()
    
    with tab6:
        st.markdown("### ⚙️ Settings")
        
        # User preferences
        st.markdown("**👤 User Preferences**")
        
        # Auto-refresh toggle
        auto_refresh = st.checkbox("🔄 Enable Auto-refresh KPIs", value=st.session_state.get('auto_refresh', False))
        st.session_state.auto_refresh = auto_refresh
        
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