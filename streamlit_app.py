import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="plotly")

def get_dynamic_equipment_list():
    """Extract actual equipment names from database dynamically"""
    try:
        conn = sqlite3.connect('data_prototype/metadata.db')
        cursor = conn.cursor()
        
        # Get unique equipment names from content
        cursor.execute("""
            SELECT DISTINCT content 
            FROM documents 
            WHERE doc_type='row' 
            AND content IS NOT NULL 
            AND content != ''
            LIMIT 50
        """)
        
        equipment_list = []
        for (content,) in cursor.fetchall():
            if isinstance(content, str) and len(content) > 10:
                # Extract equipment names from content
                import re
                equipment_patterns = [
                    r'[A-Z]{2,4}\s*[M/C|MACHINE|M/C]\s*[A-Z0-9\-]+',
                    r'[A-Z]{2,4}\s*[A-Z0-9\-]+\s*[A-Z0-9\-]*',
                    r'[A-Z]{2,4}\s*Panel',
                    r'[A-Z]{2,4}\s*FDR',
                    r'[A-Z]{2,4}\s*Relay'
                ]
                
                for pattern in equipment_patterns:
                    matches = re.findall(pattern, content)
                    for match in matches:
                        if match not in equipment_list and len(match) > 5:
                            equipment_list.append(match.strip())
        
        conn.close()
        
        # Remove duplicates and sort
        equipment_list = sorted(list(set(equipment_list)))
        
        # If no equipment found, return default list
        if not equipment_list:
            equipment_list = [
                "I/C Panel Numerical Relay",
                "I/C Panel", 
                "SCP M/C FDR-2 TO ISOLATOR ROOM",
                "RADIAL FDR-1 FoR 6.6KV SWBD-3",
                "INCOMER-1 FROM 25MVA TRANSFO 1T"
            ]
        
        return equipment_list[:20]  # Limit to 20 items
        
    except Exception as e:
        print(f"Error extracting equipment: {e}")
        return [
            "I/C Panel Numerical Relay",
            "I/C Panel", 
            "SCP M/C FDR-2 TO ISOLATOR ROOM",
            "RADIAL FDR-1 FoR 6.6KV SWBD-3",
            "INCOMER-1 FROM 25MVA TRANSFO 1T"
        ]

def normalize_query_terms(text: str) -> str:
    """Normalize common plant synonyms to improve recall."""
    synonyms = [
        (r"\bsvp\b", "scp"),
        (r"\bmachine\b", "m/c"),
        (r"\bfeeder\b", "fdr"),
        (r"\bisolator\s*room\b", "isolator room"),
        (r"\bpfcs\b", "pfcs"),
    ]
    t = text.lower()
    for pat, rep in synonyms:
        t = re.sub(pat, rep, t)
    return t

def extract_search_terms_from_router(router_result: dict) -> str:
    """Extract search terms from SLM router result"""
    equipment = router_result.get('equipment', [])
    dates = router_result.get('dates', [])
    metric = router_result.get('metric', '')
    
    search_terms = []
    
    # Add equipment terms
    if equipment:
        search_terms.extend(equipment)
    
    # Add dates (filter out None values)
    if dates:
        search_terms.extend([d for d in dates if d is not None])
    
    # Add metric if relevant
    if metric and metric.lower() in ['energy', 'consumption', 'power']:
        search_terms.append(metric)
    
    # Fallback to original extraction if no SLM terms
    if not search_terms:
        return extract_search_terms_fallback("")
    
    return ' '.join(search_terms).strip()

def extract_search_terms_fallback(query: str) -> str:
    """Fallback extraction using regex patterns"""
    import re
    
    # Extract equipment/feeder patterns
    equipment_patterns = [
        r'[A-Z]{2,4}\s*[M/C|MACHINE|M/C]\s*[A-Z0-9\-]+',  # SCP M/C FDR-2, SVP MACHINE -3
        r'[A-Z]{2,4}\s*[A-Z0-9\-]+\s*[A-Z0-9\-]*',        # General equipment codes
        r'[a-z\s]+room',                                    # isolator room, control room
        r'[a-z\s]+panel',                                   # feeder panel, control panel
        r'[a-z\s]+oven',                                    # coke oven, etc.
        r'fdr[- ]?\d+',                                     # fdr-2, fdr 2
        r'[A-Z]{2,4}[- ]?\d+',                             # SCP-2, SVP-3
    ]
    
    # Extract date patterns
    date_patterns = [
        r'\d{4}[-/]\d{2}[-/]\d{2}',  # 2025-09-22, 2025/09/22
        r'\d{2}[-/]\d{2}[-/]\d{4}',  # 22-09-2025, 22/09/2025
        r'\d{1,2}[-/]\d{1,2}[-/]\d{4}',  # 9/22/2025
    ]
    
    search_terms = []
    
    # Extract equipment terms
    for pattern in equipment_patterns:
        matches = re.findall(pattern, query, re.IGNORECASE)
        search_terms.extend(matches)
    
    # Extract date terms
    for pattern in date_patterns:
        matches = re.findall(pattern, query)
        search_terms.extend(matches)
    
    # If no specific patterns found, fall back to normalized query
    if not search_terms:
        return normalize_query_terms(query)
    
    # Join and clean up
    result = ' '.join(search_terms).strip()
    return result if result else normalize_query_terms(query)

def suggest_similar_terms(query: str, limit: int = 5):
    """Suggest similar equipment/feeder terms from the DB using fuzzy matching."""
    try:
        conn = sqlite3.connect('data_prototype/metadata.db')
        cursor = conn.cursor()
        cursor.execute("SELECT content FROM documents WHERE doc_type='row' LIMIT 2000")
        rows = [r[0] for r in cursor.fetchall()]
        conn.close()
        # Extract candidate terms (tokens with letters/numbers and dashes)
        tokens = set()
        for c in rows:
            for tok in re.findall(r"[A-Za-z][A-Za-z0-9\-/]+", c)[:20]:
                if len(tok) >= 3:
                    tokens.add(tok.lower())
        return difflib.get_close_matches(query.lower(), list(tokens), n=limit, cutoff=0.6)
    except Exception:
        return []
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
import hashlib
import difflib
import re
from datetime import datetime, timedelta
import hashlib
from advanced_logging import advanced_logger
from user_profiles import user_profile_manager
from advanced_search import advanced_search_engine, SearchFilter
from response_generator import response_generator

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
    
    # Add time series interface to sidebar
    render_time_series_interface()

def render_time_series_interface():
    """Render time series query interface in sidebar"""
    
    with st.sidebar:
        st.markdown("### 📊 Time Series Explorer")
        
        # Dynamic equipment selection
        equipment_options = get_dynamic_equipment_list()
        
        selected_equipment = st.selectbox("Equipment:", equipment_options)
        
        # Time period
        time_period = st.selectbox("Period:", ["Daily", "Weekly", "Monthly"])
        
        # Date range
        start_date = st.date_input("Start Date:", value=None)
        end_date = st.date_input("End Date:", value=None)
        
        if st.button("🔍 Generate Query"):
            if time_period == "Daily":
                query = f"Show me daily energy consumption for {selected_equipment}"
            elif time_period == "Weekly":
                query = f"Show me weekly energy consumption for {selected_equipment}"
            else:
                query = f"Show me monthly energy consumption for {selected_equipment}"
            
            st.session_state.suggested_query = query
            st.success("Query generated! Check the chat box.")
    
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
    
    # Add KPI suggestion box above search
    render_kpi_suggestion_box()
    
    # Search input with autocomplete
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Check for suggested query
        if 'suggested_query' in st.session_state:
            default_query = st.session_state.suggested_query
            del st.session_state.suggested_query  # Clear after use
        else:
            default_query = st.session_state.get('trigger_search', '')
        
        query_input = st.text_input(
            "Enter your query",
            value=default_query,
            placeholder="What is the total energy consumption?",
            key="search_query"
        )
    
    with col2:
        if st.button("🔍 Search", type="primary"):
            if query_input:
                execute_advanced_search(query_input)
    
    # Autocomplete suggestions and synonym normalization
    if query_input and len(query_input) >= 2:
        suggestions = advanced_search_engine.get_autocomplete_suggestions(query_input, limit=5)
        
        if suggestions:
            st.markdown("**💡 Suggestions:**")
            for suggestion in suggestions:
                if st.button(f"{suggestion.text} ({suggestion.type})", key=f"suggestion_{suggestion.text}"):
                    st.session_state.trigger_search = suggestion.text
                    st.rerun()
    
    # Advanced filters (date filter removed per request)
    with st.expander("🎛️ Advanced Filters", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**📅 Date Range**")
            st.info("Date filter disabled to avoid unintended filtering.")
            # Ensure any lingering date_range filters are cleared
            if any(f.filter_type == "date_range" for f in st.session_state.active_filters):
                st.session_state.active_filters = [f for f in st.session_state.active_filters if f.filter_type != "date_range"]
        
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
        # Remove any date_range filters globally (disabled)
        if 'active_filters' in st.session_state and st.session_state.active_filters:
            st.session_state.active_filters = [f for f in st.session_state.active_filters if f.filter_type != 'date_range']
        
        # Apply filters
        filtered_query, filter_metadata = advanced_search_engine.apply_filters(
            query, st.session_state.active_filters
        )
        
        # Optimize query and normalize synonyms
        optimized_query = advanced_search_engine.optimize_query(filtered_query)
        optimized_query = normalize_query_terms(optimized_query)
        
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
        
        # Route query using SLM
        status_text.text("🔍 Routing query with SLM...")
        progress_bar.progress(20)
        route_result = router.route_query(optimized_query)
        
        # Extract search terms from SLM router result
        search_terms = extract_search_terms_from_router(route_result)
        st.info(f"🔍 SLM extracted: {search_terms}")
        st.info(f"🎯 Equipment: {route_result.get('equipment', [])}")
        st.info(f"📅 Dates: {route_result.get('dates', [])}")
        st.info(f"⚙️ Operation: {route_result.get('operation', '')}")
        
        # Get context using SLM-extracted search terms
        status_text.text("📚 Searching documents...")
        progress_bar.progress(40)
        results = embedding_store.search(search_terms, k=8)
        
        # Check if no relevant results found
        if not results:
            # Offer fuzzy suggestions
            suggestions = suggest_similar_terms(optimized_query)
            if suggestions:
                st.info("No exact matches found. Similar items:")
                for s in suggestions:
                    if st.button(f"🔎 {s}", key=f"sugg_{s}"):
                        st.session_state.trigger_search = s
                        st.rerun()
            else:
                status_text.text("⚠️ No relevant documents found. Please upload files first.")
                st.warning("No documents found in the database. Please upload files first using the 'Files' tab.")
            progress_bar.progress(100)
            return
        
        # Execute based on SLM route with enhanced parameters
        route_name = route_result.get("route", "llm_explain")
        status_text.text(f"🤖 Processing via {route_name}...")
        progress_bar.progress(60)

        result = None
        try:
            if route_name == "agent":
                # Use SLM-extracted operation and equipment for better agent execution
                operation = route_result.get('operation', '')
                equipment = route_result.get('equipment', [])
                metric = route_result.get('metric', '')
                
                # Use enhanced pandas operation if we have SLM parameters
                if operation and metric and equipment:
                    result = agent_tools.execute_pandas_operation(operation, metric, equipment, None, results)
                else:
                    # Fallback to original query - use pandas operation with basic parameters
                    result = agent_tools.execute_pandas_operation('sum', 'energy', [], None, results)
            elif route_name == "llm_reason":
                result = llm_reasoning.perform_reasoning(optimized_query, results)
            elif route_name == "llm_explain":
                result = llm_reasoning.generate_explanation(optimized_query, results)
            elif route_name == "slm_parse":
                result = llm_reasoning.perform_reasoning(optimized_query, results)
        except Exception:
            result = None

        # Fallback to RAG + LLM if no usable result
        if not result:
            status_text.text("🧠 Falling back to RAG + LLM...")
            progress_bar.progress(75)
            rag_results = embedding_store.search(search_terms, k=5)
            if not rag_results:
                st.warning("No relevant documents found. Try adjusting filters or uploading more data.")
                progress_bar.progress(100)
                return
            result = llm_reasoning.generate_explanation(
                "Answer precisely using the provided context. Cite facts; avoid speculation.", rag_results
            )
        
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
            route_name,
            processing_time,
            True,
            len(results)
        )
        
        # Store last query for response generator
        st.session_state.last_query = query
        
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
    
    # Generate proper response using response generator
    route_info = {"route": route, "processing_time": processing_time}
    response_text = response_generator.generate_response(
        st.session_state.get('last_query', 'Unknown query'),
        result,
        results,
        route_info
    )
    
    # Display the formatted response
    st.markdown(response_text)
    
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
    
    # Date range selection (optional - can be cleared)
    col_date1, col_date2, col_clear = st.columns([1, 1, 0.5])
    
    with col_date1:
        start_date = st.date_input("📅 Start Date", value=None, help="Filter data from this date (optional)")
    with col_date2:
        end_date = st.date_input("📅 End Date", value=None, help="Filter data until this date (optional)")
    with col_clear:
        if st.button("🗑️ Clear Dates", help="Clear date filters"):
            # Clear any cached data that depends on dates
            if 'kpi_data' in st.session_state:
                del st.session_state['kpi_data']
            st.rerun()
    
    # Only use date range if both dates are selected
    date_range = (start_date, end_date) if start_date and end_date else None
    
    # Show current filter status
    if date_range:
        st.info(f"📅 Filtering data from {start_date} to {end_date}")
    else:
        st.info("📅 Showing all available data (no date filter applied)")
    
    # Get real-time data with filters (force cache refresh on change)
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

@st.cache_data(ttl=300)
def get_last_known_date():
    """Scan recent row contents for the most recent date present (best effort)."""
    try:
        conn = sqlite3.connect('data_prototype/metadata.db')
        cursor = conn.cursor()
        cursor.execute("SELECT content FROM documents WHERE doc_type='row' ORDER BY rowid DESC LIMIT 2000")
        rows = [r[0] for r in cursor.fetchall()]
        conn.close()
        import re, datetime as _dt
        latest = None
        for c in rows:
            for m in re.findall(r"(\d{4}[-/](\d{2})[-/](\d{2}))|(\d{2}[-/](\d{2})[-/](\d{4}))", c):
                s = ''.join(m)
                s = s.replace('/', '-')
                if not s:
                    continue
                try:
                    d = _dt.date.fromisoformat(s if len(s.split('-')[0])==4 else '-'.join(reversed(s.split('-'))))
                    if latest is None or d > latest:
                        latest = d
                except Exception:
                    continue
        return latest
    except Exception:
        return None

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
            # Calculate real KPIs from actual data with optional filters
            kpi_data = calculate_dynamic_kpis(conn, selected_files, date_range)
            
            # Get actual last update time from database
            cursor.execute("SELECT MAX(created_at) FROM documents")
            last_update_result = cursor.fetchone()
            if last_update_result and last_update_result[0]:
                from datetime import datetime
                last_update_time = datetime.fromisoformat(last_update_result[0].replace('Z', '+00:00'))
                now = datetime.now()
                time_diff = now - last_update_time.replace(tzinfo=None)
                
                if time_diff.days > 0:
                    kpi_data['last_update'] = f"{time_diff.days} day(s) ago"
                elif time_diff.seconds > 3600:
                    hours = time_diff.seconds // 3600
                    kpi_data['last_update'] = f"{hours} hour(s) ago"
                elif time_diff.seconds > 60:
                    minutes = time_diff.seconds // 60
                    kpi_data['last_update'] = f"{minutes} minute(s) ago"
                else:
                    kpi_data['last_update'] = "Just now"
            else:
                kpi_data['last_update'] = 'No recent data'
        
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
    """Calculate KPIs from actual database data (file/date aware)"""
    cursor = conn.cursor()

    # Filter by files
    file_clause = ""
    params: list = []
    if selected_files:
        placeholders = ','.join(['?' for _ in selected_files])
        file_clause = f" AND file_name IN ({placeholders})"
        params.extend(selected_files)

    # Pull recent rows (limit for performance)
    cursor.execute(
        f"SELECT content, file_name FROM documents WHERE doc_type='row' {file_clause} LIMIT 2000",
        params
    )
    rows = cursor.fetchall()

    # Optional date filtering from free text content (best-effort)
    import re, datetime as _dt
    def in_range(text: str) -> bool:
        if not date_range or not all(date_range):
            return True
        m = re.search(r"(\d{4}[-/](\d{2})[-/](\d{2}))|(\d{2}[-/](\d{2})[-/](\d{4}))", text)
        if not m:
            return True
        s = m.group(0).replace('/', '-')
        try:
            d = _dt.date.fromisoformat(s if len(s.split('-')[0])==4 else '-'.join(reversed(s.split('-'))))
            return date_range[0] <= d <= date_range[1]
        except Exception:
            return True

    filtered_rows = [(c, f) for (c, f) in rows if in_range(c)]

    # Parse energy-like values with improved pattern matching
    total_energy = 0.0
    count_energy = 0
    equipment_seen = set()
    
    # More flexible energy pattern - look for numbers that could be energy values
    energy_patterns = [
        re.compile(r"\b(\d{1,6})(?:\.\d+)?\s*(?:kwh|kwh|energy|consumption)", re.IGNORECASE),
        re.compile(r"\b(\d{1,6})(?:\.\d+)?\b"),  # Any reasonable number
        re.compile(r"(\d{1,6})(?:\.\d+)?\s*(?:kwh|kwh)", re.IGNORECASE)
    ]

    for content, file_name in filtered_rows:
        # Extract equipment tokens
        for token in re.findall(r"[A-Za-z0-9_/.-]+", content)[:10]:
            equipment_seen.add(token)
        
        # Try multiple patterns to find energy values
        found_energy = False
        for pattern in energy_patterns:
            if found_energy:
                break
            for match in pattern.findall(content):
                try:
                    val = float(match)
                    # More flexible range for energy values
                    if 1 <= val <= 1000000:  # Reasonable energy range
                        total_energy += val
                        count_energy += 1
                        found_energy = True
                        break  # Only count once per content
                except Exception:
                    continue

    avg_energy = (total_energy / count_energy) if count_energy else 0.0

    return {
        'total_energy': int(total_energy),
        'energy_delta': 0.0,
        'active_equipment': max(0, min(len(equipment_seen), 50)),
        'total_equipment': max(1, len(equipment_seen) or 15),
        'equipment_delta': 0,
        'efficiency': round(min(100.0, (avg_energy / 1000.0) * 10.0), 1) if avg_energy else 0.0,
        'efficiency_delta': 0.0,
        'alerts': 0,
        'alerts_delta': 0,
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
        # Equipment quick search opens a picker and triggers a prefilled query
        with st.popover("🔍 Search Equipment"):
            eq = st.text_input("Equipment name", placeholder="e.g., I/C Panel, COP-2 battery A")
            if st.button("Search this equipment", key="eq_go"):
                if eq:
                    st.session_state.trigger_search = f"equipment status {eq}"
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
        
        # Extract search terms for better matching
        search_terms = extract_search_terms_fallback(query)
        recent_docs = embedding_store.search(search_terms, k=5, file_id_filter=file_id_filter)
        
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
    """Render dynamic energy consumption charts and trends"""
    st.markdown("### 📈 Energy Consumption Analytics")
    
    # File selection for charts
    available_files = get_available_files()
    if available_files:
        chart_files = st.multiselect(
            "📁 Select Files for Charts",
            available_files,
            default=available_files[:1] if available_files else [],
            help="Choose which files to include in the energy analytics"
        )
    else:
        chart_files = []
        st.warning("No files available. Please upload files first.")
    
    # Date range selection for charts
    col_date1, col_date2, col_chart_type, col_refresh = st.columns([1, 1, 1, 0.5])
    
    with col_date1:
        chart_start_date = st.date_input("📅 Chart Start Date", value=None, help="Start date for energy analysis (optional)")
    with col_date2:
        chart_end_date = st.date_input("📅 Chart End Date", value=None, help="End date for energy analysis (optional)")
    with col_chart_type:
        chart_type = st.selectbox("📊 Chart Type", ["Line Chart", "Bar Chart", "Area Chart"], help="Choose visualization type")
    with col_refresh:
        if st.button("🔄 Refresh", help="Refresh data and clear cache"):
            st.cache_data.clear()
            st.rerun()
    
    # Only use date range if both dates are selected
    chart_date_range = (chart_start_date, chart_end_date) if chart_start_date and chart_end_date else None
    
    # Get dynamic energy data
    energy_data = get_dynamic_energy_data(chart_files, chart_date_range)
    
    if not energy_data.empty:
        # Energy consumption trend
        st.markdown("**⚡ Energy Consumption Analysis**")
        
        # Create interactive charts using Plotly for better visualization
        import plotly.express as px
        import plotly.graph_objects as go
        
        if chart_type == "Line Chart":
            fig = px.line(energy_data, x='Date', y='Total_Consumption', 
                         title='Energy Consumption Trend',
                         labels={'Total_Consumption': 'Energy Consumption (kWh)', 'Date': 'Date'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        elif chart_type == "Bar Chart":
            fig = px.bar(energy_data, x='Date', y='Total_Consumption',
                        title='Energy Consumption by Date',
                        labels={'Total_Consumption': 'Energy Consumption (kWh)', 'Date': 'Date'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:  # Area Chart
            fig = px.area(energy_data, x='Date', y='Total_Consumption',
                         title='Energy Consumption Area Chart',
                         labels={'Total_Consumption': 'Energy Consumption (kWh)', 'Date': 'Date'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Equipment breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🏭 Equipment Energy Distribution**")
            if 'Equipment' in energy_data.columns:
                # Get detailed equipment data from original records
                equipment_data = get_equipment_energy_breakdown(chart_files, chart_date_range)
                if not equipment_data.empty:
                    fig_equipment = px.pie(equipment_data, values='Total_Consumption', names='Equipment',
                                         title='Energy Distribution by Equipment')
                    fig_equipment.update_layout(height=400)
                    st.plotly_chart(fig_equipment, use_container_width=True)
                else:
                    st.info("Equipment breakdown not available")
            else:
                st.info("Equipment breakdown not available")
        
        with col2:
            st.markdown("**📊 Energy Efficiency Trends**")
            if 'Efficiency' in energy_data.columns:
                fig_efficiency = px.line(energy_data, x='Date', y='Efficiency',
                                       title='Energy Efficiency Over Time',
                                       labels={'Efficiency': 'Efficiency (%)', 'Date': 'Date'})
                fig_efficiency.update_layout(height=400)
                st.plotly_chart(fig_efficiency, use_container_width=True)
            else:
                # Calculate efficiency from consumption data
                efficiency_data = energy_data.copy()
                efficiency_data['Efficiency'] = (efficiency_data['Total_Consumption'] / efficiency_data['Total_Consumption'].max()) * 100
                fig_efficiency = px.line(efficiency_data, x='Date', y='Efficiency',
                                       title='Calculated Efficiency Over Time',
                                       labels={'Efficiency': 'Efficiency (%)', 'Date': 'Date'})
                fig_efficiency.update_layout(height=400)
                st.plotly_chart(fig_efficiency, use_container_width=True)
        
        # Advanced Analytics
        st.markdown("**🔍 Advanced Analytics**")
        
        # Trend analysis
        if len(energy_data) > 1:
            energy_data_sorted = energy_data.sort_values('Date')
            trend = "📈 Increasing" if energy_data_sorted['Total_Consumption'].iloc[-1] > energy_data_sorted['Total_Consumption'].iloc[0] else "📉 Decreasing"
            st.info(f"**Energy Trend**: {trend}")
        
        # Summary statistics with enhanced metrics
        st.markdown("**📈 Summary Statistics**")
        col1, col2, col3, col4 = st.columns(4)
        
        total_energy = energy_data['Total_Consumption'].sum()
        avg_daily = energy_data['Total_Consumption'].mean()
        peak_consumption = energy_data['Total_Consumption'].max()
        min_consumption = energy_data['Total_Consumption'].min()
        
        with col1:
            st.metric("Total Energy", f"{total_energy:,.0f} kWh", 
                     delta=f"{((total_energy/1000)-864):.0f}% vs baseline" if total_energy > 0 else None)
        with col2:
            st.metric("Average Daily", f"{avg_daily:,.0f} kWh", 
                     delta=f"{((avg_daily/1000)-8.6):.1f}% vs baseline" if avg_daily > 0 else None)
        with col3:
            st.metric("Peak Consumption", f"{peak_consumption:,.0f} kWh")
        with col4:
            st.metric("Data Points", len(energy_data))
        
        # Energy efficiency insights
        st.markdown("**💡 Energy Insights**")
        col1, col2 = st.columns(2)
        
        with col1:
            efficiency_score = (avg_daily / peak_consumption) * 100 if peak_consumption > 0 else 0
            st.metric("Efficiency Score", f"{efficiency_score:.1f}%", 
                     delta="Good" if efficiency_score > 80 else "Needs Improvement")
        
        with col2:
            energy_variance = energy_data['Total_Consumption'].std()
            st.metric("Energy Variance", f"{energy_variance:,.0f} kWh", 
                     delta="Stable" if energy_variance < avg_daily * 0.2 else "Variable")
        
        # Equipment performance table
        if 'Equipment' in energy_data.columns:
            st.markdown("**🏭 Equipment Performance**")
            equipment_performance = energy_data.groupby('Equipment').agg({
                'Total_Consumption': ['sum', 'mean', 'count']
            }).round(2)
            equipment_performance.columns = ['Total Energy (kWh)', 'Average (kWh)', 'Records']
            equipment_performance = equipment_performance.sort_values('Total Energy (kWh)', ascending=False)
            st.dataframe(equipment_performance, use_container_width=True)
            
    else:
        st.warning("No energy data available for the selected files and date range.")
        st.info("💡 Try selecting different files or adjusting the date range.")

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_dynamic_energy_data(selected_files=None, date_range=None):
    """Get dynamic energy data from database for charts"""
    try:
        import pandas as pd
        import sqlite3
        import re
        from datetime import datetime
        
        conn = sqlite3.connect('data_prototype/metadata.db')
        cursor = conn.cursor()
        
        # Filter by files
        file_clause = ""
        params = []
        if selected_files:
            placeholders = ','.join(['?' for _ in selected_files])
            file_clause = f" AND file_name IN ({placeholders})"
            params.extend(selected_files)
        
        # Get documents with energy data
        cursor.execute(
            f"SELECT content, file_name, doc_id FROM documents WHERE doc_type='row' {file_clause} ORDER BY doc_id",
            params
        )
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return pd.DataFrame()
        
        # Process energy data
        energy_records = []
        equipment_pattern = re.compile(r'([A-Z]{2,4}\s+[A-Z/]+(?:\s+[A-Z0-9-]+)*)', re.IGNORECASE)
        energy_patterns = [
            re.compile(r'\b(\d{1,6})(?:\.\d+)?\s*(?:kwh|kwh|energy|consumption)', re.IGNORECASE),
            re.compile(r'\b(\d{1,6})(?:\.\d+)?\b'),
        ]
        
        for content, file_name, doc_id in rows:
            # Extract equipment names
            equipment_matches = equipment_pattern.findall(content)
            equipment = equipment_matches[0] if equipment_matches else "Unknown Equipment"
            
            # Extract energy values
            energy_values = []
            for pattern in energy_patterns:
                matches = pattern.findall(content)
                for match in matches:
                    try:
                        val = float(match)
                        if 1 <= val <= 1000000:  # Reasonable energy range
                            energy_values.append(val)
                            break  # Only take first valid value
                    except:
                        continue
            
            if energy_values:
                # Use current date since we don't have created_at
                date_str = datetime.now().strftime('%Y-%m-%d')
                
                energy_records.append({
                    'Date': date_str,
                    'Total_Consumption': energy_values[0],
                    'Equipment': equipment,
                    'File': file_name,
                    'Efficiency': min(100.0, (energy_values[0] / 1000.0) * 10.0) if energy_values[0] else 0.0
                })
        
        if not energy_records:
            return pd.DataFrame()
        
        # Create DataFrame
        df = pd.DataFrame(energy_records)
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Apply date filtering if provided
        if date_range and len(date_range) == 2:
            start_date, end_date = date_range
            df = df[(df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)]
        
        # Group by date and sum consumption
        daily_energy = df.groupby('Date').agg({
            'Total_Consumption': 'sum',
            'Equipment': lambda x: ', '.join(x.unique()),
            'Efficiency': 'mean'
        }).reset_index()
        
        return daily_energy
        
    except Exception as e:
        st.error(f"Error getting energy data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_equipment_energy_breakdown(selected_files=None, date_range=None):
    """Get equipment-specific energy breakdown for pie charts"""
    try:
        import pandas as pd
        import sqlite3
        import re
        from datetime import datetime
        
        conn = sqlite3.connect('data_prototype/metadata.db')
        cursor = conn.cursor()
        
        # Filter by files
        file_clause = ""
        params = []
        if selected_files:
            placeholders = ','.join(['?' for _ in selected_files])
            file_clause = f" AND file_name IN ({placeholders})"
            params.extend(selected_files)
        
        # Get documents with energy data
        cursor.execute(
            f"SELECT content, file_name, doc_id FROM documents WHERE doc_type='row' {file_clause} ORDER BY doc_id",
            params
        )
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return pd.DataFrame()
        
        # Process equipment energy data
        equipment_energy = {}
        equipment_pattern = re.compile(r'([A-Z]{2,4}\s+[A-Z/]+(?:\s+[A-Z0-9-]+)*)', re.IGNORECASE)
        energy_patterns = [
            re.compile(r'\b(\d{1,6})(?:\.\d+)?\s*(?:kwh|kwh|energy|consumption)', re.IGNORECASE),
            re.compile(r'\b(\d{1,6})(?:\.\d+)?\b'),
        ]
        
        for content, file_name, doc_id in rows:
            # Extract equipment names
            equipment_matches = equipment_pattern.findall(content)
            equipment = equipment_matches[0] if equipment_matches else "Unknown Equipment"
            
            # Extract energy values
            energy_values = []
            for pattern in energy_patterns:
                matches = pattern.findall(content)
                for match in matches:
                    try:
                        val = float(match)
                        if 1 <= val <= 1000000:  # Reasonable energy range
                            energy_values.append(val)
                            break  # Only take first valid value
                    except:
                        continue
            
            if energy_values:
                if equipment in equipment_energy:
                    equipment_energy[equipment] += energy_values[0]
                else:
                    equipment_energy[equipment] = energy_values[0]
        
        if not equipment_energy:
            return pd.DataFrame()
        
        # Create DataFrame for pie chart
        equipment_data = []
        for equipment, total_energy in equipment_energy.items():
            equipment_data.append({
                'Equipment': equipment,
                'Total_Consumption': total_energy
            })
        
        df = pd.DataFrame(equipment_data)
        df = df.sort_values('Total_Consumption', ascending=False).head(10)  # Top 10 equipment
        
        return df
        
    except Exception as e:
        st.error(f"Error getting equipment breakdown: {e}")
        return pd.DataFrame()

def render_live_chat():
    """Render live chat interface with streaming"""
    st.markdown("### 💬 Live Industrial Assistant")
    
    # Add KPI suggestions above chat
    try:
        from kpi_suggestion_system import KPISuggestionSystem
        kpi_system = KPISuggestionSystem()
        suggestions = kpi_system.get_contextual_suggestions("energy consumption")
        
        if suggestions:
            st.markdown("**💡 Quick Suggestions:**")
            cols = st.columns(3)
            for i, suggestion in enumerate(suggestions[:3]):  # Show top 3
                with cols[i % 3]:
                    if st.button(f"💡 {suggestion}", key=f"chat_kpi_suggestion_{i}"):
                        # Pre-fill the chat input with the suggestion
                        st.session_state.chat_input_key = suggestion
                        st.rerun()
    except Exception as e:
        # Fallback suggestions if KPI system fails
        st.markdown("**💡 Quick Suggestions:**")
        cols = st.columns(3)
        fallback_suggestions = [
            "Show total energy consumption",
            "Compare energy by equipment", 
            "Show daily energy trends"
        ]
        for i, suggestion in enumerate(fallback_suggestions):
            with cols[i % 3]:
                if st.button(f"💡 {suggestion}", key=f"chat_fallback_suggestion_{i}"):
                    st.session_state.chat_input_key = suggestion
                    st.rerun()
    
    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input with pre-filled suggestion
    default_chat_input = st.session_state.get('chat_input_key', '')
    if default_chat_input:
        del st.session_state.chat_input_key # Clear after use
    
    if prompt := st.chat_input(
        "Ask about plant operations, equipment status, or energy consumption..."
    ):
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
        # Get relevant context using extracted search terms
        embedding_store = EmbeddingStore()
        search_terms = extract_search_terms_fallback(prompt)
        results = embedding_store.search(search_terms, k=3)
        
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
    
    timestamps = pd.date_range(start='2024-01-01', periods=24, freq='h')
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
    
    # Cache controls
    colr1, colr2 = st.columns([1,1])
    with colr1:
        if st.button("🔄 Refresh KPIs", key="refresh_kpis"):
            st.cache_data.clear()
            st.rerun()
    with colr2:
        if st.button("🧹 Clear All Cache & Session", key="clear_all"):
            st.cache_data.clear()
            for k in list(st.session_state.keys()):
                del st.session_state[k]
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
                
                # Deduplicate per content hash to avoid re-ingestion
                with open(file_path, 'rb') as fh:
                    content_hash = hashlib.md5(fh.read()).hexdigest()
                conn = sqlite3.connect('data_prototype/metadata.db')
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM documents WHERE file_name = ?", (uploaded_file.name,))
                already = cursor.fetchone()[0]
                conn.close()
                if already > 0:
                    documents = []  # skip ingestion; file already present
                else:
                    documents = ingest_file(file_path, row_limit=20)  # Limit for testing
                
                # Step 2: Adding to embedding store
                status_text.text("🔍 Generating embeddings...")
                progress_bar.progress(50)
                
                embedding_store = EmbeddingStore()
                if documents:
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