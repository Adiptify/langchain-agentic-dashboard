# Langchain Implementation Project

This project aims to build a web dashboard backed by a LangChain agentic system. The system will ingest various document formats (CSV, Excel, PDF, DOCX), normalize data, index semantic vectors with Ollama embeddings into FAISS, dynamically route queries between SLMs and LLMs, and execute safe tools to return hybrid answers.

## Current Status:
- Starting from a fresh codebase based on the architecture described in `cursor.md`.
- All necessary libraries and Ollama are assumed to be installed in the `mendyenv` virtual environment.
- Initial `utils.py`, `config.py`, `ingestion_pipeline.py`, `embedding_store.py`, `router.py`, `agent_tools.py`, `llm_reasoning.py`, `verifier.py`, `test_ingestion_cli.py`, and `test_query_cli.py` are implemented.
- Fixed linter errors and initial database schema issues.
- **Note**: Ingestion, especially embedding, can be time-consuming.

# Langchain Implementation Project

This project aims to build a web dashboard backed by a LangChain agentic system. The system will ingest various document formats (CSV, Excel, PDF, DOCX), normalize data, index semantic vectors with Ollama embeddings into FAISS, dynamically route queries between SLMs and LLMs, and execute safe tools to return hybrid answers.

## Current Status:
- Starting from a fresh codebase based on the architecture described in `cursor.md`.
- All necessary libraries and Ollama are assumed to be installed in the `mendyenv` virtual environment.
- All core components implemented with advanced features including user profiles, comprehensive logging, and intelligent search.

## ✅ Completed Components:

### Core System:
1. **utils.py** - Logging and utility functions
2. **config.py** - Configuration management
3. **ingestion_pipeline.py** - File ingestion with clean SLM summaries (no verbose prefixes)
4. **embedding_store.py** - FAISS vector store and Ollama embeddings
5. **router.py** - Query routing logic
6. **agent_tools.py** - Safe tool execution (pandas operations)
7. **llm_reasoning.py** - LLM reasoning and explanation
8. **verifier.py** - Output verification
9. **test_ingestion_cli.py** - CLI for testing ingestion
10. **test_query_cli.py** - CLI for testing queries

### Advanced Features:
11. **streamlit_app.py** - Advanced Streamlit web dashboard with user profiles
12. **advanced_logging.py** - Comprehensive logging system with database storage
13. **user_profiles.py** - User profile management and personalization
14. **advanced_search.py** - Advanced search with autocomplete and filters
15. **STREAMLIT_README.md** - Streamlit documentation

## 🔧 Key Features Implemented:

### SLM Integration:
- ✅ Clean row-level summarization using llama3.2:3b model
- ✅ Removed verbose prefixes like "Based on the given row..."
- ✅ Intelligent content extraction and normalization

### User Experience:
- ✅ User profiles with preferences and analytics
- ✅ Session memory and conversation history
- ✅ Advanced search with autocomplete suggestions
- ✅ Custom filters and query optimization
- ✅ Saved queries and personalization

### System Monitoring:
- ✅ Comprehensive logging with database storage
- ✅ Performance metrics and error tracking
- ✅ User activity analytics
- ✅ Real-time system monitoring dashboard

### Agent Intelligence:
- ✅ Intelligent pandas operations with energy consumption prioritization
- ✅ Dynamic query routing between SLM and LLM
- ✅ Safe tool execution with verification
- ✅ Hybrid answers with provenance

## 🚀 How to Run:

### CLI Testing:
```bash
# Activate virtual environment
.\mendyenv\Scripts\Activate.ps1

# Test ingestion
python test_ingestion_cli.py --ingest "data_prototype @Energy Consumption Daily Report MHS Ele - Copy.xlsx" --row_limit 20

# Test queries
python test_query_cli.py --query "What is the total energy consumption?" --file_id "your-file-id"
```

### Streamlit Dashboard:
```bash
# Install additional dependencies
pip install -r requirements_streamlit.txt

# Run dashboard
streamlit run streamlit_app.py
```

## 📊 Dashboard Features:

### 🔍 Advanced Search:
- Real-time autocomplete suggestions
- Intelligent query optimization
- Advanced filtering (date range, file type, content type)
- Query pattern learning and suggestions

### 👤 User Profiles:
- Personal login and profile management
- User preferences and settings
- Search history and analytics
- Saved queries and favorites

### 📈 Analytics:
- User activity tracking
- Query performance metrics
- System monitoring and error tracking
- Interactive charts and visualizations

### 💾 Data Management:
- File upload and processing
- Document indexing and search
- Source highlighting and provenance
- Export and sharing capabilities

## 🔮 Future Enhancements:
1. **Multi-modal Support**: Images, audio, video processing
2. **Advanced Memory**: Long-term persistence and learning
3. **Enterprise Features**: Authentication, roles, API integration
4. **Performance**: Caching, parallel processing, optimization
5. **Mobile**: Responsive design and mobile app