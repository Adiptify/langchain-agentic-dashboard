# 🤖 LangChain Agentic Dashboard

A comprehensive web dashboard backed by a LangChain agentic system that ingests various document formats, normalizes data using Small Language Models (SLMs), indexes semantic vectors with Ollama embeddings into FAISS, and dynamically routes queries between SLMs and larger LLMs for intelligent data analysis.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![LangChain](https://img.shields.io/badge/langchain-latest-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🌟 Features

### 🔍 **Intelligent Data Processing**
- **Multi-format Support**: CSV, Excel, PDF, DOCX, TXT files
- **SLM Integration**: Clean row-level summarization using `llama3.2:3b`
- **Semantic Search**: FAISS vector store with Ollama embeddings (`nomic-embed-text`)
- **Dynamic Routing**: Intelligent query routing between SLM and LLM models

### 👤 **User Experience**
- **Personal Profiles**: User authentication and personalization
- **Advanced Search**: Real-time autocomplete and intelligent suggestions
- **Session Memory**: Persistent conversation history and context
- **Custom Filters**: Date range, file type, content type filtering
- **Saved Queries**: Organize and reuse successful queries

### 📊 **Analytics & Monitoring**
- **User Analytics**: Personal dashboard with search statistics
- **System Monitoring**: Real-time performance metrics and error tracking
- **Comprehensive Logging**: Database-stored activity logs and analytics
- **Interactive Charts**: Visual data representation with Plotly

### 🛡️ **Safety & Verification**
- **Safe Tool Execution**: Pandas operations with verification
- **Output Verification**: SLM-based result validation
- **Error Handling**: Comprehensive error tracking and user feedback
- **Data Provenance**: Source highlighting and document attribution

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) installed and running
- Required Ollama models:
  ```bash
  ollama pull nomic-embed-text
  ollama pull llama3.2:3b
  ollama pull gpt-oss:120b-cloud
  ollama pull deepseek-v3.1:671b-cloud
  ```

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/langchain-agentic-dashboard.git
   cd langchain-agentic-dashboard
   ```

2. **Create virtual environment**
   ```bash
   python -m venv mendyenv
   # Windows
   mendyenv\Scripts\activate
   # Linux/Mac
   source mendyenv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements_streamlit.txt
   ```

4. **Run the dashboard**
   ```bash
   streamlit run streamlit_app.py
   ```

5. **Access the application**
   Open your browser to `http://localhost:8501`

## 📖 Usage

### CLI Testing

Test individual components using the command-line interface:

```bash
# Test file ingestion
python test_ingestion_cli.py --ingest "your-file.xlsx" --row_limit 20

# Test query processing
python test_query_cli.py --query "What is the total energy consumption?" --file_id "your-file-id"
```

### Web Dashboard

1. **Login**: Create a user profile or login with existing credentials
2. **Upload Files**: Drag and drop files in the Files tab
3. **Search**: Use the advanced search with autocomplete suggestions
4. **Analyze**: View results with source highlighting and verification
5. **Monitor**: Check system performance in the Monitoring tab

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   File Upload   │───▶│  SLM Processing │───▶│  FAISS Index   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Data Ingestion │───▶│ Row Summaries   │───▶│ Vector Search   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Query Router    │───▶│ Agent Tools     │───▶│ LLM Reasoning   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Result Verifier │───▶│ User Dashboard  │───▶│ Analytics       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📁 Project Structure

```
langchain-agentic-dashboard/
├── 📄 Core Components
│   ├── ingestion_pipeline.py      # File processing and SLM integration
│   ├── embedding_store.py         # FAISS vector store management
│   ├── router.py                  # Query routing logic
│   ├── agent_tools.py             # Safe tool execution
│   ├── llm_reasoning.py           # LLM reasoning and explanation
│   └── verifier.py                # Output verification
├── 🎨 User Interface
│   ├── streamlit_app.py           # Main web dashboard
│   ├── advanced_logging.py        # Comprehensive logging system
│   ├── user_profiles.py           # User management and personalization
│   └── advanced_search.py         # Search engine with autocomplete
├── 🧪 Testing
│   ├── test_ingestion_cli.py      # CLI for testing ingestion
│   └── test_query_cli.py          # CLI for testing queries
├── ⚙️ Configuration
│   ├── config.py                  # System configuration
│   ├── utils.py                   # Utility functions
│   ├── requirements.txt           # Core dependencies
│   └── requirements_streamlit.txt # UI dependencies
└── 📚 Documentation
    ├── README.md                  # This file
    ├── README.txt                 # Detailed project status
    └── STREAMLIT_README.md        # Streamlit-specific documentation
```

## 🔧 Configuration

### Ollama Models

The system uses the following Ollama models:

- **Embeddings**: `nomic-embed-text` - Fast, efficient embeddings
- **SLM Parsing**: `llama3.2:3b` - Row-level data summarization
- **LLM Reasoning**: `gpt-oss:120b-cloud` - Complex reasoning tasks
- **Alternative LLM**: `deepseek-v3.1:671b-cloud` - Backup reasoning model

### Environment Variables

Create a `.env` file for custom configuration:

```env
OLLAMA_BASE_URL=http://localhost:11434
EMBED_MODEL=nomic-embed-text
SLM_PARSE_MODEL=llama3.2:3b
LLM_REASONING_MODEL=gpt-oss:120b-cloud
DATA_DIR=./data
INDEX_DIR=./index
```

## 📊 Key Features Explained

### SLM Integration

The system uses Small Language Models for efficient row-level data processing:

```python
# Example: Clean row summarization
Input: {'feeder': 'I/C Panel', 'swb_no': 'I/C-1', '30-06-2024': 246740}
Output: "I/C Panel feeder SWB I/C-1 shows 246740 KWH on 30-06-2024"
```

### Advanced Search

- **Autocomplete**: Real-time suggestions based on query patterns
- **Filtering**: Date range, file type, content type filters
- **Optimization**: Automatic query enhancement with domain knowledge
- **Learning**: Pattern recognition from successful queries

### User Personalization

- **Profiles**: Individual user accounts with preferences
- **Analytics**: Personal search statistics and activity tracking
- **Saved Queries**: Organize and reuse successful queries
- **Session Memory**: Persistent conversation context

## 🧪 Testing

### Unit Tests

```bash
# Test ingestion pipeline
python -m pytest tests/test_ingestion.py

# Test embedding store
python -m pytest tests/test_embedding.py

# Test query routing
python -m pytest tests/test_router.py
```

### Integration Tests

```bash
# Test complete workflow
python test_ingestion_cli.py --ingest "sample_data.xlsx" --row_limit 10
python test_query_cli.py --query "total energy consumption"
```

## 📈 Performance

### Benchmarks

- **File Processing**: ~2-5 seconds per 100 rows (Excel)
- **Embedding Generation**: ~0.1-0.5 seconds per document
- **Query Processing**: ~1-3 seconds average response time
- **Search Performance**: Sub-second vector similarity search

### Optimization Tips

1. **Row Limits**: Use `--row_limit` for testing (recommended: 20-50 rows)
2. **Sheet Limits**: Excel processing limited to first 2 sheets
3. **Model Selection**: `nomic-embed-text` for faster embeddings
4. **Caching**: Query suggestions cached for 1 hour

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Development Guidelines

- Follow PEP 8 style guidelines
- Add tests for new features
- Update documentation as needed
- Ensure all tests pass before submitting

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for the agentic framework
- [Ollama](https://ollama.ai/) for local LLM hosting
- [Streamlit](https://streamlit.io/) for the web interface
- [FAISS](https://github.com/facebookresearch/faiss) for vector similarity search

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/langchain-agentic-dashboard/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/langchain-agentic-dashboard/discussions)
- **Email**: your.email@example.com

## 🔮 Roadmap

- [ ] **Multi-modal Support**: Images, audio, video processing
- [ ] **Advanced Memory**: Long-term persistence and learning
- [ ] **Enterprise Features**: Authentication, roles, API integration
- [ ] **Performance**: Caching, parallel processing, optimization
- [ ] **Mobile**: Responsive design and mobile app
- [ ] **API**: RESTful API for external integrations
- [ ] **Deployment**: Docker containers and cloud deployment guides

---

<div align="center">

**⭐ Star this repository if you found it helpful!**

Made by [Adiptify](https://github.com/Adiptify)

</div>
