# Contributing to LangChain Agentic Dashboard

Thank you for your interest in contributing to the LangChain Agentic Dashboard! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### Reporting Issues

Before creating an issue, please:

1. **Search existing issues** to avoid duplicates
2. **Check the documentation** to ensure it's not a configuration issue
3. **Provide detailed information** including:
   - OS and Python version
   - Ollama model versions
   - Error messages and logs
   - Steps to reproduce

### Suggesting Features

We welcome feature suggestions! Please:

1. **Check the roadmap** in README.md
2. **Describe the use case** and benefits
3. **Provide examples** of how it would work
4. **Consider implementation complexity**

## 🛠️ Development Setup

### Prerequisites

- Python 3.8+
- Git
- Ollama installed and running
- Required Ollama models (see README.md)

### Setup Steps

1. **Fork and clone the repository**
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

4. **Test the setup**
   ```bash
   python test_ingestion_cli.py --help
   streamlit run streamlit_app.py
   ```

## 📝 Code Guidelines

### Python Style

- Follow **PEP 8** style guidelines
- Use **type hints** for function parameters and return values
- Write **docstrings** for all functions and classes
- Keep **line length** under 88 characters (Black formatter)

### Code Structure

```python
def example_function(param1: str, param2: int = 10) -> Dict[str, Any]:
    """
    Brief description of what the function does.
    
    Args:
        param1: Description of parameter
        param2: Description with default value
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When invalid input is provided
    """
    # Implementation here
    pass
```

### File Organization

- **One class per file** for major components
- **Group related functions** in modules
- **Use descriptive names** for files and functions
- **Keep imports organized** (standard library, third-party, local)

## 🧪 Testing

### Writing Tests

Create tests in the `tests/` directory:

```python
# tests/test_ingestion.py
import pytest
from ingestion_pipeline import ingest_file

def test_ingest_csv_file():
    """Test CSV file ingestion."""
    documents = ingest_file("test_data/sample.csv", row_limit=5)
    assert len(documents) > 0
    assert all(doc.doc_type == "row" for doc in documents)
```

### Running Tests

```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest tests/test_ingestion.py

# Run with coverage
python -m pytest --cov=.
```

### Test Data

- Use **small sample files** for testing
- **Mock external dependencies** (Ollama API calls)
- **Test edge cases** and error conditions
- **Clean up** test data after tests

## 📚 Documentation

### Code Documentation

- **Docstrings**: Use Google or NumPy style
- **Comments**: Explain complex logic, not obvious code
- **Type hints**: Help with IDE support and documentation

### User Documentation

- **README.md**: Keep updated with new features
- **Code examples**: Provide working examples
- **Screenshots**: For UI changes
- **Troubleshooting**: Common issues and solutions

## 🔄 Pull Request Process

### Before Submitting

1. **Test your changes** thoroughly
2. **Update documentation** if needed
3. **Check for linting errors**
4. **Ensure all tests pass**

### PR Description

Include:

- **Summary** of changes
- **Motivation** for the change
- **Testing** performed
- **Screenshots** for UI changes
- **Breaking changes** (if any)

### Review Process

- **Automated checks** must pass
- **Code review** by maintainers
- **Testing** on different environments
- **Documentation** review

## 🏗️ Architecture Guidelines

### Component Design

- **Single Responsibility**: Each component has one clear purpose
- **Loose Coupling**: Minimize dependencies between components
- **High Cohesion**: Related functionality grouped together
- **Error Handling**: Graceful failure and user feedback

### Data Flow

```
File Upload → Ingestion → SLM Processing → Embedding → Storage
     ↓
Query Input → Router → Agent/LLM → Verification → Response
```

### Adding New Features

1. **Design the interface** first
2. **Implement core logic** with tests
3. **Add UI components** if needed
4. **Update documentation**
5. **Add configuration** options

## 🐛 Debugging

### Common Issues

1. **Ollama Connection**: Check if Ollama is running
2. **Model Availability**: Ensure required models are pulled
3. **File Permissions**: Check upload directory permissions
4. **Memory Issues**: Monitor RAM usage during processing

### Debug Tools

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Check Ollama status
ollama list

# Test model availability
python -c "import requests; print(requests.get('http://localhost:11434/api/tags').json())"
```

## 📋 Checklist

Before submitting a PR:

- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] Documentation updated
- [ ] No sensitive data in commits
- [ ] Commit messages are descriptive
- [ ] PR description is complete

## 🎯 Areas for Contribution

### High Priority

- **Performance optimization** for large files
- **Error handling** improvements
- **Test coverage** expansion
- **Documentation** improvements

### Medium Priority

- **UI/UX enhancements**
- **Additional file formats**
- **Export functionality**
- **API endpoints**

### Low Priority

- **Mobile responsiveness**
- **Theming options**
- **Advanced analytics**
- **Plugin system**

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and ideas
- **Email**: kumar.aditya.prof@gmail.com
- **Discord**: [Join our community](https://discord.gg/uDwg5v7CXt)

## 🙏 Recognition

Contributors will be:

- **Listed** in the README.md
- **Mentioned** in release notes
- **Invited** to maintainer discussions
- **Recognized** in project documentation

Thank you for contributing to the LangChain Agentic Dashboard! 🚀
