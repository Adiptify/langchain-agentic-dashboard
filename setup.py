#!/usr/bin/env python3
"""
Setup script for LangChain Agentic Dashboard
This script helps users set up the project quickly and easily.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required. Current version:", f"{version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def check_ollama():
    """Check if Ollama is installed and running."""
    print("🦙 Checking Ollama installation...")
    
    # Check if ollama command exists
    try:
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
        print(f"✅ Ollama is installed: {result.stdout.strip()}")
    except FileNotFoundError:
        print("❌ Ollama is not installed. Please install from https://ollama.ai/")
        return False
    
    # Check if Ollama is running
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama is running")
            return True
        else:
            print("❌ Ollama is not responding. Please start Ollama service")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        print("💡 Make sure Ollama is running: ollama serve")
        return False

def check_required_models():
    """Check if required Ollama models are available."""
    print("📦 Checking required Ollama models...")
    
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code != 200:
            print("❌ Cannot fetch model list from Ollama")
            return False
        
        models = response.json().get("models", [])
        model_names = [model["name"] for model in models]
        
        required_models = [
            "nomic-embed-text",
            "llama3.2:3b",
            "gpt-oss:120b-cloud",
            "deepseek-v3.1:671b-cloud"
        ]
        
        missing_models = []
        for model in required_models:
            if not any(model in name for name in model_names):
                missing_models.append(model)
        
        if missing_models:
            print("❌ Missing required models:")
            for model in missing_models:
                print(f"   - {model}")
            print("\n💡 Install missing models with:")
            for model in missing_models:
                print(f"   ollama pull {model}")
            return False
        
        print("✅ All required models are available")
        return True
        
    except Exception as e:
        print(f"❌ Error checking models: {e}")
        return False

def create_virtual_environment():
    """Create virtual environment."""
    venv_name = "mendyenv"
    
    if os.path.exists(venv_name):
        print(f"✅ Virtual environment '{venv_name}' already exists")
        return True
    
    return run_command(f"python -m venv {venv_name}", f"Creating virtual environment '{venv_name}'")

def install_dependencies():
    """Install Python dependencies."""
    if platform.system() == "Windows":
        pip_cmd = "mendyenv\\Scripts\\pip"
    else:
        pip_cmd = "mendyenv/bin/pip"
    
    # Install core dependencies
    if not run_command(f"{pip_cmd} install -r requirements.txt", "Installing core dependencies"):
        return False
    
    # Install Streamlit dependencies
    if not run_command(f"{pip_cmd} install -r requirements_streamlit.txt", "Installing Streamlit dependencies"):
        return False
    
    return True

def create_directories():
    """Create necessary directories."""
    directories = ["uploads", "data", "index"]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    return True

def test_installation():
    """Test the installation."""
    print("🧪 Testing installation...")
    
    if platform.system() == "Windows":
        python_cmd = "mendyenv\\Scripts\\python"
    else:
        python_cmd = "mendyenv/bin/python"
    
    # Test imports
    test_script = """
import sys
try:
    from ingestion_pipeline import ingest_file
    from embedding_store import EmbeddingStore
    from router import QueryRouter
    from agent_tools import AgentTools
    from llm_reasoning import LLMReasoning
    from verifier import Verifier
    print("✅ All core modules imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
"""
    
    return run_command(f'{python_cmd} -c "{test_script}"', "Testing module imports")

def main():
    """Main setup function."""
    print("🚀 LangChain Agentic Dashboard Setup")
    print("=" * 50)
    
    # Check prerequisites
    if not check_python_version():
        sys.exit(1)
    
    if not check_ollama():
        print("\n💡 Please install and start Ollama, then run this script again")
        sys.exit(1)
    
    if not check_required_models():
        print("\n💡 Please install missing models, then run this script again")
        sys.exit(1)
    
    # Setup project
    if not create_virtual_environment():
        sys.exit(1)
    
    if not install_dependencies():
        sys.exit(1)
    
    if not create_directories():
        sys.exit(1)
    
    if not test_installation():
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Activate virtual environment:")
    if platform.system() == "Windows":
        print("   mendyenv\\Scripts\\activate")
    else:
        print("   source mendyenv/bin/activate")
    
    print("2. Run the Streamlit dashboard:")
    print("   streamlit run streamlit_app.py")
    
    print("3. Or test with CLI:")
    print("   python test_ingestion_cli.py --help")
    
    print("\n📚 For more information, see README.md")

if __name__ == "__main__":
    main()
