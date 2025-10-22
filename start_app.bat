@echo off
echo Starting LangChain Agentic Dashboard...
cd /d "C:\Users\kradi\Mendygo AI\Langchain_implemenation"
call mendyenv\Scripts\activate.bat
streamlit run streamlit_app.py --server.port 8501
pause
