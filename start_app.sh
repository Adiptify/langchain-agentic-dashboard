#!/bin/bash
echo "Starting LangChain Agentic Dashboard..."
cd "C:\Users\kradi\Mendygo AI\Langchain_implemenation"
source mendyenv/Scripts/activate
streamlit run streamlit_app.py --server.port 8501
