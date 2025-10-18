# rag_ui.py
"""
Streamlit UI for RAG Agent Backend
- Upload files, run queries, view files, view logs
- Expects backend at http://localhost:8000
"""
import streamlit as st
import requests
import time
from typing import Any

BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="RAG Agent Dashboard", layout="wide")

st.title("RAG Agent Dashboard (Streamlit)")

# Sidebar: upload
st.sidebar.header("Upload Data")
uploaded = st.sidebar.file_uploader("Upload CSV/Excel/PDF/DOCX", accept_multiple_files=True)
if st.sidebar.button("Upload to backend") and uploaded:
    for f in uploaded:
        files = {"file": (f.name, f.getvalue())}
        with st.spinner(f"Uploading {f.name}..."):
            r = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=1800)
            if r.status_code == 200:
                st.sidebar.success(f"Uploaded {f.name}")
                # Force refresh of indexed files and logs after successful upload
                st.rerun()
            else:
                st.sidebar.error(f"Upload failed {f.name}: {r.text}")

# Main: Tabs
tabs = st.tabs(["Query", "Indexed Files", "Logs", "Live Status"])

with tabs[0]:
    st.header("Ask a question")
    query = st.text_area("Enter your query", height=120)
    prefer_depth = st.checkbox("Prefer deep reasoning (LLM)", value=False)
    if st.button("Run Query"):
        if not query.strip():
            st.warning("Enter a query first")
        else:
            with st.spinner("Running query..."):
                payload = {"query": query, "prefer_depth": prefer_depth}
                r = requests.post(f"{BACKEND_URL}/query", json=payload, timeout=120)
                if r.status_code != 200:
                    st.error(f"Query failed: {r.status_code} {r.text}")
                else:
                    res = r.json()
                    st.subheader("Natural Language Answer")
                    st.write(res.get("answer_text"))
                    st.subheader("Structured Output")
                    st.json(res.get("structured_output"))
                    st.subheader("Sources (top)")
                    for s in res.get("sources", []):
                        st.write(f"- file: {s.get('file_name')} doc_id: {s.get('doc_id')} excerpt: {s.get('excerpt')[:300]}")
                    st.subheader("Actions & Verifier")
                    st.json({"actions": res.get("actions_taken"), "verifier": res.get("verifier_status"), "confidence": res.get("confidence")})

with tabs[1]:
    st.header("Indexed Files")
    # Removed while True loop, rely on st.rerun() or manual refresh
    r = requests.get(f"{BACKEND_URL}/files", timeout=30)
    if r.status_code == 200:
        files = r.json().get("files", [])
        st.write(f"Indexed files: {len(files)}")
        for f in files:
            st.write(f"- {f['file_name']} (file_id: {f['file_id']}) created_at: {time.ctime(f['created_at'])}")
    else:
        st.error("Failed to fetch files")

with tabs[2]:
    st.header("Logs")
    # Removed while True loop, rely on st.rerun() or manual refresh
    r = requests.get(f"{BACKEND_URL}/logs?limit=200", timeout=30)
    if r.status_code == 200:
        logs = r.json().get("logs", [])
        st.subheader("Backend Logs")
        if logs:
            for lg in logs:
                st.write(f"[{time.ctime(lg['ts'])}] **{lg['level']}** — {lg['message']}")
                if lg.get("context"):
                    st.json(lg["context"])
        else:
            st.info("No backend logs yet.")
    else:
        st.error(f"Failed to fetch logs: {r.status_code} {r.text}")

with tabs[3]:
    st.header("Live Ingestion Status")
    # Removed while True loop, rely on st.rerun() or manual refresh
    r = requests.get(f"{BACKEND_URL}/logs?limit=5", timeout=30) # Get last 5 logs
    if r.status_code == 200:
        logs = r.json().get("logs", [])
        if logs:
            latest_log = logs[0]
            st.write(f"**Latest Activity:** [{time.ctime(latest_log['ts'])}] **{latest_log['level']}** — {latest_log['message']}")
            if latest_log.get("context"):
                st.json(latest_log["context"])
        else:
            st.info("Awaiting backend activity...")
    else:
        st.error(f"Failed to fetch status: {r.status_code} {r.text}")
