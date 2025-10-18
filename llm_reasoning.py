import json
from typing import Dict, Any, List

from config import LLM_REASON_MODEL, LLM_EXPLAIN_MODEL, OLLAMA_BASE_URL
from utils import logger
import requests

class LLMReasoning:
    """
    Handles complex reasoning and explanation tasks using larger LLMs.
    This component will take retrieved context and a query, and generate
    a comprehensive response.
    """
    def __init__(self):
        self.reasoning_model = LLM_REASON_MODEL
        self.explanation_model = LLM_EXPLAIN_MODEL
        logger.info(f"LLMReasoning initialized with reasoning model: {self.reasoning_model} and explanation model: {self.explanation_model}")

    def _call_ollama_llm(self, prompt: str, model: str, temperature: float = 0.2) -> str:
        """Helper to call Ollama LLM."""
        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": temperature, "num_gpu": 1}
                },
                timeout=180 # Longer timeout for complex LLM tasks
            )
            response.raise_for_status()
            response_data = response.json()
            return response_data.get("response", "").strip()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Ollama model {model} for reasoning/explanation: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Ollama error response status: {e.response.status_code}")
                logger.error(f"Ollama error response body: {e.response.text}")
            return f"LLM Error: Could not get response from LLM model {model}."

    def perform_reasoning(self, query: str, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generates a reasoned answer based on the query and provided context.
        """
        context_str = "\n---\n".join([f"Document ID: {d.get('doc_id', 'N/A')}\nContent: {d.get('content', 'N/A')}\nMetadata: {json.dumps(d.get('metadata', {}), indent=2)}" for d in context])

        prompt = f"""
        You are an advanced AI assistant capable of complex reasoning and analysis.
        Analyze the provided context documents to answer the following query.
        Synthesize information from multiple documents if necessary. Provide a detailed, well-reasoned answer.
        Also, clearly state the provenance (Document IDs) for the information you use.

        Query: "{query}"

        Context Documents:
        {context_str}

        Reasoned Answer:
        """
        logger.info(f"Performing reasoning for query (first 50 chars): {query[:50]}...")
        reasoned_answer = self._call_ollama_llm(prompt, self.reasoning_model)

        # Extract provenance from the LLM's response if possible, or infer from context
        provenance_docs = []
        for doc in context:
            if doc.get('doc_id') in reasoned_answer: # Simple check for now
                provenance_docs.append(doc.get('doc_id'))
        
        if not provenance_docs:
            # Fallback: if LLM didn't explicitly mention, include all context doc_ids as potential provenance
            provenance_docs = [d.get('doc_id') for d in context if d.get('doc_id')]

        return {
            "answer": reasoned_answer,
            "type": "natural_language_reasoning",
            "provenance": provenance_docs
        }

    def generate_explanation(self, query: str, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generates an explanation or summary based on the query and provided context.
        """
        context_str = "\n---\n".join([f"Document ID: {d.get('doc_id', 'N/A')}\nContent: {d.get('content', 'N/A')}\nMetadata: {json.dumps(d.get('metadata', {}), indent=2)}" for d in context])

        prompt = f"""
        You are an AI assistant designed to provide clear and concise explanations and summaries.
        Based on the following context documents, provide a comprehensive explanation or summary to answer the query.
        Ensure your explanation is easy to understand and directly addresses the user's request.
        Cite the Document IDs for the information you use.

        Query: "{query}"

        Context Documents:
        {context_str}

        Explanation/Summary:
        """
        logger.info(f"Generating explanation for query (first 50 chars): {query[:50]}...")
        explanation = self._call_ollama_llm(prompt, self.explanation_model)

        # Extract provenance
        provenance_docs = []
        for doc in context:
            if doc.get('doc_id') in explanation: # Simple check for now
                provenance_docs.append(doc.get('doc_id'))

        if not provenance_docs:
            provenance_docs = [d.get('doc_id') for d in context if d.get('doc_id')]

        return {
            "answer": explanation,
            "type": "natural_language_explanation",
            "provenance": provenance_docs
        }

# Example Usage (for testing/demonstration)
if __name__ == "__main__":
    llm_reasoning = LLMReasoning()

    # Dummy context documents for demonstration
    dummy_context = [
        {
            "doc_id": "doc1",
            "content": "The sales of product A increased by 15% in Q1 due to a successful marketing campaign.",
            "metadata": {"source": "sales_report.pdf", "page": 5}
        },
        {
            "doc_id": "doc2",
            "content": "Product B saw a 5% decrease in sales in the same quarter, possibly due to increased competition.",
            "metadata": {"source": "competitor_analysis.docx", "page": 2}
        }
    ]

    query_reason = "Why did sales of product A increase in Q1?"
    reason_result = llm_reasoning.perform_reasoning(query_reason, dummy_context)
    print(f"\nReasoning Result for \'{query_reason}\':")
    print(f"Answer: {reason_result['answer']}")
    print(f"Provenance: {reason_result['provenance']}")

    query_explain = "Summarize the sales performance of Product B in Q1."
    explain_result = llm_reasoning.generate_explanation(query_explain, dummy_context)
    print(f"\nExplanation Result for \'{query_explain}\':")
    print(f"Answer: {explain_result['answer']}")
    print(f"Provenance: {explain_result['provenance']}")
