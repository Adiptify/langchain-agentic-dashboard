import json
from typing import Dict, Any, List

from config import SLM_PARSE_MODEL, LLM_REASON_MODEL, OLLAMA_BASE_URL
from utils import logger
import requests

class Verifier:
    """
    A verification layer to check the accuracy, completeness, and safety of LLM outputs
    and tool execution results. It uses an SLM or a smaller LLM for rapid verification.
    """
    def __init__(self):
        self.verification_model = SLM_PARSE_MODEL # Using SLM for quicker verification
        logger.info(f"Verifier initialized with verification model: {self.verification_model}")

    def _call_ollama_llm(self, prompt: str, model: str, temperature: float = 0.0) -> str:
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
                timeout=60 # Shorter timeout for verification
            )
            response.raise_for_status()
            response_data = response.json()
            return response_data.get("response", "").strip()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Ollama model {model} for verification: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Ollama error response status: {e.response.status_code}")
                logger.error(f"Ollama error response body: {e.response.text}")
            return f"Verification Error: Could not get response from LLM model {model}."

    def verify_structural_parse(self, query: str, original_context: List[Dict[str, Any]], parsed_data: str) -> Dict[str, Any]:
        """
        Verifies if the structured data extracted by an SLM is accurate and complete
        relative to the original context and query.
        """
        verification_prompt = f"""
        You are an AI verifier. Your task is to assess the accuracy and completeness
        of extracted structured data against the original query and context.

        Original Query: "{query}"
        Original Context: {json.dumps(original_context, indent=2)}
        Extracted Structured Data: {parsed_data}

        Is the 'Extracted Structured Data' accurate and complete based on the 'Original Context' and 'Original Query'?
        Respond with "YES" if accurate and complete, or "NO" followed by a brief explanation of discrepancies.
        """
        logger.info(f"Verifying structural parse for query (first 50 chars): {query[:50]}...")
        verification_result = self._call_ollama_llm(verification_prompt, self.verification_model, temperature=0.1)

        if verification_result.strip().upper().startswith("YES"):
            return {"status": "verified", "message": "Structured data verified as accurate and complete."}
        else:
            return {"status": "unverified", "message": f"Verification failed: {verification_result}"}

    def verify_reasoning_explanation(self, query: str, original_context: List[Dict[str, Any]], llm_response: str) -> Dict[str, Any]:
        """
        Verifies if the reasoning or explanation provided by an LLM is consistent
        with the retrieved context and addresses the query appropriately.
        """
        verification_prompt = f"""
        You are an AI verifier. Your task is to assess the quality of an LLM's reasoning or explanation.
        Check if the LLM's response is consistent with the provided context and directly answers the query.

        Original Query: "{query}"
        Original Context: {json.dumps(original_context, indent=2)}
        LLM Response: {llm_response}

        Is the 'LLM Response' consistent with the 'Original Context' and does it appropriately answer the 'Original Query'?
        Respond with "YES" if consistent and appropriate, or "NO" followed by a brief explanation of discrepancies.
        """
        logger.info(f"Verifying reasoning/explanation for query (first 50 chars): {query[:50]}...")
        verification_result = self._call_ollama_llm(verification_prompt, self.verification_model, temperature=0.1)

        if verification_result.strip().upper().startswith("YES"):
            return {"status": "verified", "message": "Reasoning/explanation verified as consistent and appropriate."}
        else:
            return {"status": "unverified", "message": f"Verification failed: {verification_result}"}

    # def verify_tool_execution(self, query: str, tool_output: Dict[str, Any]) -> Dict[str, Any]:
    #     """
    #     Verifies if the tool execution output is logical and directly addresses the query.
    #     This might involve checking output format, expected values, etc.
    #     (Placeholder - actual implementation would depend on specific tool outputs)
    #     """
    #     # A more complex verification for tool execution might involve running a small LLM
    #     # to evaluate if the tool output logically answers the query.
    #     logger.info(f"Verifying tool execution for query (first 50 chars): {query[:50]}...")
    #     # For now, a simple placeholder.
    #     if "error" in tool_output:
    #         return {"status": "unverified", "message": f"Tool execution failed: {tool_output['error']}"}
    #     return {"status": "verified", "message": "Tool execution output seems logical."}
