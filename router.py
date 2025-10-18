import json
from typing import Dict, Any, List

from config import SLM_PARSE_MODEL, LLM_REASON_MODEL, LLM_EXPLAIN_MODEL, OLLAMA_BASE_URL
from utils import logger

class QueryRouter:
    """
    Dynamically routes incoming queries to the appropriate LLM (SLM, LLM_Reason, LLM_Explain)
    or tools based on the query's intent and keywords.
    """
    def __init__(self):
        logger.info("QueryRouter initialized.")

    def route_query(self, query: str) -> Dict[str, Any]:
        # Lowercase the query for case-insensitive matching
        lower_query = query.lower()

        # --- Routing Logic --- 
        # This is a simplified keyword-based routing. More sophisticated routing
        # would involve an LLM to determine intent or a more complex rule engine.

        # 1. Numerical / Calculation / Data Manipulation Queries (often routed to agent_tools)
        if any(keyword in lower_query for keyword in ["sum", "total", "average", "mean", "max", "min", "count", "calculate", "how many", "what is the value"]):
            logger.info(f"Routing query \'{query}\' to Agent (for tools like pandas/calculator).")
            return {"route": "agent", "model": None, "tool": "pandas_calculator"} # Placeholder for tool invocation

        # 2. Structural Parsing Queries (often routed to SLM for data extraction/normalization)
        if any(keyword in lower_query for keyword in ["extract", "structured data", "normalize", "json format", "list out", "show details of"]):
            logger.info(f"Routing query \'{query}\' to SLM for structural parsing.")
            return {"route": "slm_parse", "model": SLM_PARSE_MODEL, "tool": None}

        # 3. Reasoning / Complex Analysis Queries (routed to larger LLM_Reason)
        if any(keyword in lower_query for keyword in ["explain why", "analyze", "reasoning", "implications", "compare", "contrast", "trends"]):
            logger.info(f"Routing query \'{query}\' to LLM for reasoning/analysis.")
            return {"route": "llm_reason", "model": LLM_REASON_MODEL, "tool": None}

        # 4. Explanation / General Information Queries (routed to larger LLM_Explain)
        if any(keyword in lower_query for keyword in ["what is", "describe", "tell me about", "explain", "summarize"]):
            logger.info(f"Routing query \'{query}\' to LLM for explanation.")
            return {"route": "llm_explain", "model": LLM_EXPLAIN_MODEL, "tool": None}

        # Default route if no specific intent is detected
        logger.info(f"Default routing query \'{query}\' to LLM for explanation (no specific intent detected).")
        return {"route": "llm_explain", "model": LLM_EXPLAIN_MODEL, "tool": None}

# Example Usage (for testing/demonstration)
if __name__ == "__main__":
    router = QueryRouter()

    queries = [
        "What is the total sales for product A?",
        "Extract the structured data for the latest energy consumption report.",
        "Explain why energy consumption increased last month.",
        "Describe the daily temperature fluctuations.",
        "Show me the details of row 50.",
        "Analyze the relationship between humidity and temperature.",
        "Summarize the key findings in the document.",
        "Just a simple query."
    ]

    for q in queries:
        route = router.route_query(q)
        print(f"Query: \"{q}\" -> Route: {route['route']}, Model: {route['model']}, Tool: {route['tool']}")
