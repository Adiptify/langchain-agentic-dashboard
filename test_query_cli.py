import argparse
import json
import os
from typing import Dict, Any, List

from embedding_store import EmbeddingStore
from router import QueryRouter
from agent_tools import AgentTools
from llm_reasoning import LLMReasoning
from verifier import Verifier
from utils import logger, log_process_completion

def main():
    parser = argparse.ArgumentParser(description="CLI for querying the LangChain agentic system.")
    parser.add_argument("--query", type=str, help="The query string to process.", required=True)
    parser.add_argument("--file_id", type=str, help="Optional: The file_id to narrow down the query context.", default=None)
    parser.add_argument("--k", type=int, default=5, help="Number of documents to retrieve from FAISS.")

    args = parser.parse_args()

    # Initialize components
    embedding_store = EmbeddingStore()
    router = QueryRouter()
    agent_tools = AgentTools()
    llm_reasoning = LLMReasoning()
    verifier = Verifier()

    query_text = args.query
    file_id_filter = args.file_id

    logger.info(f"Processing query: \'{query_text}\'")

    # Step 1: Route the query
    route_decision = router.route_query(query_text)
    route_type = route_decision["route"]
    target_model = route_decision["model"]
    target_tool = route_decision["tool"]

    print(f"\n--- Query Routing ---")
    print(f"Query: \"{query_text}\"")
    print(f"Route: {route_type}, Model: {target_model if target_model else 'N/A'}, Tool: {target_tool if target_tool else 'N/A'}")

    final_result = {"answer": "", "type": "", "provenance": []}
    context_docs = []

    # Step 2: Retrieve relevant context (always perform semantic search first)
    logger.info(f"Retrieving context for query: \'{query_text}\'")
    search_results = embedding_store.search(query_text, k=args.k, file_id_filter=file_id_filter)
    if search_results:
        print(f"\n--- Retrieved Context ({len(search_results)} documents) ---")
        for i, res in enumerate(search_results):
            print(f"  Document {i+1} (ID: {res.get('doc_id')}, Type: {res.get('doc_type')}, File: {res.get('file_name')}):")
            print(f"    Content: {res.get('content')[:150]}...")
            context_docs.append(res) # Store full results for later processing
    else:
        print("\n--- No Context Retrieved ---")
        log_process_completion("Query Processing", status="failed", details=f"No context found for query: {query_text}")
        print("No relevant context documents found. Cannot proceed with answering the query.")
        return

    # Step 3: Execute based on route decision
    print(f"\n--- Executing Route: {route_type} ---")

    if route_type == "agent" and target_tool == "pandas_calculator":
        logger.info(f"Executing AgentTools.execute_pandas_query for query: {query_text}")
        # For pandas_calculator, we need to decide which file_id to operate on.
        # For simplicity, if file_id is provided, use it. Otherwise, pick the first one from context.
        if file_id_filter:
            target_file_id = file_id_filter
        elif context_docs:
            target_file_id = context_docs[0]['file_id'] # Assuming relevant data is in the first doc's file
            print(f"Using file_id {target_file_id} from retrieved context for tool execution.")
        else:
            print("No file_id specified and no context documents to infer from for tool execution.")
            log_process_completion("Query Processing", status="failed", details="No file_id for agent tool execution.")
            return

        tool_output = agent_tools.execute_pandas_query(target_file_id, query_text)
        final_result = {
            "answer": tool_output.get("result", "Tool execution failed."),
            "type": tool_output.get("type", "tool_output"),
            "provenance": tool_output.get("provenance", [])
        }
        print(f"Agent Tool Output: {json.dumps(final_result, indent=2)}")
        # Optional: Verification of tool output
        # verification_status = verifier.verify_tool_execution(query_text, tool_output)
        # print(f"Tool Output Verification: {verification_status}")

    elif route_type == "slm_parse":
        logger.info(f"Executing SLM for structural parsing for query: {query_text}")
        # For structural parsing, use the content of the top retrieved document(s)
        # as the 'original_context' for the SLM.
        # The SLM will parse the 'query_text' given the 'context_docs'.

        # For now, let's assume the SLM directly processes the query and returns structured JSON
        # A more advanced implementation would feed context_docs into the SLM call.
        slm_parsed_data = llm_reasoning._call_ollama_llm(query_text, target_model, temperature=0.1) # Re-using LLMReasoning's helper
        
        final_result = {
            "answer": slm_parsed_data,
            "type": "structured_json",
            "provenance": [d.get('doc_id') for d in context_docs]
        }
        print(f"SLM Parsed Data: {json.dumps(final_result, indent=2)}")
        # Optional: Verification of structural parse
        # verification_status = verifier.verify_structural_parse(query_text, context_docs, slm_parsed_data)
        # print(f"Structural Parse Verification: {verification_status}")

    elif route_type == "llm_reason":
        logger.info(f"Executing LLMReasoning.perform_reasoning for query: {query_text}")
        reasoning_output = llm_reasoning.perform_reasoning(query_text, context_docs)
        final_result = reasoning_output
        print(f"LLM Reasoning Output: {json.dumps(final_result, indent=2)}")
        # Optional: Verification of reasoning
        # verification_status = verifier.verify_reasoning_explanation(query_text, context_docs, reasoning_output.get("answer", ""))
        # print(f"Reasoning Verification: {verification_status}")

    elif route_type == "llm_explain":
        logger.info(f"Executing LLMReasoning.generate_explanation for query: {query_text}")
        explanation_output = llm_reasoning.generate_explanation(query_text, context_docs)
        final_result = explanation_output
        print(f"LLM Explanation Output: {json.dumps(final_result, indent=2)}")
        # Optional: Verification of explanation
        # verification_status = verifier.verify_reasoning_explanation(query_text, context_docs, explanation_output.get("answer", ""))
        # print(f"Explanation Verification: {verification_status}")

    print(f"\n--- Final Answer ---")
    print(f"Answer Type: {final_result['type']}")
    print(f"Answer: {final_result['answer']}")
    print(f"Provenance: {final_result['provenance']}")

    log_process_completion("CLI Query", details=f"Query \'{query_text}\' processed via route {route_type}. Answer type: {final_result['type']}")

if __name__ == "__main__":
    main()
