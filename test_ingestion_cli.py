import argparse
import os
import json
from ingestion_pipeline import ingest_file
from embedding_store import EmbeddingStore
from utils import logger, log_process_completion

def main():
    parser = argparse.ArgumentParser(description="CLI for ingestion and querying functionalities.")
    parser.add_argument("--ingest", type=str, help="Path to the file to ingest.")
    parser.add_argument("--query", type=str, help="A query string to search for.")
    parser.add_argument("--k", type=int, default=5, help="Number of documents to retrieve for query.")
    parser.add_argument("--row_limit", type=int, default=None, help="Optional: Limit the number of rows to ingest for tabular data.")
    args = parser.parse_args()

    embedding_store = EmbeddingStore()

    if args.ingest:
        file_path = args.ingest
        if not os.path.exists(file_path):
            print(f"Error: File not found at {file_path}")
            logger.error(f"Ingestion failed: File not found at {file_path}")
            return
        
        logger.info(f"Attempting to ingest file: {file_path}")
        documents = ingest_file(file_path, row_limit=args.row_limit) # Pass row_limit here
        if documents:
            # The file_id is common to all documents from the same file
            first_doc_file_id = documents[0].file_id if documents else "N/A"
            print(f"File ingested successfully with File ID: {first_doc_file_id}")

            embedding_store.add_documents(documents)
            logger.info(f"Ingested {len(documents)} documents.")
            log_process_completion("CLI Ingestion", details=f"Successfully ingested and embedded {len(documents)} documents from {file_path}. Embedding can be time-consuming.")
            print(f"Successfully ingested {len(documents)} documents from {file_path}. Embedding can be time-consuming, please wait...")
            
            print(f"\nStarting embedding for {len(documents)} documents...")
            # Instead of iterating through docs here, add a batching mechanism if needed
            # For now, the add_documents already handles embedding internally
            # We can add a simple counter or spinner here if add_documents were batched externally.
            
            # After successful ingestion, ask if user wants to query
            query_now = input("\nIngestion and embedding complete. Do you want to run a query now? (yes/no): ").lower()
            if query_now == 'yes':
                query_text = input("Enter your query: ")
                search_results = embedding_store.search(query_text, k=args.k)
                if search_results:
                    print(f"\nSearch Results for '{query_text}':")
                    for i, result in enumerate(search_results):
                        print(f"  Result {i+1}:")
                        print(f"    Doc ID: {result.get('doc_id')}")
                        print(f"    File Name: {result.get('file_name')}")
                        print(f"    Doc Type: {result.get('doc_type')}")
                        print(f"    Content: {result.get('content')[:200]}...")
                        print(f"    Metadata: {json.dumps(result.get('metadata'), indent=2)}")
                        print(f"    Distance: {result.get('distance'):.4f}")
                else:
                    print(f"No results found for '{query_text}'.")
            else:
                print("Exiting without running a query. You can run queries later using --query option.")

        else:
            logger.warning(f"No documents were generated or added from {file_path}.")
            log_process_completion("CLI Ingestion", status="failed", details=f"No documents generated from {file_path}")
            print(f"No documents were generated or added from {file_path}.")

    if args.query:
        query_text = args.query
        logger.info(f"Searching for query: {query_text}")
        results = embedding_store.search(query_text, k=args.k)
        
        if results:
            log_process_completion("CLI Search", details=f"Found {len(results)} results for query: {query_text}")
            print(f"Search Results for '{query_text}':")
            for i, result in enumerate(results):
                print(f"  Result {i+1}:")
                print(f"    Doc ID: {result.get('doc_id')}")
                print(f"    File Name: {result.get('file_name')}")
                print(f"    Doc Type: {result.get('doc_type')}")
                print(f"    Content: {result.get('content')[:200]}...")
                print(f"    Metadata: {json.dumps(result.get('metadata'), indent=2)}")
                print(f"    Distance: {result.get('distance'):.4f}")
        else:
            log_process_completion("CLI Search", status="no results", details=f"No results found for query: {query_text}")
            print(f"No results found for '{query_text}'.")

    if not (args.ingest or args.query):
        print("Please provide either --ingest FILE_PATH or --query QUERY_TEXT")
        parser.print_help()

if __name__ == "__main__":
    main()
