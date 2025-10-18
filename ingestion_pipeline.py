import os
import pandas as pd
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
import re
import numpy as np
import json
import requests
import time

from config import DATA_DIR, TRUSTED_FLAG_DEFAULT, TEXT_CHUNK_SIZE, TEXT_CHUNK_OVERLAP, OLLAMA_BASE_URL, SLM_PARSE_MODEL
from utils import logger, log_process_completion

class Document:
    def __init__(self, doc_id: str, file_id: str, file_name: str, doc_type: str, content: str, metadata: Dict[str, Any]):
        self.doc_id = doc_id
        self.file_id = file_id
        self.file_name = file_name
        self.doc_type = doc_type
        self.content = content
        self.metadata = metadata

    def to_dict(self):
        return {
            "doc_id": self.doc_id,
            "file_id": self.file_id,
            "file_name": self.file_name,
            "doc_type": self.doc_type,
            "content": self.content,
            "metadata": self.metadata,
        }

def _get_slm_summary(row_data: Dict[str, Any], column_types: Dict[str, str]) -> str:
    """
    Generates a concise, natural language summary of a single row using an SLM.
    Returns clean, direct summaries without prefixes or verbose explanations.
    """
    # Filter out numeric fields that are NaN for better summary generation
    filtered_row_data = {k: v for k, v in row_data.items() if pd.notna(v)}

    # Create a more direct prompt that focuses on clean output
    prompt = f"""Generate a concise summary of this data row. Return ONLY the summary text, no prefixes like "Summary:" or "Based on the given row".

Examples:
Input: {{'feeder': 'I/C Panel Numerical Relay', 'swb_no': 'I/C-1', '30-06-2024': 246740, '01-07-2024': 246885, 'difference': 145}}
Output: I/C Panel Numerical Relay feeder SWB I/C-1 shows 246740 KWH on 30-06-2024, 246885 KWH on 01-07-2024 with 145 KWH difference.

Input: {{'product': 'Laptop', 'price': 1200, 'category': 'Electronics'}}
Output: Electronics Laptop priced at $1200.

Data: {filtered_row_data}
Summary:"""

    logger.info(f"Requesting SLM summary for row (first 100 chars): {str(filtered_row_data)[:100]}...")
    start_time = time.time()
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": SLM_PARSE_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"num_gpu": 1}
            }
        )
        end_time = time.time()
        response.raise_for_status()
        response_data = response.json()
        logger.info(f"Ollama SLM response status: {response.status_code}, time: {end_time - start_time:.2f}s")
        
        if "response" in response_data and response_data["response"]:
            summary = response_data["response"].strip()
            
            # Clean up common prefixes and verbose additions
            prefixes_to_remove = [
                "summary:",
                "based on the given row of data, here is a concise and natural language summary:",
                "based on the given row,",
                "here is a concise summary:",
                "the summary is:",
                "summary of the data:",
                "data summary:",
                "row summary:",
                "concise summary:",
                "natural language summary:"
            ]
            
            summary_lower = summary.lower()
            for prefix in prefixes_to_remove:
                if summary_lower.startswith(prefix):
                    summary = summary[len(prefix):].strip()
                    break
            
            # Remove any remaining verbose additions
            if "based on" in summary_lower or "given row" in summary_lower:
                # Extract the actual summary part after common verbose phrases
                lines = summary.split('\n')
                for line in lines:
                    line = line.strip()
                    if line and not any(phrase in line.lower() for phrase in ["based on", "given row", "here is", "summary:", "the data"]):
                        summary = line
                        break
            
            logger.info(f"Cleaned SLM summary: {summary[:100]}...")
            return summary
        else:
            logger.warning(f"Ollama SLM returned no 'response' key or it was empty for row: {str(filtered_row_data)[:100]}...")
            logger.warning(f"Full Ollama SLM response: {json.dumps(response_data)}")
            return str(filtered_row_data) # Fallback to string representation

    except requests.exceptions.RequestException as e:
        end_time = time.time()
        logger.error(f"Error getting SLM summary from Ollama (took {end_time - start_time:.2f}s): {e}", exc_info=True)
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Ollama SLM error response status: {e.response.status_code}")
            logger.error(f"Ollama SLM error response body: {e.response.text}")
        return str(filtered_row_data) # Fallback to string representation

def _normalize_header(header: str) -> str:
    """
    Normalizes a header string to snake_case.
    """
    header = header.strip().lower()
    header = re.sub(r'[^a-z0-9\s_]', '', header) # Remove special characters except underscore
    header = re.sub(r'\s+', '_', header) # Replace spaces with underscores
    return header

def _detect_column_type(series: pd.Series):
    """
    Detects the type of a pandas Series.
    """
    # Try numeric
    try:
        pd.to_numeric(series)
        return "number"
    except ValueError:
        pass
    
    # Try datetime
    try:
        common_date_formats = [
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d %H:%M',
            '%Y-%m-%d',
            '%d-%m-%Y',
            '%m/%d/%Y',
            '%d/%m/%Y',
            '%Y%m%d_%H%M%S', # For formats like '20230707_000000'
            '%Y%m%d',
        ]

        best_converted_series = None
        max_valid_dates = -1

        for fmt in common_date_formats:
            converted_series = pd.to_datetime(series, format=fmt, errors='coerce')
            valid_dates_count = converted_series.count()
            if valid_dates_count > max_valid_dates:
                max_valid_dates = valid_dates_count
                best_converted_series = converted_series
        
        # If a specific format worked for more than 50% of dates, or general coerce works
        if max_valid_dates / len(series) > 0.5:
            return "datetime"
        
        # Fallback to general coercion if no specific format yielded good results but general one might
        converted_series_general = pd.to_datetime(series, errors='coerce')
        if converted_series_general.count() / len(series) > 0.5:
            return "datetime"
        
    except Exception as e:
        logger.debug(f"Error during date detection for column: {e}")
        pass

    # Default to string
    return "string"

def _process_dataframe(df: pd.DataFrame, file_id: str, file_name: str, sheet_name: str = None) -> List[Document]:
    """
    Processes a pandas DataFrame into a list of Documents (row-level and column summaries).
    """
    documents = []
    normalized_columns = {}
    column_types = {}

    # Normalize headers and detect types
    for col in df.columns:
        normalized_col = _normalize_header(str(col))
        normalized_columns[col] = normalized_col
        column_types[normalized_col] = _detect_column_type(df[col])
    
    df.rename(columns=normalized_columns, inplace=True)

    # Create row-level documents
    for idx, row in df.iterrows():
        row_id = str(uuid.uuid4())
        numeric_fields = {}

        # Dynamically build a descriptive summary for the row
        description_col = str(row.get('description', '')) # Assuming 'description' might exist
        swb_no_col = str(row.get('swb_no', ''))

        row_summary_parts = []
        if description_col: # Start with description if available
            row_summary_parts.append(f"Feeder {description_col.strip()}")
        if swb_no_col:
            row_summary_parts.append(f"location SWB no. {swb_no_col.strip()}")

        # Populate numeric_fields for agent tools
        for col_name, col_type in column_types.items():
            if col_type == "number":
                val = row.get(col_name)
                if pd.notna(val):
                    try:
                        numeric_fields[col_name] = float(val)
                    except (ValueError, TypeError):
                        numeric_fields[col_name] = np.nan # Store NaN for non-convertible numbers
                else:
                    numeric_fields[col_name] = np.nan

        date_columns = [col for col in df.columns if re.match(r'^\\d{{8}}_\\d{{6}}$', col) or re.match(r'^\\d{{2}}-\\d{{2}}-\\d{{4}}$', col)]
        date_columns.sort() # Ensure consistent order

        for i in range(len(date_columns)):
            current_date_col = date_columns[i]
            value_col = current_date_col # The date column itself holds the value
            diff_col = f"difference{i}" if f"difference{i}" in df.columns else None # Find corresponding difference column

            date_val = row.get(current_date_col)
            diff_val = row.get(diff_col)

            if pd.isna(date_val) and pd.isna(diff_val):
                continue

            formatted_date = current_date_col.replace('_', ' ').replace('000000', '').strip()
            
            if not pd.isna(date_val):
                row_summary_parts.append(f"on Date {formatted_date} {str(date_val).strip()} KWH")
            
            if diff_col and not pd.isna(diff_val):
                row_summary_parts.append(f"both day difference {str(diff_val).strip()} KWH")

        # Use SLM to generate a natural language summary for the row
        row_summary_content = _get_slm_summary(row.to_dict(), column_types)

        metadata = {
            "file_id": file_id,
            "file_name": file_name,
            "sheet": sheet_name,
            "row_id": row_id,
            "columns": list(df.columns),
            "numeric_fields": numeric_fields, # Still collect numeric fields for potential filtering
            "created_at": datetime.now().isoformat(),
            "trusted": TRUSTED_FLAG_DEFAULT,
            "doc_type": "row",
        }
        documents.append(Document(doc_id=str(uuid.uuid4()), file_id=file_id, file_name=file_name, doc_type="row", content=row_summary_content, metadata=metadata))

    # Create column-summary documents
    for col_name in df.columns:
        col_series = df[col_name]
        col_type = column_types[col_name]
        summary_content = f"Column summary for \'{col_name}\'. Data type: {col_type}. "
        
        if col_type == "number":
            if not col_series.empty and col_series.dropna().empty: # If all values are NaN
                min_val_scalar = np.nan
                max_val_scalar = np.nan
                mean_val_scalar = np.nan
            else:
                try:
                    min_val_scalar = float(col_series.min())
                except TypeError:
                    min_val_scalar = np.nan
                try:
                    max_val_scalar = float(col_series.max())
                except TypeError:
                    max_val_scalar = np.nan
                try:
                    mean_val_scalar = float(col_series.mean())
                except TypeError:
                    mean_val_scalar = np.nan

            min_str = f'{min_val_scalar:.2f}' if pd.notna(min_val_scalar) else "N/A"
            max_str = f'{max_val_scalar:.2f}' if pd.notna(max_val_scalar) else "N/A"
            mean_str = f'{mean_val_scalar:.2f}' if pd.notna(mean_val_scalar) else "N/A"

            summary_content += f"Minimum value: {min_str}, Maximum value: {max_str}, Average value: {mean_str}. "
        elif col_type == "datetime":
            # For datetime, provide range if possible
            try:
                min_date = pd.to_datetime(col_series, errors='coerce').min()
                max_date = pd.to_datetime(col_series, errors='coerce').max()
                if pd.notna(min_date) and pd.notna(max_date):
                    summary_content += f"Date range from {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}. "
            except Exception:
                pass
        
        unique_sample = col_series.nunique()
        sample_values = col_series.dropna().head(5).values.tolist()
        summary_content += f"Number of unique values: {unique_sample}. Sample values: {sample_values}."
        
        metadata = {
            "file_id": file_id,
            "file_name": file_name,
            "sheet": sheet_name,
            "column_name": col_name,
            "column_type": col_type,
            "created_at": datetime.now().isoformat(),
            "trusted": TRUSTED_FLAG_DEFAULT,
            "doc_type": "column_summary",
        }
        documents.append(Document(doc_id=str(uuid.uuid4()), file_id=file_id, file_name=file_name, doc_type="column_summary", content=summary_content, metadata=metadata))

    return documents

def _chunk_text(text: str, file_id: str, file_name: str, page_no: int = None, section_title: str = None) -> List[Document]:
    """
    Chunks text into smaller documents with overlap.
    """
    # Basic word-based chunking for now
    words = text.split()
    chunks = []
    for i in range(0, len(words), TEXT_CHUNK_SIZE - TEXT_CHUNK_OVERLAP):
        chunk_words = words[i:i + TEXT_CHUNK_SIZE]
        content = " ".join(chunk_words)
        metadata = {
            "file_id": file_id,
            "file_name": file_name,
            "page_no": page_no,
            "section_title": section_title,
            "chunk_id": str(uuid.uuid4()),
            "created_at": datetime.now().isoformat(),
            "trusted": TRUSTED_FLAG_DEFAULT,
            "doc_type": "text_chunk",
        }
        chunks.append(Document(doc_id=str(uuid.uuid4()), file_id=file_id, file_name=file_name, doc_type="text_chunk", content=content, metadata=metadata))
    return chunks

def ingest_file(file_path: str, file_id: str = None, row_limit: Optional[int] = None) -> List[Document]:
    """
    Ingests a single file, processes it based on type, and returns a list of Documents.
    """
    file_name = os.path.basename(file_path)
    if file_id is None:
        file_id = str(uuid.uuid4())

    logger.info(f"Starting ingestion for file: {file_path}")
    all_documents: List[Document] = []

    try:
        if file_path.endswith(('.csv', '.xlsx')):
            df = None
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx'):
                xl = pd.ExcelFile(file_path)
                # Process only the first two sheets as requested
                for sheet_name in xl.sheet_names[:2]: # Limit to first 2 sheets
                    df_sheet = xl.parse(sheet_name)
                    if row_limit: # Apply row limit if provided
                        df_sheet = df_sheet.head(row_limit)
                    all_documents.extend(_process_dataframe(df_sheet, file_id, file_name, sheet_name=sheet_name))
            
            if df is not None: # For CSVs that are processed directly
                if row_limit: # Apply row limit if provided
                    df = df.head(row_limit)
                all_documents.extend(_process_dataframe(df, file_id, file_name))

        elif file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                text_content = f.read()
            all_documents.extend(_chunk_text(text_content, file_id, file_name))
            log_process_completion(f"Ingestion of TXT: {file_name}", details="Processed as textual data")
        else:
            logger.warning(f"Unsupported file type for ingestion: {file_name}")
            log_process_completion(f"Ingestion of {file_name}", status="skipped", details="Unsupported file type")
            return []

        logger.info(f"Successfully ingested {len(all_documents)} documents from {file_name}")
        return all_documents

    except Exception as e:
        logger.error(f"Error ingesting file {file_name}: {e}", exc_info=True)
        log_process_completion(f"Ingestion of {file_name}", status="failed", details=str(e))
        return []
