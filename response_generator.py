"""
Response Generator for Streamlit UI
Generates proper formatted responses for the dashboard
"""

from typing import Dict, List, Any, Optional
import json

class StreamlitResponseGenerator:
    """Generate properly formatted responses for Streamlit UI"""
    
    def __init__(self):
        self.response_templates = {
            'success': self._generate_success_response,
            'error': self._generate_error_response,
            'no_data': self._generate_no_data_response,
            'partial': self._generate_partial_response
        }
    
    def generate_response(self, query: str, result: Dict[str, Any], 
                         documents: List[Dict], route_info: Dict[str, Any]) -> str:
        """Generate appropriate response based on result type"""
        
        # Handle different result types
        if not result:
            return self._generate_no_data_response(query, result, documents)
        
        # Check for error
        if result.get('error'):
            return self._generate_error_response(query, result.get('error'), documents)
        
        # Check if it's an LLM response (has 'answer' field)
        if result.get('answer'):
            return self._generate_llm_response(query, result, documents, route_info)
        
        # Check if it's an agent response (has 'value' field)
        if result.get('value') is not None:
            # Check if we have partial data
            if result.get('matched_rows', 0) < 3:
                return self._generate_partial_response(query, result, documents, route_info)
            # Full success response
            return self._generate_success_response(query, result, documents, route_info)
        
        # No data found
        return self._generate_no_data_response(query, result, documents)
    
    def _generate_success_response(self, query: str, result: Dict[str, Any], 
                                  documents: List[Dict], route_info: Dict[str, Any]) -> str:
        """Generate success response with full data"""
        
        value = result.get('value', 0)
        operation = result.get('operation', 'sum')
        matched_rows = result.get('matched_rows', 0)
        equipment_terms = result.get('equipment_terms', [])
        metric = result.get('metric', 'energy')
        
        # Format value with appropriate units
        if metric.lower() in ['energy', 'consumption', 'power']:
            unit = 'kWh'
            formatted_value = f"{value:,.2f} {unit}"
        else:
            formatted_value = f"{value:,.2f}"
        
        response = f"""
## ✅ Query Results

**Query:** {query}

**Result:** {formatted_value}

**Operation:** {operation.title()}

**Data Sources:** {matched_rows} documents analyzed

**Equipment:** {', '.join(equipment_terms) if equipment_terms else 'All equipment'}

### 📊 Key Findings:
- **Total {metric}:** {formatted_value}
- **Data processed:** {matched_rows} documents
- **Equipment analyzed:** {len(equipment_terms)} items
- **Confidence:** High (multiple data sources)

### 📋 Top Data Sources:
"""
        
        # Add top documents with better formatting
        for i, doc in enumerate(documents[:3], 1):
            content = doc.get('content', '')[:120]
            score = doc.get('score', 0)
            file_name = doc.get('file_name', 'Unknown')
            
            response += f"""
**{i}. {file_name}** (Score: {score:.3f})
> {content}...
"""
        
        return response
    
    def _generate_llm_response(self, query: str, result: Dict[str, Any], 
                              documents: List[Dict], route_info: Dict[str, Any]) -> str:
        """Generate response for LLM reasoning/explanation results"""
        
        answer = result.get('answer', '')
        response_type = result.get('type', 'unknown')
        provenance = result.get('provenance', [])
        
        response = f"""
## 🤖 AI Analysis

**Query:** {query}

**Response Type:** {response_type.replace('_', ' ').title()}

**Answer:**
{answer}

**Processing Details:**
- Route: {route_info.get('route', 'Unknown')}
- Processing Time: {route_info.get('processing_time', 0):.2f}s
- Documents Analyzed: {len(documents)}

### 📋 Data Sources:
"""
        
        # Add provenance information
        if provenance:
            for doc_id in provenance[:3]:  # Show top 3
                response += f"- Document ID: {doc_id}\n"
        
        # Add top documents with scores
        for i, doc in enumerate(documents[:3], 1):
            content = doc.get('content', '')[:120]
            score = doc.get('score', 0)
            file_name = doc.get('file_name', 'Unknown')
            
            response += f"""
**{i}. {file_name}** (Relevance: {score:.3f})
> {content}...
"""
        
        return response
    
    def _generate_partial_response(self, query: str, result: Dict[str, Any], 
                                 documents: List[Dict], route_info: Dict[str, Any]) -> str:
        """Generate response for partial data"""
        
        value = result.get('value', 0)
        operation = result.get('operation', 'sum')
        matched_rows = result.get('matched_rows', 0)
        equipment_terms = result.get('equipment_terms', [])
        metric = result.get('metric', 'energy')
        
        formatted_value = f"{value:,.2f} kWh" if metric.lower() in ['energy', 'consumption'] else f"{value:,.2f}"
        
        response = f"""
## ⚠️ Partial Results

**Query:** {query}

**Result:** {formatted_value}

**Status:** Limited data available ({matched_rows} documents)

**Equipment:** {', '.join(equipment_terms) if equipment_terms else 'All equipment'}

### 📊 Analysis:
- **Total {metric}:** {formatted_value}
- **Data sources:** {matched_rows} documents (limited)
- **Confidence:** Medium (limited data sources)

### 💡 Suggestions:
- Try broader search terms
- Check if equipment names are correct
- Consider different date ranges
"""
        
        return response
    
    def _generate_no_data_response(self, query: str, result: Dict[str, Any], 
                                   documents: List[Dict]) -> str:
        """Generate response when no data is found"""
        
        response = f"""
## ❌ No Data Found

**Query:** {query}

**Status:** No matching data found

### 🔍 Possible Reasons:
- Equipment name not found in database
- Query too specific
- Data not available for this equipment

### 💡 Suggestions:
- Try different equipment names
- Use broader search terms
- Check spelling of equipment names

### 📋 Available Equipment Examples:
- SCP M/C FDR-2
- SCP MACHINE-3  
- RADIAL FDR-1
- SCP M/C FDR-2 TO ISOLATOR ROOM
"""
        
        return response
    
    def _generate_error_response(self, query: str, error: str, 
                               documents: List[Dict]) -> str:
        """Generate error response"""
        
        response = f"""
## ❌ Query Error

**Query:** {query}

**Error:** {error}

### 🔧 Troubleshooting:
- Check query syntax
- Ensure equipment names are correct
- Try simpler queries first

### 💡 Example Queries:
- "What is the energy consumption of SCP M/C FDR-2?"
- "Show me total energy for SCP MACHINE-3"
- "What is the power consumption of RADIAL FDR-1"
"""
        
        return response
    
    def generate_summary_stats(self, results: List[Dict[str, Any]]) -> str:
        """Generate summary statistics for multiple results"""
        
        if not results:
            return "No results to summarize"
        
        total_energy = sum(r.get('value', 0) for r in results if r.get('value'))
        total_documents = sum(r.get('matched_rows', 0) for r in results)
        equipment_count = len(set(
            eq for r in results 
            for eq in r.get('equipment_terms', [])
        ))
        
        response = f"""
## 📊 Summary Statistics

**Total Energy:** {total_energy:,.2f} kWh
**Documents Processed:** {total_documents}
**Equipment Analyzed:** {equipment_count}
**Queries Processed:** {len(results)}
"""
        
        return response

# Global response generator instance
response_generator = StreamlitResponseGenerator()
