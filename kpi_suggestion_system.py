#!/usr/bin/env python3
"""
KPI Suggestion System
Smart suggestion system that provides contextual prompts based on user queries and history
"""

import sys
import os
import json
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Any
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class KPISuggestionSystem:
    def __init__(self):
        self.db_path = 'data_prototype/metadata.db'
        self.user_history_db = 'user_profiles.db'
        
    def get_contextual_suggestions(self, current_query: str = "", user_id: str = "default") -> List[Dict[str, Any]]:
        """
        Get contextual suggestions based on current query and user history
        """
        suggestions = []
        
        # If user has a current query, provide related suggestions
        if current_query:
            suggestions.extend(self._get_related_suggestions(current_query))
        
        # If no current query or few suggestions, add top used prompts
        if len(suggestions) < 3:
            suggestions.extend(self._get_top_used_prompts(user_id))
        
        # Add general energy-related suggestions
        if len(suggestions) < 3:
            suggestions.extend(self._get_general_suggestions())
        
        return suggestions[:3]  # Return top 3 suggestions
    
    def _get_related_suggestions(self, query: str) -> List[Dict[str, Any]]:
        """Get suggestions related to the current query"""
        suggestions = []
        
        # Extract equipment name from query for dynamic suggestions
        equipment_name = self._extract_equipment_from_query(query)
        
        # Equipment-based suggestions
        if any(term in query.lower() for term in ['scp', 'fdr', 'panel', 'relay', 'ic', 'machine', 'equipment']):
            if equipment_name:
                suggestions.extend([
                    {
                        "prompt": f"Show me daily consumption for {equipment_name}",
                        "category": "Time Series",
                        "icon": "[CHART]",
                        "description": "Get daily readings and trends"
                    },
                    {
                        "prompt": f"What is the total consumption for {equipment_name}?",
                        "category": "Summary",
                        "icon": "[LIGHTNING]",
                        "description": "Calculate total usage"
                    },
                    {
                        "prompt": f"Show me efficiency metrics for {equipment_name}",
                        "category": "Analytics",
                        "icon": "[TREND]",
                        "description": "Analyze efficiency and performance"
                    }
                ])
            else:
                suggestions.extend([
                    {
                        "prompt": "Show me daily consumption for this equipment",
                        "category": "Time Series",
                        "icon": "[CHART]",
                        "description": "Get daily readings and trends"
                    },
                    {
                        "prompt": "What is the total consumption for this equipment?",
                        "category": "Summary",
                        "icon": "[LIGHTNING]",
                        "description": "Calculate total usage"
                    },
                    {
                        "prompt": "Show me efficiency metrics for this equipment",
                        "category": "Analytics",
                        "icon": "[TREND]",
                        "description": "Analyze efficiency and performance"
                    }
                ])
        
        # Energy-related suggestions
        if any(term in query.lower() for term in ['energy', 'consumption', 'kwh', 'power']):
            suggestions.extend([
                {
                    "prompt": "Show me energy consumption trends over time",
                    "category": "Trends",
                    "icon": "[TREND]",
                    "description": "Visualize energy usage patterns"
                },
                {
                    "prompt": "What is the peak energy consumption period?",
                    "category": "Peak Analysis",
                    "icon": "[SEARCH]",
                    "description": "Identify high consumption periods"
                },
                {
                    "prompt": "Compare energy consumption across different equipment",
                    "category": "Comparison",
                    "icon": "[BALANCE]",
                    "description": "Compare multiple equipment performance"
                }
            ])
        
        # Date-related suggestions
        if any(term in query.lower() for term in ['daily', 'weekly', 'monthly', 'date', 'period']):
            suggestions.extend([
                {
                    "prompt": "Show me weekly energy consumption summary",
                    "category": "Weekly Report",
                    "icon": "[CALENDAR]",
                    "description": "Get weekly energy overview"
                },
                {
                    "prompt": "What is the monthly energy consumption trend?",
                    "category": "Monthly Analysis",
                    "icon": "[CHART]",
                    "description": "Analyze monthly patterns"
                },
                {
                    "prompt": "Show me energy consumption by date range",
                    "category": "Custom Period",
                    "icon": "[DATE]",
                    "description": "Query specific date ranges"
                }
            ])
        
        return suggestions
    
    def _extract_equipment_from_query(self, query: str) -> str:
        """Extract equipment name from query for dynamic suggestions"""
        import re
        
        # Common equipment patterns
        equipment_patterns = [
            r'[A-Z]{2,4}\s*[M/C|MACHINE|M/C]\s*[A-Z0-9\-]+',
            r'[A-Z]{2,4}\s*[A-Z0-9\-]+\s*[A-Z0-9\-]*',
            r'[A-Z]{2,4}\s*Panel',
            r'[A-Z]{2,4}\s*FDR',
            r'[A-Z]{2,4}\s*Relay'
        ]
        
        for pattern in equipment_patterns:
            matches = re.findall(pattern, query)
            if matches:
                return matches[0].strip()
        
        return ""
    
    def _get_top_used_prompts(self, user_id: str) -> List[Dict[str, Any]]:
        """Get top used prompts by user"""
        try:
            conn = sqlite3.connect(self.user_history_db)
            cursor = conn.cursor()
            
            # Get user's query history
            cursor.execute("""
                SELECT query, COUNT(*) as usage_count 
                FROM query_history 
                WHERE user_id = ? 
                GROUP BY query 
                ORDER BY usage_count DESC 
                LIMIT 3
            """, (user_id,))
            
            results = cursor.fetchall()
            suggestions = []
            
            for query, count in results:
                suggestions.append({
                    "prompt": query,
                    "category": "Frequently Used",
                    "icon": "[STAR]",
                    "description": f"Used {count} times"
                })
            
            conn.close()
            return suggestions
            
        except Exception as e:
            print(f"Error getting user history: {e}")
            return []
    
    def _get_general_suggestions(self) -> List[Dict[str, Any]]:
        """Get general energy-related suggestions"""
        return [
                {
                    "prompt": "Show me total energy consumption for all equipment",
                    "category": "Overview",
                    "icon": "[FACTORY]",
                    "description": "Get system-wide energy overview"
                },
                {
                    "prompt": "What are the top energy consuming equipment?",
                    "category": "Ranking",
                    "icon": "[TROPHY]",
                    "description": "Identify highest consumers"
                },
                {
                    "prompt": "Show me energy consumption by equipment type",
                    "category": "Categorization",
                    "icon": "[LIST]",
                    "description": "Group by equipment categories"
                }
        ]
    
    def log_user_query(self, user_id: str, query: str):
        """Log user query for history tracking"""
        try:
            conn = sqlite3.connect(self.user_history_db)
            cursor = conn.cursor()
            
            # Create table if not exists
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS query_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    query TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Insert query
            cursor.execute("""
                INSERT INTO query_history (user_id, query) 
                VALUES (?, ?)
            """, (user_id, query))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error logging query: {e}")

def test_kpi_suggestion_system():
    """Test the KPI suggestion system"""
    print("TESTING KPI SUGGESTION SYSTEM")
    print("=" * 50)
    
    kpi_system = KPISuggestionSystem()
    
    # Test scenarios
    test_scenarios = [
        ("", "No current query"),
        ("SCP M/C FDR-2 energy consumption", "Equipment-specific query"),
        ("daily energy readings", "Time-based query"),
        ("energy consumption trends", "Analytics query")
    ]
    
    for query, description in test_scenarios:
        print(f"\n--- {description} ---")
        print(f"Query: '{query}'")
        
        suggestions = kpi_system.get_contextual_suggestions(query)
        
        for i, suggestion in enumerate(suggestions, 1):
            print(f"{i}. {suggestion['icon']} {suggestion['prompt']}")
            print(f"   Category: {suggestion['category']}")
            print(f"   Description: {suggestion['description']}")
            print()
    
    return True

if __name__ == "__main__":
    test_kpi_suggestion_system()
