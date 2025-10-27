#!/usr/bin/env python3
"""
Comprehensive System Testing CLI
Tests all components: SLM parsing, embeddings, database, and KPIs
"""

import sys
import os
import json
import requests
import sqlite3
import pandas as pd
from datetime import datetime
import re

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_ollama_connection():
    """Test Ollama connection and models"""
    print("🔍 Testing Ollama Connection...")
    
    try:
        # Test basic connection
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print(f"✅ Ollama connected. Available models: {len(models)}")
            for model in models[:5]:  # Show first 5 models
                print(f"   - {model.get('name', 'Unknown')}")
            return True
        else:
            print(f"❌ Ollama connection failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Ollama connection error: {e}")
        return False

def test_slm_parsing():
    """Test SLM JSON parsing with various queries"""
    print("\n🤖 Testing SLM Parsing...")
    
    test_queries = [
        "what is the energy consumption of SCP M/C FDR-2",
        "show me total energy for SVP machine -3",
        "average consumption of I/C Panel",
        "energy usage from 2025-01-01 to 2025-01-31"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Test Query {i}: {query} ---")
        
        try:
            # Call SLM directly with streaming disabled
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3.2:3b",
                    "prompt": f"""You are a strict parser for industrial plant data queries. Given a user's question, output a single JSON object with schema:
{{ 
  "intent": "EXACT_FETCH|FILTER_LIST|AGGREGATE|COMPARE|EXPLAIN|OPINION|UNKNOWN",
  "operation": "sum|avg|max|min|count|list|filter",
  "metric": "energy|consumption|power|temperature|pressure",
  "equipment": ["SCP M/C FDR-2", "SVP machine -3", "isolator room"],
  "dates": ["2025-09-22", "2025-10-22"],
  "filters": [{{"column":"", "op":"equals|contains|greater_than", "value":""}}],
  "group_by": [],
  "limit": 100,
  "confidence": 0.95
}}

Now parse this query: "{query}"

Only output JSON.""",
                    "max_tokens": 512,
                    "temperature": 0.0,
                    "stream": False  # Disable streaming
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                raw_text = data.get("response", "")
                print(f"Raw response: {raw_text[:200]}...")
                
                # Try to parse JSON
                try:
                    parsed = json.loads(raw_text)
                    print(f"✅ JSON parsed successfully: {parsed}")
                except json.JSONDecodeError as e:
                    print(f"❌ JSON parsing failed: {e}")
                    print(f"Raw text: {raw_text}")
            else:
                print(f"❌ SLM request failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ SLM test failed: {e}")

def test_embeddings():
    """Test embedding generation"""
    print("\n🔗 Testing Embeddings...")
    
    test_texts = [
        "SCP M/C FDR-2 energy consumption 150 kWh",
        "SVP machine -3 power usage 200 kW",
        "I/C Panel voltage 440V current 50A"
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n--- Test Embedding {i}: {text} ---")
        
        try:
            response = requests.post(
                "http://localhost:11434/api/embeddings",
                json={
                    "model": "qwen3-embedding:0.6b",
                    "prompt": text
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                embedding = data.get("embedding", [])
                if embedding:
                    print(f"✅ Embedding generated: {len(embedding)} dimensions")
                    print(f"   First 5 values: {embedding[:5]}")
                else:
                    print("❌ Empty embedding returned")
                    print(f"   Full response: {data}")
            else:
                print(f"❌ Embedding request failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Embedding test failed: {e}")

def test_database():
    """Test database connection and data"""
    print("\n🗄️ Testing Database...")
    
    try:
        conn = sqlite3.connect('data_prototype/metadata.db')
        cursor = conn.cursor()
        
        # Check if documents table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='documents'")
        if cursor.fetchone():
            print("✅ Documents table exists")
            
            # Count total documents
            cursor.execute("SELECT COUNT(*) FROM documents")
            total_docs = cursor.fetchone()[0]
            print(f"   Total documents: {total_docs}")
            
            # Count by type
            cursor.execute("SELECT doc_type, COUNT(*) FROM documents GROUP BY doc_type")
            doc_types = cursor.fetchall()
            print("   Document types:")
            for doc_type, count in doc_types:
                print(f"     - {doc_type}: {count}")
            
            # Sample some content
            cursor.execute("SELECT content, doc_type FROM documents LIMIT 3")
            samples = cursor.fetchall()
            print("   Sample content:")
            for i, (content, doc_type) in enumerate(samples, 1):
                print(f"     {i}. [{doc_type}] {content[:100]}...")
                
        else:
            print("❌ Documents table does not exist")
            
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

def test_kpi_calculation():
    """Test KPI calculation with real data"""
    print("\n📊 Testing KPI Calculation...")
    
    try:
        conn = sqlite3.connect('data_prototype/metadata.db')
        cursor = conn.cursor()
        
        # Get sample data
        cursor.execute("SELECT content, file_name FROM documents WHERE doc_type='row' LIMIT 10")
        rows = cursor.fetchall()
        
        if not rows:
            print("❌ No row data found in database")
            return False
            
        print(f"✅ Found {len(rows)} row documents")
        
        # Test energy pattern matching
        energy_patterns = [
            re.compile(r'\b(\d{1,4}(?:\.\d{1,2})?)\s*(?:kwh|kwh|energy|consumption)', re.IGNORECASE),
            re.compile(r'energy[:\s]*(\d{1,4}(?:\.\d{1,2})?)', re.IGNORECASE),
            re.compile(r'consumption[:\s]*(\d{1,4}(?:\.\d{1,2})?)', re.IGNORECASE),
        ]
        
        total_energy = 0.0
        energy_values = []
        equipment_seen = set()
        
        for content, file_name in rows:
            print(f"   Processing: {content[:50]}...")
            
            # Extract equipment
            equipment_tokens = re.findall(r'[A-Z]{2,4}\s+[A-Z/]+(?:\s+[A-Z0-9-]+)*', content)
            for token in equipment_tokens:
                equipment_seen.add(token)
            
            # Extract energy values
            for pattern in energy_patterns:
                matches = pattern.findall(content)
                for match in matches:
                    try:
                        val = float(match)
                        if 10 <= val <= 10000:
                            energy_values.append(val)
                            total_energy += val
                            print(f"     Found energy: {val} kWh")
                            break
                    except:
                        continue
        
        print(f"✅ Energy extraction results:")
        print(f"   Total energy: {total_energy:.2f} kWh")
        print(f"   Energy values found: {len(energy_values)}")
        print(f"   Equipment found: {len(equipment_seen)}")
        print(f"   Equipment list: {list(equipment_seen)[:5]}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ KPI calculation test failed: {e}")
        return False

def test_router_integration():
    """Test router integration"""
    print("\n🛣️ Testing Router Integration...")
    
    try:
        from router import QueryRouter
        
        router = QueryRouter()
        
        test_queries = [
            "what is the energy consumption of SCP M/C FDR-2",
            "show me total energy for SVP machine -3",
            "average consumption of I/C Panel"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- Router Test {i}: {query} ---")
            
            try:
                result = router.route_query(query)
                print(f"✅ Router result: {result}")
                
                # Check if we got proper equipment extraction
                equipment = result.get('equipment', [])
                if equipment:
                    print(f"   Equipment extracted: {equipment}")
                else:
                    print("   ⚠️  No equipment extracted")
                    
            except Exception as e:
                print(f"❌ Router test failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Router integration test failed: {e}")
        return False

def test_streamlit_components():
    """Test Streamlit component imports"""
    print("\n🌊 Testing Streamlit Components...")
    
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
        
        # Test deprecated methods
        if hasattr(st, 'rerun'):
            print("✅ st.rerun() available")
        else:
            print("❌ st.rerun() not available")
            
        if hasattr(st, 'experimental_rerun'):
            print("⚠️  st.experimental_rerun() still exists (deprecated)")
        else:
            print("✅ st.experimental_rerun() removed")
            
        return True
        
    except Exception as e:
        print(f"❌ Streamlit test failed: {e}")
        return False

def test_pandas_dataframe():
    """Test pandas DataFrame boolean evaluation"""
    print("\n🐼 Testing Pandas DataFrame...")
    
    try:
        import pandas as pd
        
        # Test empty DataFrame
        empty_df = pd.DataFrame()
        print(f"✅ Empty DataFrame: {empty_df.empty}")
        
        # Test non-empty DataFrame
        test_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        print(f"✅ Non-empty DataFrame: {test_df.empty}")
        
        # Test the problematic pattern
        try:
            if test_df and len(test_df) > 0:
                print("❌ Old pattern still works (unexpected)")
        except ValueError as e:
            print(f"✅ Old pattern correctly fails: {e}")
        
        # Test the correct pattern
        if not test_df.empty:
            print("✅ New pattern works correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Pandas test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 COMPREHENSIVE SYSTEM TESTING CLI")
    print("=" * 50)
    
    tests = [
        ("Ollama Connection", test_ollama_connection),
        ("SLM Parsing", test_slm_parsing),
        ("Embeddings", test_embeddings),
        ("Database", test_database),
        ("KPI Calculation", test_kpi_calculation),
        ("Router Integration", test_router_integration),
        ("Streamlit Components", test_streamlit_components),
        ("Pandas DataFrame", test_pandas_dataframe)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

