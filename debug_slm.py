#!/usr/bin/env python3
"""
Simple SLM Debug Tool
Shows raw SLM responses for debugging JSON parsing issues
"""

import requests
import json
import re

def debug_slm_response(query):
    """Debug SLM response for a given query"""
    print(f"🤖 Debugging SLM Response for: {query}")
    print("=" * 60)
    
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2:3b",
                "prompt": f"""Output only a JSON object for this query: "{query}"

Format: {{"intent": "EXACT_FETCH", "equipment": ["SCP M/C FDR-2"], "confidence": 0.9}}

Only output JSON, nothing else.""",
                "max_tokens": 100,
                "temperature": 0.0,
                "stream": False  # Disable streaming
            },
            timeout=10
        )
        
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            # Get raw text first
            raw_text = response.text
            print("\nRAW RESPONSE TEXT:")
            print("-" * 40)
            print(repr(raw_text))
            print("-" * 40)
            
            # Try to parse as JSON
            try:
                data = json.loads(raw_text)
                print("✅ Response parsed as JSON successfully")
                print(f"JSON keys: {list(data.keys())}")
                
                if "response" in data:
                    slm_response = data["response"]
                    print(f"\nSLM response: {repr(slm_response)}")
                    
                    # Try to extract JSON from SLM response
                    start = slm_response.find('{')
                    end = slm_response.rfind('}')
                    if start != -1 and end != -1:
                        json_str = slm_response[start:end+1]
                        print(f"\nExtracted JSON: {repr(json_str)}")
                        try:
                            parsed = json.loads(json_str)
                            print(f"✅ SLM JSON parsed: {parsed}")
                            return parsed
                        except json.JSONDecodeError as e:
                            print(f"❌ SLM JSON parse error: {e}")
                            return None
                    else:
                        print("❌ No JSON found in SLM response")
                        return None
                else:
                    print("❌ No 'response' key in JSON")
                    return None
                    
            except json.JSONDecodeError as e:
                print(f"❌ Response JSON parse error: {e}")
                print(f"Raw response: {raw_text}")
                return None
        else:
            print(f"❌ Request failed: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Test with various queries"""
    test_queries = [
        "energy consumption SCP M/C FDR-2",
        "total energy for SVP machine -3",
        "average consumption of I/C Panel"
    ]
    
    for query in test_queries:
        print(f"\n{'='*80}")
        result = debug_slm_response(query)
        if result:
            print(f"✅ Successfully parsed: {result}")
        else:
            print("❌ Failed to parse")
        print(f"{'='*80}")

if __name__ == "__main__":
    main()

