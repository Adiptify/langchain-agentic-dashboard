#!/usr/bin/env python3
"""
Embedding Debug Tool
Tests embedding generation and shows detailed responses
"""

import requests
import json

def debug_embedding(text):
    """Debug embedding generation for given text"""
    print(f"🔗 Debugging Embedding for: {text}")
    print("=" * 60)
    
    try:
        response = requests.post(
            "http://localhost:11434/api/embeddings",
            json={
                "model": "qwen3-embedding:0.6b",
                "prompt": text
            },
            timeout=30
        )
        
        print(f"Response status: {response.status_code}")
        print(f"Response headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            try:
                data = response.json()
                print(f"✅ Response parsed as JSON successfully")
                print(f"JSON keys: {list(data.keys())}")
                
                embedding = data.get("embedding", [])
                if embedding:
                    print(f"✅ Embedding generated: {len(embedding)} dimensions")
                    print(f"First 5 values: {embedding[:5]}")
                    print(f"Last 5 values: {embedding[-5:]}")
                    print(f"Min value: {min(embedding):.6f}")
                    print(f"Max value: {max(embedding):.6f}")
                    return embedding
                else:
                    print("❌ Empty embedding returned")
                    print(f"Full response: {data}")
                    return None
            except json.JSONDecodeError as e:
                print(f"❌ Response JSON parse error: {e}")
                print(f"Raw response: {response.text}")
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
    """Test with various texts"""
    test_texts = [
        "SCP M/C FDR-2 energy consumption 150 kWh",
        "SVP machine -3 power usage 200 kW",
        "I/C Panel voltage 440V current 50A",
        "energy consumption",
        "equipment status"
    ]
    
    for text in test_texts:
        print(f"\n{'='*80}")
        embedding = debug_embedding(text)
        if embedding:
            print(f"✅ Successfully generated embedding with {len(embedding)} dimensions")
        else:
            print("❌ Failed to generate embedding")
        print(f"{'='*80}")

if __name__ == "__main__":
    main()

