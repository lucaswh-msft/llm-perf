#!/usr/bin/env python3
"""Debug script to find the correct Azure endpoint format"""

import urllib.request
import json
import os

ENDPOINT_URL = os.environ.get('AZURE_ENDPOINT_URL', 'https://your-endpoint.eastus2.inference.ml.azure.com/score')
API_KEY = os.environ.get('AZURE_API_KEY', 'your-api-key-here')

# Test different request formats
test_formats = [
    {
        "name": "Format 1: Direct prompt",
        "data": {
            "prompt": "What is 2+2?",
            "max_tokens": 50,
            "temperature": 0.7
        }
    },
    {
        "name": "Format 2: Input data with input_string",
        "data": {
            "input_data": {
                "input_string": ["What is 2+2?"],
                "parameters": {
                    "max_new_tokens": 50,
                    "temperature": 0.7
                }
            }
        }
    },
    {
        "name": "Format 3: Input data with data array",
        "data": {
            "input_data": {
                "data": ["What is 2+2?"],
                "params": {
                    "max_new_tokens": 50,
                    "temperature": 0.7
                }
            }
        }
    },
    {
        "name": "Format 4: Inputs array",
        "data": {
            "inputs": ["What is 2+2?"],
            "parameters": {
                "max_new_tokens": 50,
                "temperature": 0.7
            }
        }
    },
    {
        "name": "Format 5: Messages format (chat)",
        "data": {
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "max_tokens": 50,
            "temperature": 0.7
        }
    }
]

def test_format(format_data):
    """Test a specific request format"""
    print(f"\n{'='*70}")
    print(f"Testing: {format_data['name']}")
    print(f"{'='*70}")
    print(f"Request body: {json.dumps(format_data['data'], indent=2)}")
    
    body = str.encode(json.dumps(format_data['data']))
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',  # Non-streaming first
        'Authorization': f'Bearer {API_KEY}'
    }
    
    req = urllib.request.Request(ENDPOINT_URL, body, headers)
    
    try:
        response = urllib.request.urlopen(req)
        result = response.read().decode('utf-8')
        
        print(f"\n✓ SUCCESS - Status: {response.status}")
        print(f"Response: {result[:500]}")
        
        # Try to parse as JSON
        try:
            parsed = json.loads(result)
            print(f"\nParsed JSON keys: {list(parsed.keys())}")
            if parsed:
                print(f"Full response: {json.dumps(parsed, indent=2)[:1000]}")
        except:
            print("Response is not JSON")
            
    except urllib.error.HTTPError as error:
        print(f"\n✗ FAILED - Status: {error.code}")
        print(f"Error: {error.read().decode('utf8', 'ignore')[:500]}")
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")

if __name__ == "__main__":
    print("Testing different Azure endpoint formats...")
    
    for format_data in test_formats:
        test_format(format_data)
    
    print("\n" + "="*70)
    print("Testing complete. Look for the format that returns actual content.")
    print("="*70)
