#!/usr/bin/env python3
"""
Validate Azure endpoint before running full performance tests.
Tests basic connectivity, token counting, and response format.
"""

import urllib.request
import json
import time
from typing import Dict, List
import sys
import os

# Configuration
ENDPOINT_URL = os.environ.get('AZURE_ENDPOINT_URL', 'https://your-endpoint.eastus2.inference.ml.azure.com/score')
API_KEY = os.environ.get('AZURE_API_KEY', 'your-api-key-here')
USE_STREAMING = True  # Should be True for performance testing
INPUT_FILE = 'inputs_1.json'

# Validation parameters
INPUT_TOKENS_TARGET = 2500
OUTPUT_TOKENS_TARGET = 200
TEST_REQUESTS = 5


def load_sample_inputs(file_path: str, num_samples: int = 5) -> List[Dict]:
    """Load sample inputs from input file"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            return data['data'][:num_samples]
    except FileNotFoundError:
        print(f"Error: {file_path} not found. Generate inputs first.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error parsing {file_path}: {e}")
        sys.exit(1)


def estimate_tokens(text: str) -> int:
    """Rough token estimation (1 token ≈ 0.75 words)"""
    words = len(text.split())
    return int(words / 0.75)


def make_request(input_text: str, max_tokens: int = 200) -> Dict:
    """Make a single request to the endpoint"""
    
    # Format request based on Azure ML deployment expectations
    # The endpoint requires chat format with role/content structure
    data = {
        "input_data": {
            "input_string": [
                {"role": "user", "content": input_text}
            ],
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True
            }
        }
    }
    
    body = str.encode(json.dumps(data))
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'text/event-stream' if USE_STREAMING else 'application/json',
        'Authorization': f'Bearer {API_KEY}'
    }
    
    req = urllib.request.Request(ENDPOINT_URL, body, headers)
    
    start_time = time.time()
    
    try:
        response = urllib.request.urlopen(req)
        
        if USE_STREAMING:
            full_response = ""
            first_token_time = None
            
            for line in response:
                decoded_line = line.decode('utf-8').strip()
                
                if decoded_line.startswith('data: '):
                    data_content = decoded_line[6:]
                    
                    if data_content != '[DONE]':
                        try:
                            chunk = json.loads(data_content)
                            
                            # Azure ML format: {"output": "text"}
                            token = None
                            if 'output' in chunk:
                                token = chunk.get('output', '')
                            elif 'choices' in chunk:
                                token = chunk['choices'][0].get('delta', {}).get('content', '')
                            elif 'text' in chunk:
                                token = chunk.get('text', '')
                            
                            if token:
                                full_response += token
                                if first_token_time is None:
                                    first_token_time = time.time()
                        except json.JSONDecodeError:
                            pass
                elif decoded_line.startswith('{'):
                    # Sometimes responses come without 'data: ' prefix
                    try:
                        chunk = json.loads(decoded_line)
                        if 'output' in chunk:
                            token = chunk.get('output', '')
                            if token:
                                full_response += token
                                if first_token_time is None:
                                    first_token_time = time.time()
                    except json.JSONDecodeError:
                        pass
            
            end_time = time.time()
            
            return {
                'success': True,
                'response': full_response,
                'total_latency': end_time - start_time,
                'time_to_first_token': first_token_time - start_time if first_token_time else None,
                'output_tokens': estimate_tokens(full_response)
            }
        else:
            result = response.read()
            end_time = time.time()
            
            response_data = json.loads(result)
            
            # Azure ML format: {"output": "text"}
            response_text = ""
            if 'output' in response_data:
                response_text = response_data.get('output', '')
            elif 'choices' in response_data:
                response_text = response_data['choices'][0].get('message', {}).get('content', '')
            elif 'text' in response_data:
                response_text = response_data.get('text', '')
            
            return {
                'success': True,
                'response': response_text,
                'total_latency': end_time - start_time,
                'time_to_first_token': None,
                'output_tokens': estimate_tokens(response_text)
            }
            
    except urllib.error.HTTPError as error:
        return {
            'success': False,
            'error_code': error.code,
            'error_message': error.read().decode("utf8", 'ignore'),
            'total_latency': time.time() - start_time
        }


def validate_endpoint():
    """Run validation tests"""
    
    print("=" * 70)
    print("AZURE ENDPOINT VALIDATION")
    print("=" * 70)
    
    # Check API key
    if not API_KEY:
        print("❌ ERROR: API_KEY not set")
        sys.exit(1)
    
    print(f"✓ Endpoint: {ENDPOINT_URL}")
    print(f"✓ Streaming: {USE_STREAMING}")
    print()

    # Load test inputs
    print("Loading test inputs...")
    inputs = load_sample_inputs(INPUT_FILE, TEST_REQUESTS)
    print(f"✓ Loaded {len(inputs)} test inputs")
    print()
    
    # Validate input token counts
    print("Validating input token counts...")
    for i, input_data in enumerate(inputs):
        token_count = estimate_tokens(input_data['text_input'])
        status = "✓" if abs(token_count - INPUT_TOKENS_TARGET) < 200 else "⚠"
        print(f"  {status} Input {i+1}: ~{token_count} tokens (target: {INPUT_TOKENS_TARGET})")
    print()
    
    # Run test requests
    print("Running test requests...")
    print("-" * 70)
    
    results = []
    for i, input_data in enumerate(inputs):
        print(f"\nTest {i+1}/{len(inputs)}:")
        result = make_request(input_data['text_input'], input_data.get('max_tokens', 200))
        results.append(result)
        
        if result['success']:
            print(f"  ✓ Status: SUCCESS")
            print(f"  ✓ Total latency: {result['total_latency']:.2f}s")
            if result['time_to_first_token']:
                print(f"  ✓ Time to first token: {result['time_to_first_token']:.2f}s")
            print(f"  ✓ Output tokens: ~{result['output_tokens']} (target: {OUTPUT_TOKENS_TARGET})")
            print(f"  ✓ Response preview: {result['response'][:100]}...")
        else:
            print(f"  ❌ Status: FAILED")
            print(f"  ❌ Error code: {result['error_code']}")
            print(f"  ❌ Error: {result['error_message'][:200]}")
    
    print()
    print("-" * 70)
    
    # Summary
    successful = sum(1 for r in results if r['success'])
    print(f"\nSUMMARY:")
    print(f"  Success rate: {successful}/{len(results)} ({100*successful/len(results):.1f}%)")
    
    if successful > 0:
        latencies = [r['total_latency'] for r in results if r['success']]
        print(f"  Average latency: {sum(latencies)/len(latencies):.2f}s")
        print(f"  Min latency: {min(latencies):.2f}s")
        print(f"  Max latency: {max(latencies):.2f}s")
    
    if successful == len(results):
        print("\n✓ All validation tests passed! Ready for performance testing.")
        return 0
    else:
        print("\n⚠ Some tests failed. Fix issues before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(validate_endpoint())