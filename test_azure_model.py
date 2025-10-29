#!/usr/bin/env python3
"""
Performance testing script for Azure Llama 3.3 70B deployment.
Supports both MaaS (serverless) and PTU (provisioned) deployments.
"""

import urllib.request
import json
import time
import concurrent.futures
import statistics
from datetime import datetime
from typing import List, Dict
import csv
from openai import OpenAI
import os

# ============================================================================
# DEPLOYMENT CONFIGURATION
# ============================================================================
# Deployment Type: 'maas' (serverless) or 'ptu' (provisioned with PTU control)
DEPLOYMENT_TYPE = 'maas'

# Test Mode: 'quick' (5 min validation) or 'full' (30 min production test)
TEST_MODE = 'quick'

# Precision: Only relevant for PTU deployments
# 'fp8' = 8-bit floating point (faster, less memory)
# 'bf16' = 16-bit brain floating point (higher quality)
# 'maas' = serverless (precision determined by Azure)
PRECISION = 'maas' if DEPLOYMENT_TYPE == 'maas' else 'bf16'

# ============================================================================
# ENDPOINT CONFIGURATION
# Available urls:
# 1. 'https://lucaswh-fde1-l3370bi-7.eastus2.inference.ml.azure.com/score' (request mode)
# 2. 'https://nazimi-fde-resource.services.ai.azure.com/openai/v1/' (sdk mode)
# ============================================================================
# Request Mode: 'sdk' (OpenAI client) or 'request' (urllib for Azure ML endpoints)
REQUEST_MODE = 'sdk'

# For SDK mode (OpenAI-compatible endpoints)
ENDPOINT_URL = os.environ.get('AZURE_ENDPOINT_URL', 'https://your-resource.services.ai.azure.com/openai/v1/')
MODEL_NAME = os.environ.get('AZURE_MODEL_NAME', 'Llama-3.3-70B-Instruct')  # Used for SDK mode

# For request mode (Azure ML endpoints) 
# ENDPOINT_URL = 'https://your-endpoint.eastus2.inference.ml.azure.com/score'

API_KEY = os.environ.get('AZURE_API_KEY', 'your-api-key-here')
INPUT_FILE = 'inputs_100.json'

# ============================================================================
# TEST PARAMETERS (Bank of America Requirements)
# ============================================================================
INPUT_TOKENS = 2500
OUTPUT_TOKENS = 200
REQUESTS_PER_SECOND = 15 # Target: 23 req/s

# ============================================================================
# PTU ALLOCATION STRATEGY (only for DEPLOYMENT_TYPE='ptu')
# ============================================================================
PTU_DEFAULTS = {
    'fp8': {1: 200, 2: 150, 3: 100, 4: 80, 5: 60, 6: 50},
    'bf16': {1: 300, 2: 225, 3: 150, 4: 120, 5: 90, 6: 75}
}

# Override PTU allocation for testing (None = use defaults)
PTU_OVERRIDE = None

# ============================================================================
# TEST MATRIX CONFIGURATION
# ============================================================================
if TEST_MODE == 'quick':
    # Quick validation test (5 minutes)
    TEST_DURATION_SECONDS = 5 * 60
    LATENCY_TARGETS = [6]  # Test single target
    RUNS_PER_TARGET = 1
    
elif TEST_MODE == 'full':
    # Full production test suite (30 minutes per run)
    TEST_DURATION_SECONDS = 30 * 60
    LATENCY_TARGETS = [1, 2, 3, 4, 5, 6]  # All targets
    RUNS_PER_TARGET = 3
    
else:
    raise ValueError(f"Invalid TEST_MODE: {TEST_MODE}. Use 'quick' or 'full'")


class PerformanceTest:
    def __init__(self, precision: str, latency_target: float, run_number: int, ptu_count: int):
        self.precision = precision
        self.latency_target = latency_target
        self.run_number = run_number
        self.ptu_count = ptu_count
        self.results = []
        self.errors = []
        self.start_time = None
        self.end_time = None
        
        # Initialize OpenAI client for SDK mode
        if REQUEST_MODE == 'sdk':
            self.client = OpenAI(
                base_url=ENDPOINT_URL,
                api_key=API_KEY
            )
        else:
            self.client = None
        
    def make_request_sdk(self, input_text: str) -> Dict:
        """Make request using OpenAI SDK (for chat completion endpoints)"""
        request_start = time.time()
        first_token_time = None
        
        try:
            # Use streaming to capture TTFT
            stream = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "user", "content": input_text}
                ],
                max_tokens=OUTPUT_TOKENS,
                temperature=0.7,
                top_p=0.9,
                stream=True
            )
            
            # Process streaming response
            for chunk in stream:
                if first_token_time is None and chunk.choices and chunk.choices[0].delta.content:
                    first_token_time = time.time()
            
            request_end = time.time()
            
            return {
                'success': True,
                'total_latency': request_end - request_start,
                'ttft': first_token_time - request_start if first_token_time else None,
                'timestamp': request_start
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': request_start,
                'total_latency': time.time() - request_start
            }
    
    def make_request_urllib(self, input_text: str) -> Dict:
        """Make request using urllib (for Azure ML endpoints)"""
        data = {
            "input_data": {
                "input_string": [
                    {"role": "user", "content": input_text}
                ],
                "parameters": {
                    "max_new_tokens": OUTPUT_TOKENS,
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            }
        }
        
        body = str.encode(json.dumps(data))
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'text/event-stream',
            'Authorization': f'Bearer {API_KEY}'
        }
        
        req = urllib.request.Request(ENDPOINT_URL, body, headers)
        request_start = time.time()
        
        try:
            response = urllib.request.urlopen(req)
            
            # Measure streaming response
            first_token_time = None
            
            for line in response:
                decoded_line = line.decode('utf-8').strip()
                
                # Azure ML format - look for {"output": "..."} or data: {...}
                if decoded_line.startswith('data: '):
                    data_content = decoded_line[6:]
                    if data_content != '[DONE]' and first_token_time is None:
                        try:
                            chunk = json.loads(data_content)
                            if 'output' in chunk and chunk['output']:
                                first_token_time = time.time()
                        except json.JSONDecodeError:
                            pass
                elif decoded_line.startswith('{') and first_token_time is None:
                    try:
                        chunk = json.loads(decoded_line)
                        if 'output' in chunk and chunk['output']:
                            first_token_time = time.time()
                    except json.JSONDecodeError:
                        pass
            
            request_end = time.time()
            
            return {
                'success': True,
                'total_latency': request_end - request_start,
                'ttft': first_token_time - request_start if first_token_time else None,
                'timestamp': request_start
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': request_start,
                'total_latency': time.time() - request_start
            }
        
    def make_request(self, input_text: str) -> Dict:
        """Make single request and measure latency (dispatches to appropriate method)"""
        if REQUEST_MODE == 'sdk':
            return self.make_request_sdk(input_text)
        else:
            return self.make_request_urllib(input_text)
    
    def run_test(self, inputs: List[Dict]):
        """Run load test for specified duration with concurrent requests"""
        print(f"\n{'='*70}")
        print(f"Starting Test: {self.precision} @ {self.latency_target}s target (Run {self.run_number})")
        if DEPLOYMENT_TYPE == 'ptu':
            print(f"PTU Allocation: {self.ptu_count}")
        else:
            print(f"Deployment: MaaS (serverless)")
        print(f"{'='*70}\n")
        
        self.start_time = time.time()
        end_time = self.start_time + TEST_DURATION_SECONDS
        request_interval = 1.0 / REQUESTS_PER_SECOND
        
        request_count = 0
        input_index = 0
        
        # Use ThreadPoolExecutor for concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=200) as executor:
            futures = []

            # Submit requests at REQUESTS_PER_SECOND without waiting for completion
            while time.time() < end_time:
                iteration_start = time.time()
                
                # Get next input (cycle through inputs)
                input_data = inputs[input_index % len(inputs)]
                input_index += 1
                
                # Submit request asynchronously
                future = executor.submit(self.make_request, input_data['text_input'])
                futures.append(future)

                request_count += 1
                
                # FAIL FAST: Check first 10 requests for critical errors
                if request_count == 10:
                    print("Checking first 10 requests for errors...")
                    # Wait for first 10 requests to complete
                    for f in futures[:10]:
                        f.result(timeout=30)  # Wait up to 30s for each
                    
                    early_results = [f.result() for f in futures[:10]]
                    early_errors = [r for r in early_results if not r['success']]
                    early_success = len(early_results) - len(early_errors)
                    
                    print(f"First 10 results: {early_success} success, {len(early_errors)} errors")
                    
                    if len(early_errors) >= 8:  # 80% failure rate
                        print(f"\n{'='*70}")
                        print("CRITICAL ERROR: 8+ failures in first 10 requests!")
                        print(f"Sample error: {early_errors[0]['error']}")
                        print("Aborting test - check endpoint configuration")
                        print(f"{'='*70}\n")
                        executor.shutdown(wait=False, cancel_futures=True)
                        raise Exception(f"Endpoint failed: {early_errors[0]['error']}")

                # Progress update every 20 seconds
                if request_count % (REQUESTS_PER_SECOND * 20) == 0:
                    elapsed = time.time() - self.start_time
                    progress = (elapsed / TEST_DURATION_SECONDS) * 100
                    completed = sum(1 for f in futures if f.done())
                    in_flight = request_count - completed
                    print(f"Progress: {progress:.1f}% | Submitted: {request_count} | Completed: {completed} | In-flight: {in_flight}")
                
                # Sleep to maintain request submission rate at 23/s
                elapsed = time.time() - iteration_start
                sleep_time = max(0, request_interval - elapsed)
                time.sleep(sleep_time)
            
            print("\nWaiting for all requests to complete...")
            
            # Collect results as they complete
            for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
                result = future.result()
                
                if result['success']:
                    self.results.append(result)
                else:
                    self.errors.append(result)
                    # Print first 3 errors
                    if len(self.errors) <= 3:
                        print(f"Error: {result['error'][:200]}")
                
                # Progress update during collection
                if i % 100 == 0 or i == len(futures):
                    print(f"Collected: {i}/{len(futures)} | Success: {len(self.results)} | Errors: {len(self.errors)}")
        
        self.end_time = time.time()
        self.calculate_metrics()
    
    def calculate_metrics(self):
        """Calculate P50, P75, P90 latencies"""
        if not self.results:
            print("\n" + "="*70)
            print("TEST FAILED - NO SUCCESSFUL REQUESTS")
            print("="*70)
            print(f"Total requests: {len(self.errors)}")
            print(f"All requests failed!")
            if self.errors:
                print(f"Sample error: {self.errors[0]['error'][:200]}")
            print("="*70)
            return None
        
        latencies = [r['total_latency'] for r in self.results]
        latencies.sort()
        
        n = len(latencies)
        p50 = latencies[int(n * 0.50)]
        p75 = latencies[int(n * 0.75)]
        p90 = latencies[int(n * 0.90)]
        
        print(f"\n{'='*70}")
        print("TEST RESULTS")
        print(f"{'='*70}")
        print(f"Duration: {self.end_time - self.start_time:.1f}s")
        print(f"Total requests: {len(self.results) + len(self.errors)}")
        print(f"Successful: {len(self.results)}")
        print(f"Failed: {len(self.errors)}")
        print(f"Success rate: {len(self.results)/(len(self.results) + len(self.errors))*100:.1f}%")
        print("\nLatency Percentiles:")
        print(f"  P50: {p50:.3f}s")
        print(f"  P75: {p75:.3f}s")
        print(f"  P90: {p90:.3f}s")
        print(f"\nTarget: {self.latency_target}s")
        
        # Check if target met
        if p90 <= self.latency_target:
            print(f"✓ Target MET (P90 <= {self.latency_target}s)")
        else:
            print(f"✗ Target MISSED (P90 > {self.latency_target}s)")
        
        return {
            'p50': p50,
            'p75': p75,
            'p90': p90,
            'total_requests': len(self.results) + len(self.errors),
            'successful_requests': len(self.results),
            'failed_requests': len(self.errors)
        }
    
    def save_results(self, output_file: str) -> bool:
        """Save results to CSV. Returns True if saved, False if skipped."""
        metrics = self.calculate_metrics()
        
        # Don't save if no successful requests
        if metrics is None:
            print(f"\n⚠ Skipping CSV save - no successful requests")
            return False
        
        with open(output_file, 'a', newline='') as f:
            writer = csv.writer(f)
            
            # Write header if file is empty
            if f.tell() == 0:
                # Use deployment_type for MaaS, precision for PTU
                if DEPLOYMENT_TYPE == 'maas':
                    writer.writerow([
                        'deployment_type', 'latency_target', 'run_number', 'requests_per_sec',
                        'p50_latency', 'p75_latency', 'p90_latency',
                        'total_requests', 'successful_requests', 'failed_requests',
                        'timestamp'
                    ])
                else:
                    writer.writerow([
                        'precision', 'latency_target', 'run_number', 'ptu_count',
                        'p50_latency', 'p75_latency', 'p90_latency',
                        'total_requests', 'successful_requests', 'failed_requests',
                        'timestamp'
                    ])
            
            # For MaaS, use REQUESTS_PER_SECOND instead of PTU count
            capacity_metric = REQUESTS_PER_SECOND if DEPLOYMENT_TYPE == 'maas' else self.ptu_count
            
            writer.writerow([
                self.precision, self.latency_target, self.run_number, capacity_metric,
                metrics['p50'], metrics['p75'], metrics['p90'],
                metrics['total_requests'], metrics['successful_requests'], metrics['failed_requests'],
                datetime.now().isoformat()
            ])
        
        return True


def load_inputs(file_path: str = INPUT_FILE) -> List[Dict]:
    """Load test inputs"""
    with open(file_path, 'r') as f:
        data = json.load(f)
        return data['data']


def run_full_test_suite():
    """Execute complete test matrix"""
    
    if not API_KEY:
        raise Exception("API_KEY must be set!")
    
    print("\n" + "="*70)
    print("LLAMA 3.3 70B PERFORMANCE TEST SUITE")
    print("="*70)
    print(f"Configuration:")
    print(f"  Request Mode: {REQUEST_MODE.upper()}")
    print(f"  Deployment: {DEPLOYMENT_TYPE.upper()}")
    print(f"  Precision: {PRECISION}")
    print(f"  Endpoint: {ENDPOINT_URL}")
    if REQUEST_MODE == 'sdk':
        print(f"  Model: {MODEL_NAME}")
    print(f"  Input tokens: {INPUT_TOKENS}")
    print(f"  Output tokens: {OUTPUT_TOKENS}")
    print(f"  Request rate: {REQUESTS_PER_SECOND} req/s")
    print(f"  Test duration: {TEST_DURATION_SECONDS/60:.0f} minutes per run")
    print(f"  Total test runs: {len(LATENCY_TARGETS) * RUNS_PER_TARGET}")
    print("="*70)
    
    # Load inputs
    inputs = load_inputs()
    print(f"\n✓ Loaded {len(inputs)} test inputs")
    
    # Create output directory if it doesn't exist
    import os
    os.makedirs('output', exist_ok=True)
    
    output_file = f"output/performance_results_{DEPLOYMENT_TYPE}_{PRECISION}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    # Track if any results were saved
    results_saved = False
    
    # Run test matrix
    for latency_target in LATENCY_TARGETS:
        # Get PTU allocation for PTU deployments (not applicable for MaaS)
        if DEPLOYMENT_TYPE == 'ptu':
            ptu_count = PTU_OVERRIDE if PTU_OVERRIDE is not None else PTU_DEFAULTS[PRECISION][latency_target]
            source = "override" if PTU_OVERRIDE is not None else "default"
            print(f"\n{'='*70}")
            print(f"Testing {latency_target}s latency target")
            print(f"PTU allocation: {ptu_count} ({source})")
            print(f"{'='*70}")
        else:
            ptu_count = 0  # Not applicable for MaaS
            print(f"\n{'='*70}")
            print(f"Testing {latency_target}s latency target (MaaS serverless)")
            print(f"{'='*70}")
        
        for run_num in range(1, RUNS_PER_TARGET + 1):
            test = PerformanceTest(PRECISION, latency_target, run_num, ptu_count)
            test.run_test(inputs)
            if test.save_results(output_file):
                results_saved = True
            
            # Cooldown between runs
            if run_num < RUNS_PER_TARGET:
                print("\nCooldown period (5 minutes)...")
                time.sleep(300)
    
    print(f"\n{'='*70}")
    if results_saved:
        print(f"✓ Test suite complete! Results saved to: {output_file}")
    else:
        print("✗ Test suite complete! No results saved - all tests failed")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    run_full_test_suite()