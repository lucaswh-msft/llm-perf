# LLM Performance Testing

Performance testing scripts for Azure LLM deployments (Llama 3.3 70B).

## Setup

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your Azure credentials:
   - `AZURE_ENDPOINT_URL`: Your Azure endpoint URL
   - `AZURE_API_KEY`: Your Azure API key
   - `AZURE_MODEL_NAME`: Your model name (e.g., Llama-3.3-70B-Instruct)

3. Set environment variables:
   ```bash
   export $(cat .env | xargs)
   ```

## Running Scripts

```bash
# Validate endpoint connectivity
python3 validate_endpoint.py

# Run performance tests
python3 test_azure_model.py

# Debug endpoint formats
python3 debug_endpoint.py
```

## Scripts

- `test_azure_model.py`: Main performance testing script
- `validate_endpoint.py`: Validate endpoint connectivity and response format
- `debug_endpoint.py`: Debug script for testing different request formats

## Input Files

- `inputs_1.json`: Single request for testing
- `inputs_5.json`: 5 requests for testing
- `inputs_90.json`: 90 requests for testing
- `inputs_100.json`: 100 requests for testing

## Output

Performance results are saved in the `output/` directory as CSV files.
