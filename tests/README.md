# Tests

## Smoke Tests

### smoke_test_mock.py
Mock version that demonstrates the debate structure without requiring API calls. 
Run this to verify the code structure is correct:
```bash
python3 tests/smoke_test_mock.py
```

### smoke_test.py
Live API test using OpenRouter free models. Requires OPENROUTER_API_KEY environment variable.

**Note**: Free OpenRouter models may be rate-limited or temporarily unavailable. 
If the test fails with API errors, the mock test validates the structure is correct.

To run with API:
```bash
source /path/to/.env  # Contains OPENROUTER_API_KEY
python3 tests/smoke_test.py
```

The test will try multiple free models in order:
1. mistralai/mistral-7b-instruct:free
2. meta-llama/llama-3-8b-instruct:free
3. google/gemma-2-9b-it:free
