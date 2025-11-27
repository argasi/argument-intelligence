import requests
import os
import json
import time

# --- Configuration ---
# NOTE: Replace with your actual OpenRouter API key.
# It is highly recommended to set this as an environment variable in a real-world scenario.
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "YOUR_OPENROUTER_API_KEY_HERE")
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Define the models to be benchmarked (OpenRouter Model IDs)
# Expanded list includes models from OpenAI, Anthropic, Google, Meta, Mistral, and Qwen.
BENCHMARK_MODELS = [
    # Tier 1: Flagship Models
    "openai/gpt-4o",
    "anthropic/claude-3-opus",
    "google/gemini-2.5-pro",
    
    # Tier 2: High Performance / High Cost-Efficiency
    "mistralai/mixtral-8x22b-instruct",
    "openai/gpt-4-turbo",
    "anthropic/claude-3-sonnet",
    "qwen/qwen-2-72b-instruct",
    
    # Tier 3: Good Cost-Efficiency / Mid-Range Performance
    "google/gemini-2.5-flash",
    "openai/gpt-3.5-turbo",
    "anthropic/claude-3-haiku",
    "meta-llama/llama-3-70b-instruct",
    "mistralai/mistral-large",
    "databricks/dbrx-instruct",
    
    # Tier 4: Fast / Lowest Cost
    "mistralai/mixtral-8x7b-instruct",
    "meta-llama/llama-3-8b-instruct",
    "qwen/qwen-2-7b-instruct",
    "nousresearch/nous-hermes-2-mixtral-8x7b-dpo",
    "cohere/command-r",
    "perplexity/pplx-7b-online",
    "01-ai/yi-large",
    "google/gemma-7b-it",
    "01-ai/yi-34b-chat",
]

# Define the Arguments Intelligence prompts (Scenarios)
ARGSI_PROMPTS = {
    "Analyze EV Pros & Cons (Balanced Argument)": "Perform a comprehensive, balanced analysis of the current market pros and cons of electric vehicles, focusing on total cost of ownership vs. environmental impact.",
    "Evaluate PESTLE for City Relocation": "Apply the PESTLE framework to analyze relocating a young professional to Austin, Texas. Detail one major risk and one major opportunity for each factor.",
    "SWOT Analysis for New Side Project": "Conduct a rapid SWOT analysis for launching a new AI-powered educational side project aimed at high school students.",
}

# --- Core Benchmarking Function ---

def benchmark_llm(model_id: str, prompt: str) -> dict:
    """
    Sends a request to the OpenRouter API and extracts relevant metrics.
    
    Args:
        model_id: The OpenRouter model ID (e.g., 'openai/gpt-4o').
        prompt: The user query.
        
    Returns:
        A dictionary containing tokens, cost, and latency, or None on failure.
    """
    if OPENROUTER_API_KEY == "YOUR_OPENROUTER_API_KEY_HERE":
        print("ERROR: Please set a valid OPENROUTER_API_KEY before running the benchmark.")
        return None

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Set max_tokens high to capture detailed, argument-rich responses
    payload = {
        "model": model_id,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.5,
        "max_tokens": 800
    }
    
    print(f"\n-> Benchmarking {model_id}...")
    start_time = time.time()
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        end_time = time.time()
        
        data = response.json()
        latency = end_time - start_time
        
        usage = data.get('usage', {})
        input_tokens = usage.get('prompt_tokens', 0)
        output_tokens = usage.get('completion_tokens', 0)
        cost_usd = usage.get('cost', 0)
        
        total_tokens = input_tokens + output_tokens
        cost_per_token = cost_usd / total_tokens if total_tokens > 0 else 0
        
        # SIMULATED Argument Intelligence Score (Heuristic based on model tier and size)
        model_index = BENCHMARK_MODELS.index(model_id)
        # Score is simulated based on position in the curated list
        ai_score = 99 - (model_index * 2) if model_index < 10 else 75 - (model_index - 10) * 1.5
        ai_score = int(max(40, ai_score)) # Min score of 40
        
        return {
            "llmName": model_id.split('/')[-1].replace('-', ' ').title(),
            "argumentIntelligenceScore": ai_score,
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
            "latencySeconds": round(latency, 2),
            "costPerToken": round(cost_per_token, 8),
        }

    except requests.exceptions.RequestException as e:
        print(f"Error during API call for {model_id}: {e}")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON response for {model_id}.")
        return None

# --- Main Execution ---

def run_full_benchmark():
    """Executes the full benchmark and prints the results in the required JSON format."""
    
    full_results = []
    
    # Iterate through each defined scenario (prompt)
    for prompt_name, prompt_text in ARGSI_PROMPTS.items():
        scenario_results = []
        
        # We only run the top 5-6 models in this sample script execution for brevity
        # but the full list is defined above.
        models_to_run = BENCHMARK_MODELS[:6] 
        
        for model_id in models_to_run:
            result = benchmark_llm(model_id, prompt_text)
            if result:
                scenario_results.append(result)
        
        # Store results for the current scenario
        full_results.append({
            "promptName": prompt_name,
            "results": scenario_results
        })

    # Output the final JSON in the exact structure required by the frontend
    print("\n" + "="*80)
    print("COPY THE FOLLOWING JSON DATA AND PASTE IT INTO 'docs/index.html' IN THE MOCK_BENCHMARK_RESULTS VARIABLE:")
    print("="*80)
    
    print(json.dumps(full_results, indent=4))
    
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE (Sampled first 6 models in this run).")

if __name__ == "__main__":
    run_full_benchmark()