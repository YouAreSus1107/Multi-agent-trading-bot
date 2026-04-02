import os
import time
from dotenv import load_dotenv

# Load ENV
env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(env_path, override=True)

# Test function
def test_provider(name, base_url, model, api_key_var):
    api_key = os.getenv(api_key_var)
    if not api_key:
        print(f"[FAIL] {name:<12} - No API Key found for {api_key_var}")
        return False

    print(f"[TEST] {name:<12} - Key found. Testing connection...")
    try:
        if name == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(model=model, google_api_key=api_key, max_retries=1)
        else:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model=model, 
                api_key=api_key, 
                base_url=base_url if base_url else None,
                max_retries=1
            )
        
        start = time.time()
        res = llm.invoke("Hi. Respond with exactly one word: Test")
        dur = time.time() - start
        if "Test" in res.content or "test" in res.content.lower():
            print(f" [OK]  {name:<12} - Success in {dur:.2f}s")
            return True
        else:
            print(f" [WARN] {name:<12} - Unexpected response: {res.content}")
            return False
            
    except Exception as e:
        print(f" [ERR] {name:<12} - Failed: {e}")
        return False

print("=== LLM API Key Test ===")
test_provider("openai", None, "gpt-4o-mini", "OPENAI_API_KEY")
test_provider("groq", "https://api.groq.com/openai/v1", "llama-3.3-70b-versatile", "GROQ_API_KEY")
test_provider("gemini", None, "gemini-2.0-flash-lite", "GEMINI_API_KEY")
test_provider("cerebras", "https://api.cerebras.ai/v1", "llama-3.3-70b", "CEREBRAS_API_KEY")
test_provider("openrouter", "https://openrouter.ai/api/v1", "meta-llama/llama-3.3-70b-instruct:free", "OPENROUTER_API_KEY")
test_provider("nvidia", "https://integrate.api.nvidia.com/v1", "meta/llama-3.3-70b-instruct", "NVIDIA_NIM_API_KEY")
test_provider("puter", "https://api.puter.com/puterai/openai/v1/", "gpt-4o-mini", "PUTER_API_TOKEN")
