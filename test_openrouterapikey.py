import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in .env file")

url = "https://openrouter.ai/api/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    # Optional
    "HTTP-Referer": "http://localhost",
    "X-OpenRouter-Title": "API Test",
}

payload = {
    "model": "openrouter/free",  # or any model available to your API key
    "messages": [
        {
            "role": "user",
            "content": "What is RAG?"
        }
    ]
}

try:
    response = requests.post(url, headers=headers, json=payload, timeout=30)

    print(f"Status Code: {response.status_code}")
    print("-" * 50)

    if response.status_code == 200:
        data = response.json()
        print("✅ API Key is valid!\n")
        print("Response:")
        print(data["choices"][0]["message"]["content"])
    else:
        print("❌ API request failed.")
        print(response.text)

except requests.exceptions.RequestException as e:
    print("Request Error:", e)