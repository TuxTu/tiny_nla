import httpx, os, json

r = httpx.post(
    "https://api.deepseek.com/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}",
        "Content-Type": "application/json",
    },
    json={
        "model": "deepseek-v4-flash",
        "messages": [{"role": "user", "content": "Say hello"}],
        "max_tokens": 20,
        "thinking": {"type": "disabled"},
    },
    timeout=30,
)
print("status:", r.status_code)
print(json.dumps(r.json(), indent=2))
