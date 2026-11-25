"""Thin GLM client (OpenAI-compatible API) with retry + JSON extraction."""
import json
import time

from openai import OpenAI

from .config import GLM_API_KEY, GLM_BASE_URL, GLM_MODEL

_client = None


def client() -> OpenAI:
    global _client
    if _client is None:
        if not GLM_API_KEY:
            raise RuntimeError("GLM_API_KEY not set; check .env")
        _client = OpenAI(api_key=GLM_API_KEY, base_url=GLM_BASE_URL)
    return _client


def chat(
    system: str,
    user: str,
    max_retries: int = 5,
    temperature: float = 0.0,
    timeout: float = 60.0,
) -> str:
    """Single chat completion, returns assistant text. Retries on transient errors."""
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client().chat.completions.create(
                model=GLM_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                timeout=timeout,
            )
            return resp.choices[0].message.content
        except Exception as e:  # noqa: BLE001 - log and retry any transient API error
            last_err = e
            wait = min(2 ** attempt * 2, 30)
            print(f"[glm] attempt {attempt + 1} failed ({type(e).__name__}: {e}); retry in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"GLM call failed after {max_retries} retries: {last_err}")


def chat_json(system: str, user: str, **kw) -> dict:
    """Chat completion that must return a JSON object; extracts it robustly."""
    text = chat(system, user, **kw)
    return extract_json(text)


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model response."""
    # Fast path: clean response.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Strip code fences.
    if "```" in text:
        for chunk in text.split("```"):
            chunk = chunk.strip()
            if chunk.startswith("json"):
                chunk = chunk[4:].strip()
            try:
                return json.loads(chunk)
            except json.JSONDecodeError:
                continue
    # Fall back to first {...} block.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not parse JSON from GLM response: {text[:300]!r}")
