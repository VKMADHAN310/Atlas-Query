import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file in the same directory
import os
import time
from typing import Optional, Tuple

from openai import OpenAI
from dotenv import load_dotenv


def call_hf_chat(system: str, prompt: str, model: Optional[str] = None) -> Tuple[str, int]:
    """
    Call Hugging Face Inference Router using the OpenAI-compatible client.
    Returns (text, duration_ms).
    """
    load_dotenv()
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN not found in environment or .env")

    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=hf_token,
    )

    # Default to a capable instruction-tuned Llama on HF router if none provided
    hf_model = model or os.getenv("HF_MODEL", "meta-llama/Llama-3.3-70B-Instruct:cerebras")

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    t0 = time.perf_counter_ns()
    completion = client.chat.completions.create(
        model=hf_model,
        messages=messages,
        temperature=0,
        max_tokens=256,
    )
    t1 = time.perf_counter_ns()

    msg = completion.choices[0].message
    content = msg.content if hasattr(msg, "content") else (msg.get("content") if isinstance(msg, dict) else "")
    return content or "", int((t1 - t0) / 1e6)


if __name__ == "__main__":
    # Simple manual test
    txt, ms = call_hf_chat(system="You are concise.", prompt="What is POSTGIS?", model=None)
    print(f"HF took {ms} ms\n---\n{txt}")