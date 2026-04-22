from __future__ import annotations

import json

import httpx

from rag_chatbot.config import get_settings


async def generate_text(prompt: str) -> str:
    settings = get_settings()
    provider = settings.llm_provider.lower()

    if provider == "ollama":
        payload = {
            "model": settings.ollama_model,
            "prompt": prompt,
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(f"{settings.ollama_base_url}/api/generate", json=payload)
            response.raise_for_status()
            data = response.json()
        return data.get("response", "")

    if provider == "openrouter":
        if not settings.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY is required when LLM_PROVIDER=openrouter")

        headers = {
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "Content-Type": "application/json",
        }
        if settings.openrouter_site_url:
            headers["HTTP-Referer"] = settings.openrouter_site_url
        if settings.openrouter_app_name:
            headers["X-Title"] = settings.openrouter_app_name

        payload = {
            "model": settings.openrouter_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                f"{settings.openrouter_base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        choices = data.get("choices", [])
        if not choices:
            return ""
        message = choices[0].get("message", {})
        return message.get("content", "") or ""

    raise ValueError(f"Unsupported LLM_PROVIDER: {settings.llm_provider}")


async def stream_text(prompt: str):
    settings = get_settings()
    provider = settings.llm_provider.lower()

    if provider == "ollama":
        payload = {
            "model": settings.ollama_model,
            "prompt": prompt,
            "stream": True,
        }
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream("POST", f"{settings.ollama_base_url}/api/generate", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    yield line
        return

    if provider == "openrouter":
        if not settings.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY is required when LLM_PROVIDER=openrouter")

        headers = {
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "Content-Type": "application/json",
        }
        if settings.openrouter_site_url:
            headers["HTTP-Referer"] = settings.openrouter_site_url
        if settings.openrouter_app_name:
            headers["X-Title"] = settings.openrouter_app_name

        payload = {
            "model": settings.openrouter_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        }
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST",
                f"{settings.openrouter_base_url}/chat/completions",
                headers=headers,
                json=payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if data == "[DONE]":
                        break
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    choices = event.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    text = delta.get("content")
                    if text:
                        yield text
        return

    raise ValueError(f"Unsupported LLM_PROVIDER: {settings.llm_provider}")
