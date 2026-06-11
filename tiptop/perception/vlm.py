"""Vision-language model router (Gemini or OpenAI)."""

import logging
import os

from PIL import Image

from tiptop.config import tiptop_cfg

_log = logging.getLogger(__name__)

DEFAULT_MODELS = {
    "gemini": "gemini-robotics-er-1.6-preview",
    "openai": "gpt-4o",
}


def get_vlm_settings() -> tuple[str, str]:
    """Return (provider, model) from env overrides or tiptop.yml."""
    cfg = tiptop_cfg()
    vlm_cfg = cfg.perception.vlm if "vlm" in cfg.perception else {}
    provider = os.environ.get("TIPTOP_VLM_PROVIDER", vlm_cfg.get("provider", "openai")).lower()
    model = os.environ.get("TIPTOP_VLM_MODEL", vlm_cfg.get("model", DEFAULT_MODELS.get(provider, "gpt-4o")))
    if provider not in DEFAULT_MODELS:
        raise ValueError(f"Unknown VLM provider '{provider}'. Use 'openai' or 'gemini'.")
    return provider, model


def detect_and_translate(
    image: Image.Image,
    task_instruction: str,
    temperature: float | None = None,
) -> tuple[list[dict], list[dict]]:
    provider, model = get_vlm_settings()
    _log.info(f"VLM detect_and_translate via {provider} ({model})")
    if provider == "openai":
        from tiptop.perception.openai_vision import detect_and_translate as _fn

        return _fn(image, task_instruction, model_id=model, temperature=temperature)
    from tiptop.perception.gemini import detect_and_translate as _fn

    return _fn(image, task_instruction, model_id=model, temperature=temperature)


async def detect_and_translate_async(
    image: Image.Image,
    task_instruction: str,
    temperature: float | None = None,
) -> tuple[list[dict], list[dict]]:
    provider, model = get_vlm_settings()
    _log.info(f"VLM detect_and_translate_async via {provider} ({model})")
    if provider == "openai":
        from tiptop.perception.openai_vision import detect_and_translate_async as _fn

        return await _fn(image, task_instruction, model_id=model, temperature=temperature)
    from tiptop.perception.gemini import detect_and_translate_async as _fn

    return await _fn(image, task_instruction, model_id=model, temperature=temperature)


def detect_bboxes_with_prompt(
    image: Image.Image,
    prompt: str,
    temperature: float | None = 0.0,
) -> list | dict:
    provider, model = get_vlm_settings()
    _log.info(f"VLM bbox detection via {provider} ({model})")
    if provider == "openai":
        from tiptop.perception.openai_vision import detect_bboxes_with_prompt as _fn

        return _fn(image, prompt, model_id=model, temperature=temperature)
    from tiptop.perception.gemini import gemini_client
    from tiptop.perception.vlm_common import load_json

    client = gemini_client()
    from google.genai import types

    response = client.models.generate_content(
        model=model,
        contents=[image, prompt],
        config=types.GenerateContentConfig(
            temperature=temperature, thinking_config=types.ThinkingConfig(thinking_budget=0)
        ),
    )
    return load_json(response.text)
