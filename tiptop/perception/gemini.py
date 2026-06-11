import logging
from functools import cache

from PIL import Image
from google import genai
from google.genai import types

from tiptop.perception.vlm_common import load_prompt, parse_detect_and_translate_response

_log = logging.getLogger(__name__)


@cache
def gemini_client() -> genai.Client:
    return genai.Client()


def detect_and_translate(
    image: Image.Image,
    task_instruction: str,
    client: genai.Client | None = None,
    model_id: str = "gemini-robotics-er-1.6-preview",
    temperature: float | None = None,
) -> tuple[list[dict], list[dict]]:
    client = client or gemini_client()
    prompt = load_prompt("detect_and_translate").format(task_instruction=task_instruction)
    response = client.models.generate_content(
        model=model_id,
        contents=[image, prompt],
        config=types.GenerateContentConfig(
            temperature=temperature, thinking_config=types.ThinkingConfig(thinking_budget=0)
        ),
    )
    return parse_detect_and_translate_response(response.text, "Gemini")


async def detect_and_translate_async(
    image: Image.Image,
    task_instruction: str,
    client: genai.Client | None = None,
    model_id: str = "gemini-robotics-er-1.6-preview",
    temperature: float | None = None,
) -> tuple[list[dict], list[dict]]:
    client = client or gemini_client()
    prompt = load_prompt("detect_and_translate").format(task_instruction=task_instruction)
    response = await client.aio.models.generate_content(
        model=model_id,
        contents=[image, prompt],
        config=types.GenerateContentConfig(
            temperature=temperature, thinking_config=types.ThinkingConfig(thinking_budget=0)
        ),
    )
    return parse_detect_and_translate_response(response.text, "Gemini")
