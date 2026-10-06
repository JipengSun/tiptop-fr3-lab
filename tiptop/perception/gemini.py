import logging
import os
from functools import cache
from pathlib import Path

from PIL import Image
from google import genai
from google.genai import types

from tiptop.perception.bbox_utils import apply_gemini_norm_bboxes
from tiptop.perception.vlm_common import load_prompt, parse_detect_and_translate_response

_log = logging.getLogger(__name__)
_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


def _load_env_file() -> None:
    """Load gitignored tiptop/.env when GOOGLE_API_KEY is not already set."""
    if os.environ.get("GOOGLE_API_KEY") or not _ENV_FILE.is_file():
        return
    for line in _ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key.startswith("export "):
            key = key.removeprefix("export ").strip()
        if key and key not in os.environ:
            os.environ[key] = value


@cache
def gemini_client() -> genai.Client:
    _load_env_file()
    if not os.environ.get("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY is not set. Export it or add to tiptop/.env.")
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
    bboxes, grounded_atoms = parse_detect_and_translate_response(response.text, "Gemini")
    apply_gemini_norm_bboxes(bboxes, image.size[0], image.size[1])
    return bboxes, grounded_atoms


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
    bboxes, grounded_atoms = parse_detect_and_translate_response(response.text, "Gemini")
    apply_gemini_norm_bboxes(bboxes, image.size[0], image.size[1])
    return bboxes, grounded_atoms
