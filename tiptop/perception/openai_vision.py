import asyncio
import base64
import logging
import os
from functools import cache
from io import BytesIO
from pathlib import Path

from PIL import Image
from openai import AsyncOpenAI, OpenAI

from tiptop.perception.bbox_utils import boxes_containment_score, normalize_openai_bboxes
from tiptop.perception.vlm_common import (
    load_json,
    load_prompt,
    parse_detect_and_translate_response,
)

_log = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-4o"
_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


def _load_env_file() -> None:
    """Load gitignored tiptop/.env when OPENAI_API_KEY is not already set."""
    if os.environ.get("OPENAI_API_KEY") or not _ENV_FILE.is_file():
        return
    for line in _ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@cache
def openai_client() -> OpenAI:
    _load_env_file()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Export it in your shell or ~/.bashrc.")
    return OpenAI(api_key=api_key)


@cache
def openai_async_client() -> AsyncOpenAI:
    _load_env_file()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Export it in your shell or ~/.bashrc.")
    return AsyncOpenAI(api_key=api_key)


def _pil_to_data_url(image: Image.Image) -> str:
    buf = BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _image_content(image: Image.Image, prompt: str) -> list:
    return [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": _pil_to_data_url(image), "detail": "high"}},
    ]


def _openai_detect_prompt(image: Image.Image, task_instruction: str) -> str:
    w, h = image.size
    return load_prompt("detect_and_translate_openai").format(
        task_instruction=task_instruction,
        image_width=w,
        image_height=h,
    )


def _normalize_openai_bboxes(result: list | dict, img_width: int, img_height: int) -> None:
    if isinstance(result, dict):
        items = result.get("bboxes", [])
    else:
        items = result
    normalize_openai_bboxes(items, img_width, img_height)


def _spatial_retry_prompt(base_prompt: str, task_instruction: str) -> str:
    return (
        f"{base_prompt}\n\n"
        "IMPORTANT CORRECTION: Your previous bounding boxes were spatially inconsistent. "
        f"Re-check the image carefully for task: {task_instruction!r}. "
        "Return pixel coordinates [xmin, ymin, xmax, ymax] that TIGHTLY fit each object. "
        "If the task mentions an object inside a box/container, the inner object boxes must lie "
        "fully inside the container box."
    )


def _maybe_retry_for_containment(
    image: Image.Image,
    prompt: str,
    task_instruction: str,
    bboxes: list[dict],
    grounded_atoms: list[dict],
    model_id: str,
    temperature: float | None,
) -> tuple[list[dict], list[dict]]:
    inner_labels = [b["label"] for b in bboxes if "peach" in b["label"] or "apple" in b["label"]]
    if not inner_labels or "box" not in {b["label"] for b in bboxes}:
        return bboxes, grounded_atoms
    score = boxes_containment_score(bboxes, "box", inner_labels)
    if score >= 1.0:
        return bboxes, grounded_atoms
    _log.warning(f"OpenAI boxes failed container containment ({score:.0%}); retrying detection once")
    retry_prompt = _spatial_retry_prompt(prompt, task_instruction)
    response_text = _generate_vision_json(image, retry_prompt, model_id=model_id, temperature=temperature)
    bboxes, grounded_atoms = parse_detect_and_translate_response(response_text, "OpenAI")
    _normalize_openai_bboxes(bboxes, image.size[0], image.size[1])
    return bboxes, grounded_atoms


async def _maybe_retry_for_containment_async(
    image: Image.Image,
    prompt: str,
    task_instruction: str,
    bboxes: list[dict],
    grounded_atoms: list[dict],
    model_id: str,
    temperature: float | None,
) -> tuple[list[dict], list[dict]]:
    inner_labels = [b["label"] for b in bboxes if "peach" in b["label"] or "apple" in b["label"]]
    if not inner_labels or "box" not in {b["label"] for b in bboxes}:
        return bboxes, grounded_atoms
    score = boxes_containment_score(bboxes, "box", inner_labels)
    if score >= 1.0:
        return bboxes, grounded_atoms
    _log.warning(f"OpenAI boxes failed container containment ({score:.0%}); retrying detection once")
    retry_prompt = _spatial_retry_prompt(prompt, task_instruction)
    response_text = await _generate_vision_json_async(
        image, retry_prompt, model_id=model_id, temperature=temperature
    )
    bboxes, grounded_atoms = parse_detect_and_translate_response(response_text, "OpenAI")
    _normalize_openai_bboxes(bboxes, image.size[0], image.size[1])
    return bboxes, grounded_atoms


def _generate_vision_json(
    image: Image.Image,
    prompt: str,
    model_id: str = DEFAULT_MODEL,
    temperature: float | None = 0.0,
) -> str:
    client = openai_client()
    response = client.chat.completions.create(
        model=model_id,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "user",
                "content": _image_content(image, prompt),
            }
        ],
    )
    text = response.choices[0].message.content
    if not text:
        raise RuntimeError("OpenAI vision returned an empty response")
    return text


async def _generate_vision_json_async(
    image: Image.Image,
    prompt: str,
    model_id: str = DEFAULT_MODEL,
    temperature: float | None = 0.0,
) -> str:
    client = openai_async_client()
    response = await client.chat.completions.create(
        model=model_id,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "user",
                "content": _image_content(image, prompt),
            }
        ],
    )
    text = response.choices[0].message.content
    if not text:
        raise RuntimeError("OpenAI vision returned an empty response")
    return text


def detect_and_translate(
    image: Image.Image,
    task_instruction: str,
    model_id: str = DEFAULT_MODEL,
    temperature: float | None = 0.0,
) -> tuple[list[dict], list[dict]]:
    prompt = _openai_detect_prompt(image, task_instruction)
    response_text = _generate_vision_json(image, prompt, model_id=model_id, temperature=temperature)
    bboxes, grounded_atoms = parse_detect_and_translate_response(response_text, "OpenAI")
    _normalize_openai_bboxes(bboxes, image.size[0], image.size[1])
    return _maybe_retry_for_containment(
        image, prompt, task_instruction, bboxes, grounded_atoms, model_id, temperature
    )


async def detect_and_translate_async(
    image: Image.Image,
    task_instruction: str,
    model_id: str = DEFAULT_MODEL,
    temperature: float | None = 0.0,
) -> tuple[list[dict], list[dict]]:
    prompt = _openai_detect_prompt(image, task_instruction)
    response_text = await _generate_vision_json_async(image, prompt, model_id=model_id, temperature=temperature)
    bboxes, grounded_atoms = parse_detect_and_translate_response(response_text, "OpenAI")
    _normalize_openai_bboxes(bboxes, image.size[0], image.size[1])
    return await _maybe_retry_for_containment_async(
        image, prompt, task_instruction, bboxes, grounded_atoms, model_id, temperature
    )


def detect_bboxes_with_prompt(
    image: Image.Image,
    prompt: str,
    model_id: str = DEFAULT_MODEL,
    temperature: float | None = 0.0,
) -> list:
    """Run a custom bbox-detection prompt. Returns parsed JSON (list or dict)."""
    response_text = _generate_vision_json(image, prompt, model_id=model_id, temperature=temperature)
    result = load_json(response_text)
    _normalize_openai_bboxes(result, image.size[0], image.size[1])
    return result
