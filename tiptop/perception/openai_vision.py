import base64
import logging
import os
from functools import cache
from io import BytesIO
from pathlib import Path

from PIL import Image
from openai import AsyncOpenAI, OpenAI

from tiptop.perception.bbox_utils import apply_gemini_norm_bboxes, boxes_containment_score
from tiptop.perception.openai_schemas import BboxListResponse, DetectAndTranslateResponse, DetectedBBox
from tiptop.perception.vlm_common import load_prompt

_log = logging.getLogger(__name__)

# gpt-4o alias does not support Structured Outputs; use a dated snapshot or gpt-4.1.
DEFAULT_MODEL = "gpt-4o-2024-08-06"
STRUCTURED_OUTPUT_MODELS = {
    "gpt-4o-2024-08-06",
    "gpt-4o-2024-11-20",
    "gpt-4o-mini",
    "gpt-4o-mini-2024-07-18",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
}
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


def _supports_structured_outputs(model_id: str) -> bool:
    if model_id in STRUCTURED_OUTPUT_MODELS:
        return True
    return model_id.startswith("gpt-4.1") or "2024-08" in model_id or "2024-11" in model_id


def _pil_to_data_url(image: Image.Image) -> str:
    buf = BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _vision_user_content(image: Image.Image, prompt: str) -> list:
    return [
        {"type": "text", "text": prompt},
        {
            "type": "image_url",
            "image_url": {"url": _pil_to_data_url(image), "detail": "high"},
        },
    ]


def _detect_prompt(task_instruction: str) -> str:
    return load_prompt("detect_and_translate").format(task_instruction=task_instruction)


def _spatial_retry_suffix(task_instruction: str) -> str:
    return (
        "\n\nIMPORTANT CORRECTION: Previous boxes were spatially wrong. "
        f"Re-check the image for task: {task_instruction!r}. "
        "Each box must TIGHTLY fit its object on the 1000x1000 grid. "
        "If the task mentions an object inside a container, inner boxes must lie fully inside the container."
    )


def _structured_bbox_to_dict(det: DetectedBBox) -> dict:
    return {
        "label": det.label,
        "box_2d": [det.ymin, det.xmin, det.ymax, det.xmax],
    }


def _response_to_bboxes_and_atoms(parsed: DetectAndTranslateResponse) -> tuple[list[dict], list[dict]]:
    bboxes = [_structured_bbox_to_dict(b) for b in parsed.bboxes]
    grounded_atoms = [
        {"predicate": spec.name, "args": spec.args}
        for spec in parsed.predicates
        if spec.name and spec.args
    ]
    return bboxes, grounded_atoms


def _finalize_bboxes(bboxes: list[dict], img_width: int, img_height: int) -> None:
    apply_gemini_norm_bboxes(bboxes, img_width, img_height)


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
    return _detect_and_translate_structured(
        image, prompt + _spatial_retry_suffix(task_instruction), model_id, temperature
    )


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
    return await _detect_and_translate_structured_async(
        image, prompt + _spatial_retry_suffix(task_instruction), model_id, temperature
    )


def _detect_and_translate_structured(
    image: Image.Image,
    prompt: str,
    model_id: str,
    temperature: float | None,
) -> tuple[list[dict], list[dict]]:
    if not _supports_structured_outputs(model_id):
        raise ValueError(
            f"Model {model_id!r} does not support Structured Outputs for vision bboxes. "
            f"Set perception.vlm.model to {DEFAULT_MODEL!r} or gpt-4.1 in tiptop.yml."
        )
    client = openai_client()
    response = client.beta.chat.completions.parse(
        model=model_id,
        temperature=temperature,
        messages=[
            {
                "role": "system",
                "content": (
                    "You detect objects in robot workspace images and translate tasks into predicates. "
                    "Bounding boxes use a 1000x1000 normalized grid: ymin/xmin are the top-left corner, "
                    "ymax/xmax are the bottom-right corner. Coordinates are relative to the full image "
                    "(0=top/left, 1000=bottom/right), not raw pixels."
                ),
            },
            {"role": "user", "content": _vision_user_content(image, prompt)},
        ],
        response_format=DetectAndTranslateResponse,
    )
    parsed = response.choices[0].message.parsed
    if parsed is None:
        refusal = response.choices[0].message.refusal
        raise RuntimeError(f"OpenAI vision refused or returned empty structured output: {refusal}")
    bboxes, grounded_atoms = _response_to_bboxes_and_atoms(parsed)
    _finalize_bboxes(bboxes, image.size[0], image.size[1])
    return bboxes, grounded_atoms


async def _detect_and_translate_structured_async(
    image: Image.Image,
    prompt: str,
    model_id: str,
    temperature: float | None,
) -> tuple[list[dict], list[dict]]:
    if not _supports_structured_outputs(model_id):
        raise ValueError(
            f"Model {model_id!r} does not support Structured Outputs for vision bboxes. "
            f"Set perception.vlm.model to {DEFAULT_MODEL!r} or gpt-4.1 in tiptop.yml."
        )
    client = openai_async_client()
    response = await client.beta.chat.completions.parse(
        model=model_id,
        temperature=temperature,
        messages=[
            {
                "role": "system",
                "content": (
                    "You detect objects in robot workspace images and translate tasks into predicates. "
                    "Bounding boxes use a 1000x1000 normalized grid: ymin/xmin are the top-left corner, "
                    "ymax/xmax are the bottom-right corner. Coordinates are relative to the full image "
                    "(0=top/left, 1000=bottom/right), not raw pixels."
                ),
            },
            {"role": "user", "content": _vision_user_content(image, prompt)},
        ],
        response_format=DetectAndTranslateResponse,
    )
    parsed = response.choices[0].message.parsed
    if parsed is None:
        refusal = response.choices[0].message.refusal
        raise RuntimeError(f"OpenAI vision refused or returned empty structured output: {refusal}")
    bboxes, grounded_atoms = _response_to_bboxes_and_atoms(parsed)
    _finalize_bboxes(bboxes, image.size[0], image.size[1])
    return bboxes, grounded_atoms


def _detect_bboxes_structured(
    image: Image.Image,
    prompt: str,
    model_id: str,
    temperature: float | None,
) -> list[dict]:
    if not _supports_structured_outputs(model_id):
        raise ValueError(
            f"Model {model_id!r} does not support Structured Outputs. Use {DEFAULT_MODEL!r} or gpt-4.1."
        )
    client = openai_client()
    response = client.beta.chat.completions.parse(
        model=model_id,
        temperature=temperature,
        messages=[
            {
                "role": "system",
                "content": (
                    "Return bounding boxes on a 1000x1000 grid (ymin, xmin, ymax, xmax). "
                    "0=top/left, 1000=bottom/right of the image."
                ),
            },
            {"role": "user", "content": _vision_user_content(image, prompt)},
        ],
        response_format=BboxListResponse,
    )
    parsed = response.choices[0].message.parsed
    if parsed is None:
        raise RuntimeError("OpenAI vision returned empty structured bbox response")
    bboxes = [_structured_bbox_to_dict(b) for b in parsed.bboxes]
    _finalize_bboxes(bboxes, image.size[0], image.size[1])
    return bboxes


def detect_and_translate(
    image: Image.Image,
    task_instruction: str,
    model_id: str = DEFAULT_MODEL,
    temperature: float | None = 0.0,
) -> tuple[list[dict], list[dict]]:
    prompt = _detect_prompt(task_instruction)
    bboxes, grounded_atoms = _detect_and_translate_structured(image, prompt, model_id, temperature)
    return _maybe_retry_for_containment(
        image, prompt, task_instruction, bboxes, grounded_atoms, model_id, temperature
    )


async def detect_and_translate_async(
    image: Image.Image,
    task_instruction: str,
    model_id: str = DEFAULT_MODEL,
    temperature: float | None = 0.0,
) -> tuple[list[dict], list[dict]]:
    prompt = _detect_prompt(task_instruction)
    bboxes, grounded_atoms = await _detect_and_translate_structured_async(image, prompt, model_id, temperature)
    return await _maybe_retry_for_containment_async(
        image, prompt, task_instruction, bboxes, grounded_atoms, model_id, temperature
    )


def detect_bboxes_with_prompt(
    image: Image.Image,
    prompt: str,
    model_id: str = DEFAULT_MODEL,
    temperature: float | None = 0.0,
) -> list:
    """Run a custom bbox-detection prompt. Returns list of bbox dicts."""
    return _detect_bboxes_structured(image, prompt, model_id, temperature)
