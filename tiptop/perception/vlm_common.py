import json
import logging
import re
from functools import cache
from pathlib import Path

_log = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"


@cache
def load_prompt(prompt_name: str) -> str:
    """Load a prompt template from the prompts directory."""
    return (_PROMPTS_DIR / f"{prompt_name}.txt").read_text(encoding="utf-8").strip()


def load_json(response_text: str) -> list | dict:
    """Extract JSON string from code fencing if present."""
    cleaned_text = response_text.strip()
    if cleaned_text.startswith("```json"):
        cleaned_text = cleaned_text.replace("```json", "").replace("```", "")
    elif cleaned_text.startswith("```"):
        cleaned_text = cleaned_text.replace("```", "")

    try:
        return json.loads(cleaned_text)
    except json.decoder.JSONDecodeError:
        _log.error(f"Invalid JSON: {cleaned_text}")
        raise


def resolve_box_2d(box_2d: list, prefer_openai: bool = False) -> list:
    """Normalize box to [ymin, xmin, ymax, xmax] in 0-1000.

    Gemini uses [ymin, xmin, ymax, xmax]. OpenAI often returns [xmin, ymin, xmax, ymax].
    When both interpretations are valid, prefer the provider-specific ordering when
    ``prefer_openai`` is set; otherwise pick OpenAI-style if they disagree substantially.
    """
    if len(box_2d) != 4:
        return box_2d
    a, b, c, d = box_2d

    def as_gemini(ymin, xmin, ymax, xmax):
        if ymax <= ymin or xmax <= xmin:
            return None
        area = (ymax - ymin) * (xmax - xmin)
        if area > 850_000:
            return None
        return [ymin, xmin, ymax, xmax]

    gemini = as_gemini(a, b, c, d)
    openai = as_gemini(b, a, d, c)
    if gemini and not openai:
        return gemini
    if openai and not gemini:
        return openai
    if gemini and openai:
        if prefer_openai:
            return openai
        g_center = ((a + c) / 2, (b + d) / 2)
        o_center = ((b + d) / 2, (a + c) / 2)
        dist = abs(g_center[0] - o_center[0]) + abs(g_center[1] - o_center[1])
        return openai if dist > 150 else gemini
    return box_2d


def normalize_bboxes(bboxes: list[dict], prefer_openai: bool = False) -> None:
    """In-place normalize all box_2d entries to [ymin, xmin, ymax, xmax]."""
    for bbox in bboxes:
        if "box_2d" in bbox:
            bbox["box_2d"] = resolve_box_2d(bbox["box_2d"], prefer_openai=prefer_openai)


def parse_detect_and_translate_response(response_text: str, provider: str) -> tuple[list, list]:
    """Parse VLM response text into bboxes and grounded atoms."""
    try:
        result = load_json(response_text)
    except Exception:
        raise ValueError(
            f"{provider} returned a non-JSON response; check for a discrepancy in your image: {response_text}"
        )
    bboxes = result.get("bboxes", [])
    grounded_atoms = [
        {"predicate": spec["name"], "args": spec["args"]}
        for spec in result.get("predicates", [])
        if spec.get("name") and spec.get("args")
    ]
    return bboxes, grounded_atoms


def _match_task_object(task_instruction: str, movable_labels: list[str]) -> str | None:
    """Pick the detected label that best matches nouns in the task instruction."""
    task_lower = task_instruction.lower()
    best_label = None
    best_score = 0
    for label in movable_labels:
        stem = re.sub(r"_\d+$", "", label).replace("_", " ").lower()
        name = label.replace("_", " ").lower()
        if stem and stem in task_lower:
            score = len(stem)
        elif name in task_lower:
            score = len(name)
        else:
            continue
        if score > best_score:
            best_score = score
            best_label = label
    return best_label


def sanitize_grounded_atoms(
    grounded_atoms: list[dict],
    known_labels: set[str],
    task_instruction: str | None = None,
) -> list[dict]:
    """Drop predicates that reference undetected objects; infer holding for pick tasks when needed."""
    valid_atoms: list[dict] = []
    for atom in grounded_atoms:
        args = atom.get("args", [])
        unknown = [arg for arg in args if arg not in known_labels]
        if unknown:
            _log.warning(
                f"Dropping goal atom {atom.get('predicate')}({', '.join(args)}): "
                f"unknown object(s) {unknown}. Known: {sorted(known_labels)}"
            )
            continue
        valid_atoms.append(atom)

    has_goal = any(atom.get("predicate") in {"on", "holding"} for atom in valid_atoms)
    if has_goal or not task_instruction:
        return valid_atoms

    pick_verbs = ("pick", "grab", "take", "get", "lift")
    if not any(verb in task_instruction.lower() for verb in pick_verbs):
        return valid_atoms

    movable_labels = sorted(label for label in known_labels if label != "table")
    matched = _match_task_object(task_instruction, movable_labels)
    if matched:
        _log.warning(
            f"No valid goal atoms after filtering; inferring holding({matched}) from task: {task_instruction!r}"
        )
        return [{"predicate": "holding", "args": [matched]}]

    _log.warning(f"No valid goal atoms and could not infer pick target from task: {task_instruction!r}")
    return valid_atoms
