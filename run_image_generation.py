#!/usr/bin/env python3
"""Generate a compositional scene image and isolated object images."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from gemini_utils import (
    DEFAULT_GEMINI_IMAGE_MODEL,
    DEFAULT_GEMINI_IMAGE_TIMEOUT,
    DEFAULT_GEMINI_RETRIES,
    DEFAULT_GEMINI_TEXT_MODEL,
    DEFAULT_GEMINI_TIMEOUT,
    generate_image,
    generate_json,
    log,
    validate_png,
)

CATEGORIES = {
    "stationery": "stationery",
    "kitchenware": "kitchenware",
    "food": "food",
    "household items": "household_items",
    "electronics": "electronics",
    "apparel": "apparel",
    "decor": "decor",
    "toys": "toys",
    "outdoortools": "outdoortools",
}
DEFAULT_SCENE_PROMPT = "A wooden chair with a pink cushion on it."
ANALYSIS_FAILURE_EXIT_CODE = int(os.getenv("INTERACT3D_ANALYSIS_FAILURE_EXIT_CODE", "42"))
GEOMETRY_REQUIREMENT = "Complete object on a pure white background (#FFFFFF), 1K resolution, centered and fully visible with no cropping. Reveal full geometry including occluded and hidden parts. Even soft lighting, no harsh shadows or reflections, no text or extra objects."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-prompt", default=DEFAULT_SCENE_PROMPT)
    parser.add_argument("--outputs-root", default="outputs")
    # Category for output-folder classification. When given, it is used directly
    # (no LLM category detection); when omitted, the category is inferred by the
    # model. Objects and scene slug are always inferred by the model.
    parser.add_argument("--category", default=None)
    parser.add_argument("--gemini-text-model", default=DEFAULT_GEMINI_TEXT_MODEL)
    parser.add_argument("--gemini-image-model", default=DEFAULT_GEMINI_IMAGE_MODEL)
    parser.add_argument("--gemini-retries", default=DEFAULT_GEMINI_RETRIES, type=int)
    parser.add_argument("--gemini-timeout", default=DEFAULT_GEMINI_TIMEOUT, type=int)
    parser.add_argument("--image-timeout", default=DEFAULT_GEMINI_IMAGE_TIMEOUT, type=int)
    return parser.parse_args()


def run_gemini_json(args: argparse.Namespace, prompt: str, schema: dict[str, Any], label: str) -> dict[str, Any]:
    return generate_json(
        prompt=prompt,
        schema=schema,
        model=args.gemini_text_model,
        label=label,
        retries=args.gemini_retries,
        timeout=args.gemini_timeout,
    )


def with_geometry_requirement(prompt: str) -> str:
    return prompt if GEOMETRY_REQUIREMENT in prompt else f"{prompt.strip()} {GEOMETRY_REQUIREMENT}"


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "composition"


def ensure_category_dirs(root: Path) -> None:
    for folder in CATEGORIES.values():
        (root / folder).mkdir(parents=True, exist_ok=True)


def normalize_category(category: str) -> str:
    key = category.lower().replace("_", " ").strip()
    return CATEGORIES.get(key, CATEGORIES.get(key.replace(" ", ""), "household_items"))


def next_output_dir(root: Path, category_folder: str, scene_slug: str) -> Path:
    category_dir = root / category_folder
    max_id = 0
    for path in category_dir.iterdir():
        if path.is_dir() and re.match(r"^\d{5}_", path.name):
            max_id = max(max_id, int(path.name[:5]))
    output_dir = category_dir / f"{max_id + 1:05d}_{slugify(scene_slug)}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def is_placeholder_object(value: Any) -> bool:
    text = str(value or "").strip().lower().replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text)
    return not text or bool(re.fullmatch(r"object(?:\s+[a-z])?", text)) or text in {"unknown", "none", "n/a", "placeholder"}


def prompt_analysis_failure_reasons(info: dict[str, str]) -> list[str]:
    reasons = []
    for key in ("object_a", "object_b"):
        if is_placeholder_object(info.get(key, "")):
            reasons.append(f"{key}={info.get(key, '')!r}")
    if slugify(str(info.get("scene_slug", ""))) in {"object_a_object_b", "object_b_object_a", "object_a", "object_b", "object"}:
        reasons.append(f"scene_slug={info.get('scene_slug', '')!r}")
    return reasons


def analyze_prompt(args: argparse.Namespace, scene_prompt: str) -> dict[str, str]:
    categories = ", ".join(CATEGORIES)
    prompt = f"""
You are extracting the two main physical objects from a compositional 3D generation prompt so each can be reconstructed independently. Return only JSON.
Prompt: {scene_prompt}
Categories: {categories}

How to identify object_a and object_b:
- They must be two concrete, tangible, countable physical objects that can each stand alone as a 3D model.
- Ignore verbs/actions (e.g. "being inserted", "placed on"), spatial words, and abstract, figurative, or marketing phrases (e.g. "that attracts wealth", "of happiness"). Extract the underlying physical object only.
- Strip leading articles and adjectives that do not change the object's identity; output a short singular noun phrase (lowercase).
- If the prompt truly contains only one object, set object_b to the most reasonable companion object implied by the scene; never leave a value as a placeholder like "object a".

Examples:
- Prompt "A coin is being inserted into a piggy bank that attracts wealth." -> object_a: "coin", object_b: "piggy bank"
- Prompt "A steaming cup of coffee next to a croissant." -> object_a: "coffee cup", object_b: "croissant"
- Prompt "A wooden chair with a pink cushion on it." -> object_a: "wooden chair", object_b: "cushion"

Required keys:
- category: exactly one category from the list
- object_a: the first main physical object (concise lowercase noun phrase)
- object_b: the second main physical object (concise lowercase noun phrase)
- scene_slug: a short lowercase English slug describing both objects
""".strip()
    schema = {
        "type": "object",
        "properties": {
            "category": {"type": "string", "enum": list(CATEGORIES)},
            "object_a": {"type": "string", "description": "First concrete physical object, concise lowercase noun phrase, no articles/actions/abstract words."},
            "object_b": {"type": "string", "description": "Second concrete physical object, concise lowercase noun phrase, no articles/actions/abstract words."},
            "scene_slug": {"type": "string", "description": "Short lowercase slug combining both objects, e.g. coin_piggy_bank."},
        },
        "required": ["category", "object_a", "object_b", "scene_slug"],
    }
    data = run_gemini_json(args, prompt, schema, "prompt analysis")

    def clean_object(value: Any, fallback: str) -> str:
        text = re.sub(r"^\s*(a|an|the)\s+", "", str(value or "").strip().lower())
        # Reject empty or placeholder-like outputs such as "object a".
        if not text or re.fullmatch(r"object[\s_]*[a-z]?", text):
            return fallback
        return text

    object_a = clean_object(data.get("object_a"), "object a")
    object_b = clean_object(data.get("object_b"), "object b")
    return {
        "category": str(data.get("category", "household items")),
        "object_a": object_a,
        "object_b": object_b,
        "scene_slug": slugify(str(data.get("scene_slug") or f"{object_a}_{object_b}")),
    }


def edit_prompt(target: str, removed: str) -> str:
    prompt = (
        f"Remove the {removed} from the image and inpaint the {target}. "
        "Keep everything else unchanged, white background, 1K resolution."
    )
    # return with_geometry_requirement(prompt)
    return prompt


def run_gemini_image(args: argparse.Namespace, prompt: str, output_path: Path, source: Path | None = None) -> None:
    generate_image(
        prompt=prompt,
        output_path=output_path,
        source=source,
        model=args.gemini_image_model,
        label="image edit" if source else "text-to-image",
        retries=args.gemini_retries,
        timeout=args.image_timeout,
    )
    validate_png(output_path)


def main() -> None:
    args = parse_args()
    outputs_root = Path(args.outputs_root)
    ensure_category_dirs(outputs_root)

    info = analyze_prompt(args, args.scene_prompt)
    failure_reasons = prompt_analysis_failure_reasons(info)
    if failure_reasons:
        log(f"prompt analysis failed; skipping sample: {', '.join(failure_reasons)}; analysis={json.dumps(info, ensure_ascii=False)}")
        sys.exit(ANALYSIS_FAILURE_EXIT_CODE)

    # Prefer the user-provided category; fall back to the model's guess.
    category = args.category or info["category"]
    category_source = "provided" if args.category else "llm"
    category_folder = normalize_category(category)
    output_dir = next_output_dir(outputs_root, category_folder, info["scene_slug"])
    paths = {
        "compositional_scene": output_dir / "compositional_scene.png",
        "object_a": output_dir / "object_a.png",
        "object_b": output_dir / "object_b.png",
    }
    prompts = {
        "scene": with_geometry_requirement(args.scene_prompt),
        "object_a": edit_prompt(info["object_a"], info["object_b"]),
        "object_b": edit_prompt(info["object_b"], info["object_a"]),
    }
    run_gemini_image(args, prompts["scene"], paths["compositional_scene"])
    run_gemini_image(args, prompts["object_a"], paths["object_a"], paths["compositional_scene"])
    run_gemini_image(args, prompts["object_b"], paths["object_b"], paths["compositional_scene"])

    metadata = {
        "models": {
            "gemini_text": args.gemini_text_model,
            "gemini_image": args.gemini_image_model,
        },
        "input": {"scene_prompt": args.scene_prompt, "category": args.category},
        "analysis": {**info, "category": category, "category_source": category_source, "category_folder": category_folder},
        "prompts": prompts,
        "outputs": {name: str(path) for name, path in paths.items()},
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata["outputs"], indent=2))
    # Final stdout line: the created directory, for batch drivers to capture.
    print(output_dir)


if __name__ == "__main__":
    main()
