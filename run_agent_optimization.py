#!/usr/bin/env python3
"""Prepare one agent-optimization retry for a failed registration sample."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from gemini_utils import DEFAULT_GEMINI_RETRIES, DEFAULT_GEMINI_TIMEOUT, DEFAULT_GEMINI_VISION_MODEL
from run_image_generation import (
    DEFAULT_GEMINI_IMAGE_MODEL,
    DEFAULT_GEMINI_IMAGE_TIMEOUT,
    GEOMETRY_REQUIREMENT,
    run_gemini_image,
    slugify,
)
from run_partfield_segmentation import run_gemini_image_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--attempt", default=1, type=int)
    parser.add_argument("--anchor", choices=["object_a", "object_b"])
    parser.add_argument("--gemini-vision-model", default=DEFAULT_GEMINI_VISION_MODEL)
    parser.add_argument("--gemini-image-model", default=DEFAULT_GEMINI_IMAGE_MODEL)
    parser.add_argument("--gemini-retries", default=DEFAULT_GEMINI_RETRIES, type=int)
    parser.add_argument("--gemini-timeout", default=DEFAULT_GEMINI_TIMEOUT, type=int)
    parser.add_argument("--image-timeout", default=DEFAULT_GEMINI_IMAGE_TIMEOUT, type=int)
    return parser.parse_args()


def load_metadata(input_dir: Path) -> dict[str, Any]:
    path = input_dir / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def source_sample_dir(input_dir: Path) -> Path:
    if input_dir.name.startswith("iter_") and input_dir.parent.name == "agent_optimization":
        return input_dir.parent.parent
    return input_dir


def retry_root(input_dir: Path) -> Path:
    if input_dir.name.startswith("iter_") and input_dir.parent.name == "agent_optimization":
        return input_dir.parent
    return input_dir / "agent_optimization"


def retry_slug(input_dir: Path) -> str:
    source = source_sample_dir(input_dir)
    return slugify(f"{source.parent.name}_{source.name}")[:80]


def next_output_dir(input_dir: Path, requested: Path | None, attempt: int) -> tuple[Path, int]:
    root = retry_root(input_dir)
    actual_attempt = attempt
    output_dir = requested or root / f"iter_{actual_attempt:02d}_{retry_slug(input_dir)}"
    while output_dir.exists():
        actual_attempt += 1
        output_dir = root / f"iter_{actual_attempt:02d}_{retry_slug(input_dir)}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir, actual_attempt


def object_label(metadata: dict[str, Any], key: str) -> str:
    return str(metadata.get("analysis", {}).get(key) or key).strip() or key


def compact_registration(registration: dict[str, Any]) -> dict[str, Any]:
    collision = registration.get("collision_refinement", {})
    return {
        "status": registration.get("status"),
        "anchor_object": registration.get("anchor_object"),
        "remaining_object": registration.get("remaining_object"),
        "penetration_check": registration.get("penetration_check", {}),
        "collision_final": collision.get("final", {}),
        "selected_candidate": collision.get("selected_candidate"),
        "contact_finalization": collision.get("contact_finalization", {}),
    }


def pick_objects(metadata: dict[str, Any], override: str | None) -> tuple[str, str]:
    registration = metadata.get("registration", {})
    anchor = override or registration.get("anchor_object") or "object_a"
    if anchor not in {"object_a", "object_b"}:
        anchor = "object_a"
    remain = "object_b" if anchor == "object_a" else "object_a"
    return anchor, remain


def retry_history(input_dir: Path) -> list[dict[str, Any]]:
    root = retry_root(input_dir)
    history = []
    if not root.exists():
        return history
    for path in sorted(root.glob("iter_*/metadata.json"))[-6:]:
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        reg = meta.get("registration", {})
        pc = reg.get("penetration_check", {})
        history.append(
            {
                "dir": path.parent.name,
                "status": reg.get("status"),
                "max_penetration_ratio": pc.get("max_penetration_ratio"),
                "violations": pc.get("violations"),
                "diagnosis": meta.get("agent_optimization", {}).get("diagnosis", ""),
            }
        )
    return history


def contact_policy(anchor_label: str, remain_label: str, attempt: int) -> str:
    shrink = min(45, 18 + max(attempt - 1, 0) * 9)
    lift = min(8, 2 + max(attempt - 1, 0) * 2)
    return f"""
Collision-prevention policy for this retry:
- Make the regenerated {remain_label} visibly smaller than the failed version, about {shrink}% less footprint/width if the previous attempt penetrated.
- Keep every part of the {remain_label} above the {anchor_label}'s support surface; do not allow hidden geometry below the contact plane.
- Use a simple, flat, explicit contact underside: small flat feet, a thin flat pad, or a narrow stable base. Avoid bulky side panels, thick plinths, rounded bottoms, downward protrusions, or invisible support geometry.
- Leave a tiny visible clearance/contact shadow of about {lift} cm in scene scale so the objects are separable after 3D reconstruction.
- Prefer simple rigid geometry with connected parts; avoid fragmented floating pieces, tangled thin wires at the contact zone, or broad area-contact that can sink into the anchor.
- If this is a table/shelf/flat-panel support, place the moving object near the center of the top surface, fully above the tabletop/panel, with the bottom silhouette completely visible.
""".strip()


def propose_prompts(args: argparse.Namespace, metadata: dict[str, Any], input_dir: Path, anchor_key: str, remain_key: str) -> dict[str, str]:
    anchor_label = object_label(metadata, anchor_key)
    remain_label = object_label(metadata, remain_key)
    scene_prompt = metadata.get("input", {}).get("scene_prompt", "")
    registration = compact_registration(metadata.get("registration", {}))
    history = retry_history(input_dir)
    policy = contact_policy(anchor_label, remain_label, args.attempt)
    images = [input_dir / "compositional_scene.png", input_dir / f"{anchor_key}.png", input_dir / f"{remain_key}.png"]
    missing = [str(path) for path in images if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing VLM reference images: {missing}")

    prompt = f"""
The previous Interact3D registration failed because final residual penetration was too high.
Your goal is not only to make a plausible picture, but to make a new 3D-reconstructable scene whose final meshes are separable and collision-free.

Original user prompt: {scene_prompt}
Fixed anchor object: {anchor_key} = {anchor_label}
Moving object to regenerate: {remain_key} = {remain_label}
Current failed registration JSON: {json.dumps(registration, ensure_ascii=False)}
Previous agent retry history JSON: {json.dumps(history, ensure_ascii=False)}
{policy}

Images are provided in this order:
1. previous compositional scene;
2. fixed anchor object image, which must be preserved exactly;
3. previous moving object image, which should be redesigned if it caused penetration.

Return only JSON with string keys:
- diagnosis: concise reason for the collision or geometry incompatibility, including what must change this time.
- scene_edit_prompt: image-edit instruction that uses the fixed anchor image as source and adds exactly one safer, smaller, compatible moving object.
- moving_edit_prompt: image-edit instruction that uses the new scene image as source, removes the anchor, and reconstructs a complete standalone moving object with no hidden downward protrusions.
- scene_slug: short lowercase retry slug.

Prompt requirements:
- Preserve only the anchor geometry, material, viewing direction, and scale impression.
- Do NOT preserve the failed moving object's scale or bottom/contact geometry if it caused penetration.
- Explicitly specify a smaller footprint, a flat visible underside/contact patch, and that the entire moving object sits above the support plane.
- Avoid vague words; specify placement, orientation, support/contact surface, clearance/shadow, and which parts must not intersect the anchor.
- Keep a pure white background, no text, no extra objects.
""".strip()
    data = run_gemini_image_json(
        args.gemini_vision_model,
        prompt,
        images,
        "agent prompt proposal",
        args.gemini_retries,
        args.gemini_timeout,
    )
    fallback_scene = (
        f"Edit the source image by preserving the {anchor_label} exactly and adding exactly one smaller {remain_label} "
        f"in the intended relationship: {scene_prompt}. The {remain_label} must sit fully above the {anchor_label} support surface, "
        "with a flat visible underside/contact patch, a small contact shadow, reduced footprint, and no hidden downward protrusions or intersections."
    )
    fallback_moving = (
        f"Remove the {anchor_label} completely and inpaint a complete standalone {remain_label}. "
        "Make the bottom/contact geometry simple, flat, fully visible, and slightly smaller than the failed version; "
        "do not include bulky plinths, broad side panels, rounded sunken bases, or hidden downward supports."
    )
    return {
        "diagnosis": str(data.get("diagnosis") or "Previous registration left excessive residual penetration."),
        "scene_edit_prompt": str(data.get("scene_edit_prompt") or data.get("scene_prompt") or fallback_scene),
        "moving_edit_prompt": str(data.get("moving_edit_prompt") or data.get("moving_object_prompt") or fallback_moving),
        "scene_slug": slugify(str(data.get("scene_slug") or f"agent_{anchor_label}_{remain_label}")),
        "contact_policy": policy,
        "retry_history": history,
    }


def copy_anchor_assets(input_dir: Path, output_dir: Path, anchor_key: str) -> None:
    for suffix in (".png", ".glb"):
        source = input_dir / f"{anchor_key}{suffix}"
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copy2(source, output_dir / source.name)


def cleanup_nonstandard_artifacts(output_dir: Path) -> None:
    keep = {
        "compositional_scene.png",
        "object_a.png",
        "object_b.png",
        "compositional_scene.glb",
        "object_a.glb",
        "object_b.glb",
        "metadata.json",
    }
    for path in output_dir.iterdir():
        if path.is_file() and path.suffix.lower() in {".png", ".glb"} and path.name not in keep:
            path.unlink(missing_ok=True)


def write_metadata(
    output_dir: Path,
    source_dir: Path,
    metadata: dict[str, Any],
    args: argparse.Namespace,
    anchor_key: str,
    remain_key: str,
    proposal: dict[str, str],
    scene_prompt: str,
    moving_prompt: str,
) -> None:
    analysis = dict(metadata.get("analysis", {}))
    analysis.setdefault(anchor_key, object_label(metadata, anchor_key))
    analysis.setdefault(remain_key, object_label(metadata, remain_key))
    analysis["scene_slug"] = proposal["scene_slug"]
    payload = {
        "models": {
            "gemini_vision": args.gemini_vision_model,
            "gemini_image": args.gemini_image_model,
        },
        "input": dict(metadata.get("input", {})),
        "analysis": analysis,
        "prompts": {
            "scene": scene_prompt,
            anchor_key: f"fixed anchor copied from {source_dir / f'{anchor_key}.png'}",
            remain_key: moving_prompt,
        },
        "outputs": {
            "compositional_scene": str(output_dir / "compositional_scene.png"),
            "object_a": str(output_dir / "object_a.png"),
            "object_b": str(output_dir / "object_b.png"),
        },
        "agent_optimization": {
            "source_dir": str(source_dir),
            "attempt": args.attempt,
            "anchor_object": anchor_key,
            "remaining_object": remain_key,
            "diagnosis": proposal["diagnosis"],
            "contact_policy": proposal.get("contact_policy", ""),
            "retry_history": proposal.get("retry_history", []),
            "previous_registration": compact_registration(metadata.get("registration", {})),
            "fixed_assets": {
                "image": str(output_dir / f"{anchor_key}.png"),
                "glb": str(output_dir / f"{anchor_key}.glb"),
            },
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    metadata = load_metadata(input_dir)
    anchor_key, remain_key = pick_objects(metadata, args.anchor)
    output_dir, actual_attempt = next_output_dir(input_dir, args.output_dir, args.attempt)
    output_dir = output_dir.resolve()
    args.attempt = actual_attempt

    proposal = propose_prompts(args, metadata, input_dir, anchor_key, remain_key)
    copy_anchor_assets(input_dir, output_dir, anchor_key)

    scene_prompt = f"{proposal['scene_edit_prompt']} {GEOMETRY_REQUIREMENT}"
    moving_prompt = f"{proposal['moving_edit_prompt']} {GEOMETRY_REQUIREMENT}"
    run_gemini_image(args, scene_prompt, output_dir / "compositional_scene.png", output_dir / f"{anchor_key}.png")
    run_gemini_image(args, moving_prompt, output_dir / f"{remain_key}.png", output_dir / "compositional_scene.png")
    cleanup_nonstandard_artifacts(output_dir)
    write_metadata(output_dir, input_dir, metadata, args, anchor_key, remain_key, proposal, scene_prompt, moving_prompt)

    print(json.dumps({"output_dir": str(output_dir), "anchor_object": anchor_key, "remaining_object": remain_key}, indent=2))
    print(output_dir)


if __name__ == "__main__":
    main()
