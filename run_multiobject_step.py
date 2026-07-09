#!/usr/bin/env python3
"""Prepare one sequential multi-object composition step."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from run_image_generation import (
    DEFAULT_GEMINI_IMAGE_MODEL,
    DEFAULT_GEMINI_IMAGE_TIMEOUT,
    DEFAULT_GEMINI_RETRIES,
    GEOMETRY_REQUIREMENT,
    run_gemini_image,
    slugify,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-glb", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--object-name", required=True)
    parser.add_argument("--scene-prompt", required=True)
    parser.add_argument("--anchor-label", default="existing composition")
    parser.add_argument("--blender", default=os.getenv("BLENDER", "blender"))
    parser.add_argument("--gemini-image-model", default=DEFAULT_GEMINI_IMAGE_MODEL)
    parser.add_argument("--gemini-retries", default=DEFAULT_GEMINI_RETRIES, type=int)
    parser.add_argument("--image-timeout", default=DEFAULT_GEMINI_IMAGE_TIMEOUT, type=int)
    return parser.parse_args()


def render_glb(glb_path: Path, output_png: Path, blender: str) -> None:
    script = f"""
import bpy
from mathutils import Vector
bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete()
bpy.ops.import_scene.gltf(filepath={json.dumps(str(glb_path.resolve()))})
objs = [o for o in bpy.context.scene.objects if o.type == 'MESH']
if not objs:
    raise RuntimeError('no mesh objects imported')
bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.cycles.device = 'CPU'
bpy.context.scene.cycles.samples = 32
bpy.context.scene.world.color = (1, 1, 1)
bpy.context.scene.view_settings.view_transform = 'Standard'
points = [o.matrix_world @ Vector(corner) for o in objs for corner in o.bound_box]
center = sum(points, Vector()) / len(points)
radius = max((p - center).length for p in points) or 1.0
bpy.ops.object.light_add(type='AREA', location=center + Vector((0, -3 * radius, 4 * radius)))
bpy.context.object.data.energy = 650
bpy.ops.object.camera_add(location=center + Vector((0, -3.0 * radius, 1.6 * radius)))
camera = bpy.context.object
camera.rotation_euler = (center - camera.location).to_track_quat('-Z', 'Y').to_euler()
bpy.context.scene.camera = camera
bpy.context.scene.render.resolution_x = 1024
bpy.context.scene.render.resolution_y = 1024
bpy.context.scene.render.filepath = {json.dumps(str(output_png.resolve()))}
bpy.ops.render.render(write_still=True)
"""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as handle:
        handle.write(script)
        script_path = handle.name
    try:
        subprocess.run([blender, "-b", "--factory-startup", "--python", script_path], check=True)
    finally:
        Path(script_path).unlink(missing_ok=True)


def cleanup_nonstandard_artifacts(output_dir: Path) -> None:
    keep = {"compositional_scene.png", "object_a.png", "object_b.png", "object_a.glb", "metadata.json"}
    for path in output_dir.iterdir():
        if path.is_file() and path.suffix.lower() in {".png", ".glb"} and path.name not in keep:
            path.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    anchor_glb = args.anchor_glb.expanduser().resolve()
    if not anchor_glb.exists():
        raise FileNotFoundError(anchor_glb)

    paths = {
        "anchor_glb": output_dir / "object_a.glb",
        "anchor_png": output_dir / "object_a.png",
        "scene_png": output_dir / "compositional_scene.png",
        "object_png": output_dir / "object_b.png",
    }
    shutil.copy2(anchor_glb, paths["anchor_glb"])
    render_glb(paths["anchor_glb"], paths["anchor_png"], args.blender)

    scene_prompt = (
        f"Use the source image as the fixed anchor: {args.anchor_label}. Preserve the anchor geometry, pose, material, scale, and camera view exactly. "
        f"Add exactly one new object: {args.object_name}. Scene requirement: {args.scene_prompt}. "
        "The new object must have a simple visible contact region, no hidden downward protrusions, no intersection with the anchor, and a small contact shadow/clearance. "
        f"{GEOMETRY_REQUIREMENT}"
    )
    object_prompt = (
        f"Use the new scene image as source. Remove the entire anchor composition ({args.anchor_label}) and reconstruct only one complete standalone {args.object_name}. "
        "Keep the added object un-cropped, centered, complete, with flat/simple contact geometry and no extra objects. "
        f"{GEOMETRY_REQUIREMENT}"
    )
    run_gemini_image(args, scene_prompt, paths["scene_png"], paths["anchor_png"])
    run_gemini_image(args, object_prompt, paths["object_png"], paths["scene_png"])
    cleanup_nonstandard_artifacts(output_dir)

    metadata = {
        "models": {
            "gemini_image": args.gemini_image_model,
        },
        "input": {"scene_prompt": args.scene_prompt, "anchor_glb": str(anchor_glb)},
        "analysis": {
            "category": "multiobject",
            "object_a": args.anchor_label,
            "object_b": args.object_name,
            "scene_slug": slugify(f"{args.anchor_label}_{args.object_name}"),
            "category_source": "multiobject_step",
            "category_folder": "multiobject",
        },
        "prompts": {"scene": scene_prompt, "object_a": "rendered fixed anchor composition", "object_b": object_prompt},
        "outputs": {
            "compositional_scene": str(paths["scene_png"]),
            "object_a": str(paths["anchor_png"]),
            "object_b": str(paths["object_png"]),
        },
        "multiobject_step": {
            "anchor_label": args.anchor_label,
            "added_object": args.object_name,
            "anchor_glb": str(paths["anchor_glb"]),
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(output_dir)


if __name__ == "__main__":
    main()
