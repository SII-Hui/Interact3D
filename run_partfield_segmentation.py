#!/usr/bin/env python3
"""Segment the compositional mesh with PartField and build two guidance meshes."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from PIL import Image, ImageDraw

from gemini_utils import (
    DEFAULT_GEMINI_RETRIES,
    DEFAULT_GEMINI_TIMEOUT,
    DEFAULT_GEMINI_VISION_MODEL,
    generate_json,
)

DEFAULT_PARTFIELD_DIR = Path(os.getenv("INTERACT3D_PARTFIELD_DIR", "PartField"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--partfield-dir", default=DEFAULT_PARTFIELD_DIR, type=Path)
    parser.add_argument("--model-ckpt", default=None, type=Path)
    parser.add_argument("--gemini-vision-model", default=DEFAULT_GEMINI_VISION_MODEL)
    parser.add_argument("--gemini-retries", default=DEFAULT_GEMINI_RETRIES, type=int)
    parser.add_argument("--gemini-timeout", default=DEFAULT_GEMINI_TIMEOUT, type=int)
    parser.add_argument("--num-parts", default=7, type=int)
    parser.add_argument("--blender", default=os.getenv("BLENDER", "blender"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-intermediates", action="store_true")
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[partfield] {message}", file=sys.stderr, flush=True)


def run_gemini_image_json(
    model: str,
    prompt: str,
    image_paths: Path | list[Path],
    label: str,
    retries: int,
    timeout: int,
) -> dict[str, Any]:
    return generate_json(
        prompt=prompt,
        image_paths=image_paths,
        model=model,
        label=label,
        retries=retries,
        timeout=timeout,
    )


def load_metadata(input_dir: Path) -> dict[str, Any]:
    path = input_dir / "metadata.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def mesh_from_any(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="scene")
    if isinstance(loaded, trimesh.Trimesh):
        return loaded
    meshes = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
    if not meshes:
        raise ValueError(f"No mesh geometry found in {path}")
    return trimesh.util.concatenate(meshes)


def run_partfield(args: argparse.Namespace, work_dir: Path) -> tuple[Path, Path]:
    repo = args.partfield_dir.expanduser().resolve()
    ckpt = (args.model_ckpt or repo / "model" / "model_objaverse.ckpt").expanduser().resolve()
    if not (repo / "partfield_inference.py").exists():
        raise FileNotFoundError(f"PartField repo not found: {repo}")
    if not ckpt.exists():
        raise FileNotFoundError(f"PartField checkpoint not found: {ckpt}")

    source_dir = work_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.input_dir / "compositional_scene.glb", source_dir / "compositional_scene.glb")

    run_name = f"interact3d_{args.input_dir.name}"
    feature_root = repo / "exp_results" / "partfield_features" / run_name
    cluster_root = work_dir / "clustering"
    commands = [
        [
            sys.executable,
            "partfield_inference.py",
            "-c",
            "configs/final/demo.yaml",
            "--opts",
            "continue_ckpt",
            str(ckpt),
            "result_name",
            f"partfield_features/{run_name}",
            "dataset.data_path",
            str(source_dir),
            "preprocess_mesh",
            "True",
        ],
        [
            sys.executable,
            "run_part_clustering.py",
            "--root",
            str(feature_root),
            "--dump_dir",
            str(cluster_root),
            "--source_dir",
            str(source_dir),
            "--use_agglo",
            "True",
            "--max_num_clusters",
            str(args.num_parts),
            "--option",
            "1",
            "--with_knn",
            "True",
        ],
    ]
    for command in commands:
        subprocess.run(command, cwd=repo, check=True)
    return feature_root / "input_compositional_scene_0.ply", cluster_root / "cluster_out" / f"compositional_scene_0_{args.num_parts:02d}.npy"


def export_parts(mesh_path: Path, labels_path: Path, output_dir: Path) -> list[Path]:
    mesh = mesh_from_any(mesh_path)
    labels = np.load(labels_path).reshape(-1)
    if labels.shape[0] != len(mesh.faces):
        raise ValueError(f"Label count {labels.shape[0]} does not match face count {len(mesh.faces)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    part_paths = []
    for index, label in enumerate(sorted(np.unique(labels))):
        face_ids = np.flatnonzero(labels == label)
        part = mesh.submesh([face_ids], append=True, repair=False)
        part_path = output_dir / f"part_{index:02d}.glb"
        part.export(part_path)
        part_paths.append(part_path)
    return part_paths


def render_parts_with_blender(part_paths: list[Path], render_dir: Path, blender: str) -> list[Path]:
    render_dir.mkdir(parents=True, exist_ok=True)
    script = f"""
import bpy, math
from mathutils import Vector
parts = {json.dumps([str(p.resolve()) for p in part_paths])}
out_dir = {json.dumps(str(render_dir.resolve()))}
colors = [(0.9,0.2,0.2,1),(0.2,0.6,1,1),(0.2,0.8,0.3,1),(1,0.7,0.1,1),(0.7,0.3,1,1),(0.2,0.9,0.9,1),(1,0.4,0.7,1)]
bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.cycles.device = 'CPU'
bpy.context.scene.cycles.samples = 16
bpy.context.scene.cycles.use_denoising = False
bpy.context.scene.world.color = (1, 1, 1)
bpy.context.scene.view_settings.view_transform = 'Standard'
def clear():
    bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete()
def look_at(obj, target):
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
for i, path in enumerate(parts):
    clear()
    bpy.ops.import_scene.gltf(filepath=path)
    objs = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    mat = bpy.data.materials.new('part_mat'); mat.diffuse_color = colors[i % len(colors)]
    mat.use_nodes = True
    mat.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = colors[i % len(colors)]
    for o in objs: o.data.materials.append(mat)
    bpy.ops.object.light_add(type='AREA', location=(0, -3, 4)); bpy.context.object.data.energy = 500
    points = [o.matrix_world @ Vector(corner) for o in objs for corner in o.bound_box]
    center = sum(points, Vector()) / len(points)
    radius = max((p - center).length for p in points) or 1
    bpy.ops.object.camera_add(location=center + Vector((0, -2.8 * radius, 1.5 * radius)))
    bpy.context.scene.camera = bpy.context.object; look_at(bpy.context.object, center)
    bpy.context.scene.render.resolution_x = 512; bpy.context.scene.render.resolution_y = 512
    bpy.context.scene.render.filepath = f'{{out_dir}}/part_{{i:02d}}.png'
    bpy.ops.render.render(write_still=True)
"""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as handle:
        handle.write(script)
        script_path = handle.name
    try:
        subprocess.run([blender, "-b", "--factory-startup", "--python", script_path], check=True)
    finally:
        Path(script_path).unlink(missing_ok=True)
    return [render_dir / f"part_{i:02d}.png" for i in range(len(part_paths))]


def render_context_parts_with_blender(part_paths: list[Path], render_dir: Path, blender: str) -> list[Path]:
    render_dir.mkdir(parents=True, exist_ok=True)
    script = f"""
import bpy
from mathutils import Vector
parts = {json.dumps([str(p.resolve()) for p in part_paths])}
out_dir = {json.dumps(str(render_dir.resolve()))}
red = (1.0, 0.05, 0.02, 1.0)
gray = (0.78, 0.78, 0.78, 0.35)
bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.cycles.device = 'CPU'
bpy.context.scene.cycles.samples = 16
bpy.context.scene.cycles.use_denoising = False
bpy.context.scene.world.color = (1, 1, 1)
bpy.context.scene.view_settings.view_transform = 'Standard'
def clear():
    bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete()
def mat(name, color):
    m = bpy.data.materials.new(name); m.diffuse_color = color; m.use_nodes = True
    bsdf = m.node_tree.nodes['Principled BSDF']; bsdf.inputs['Base Color'].default_value = color; bsdf.inputs['Alpha'].default_value = color[3]
    m.blend_method = 'BLEND'; return m
def look_at(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat('-Z', 'Y').to_euler()
for selected in range(len(parts)):
    clear(); red_mat = mat('selected', red); gray_mat = mat('context', gray); objs = []
    for i, path in enumerate(parts):
        before = set(bpy.context.scene.objects)
        bpy.ops.import_scene.gltf(filepath=path)
        imported = [o for o in bpy.context.scene.objects if o not in before and o.type == 'MESH']
        for o in imported: o.data.materials.append(red_mat if i == selected else gray_mat)
        objs.extend(imported)
    bpy.ops.object.light_add(type='AREA', location=(0, -3, 4)); bpy.context.object.data.energy = 500
    points = [o.matrix_world @ Vector(corner) for o in objs for corner in o.bound_box]
    center = sum(points, Vector()) / len(points); radius = max((p - center).length for p in points) or 1
    bpy.ops.object.camera_add(location=center + Vector((0, -2.8 * radius, 1.5 * radius)))
    bpy.context.scene.camera = bpy.context.object; look_at(bpy.context.object, center)
    bpy.context.scene.render.resolution_x = 512; bpy.context.scene.render.resolution_y = 512
    bpy.context.scene.render.filepath = f'{{out_dir}}/context_part_{{selected:02d}}.png'
    bpy.ops.render.render(write_still=True)
"""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as handle:
        handle.write(script)
        script_path = handle.name
    try:
        subprocess.run([blender, "-b", "--factory-startup", "--python", script_path], check=True)
    finally:
        Path(script_path).unlink(missing_ok=True)
    return [render_dir / f"context_part_{i:02d}.png" for i in range(len(part_paths))]


def make_contact_sheet(images: list[Path], output_path: Path) -> Path:
    thumbs = [Image.open(path).convert("RGB") for path in images]
    w, h = thumbs[0].size
    sheet = Image.new("RGB", (4 * w, 2 * (h + 28)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, img in enumerate(thumbs):
        x, y = (i % 4) * w, (i // 4) * (h + 28)
        sheet.paste(img, (x, y + 28))
        draw.text((x + 8, y + 6), f"part_{i:02d}", fill=(0, 0, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return output_path


def classify_parts(
    args: argparse.Namespace,
    reference_images: list[Path],
    sheet: Path,
    context_sheet: Path,
    object_a: str,
    object_b: str,
    num_parts: int,
) -> dict[str, str]:
    prompt = f"""
You must assign each PartField segment to exactly one object.

Images are provided in this exact order:
1. compositional_scene: both objects together; use it to understand the intended spatial relationship.
2. object_a reference: {object_a}; use it for appearance and shape cues.
3. object_b reference: {object_b}; use it for appearance and shape cues.
4. isolated_parts_sheet: tiles labeled part_00 to part_{num_parts - 1:02d}; use it to inspect each segment's standalone geometry.
5. context_parts_sheet: same tile order and labels as image 4; in each tile, the target segment is red and all other scene geometry is light gray. Use it to locate the segment in the full compositional mesh.

Decision rules:
- Assign a part to the object it belongs to in the original compositional scene, not to the closest object in the tile.
- Treat the red geometry in image 5 as the only segment being classified for that tile; gray geometry is context only.
- Use image 5 for spatial location, images 2-3 for object identity, and image 1 to resolve object-object relationship.
- If isolated shape is ambiguous, prefer the contextual location in image 5 and the relationship in image 1.
- Ignore PartField fragmentation artifacts; several parts may belong to the same object.

Labels:
- object_a = {object_a}
- object_b = {object_b}

Return only one JSON object with exactly the keys part_00 through part_{num_parts - 1:02d}; each value must be either "object_a" or "object_b". Do not include markdown, explanations, confidence scores, or extra keys.
""".strip()
    assignment = run_gemini_image_json(
        args.gemini_vision_model,
        prompt,
        [*reference_images, sheet, context_sheet],
        "part grouping",
        args.gemini_retries,
        args.gemini_timeout,
    )
    expected = {f"part_{i:02d}" for i in range(num_parts)}
    if set(assignment) != expected or any(v not in {"object_a", "object_b"} for v in assignment.values()):
        raise ValueError(f"Invalid Gemini assignment: {assignment}")
    return {key: str(value) for key, value in assignment.items()}


def remove_small_components(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    components = list(mesh.split(only_watertight=False))
    if len(components) <= 1:
        return mesh, {"components_before": len(components), "components_after": len(components), "removed_faces": 0}

    areas = np.array([float(c.area) for c in components])
    faces = np.array([len(c.faces) for c in components])
    min_area = max(float(areas.sum()) * 0.001, float(areas.max()) * 0.005)
    min_faces = max(24, int(faces.sum() * 0.001))
    keep = (areas >= min_area) | (faces >= min_faces)
    if not keep.any():
        keep[int(np.argmax(areas))] = True

    kept = [component for component, flag in zip(components, keep) if flag]
    cleaned = trimesh.util.concatenate(kept) if len(kept) > 1 else kept[0]
    return cleaned, {
        "components_before": int(len(components)),
        "components_after": int(len(kept)),
        "removed_faces": int(faces[~keep].sum()),
    }


def merge_parts(part_paths: list[Path], assignment: dict[str, str], output_dir: Path) -> tuple[dict[str, str], dict[str, Any]]:
    outputs, cleanup = {}, {}
    for label in ("object_a", "object_b"):
        selected = [mesh_from_any(path) for path in part_paths if assignment[path.stem] == label]
        if not selected:
            raise ValueError(f"No PartField segments assigned to {label}")
        output_path = output_dir / f"guidance_{label}.glb"
        merged = trimesh.util.concatenate(selected)
        cleaned, stats = remove_small_components(merged)
        cleaned.export(output_path)
        outputs[label] = str(output_path)
        cleanup[label] = stats
    return outputs, cleanup


def update_metadata(input_dir: Path, payload: dict[str, Any]) -> None:
    path = input_dir / "metadata.json"
    metadata = load_metadata(input_dir)
    metadata["partfield_segmentation"] = payload
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def cleanup_intermediates(*dirs: Path) -> None:
    for directory in dirs:
        shutil.rmtree(directory, ignore_errors=True)


def main() -> None:
    args = parse_args()
    scene_glb = args.input_dir / "compositional_scene.glb"
    if not scene_glb.exists():
        raise FileNotFoundError(scene_glb)

    metadata = load_metadata(args.input_dir)
    analysis = metadata.get("analysis", {})
    object_a = analysis.get("object_a", "object_a")
    object_b = analysis.get("object_b", "object_b")
    work_dir = args.input_dir / "partfield_work"
    parts_dir = args.input_dir / "partfield_parts"
    render_dir = args.input_dir / "partfield_renders"

    part_paths = sorted(parts_dir.glob("part_*.glb")) if parts_dir.exists() and not args.overwrite else []
    if len(part_paths) != args.num_parts:
        shutil.rmtree(parts_dir, ignore_errors=True)
        feature_mesh, labels = run_partfield(args, work_dir)
        part_paths = export_parts(feature_mesh, labels, parts_dir)
    if len(part_paths) != args.num_parts:
        raise RuntimeError(f"Expected {args.num_parts} PartField parts, got {len(part_paths)}")

    render_paths = render_parts_with_blender(part_paths, render_dir, args.blender)
    context_paths = render_context_parts_with_blender(part_paths, render_dir, args.blender)
    sheet = make_contact_sheet(render_paths, render_dir / "contact_sheet.png")
    context_sheet = make_contact_sheet(context_paths, render_dir / "contact_sheet_context.png")
    reference_images = [args.input_dir / name for name in ("compositional_scene.png", "object_a.png", "object_b.png")]
    missing = [path for path in reference_images if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing reference images for part grouping: {missing}")
    assignment = classify_parts(args, reference_images, sheet, context_sheet, object_a, object_b, len(part_paths))
    guidance, component_cleanup = merge_parts(part_paths, assignment, args.input_dir)
    payload = {
        "input_mesh": str(scene_glb),
        "num_parts": len(part_paths),
        "object_labels": {"object_a": object_a, "object_b": object_b},
        "vlm_model": args.gemini_vision_model,
        "reference_images": [str(path) for path in reference_images],
        "assignment": assignment,
        "guidance_meshes": guidance,
        "component_cleanup": component_cleanup,
        "intermediates_kept": bool(args.keep_intermediates),
    }
    if args.keep_intermediates:
        payload.update(
            {
                "part_paths": [str(path) for path in part_paths],
                "render_paths": [str(path) for path in render_paths],
                "context_render_paths": [str(path) for path in context_paths],
                "contact_sheet": str(sheet),
                "context_contact_sheet": str(context_sheet),
            }
        )
    else:
        cleanup_intermediates(parts_dir, render_dir, work_dir)
    update_metadata(args.input_dir, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
