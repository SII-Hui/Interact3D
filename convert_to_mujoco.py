#!/usr/bin/env python3
"""Convert an Interact3D sample folder to a MuJoCo MJCF scene with preserved visual materials.

By default this creates an interaction-only scene: zero gravity, no floor, and no
contacts, so all free bodies stay still until the user manipulates them.
"""

from __future__ import annotations

import argparse
import html
import json
import shutil
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from PIL import Image
from trimesh.exchange import obj as obj_exporter

DEFAULT_SAMPLE_DIR = Path("outputs/food/00001_lemon_picnic_basket")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=DEFAULT_SAMPLE_DIR, type=Path)
    parser.add_argument("--dynamic-object", choices=["object_a", "object_b", "both", "none"], default="both")
    parser.add_argument("--mass", default=0.3, type=float)
    parser.add_argument("--gravity", nargs=3, type=float, default=(0.0, 0.0, 0.0), metavar=("GX", "GY", "GZ"))
    parser.add_argument("--include-floor", action="store_true")
    parser.add_argument("--enable-contact", action="store_true")
    return parser.parse_args()


def fmt_vec(values: np.ndarray | list[float]) -> str:
    return " ".join(f"{float(v):.10g}" for v in values)


def xml_attr(value: str) -> str:
    return html.escape(value, quote=True)


def load_visual_meshes(path: Path) -> list[trimesh.Trimesh]:
    loaded = trimesh.load(path, force="scene")
    if isinstance(loaded, trimesh.Trimesh):
        meshes = [loaded]
    else:
        dumped = loaded.dump(concatenate=False)
        meshes = [dumped] if isinstance(dumped, trimesh.Trimesh) else list(dumped)
    result = []
    for mesh in meshes:
        if isinstance(mesh, trimesh.Trimesh) and mesh.faces.size:
            result.append(mesh.copy())
    if not result:
        raise ValueError(f"No mesh geometry found in {path}")
    return result


def material_rgba(mesh: trimesh.Trimesh) -> str:
    visual = getattr(mesh, "visual", None)
    material = getattr(visual, "material", None)
    color = getattr(material, "baseColorFactor", None)
    if color is None and hasattr(visual, "vertex_colors") and len(visual.vertex_colors):
        color = np.asarray(visual.vertex_colors, dtype=np.float64).mean(axis=0)
    if color is None:
        return "1 1 1 1"
    arr = np.asarray(color, dtype=np.float64).reshape(-1)
    if arr.size == 3:
        arr = np.concatenate([arr, [255.0 if arr.max(initial=0) > 1.0 else 1.0]])
    arr = arr[:4]
    if arr.max(initial=0) > 1.0:
        arr = arr / 255.0
    arr = np.clip(arr, 0.0, 1.0)
    return fmt_vec(arr)


def unique_obj_export(mesh: trimesh.Trimesh, mesh_dir: Path, asset_name: str) -> dict[str, Any]:
    """Export one visual mesh as OBJ without sharing material/texture filenames."""
    obj_text, resources = obj_exporter.export_obj(
        mesh,
        include_normals=True,
        include_color=True,
        include_texture=True,
        return_texture=True,
        write_texture=False,
        mtl_name=f"{asset_name}.mtl",
    )

    texture_file: str | None = None
    mtl_name = f"{asset_name}.mtl"
    mtl_bytes = resources.get(mtl_name)
    if mtl_bytes is None:
        mtl_candidates = [k for k in resources if k.lower().endswith(".mtl")]
        if mtl_candidates:
            mtl_name = mtl_candidates[0]
            mtl_bytes = resources[mtl_name]

    replacement_names: dict[str, str] = {}
    for original_name, data in resources.items():
        if original_name.lower().endswith(".mtl"):
            continue
        suffix = Path(original_name).suffix.lower()
        if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
            png_name = f"{asset_name}_{Path(original_name).stem}.png"
            Image.open(BytesIO(data)).convert("RGBA").save(mesh_dir / png_name)
            replacement_names[original_name] = png_name
            texture_file = png_name
        else:
            out_name = f"{asset_name}_{Path(original_name).name}"
            (mesh_dir / out_name).write_bytes(data)
            replacement_names[original_name] = out_name

    if mtl_bytes is not None:
        mtl_text = mtl_bytes.decode("utf-8", errors="replace")
        for old, new in replacement_names.items():
            mtl_text = mtl_text.replace(old, new)
        # Keep the OBJ-side material file too. MuJoCo materials are declared
        # explicitly below, but this preserves a faithful standalone OBJ asset.
        (mesh_dir / f"{asset_name}.mtl").write_text(mtl_text, encoding="utf-8")

    (mesh_dir / f"{asset_name}.obj").write_text(obj_text, encoding="utf-8")
    return {
        "asset_name": asset_name,
        "obj_file": f"{asset_name}.obj",
        "texture_file": texture_file,
        "rgba": material_rgba(mesh),
    }


def transformed_parts(src_dir: Path, name: str, transform: np.ndarray) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    meshes = load_visual_meshes(src_dir / f"{name}.glb")
    for mesh in meshes:
        mesh.apply_transform(transform)
    bounds = np.array([mesh.bounds for mesh in meshes])
    center = np.array([bounds[:, 0, :].min(axis=0), bounds[:, 1, :].max(axis=0)]).mean(axis=0)
    for mesh in meshes:
        mesh.apply_translation(-center)
    return meshes, center


def main() -> None:
    args = parse_args()
    src_dir = args.input_dir.expanduser().resolve()
    out_dir = src_dir / "mujoco"
    mesh_dir = out_dir / "meshes"
    if mesh_dir.exists():
        shutil.rmtree(mesh_dir)
    mesh_dir.mkdir(parents=True, exist_ok=True)

    meta = json.loads((src_dir / "metadata.json").read_text(encoding="utf-8"))
    transforms = meta["registration"]["transforms"]

    centers: dict[str, np.ndarray] = {}
    exported: dict[str, list[dict[str, Any]]] = {}
    for object_name in ["object_a", "object_b"]:
        mat = np.asarray(transforms[object_name]["matrix"], dtype=np.float64)
        meshes, center = transformed_parts(src_dir, object_name, mat)
        centers[object_name] = center
        exported[object_name] = [
            unique_obj_export(mesh, mesh_dir, f"{object_name}_{index:02d}")
            for index, mesh in enumerate(meshes)
        ]

    asset_lines = []
    for parts in exported.values():
        for part in parts:
            if part["texture_file"]:
                asset_lines.append(f'    <texture name="{xml_attr(part["asset_name"])}_tex" type="2d" file="{xml_attr(part["texture_file"])}"/>')
                asset_lines.append(f'    <material name="{xml_attr(part["asset_name"])}_mat" texture="{xml_attr(part["asset_name"])}_tex" rgba="1 1 1 1" specular="0.25" shininess="0.5"/>')
            else:
                asset_lines.append(f'    <material name="{xml_attr(part["asset_name"])}_mat" rgba="{xml_attr(part["rgba"])}" specular="0.25" shininess="0.5"/>')
            asset_lines.append(f'    <mesh name="{xml_attr(part["asset_name"])}_mesh" file="{xml_attr(part["obj_file"])}"/>')

    def body_xml(object_name: str, label: str) -> str:
        is_dynamic = args.dynamic_object == "both" or args.dynamic_object == object_name
        lines = [f'    <body name="{xml_attr(label)}" pos="{fmt_vec(centers[object_name])}">']
        if is_dynamic:
            lines.append("      <freejoint/>")
        mass = max(float(args.mass), 1e-8) / max(len(exported[object_name]), 1)
        for part in exported[object_name]:
            attrs = [
                f'type="mesh"',
                f'mesh="{xml_attr(part["asset_name"])}_mesh"',
                f'material="{xml_attr(part["asset_name"])}_mat"',
                'friction="1 0.005 0.0001"',
            ]
            if not args.enable_contact:
                attrs.extend(['contype="0"', 'conaffinity="0"'])
            if is_dynamic:
                attrs.append(f'mass="{mass:.10g}"')
            lines.append(f"      <geom {' '.join(attrs)}/>")
        lines.append("    </body>")
        return "\n".join(lines)

    floor_xml = (
        '    <geom name="floor" type="plane" size="5 5 0.05" rgba="0.8 0.8 0.8 1" contype="0" conaffinity="0"/>\n'
        if args.include_floor and not args.enable_contact
        else '    <geom name="floor" type="plane" size="5 5 0.05" rgba="0.8 0.8 0.8 1"/>\n'
        if args.include_floor
        else ""
    )

    xml = f"""<mujoco model="interact3d_scene">
  <compiler angle="radian" meshdir="meshes" texturedir="meshes" autolimits="true"/>
  <option timestep="0.002" gravity="{fmt_vec(args.gravity)}"/>

  <asset>
{chr(10).join(asset_lines)}
  </asset>

  <worldbody>
    <light pos="0 -3 4" dir="0 0 -1"/>
{floor_xml}
{body_xml('object_b', 'object_b_table')}

{body_xml('object_a', 'object_a_salt_lamp')}
  </worldbody>
</mujoco>
"""

    out_dir.mkdir(parents=True, exist_ok=True)
    scene_path = out_dir / "scene.xml"
    scene_path.write_text(xml, encoding="utf-8")
    print(scene_path)


if __name__ == "__main__":
    main()
