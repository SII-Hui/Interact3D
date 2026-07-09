#!/usr/bin/env python3
"""Run local TRELLIS.2 image-to-3D generation for all PNG files in a directory."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from PIL import Image

DEFAULT_MODEL = os.getenv("INTERACT3D_TRELLIS2_MODEL", "microsoft/TRELLIS.2-4B")
DEFAULT_TRELLIS2_DIR = Path(os.getenv("INTERACT3D_TRELLIS2_DIR", "TRELLIS.2"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--trellis2-dir", default=DEFAULT_TRELLIS2_DIR, type=Path)
    parser.add_argument("--decimation-target", type=int, default=200_000)
    parser.add_argument("--texture-size", type=int, default=4096)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def add_trellis2_to_path(trellis2_dir: Path) -> Path:
    repo_dir = trellis2_dir.expanduser().resolve()
    if not (repo_dir / "trellis2").is_dir():
        raise ModuleNotFoundError(
            f"Cannot find TRELLIS.2 package under {repo_dir}. "
            "Clone TRELLIS.2 there or pass --trellis2-dir /path/to/TRELLIS.2."
        )
    sys.path.insert(0, str(repo_dir))
    return repo_dir


def load_pipeline(model: str, trellis2_dir: Path):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    repo_dir = add_trellis2_to_path(trellis2_dir)
    try:
        from trellis2.pipelines import Trellis2ImageTo3DPipeline
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Failed to import TRELLIS.2. Run its setup script in the current conda environment first."
        ) from error

    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(model)
    pipeline.cuda()
    return pipeline, repo_dir


def export_glb(mesh: Any, output_path: Path, decimation_target: int, texture_size: int) -> None:
    import o_voxel

    mesh.simplify(16_777_216)
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=decimation_target,
        texture_size=texture_size,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        verbose=True,
    )
    # Prefer WebP textures (smaller files), but fall back to the default PNG
    # textures if the local Pillow build lacks full WebP support (e.g.
    # "module 'PIL._webp' has no attribute 'HAVE_WEBPANIM'").
    try:
        glb.export(str(output_path), extension_webp=True)
    except (AttributeError, ImportError, ValueError) as error:
        print(
            f"[export_glb] WebP texture export failed ({error}); "
            "falling back to PNG textures.",
            file=sys.stderr,
        )
        glb.export(str(output_path))


def update_metadata(input_dir: Path, payload: dict[str, Any]) -> None:
    path = input_dir / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    metadata["trellis2_generation"] = payload
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.input_dir.is_dir():
        raise NotADirectoryError(args.input_dir)

    images = sorted(args.input_dir.glob("*.png"))
    if not images:
        raise FileNotFoundError(f"No PNG files found in {args.input_dir}")

    pipeline, repo_dir = load_pipeline(args.model, args.trellis2_dir)
    generated, skipped = {}, []

    for image_path in images:
        output_path = image_path.with_suffix(".glb")
        if output_path.exists() and not args.overwrite:
            skipped.append(str(output_path))
            continue
        image = Image.open(image_path).convert("RGBA")
        mesh = pipeline.run(image)[0]
        export_glb(mesh, output_path, args.decimation_target, args.texture_size)
        generated[str(image_path)] = str(output_path)

    update_metadata(
        args.input_dir,
        {
            "model": args.model,
            "trellis2_dir": str(repo_dir),
            "input_dir": str(args.input_dir),
            "decimation_target": args.decimation_target,
            "texture_size": args.texture_size,
            "generated": generated,
            "skipped": skipped,
        },
    )
    print(json.dumps({"generated": generated, "skipped": skipped}, indent=2))


if __name__ == "__main__":
    main()
