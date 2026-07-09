#!/usr/bin/env python3
"""Rebuild a multi-object registered scene while preserving the fixed anchor GLB visuals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import trimesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    return parser.parse_args()


def load_metadata(input_dir: Path) -> dict[str, Any]:
    path = input_dir / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def matrix(metadata: dict[str, Any], key: str) -> np.ndarray:
    return np.asarray(metadata["registration"]["transforms"][key]["matrix"], dtype=np.float64)


def transformed_meshes(path: Path, transform: np.ndarray) -> list[trimesh.Trimesh]:
    loaded = trimesh.load(path, force="scene")
    if isinstance(loaded, trimesh.Trimesh):
        meshes = [loaded]
    else:
        dumped = loaded.dump(concatenate=False)
        meshes = [dumped] if isinstance(dumped, trimesh.Trimesh) else list(dumped)
    output = []
    for mesh in meshes:
        if not isinstance(mesh, trimesh.Trimesh) or mesh.faces.size == 0:
            continue
        item = mesh.copy()
        item.apply_transform(transform)
        output.append(item)
    if not output:
        raise ValueError(f"No mesh geometry found in {path}")
    return output


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    metadata = load_metadata(input_dir)
    registration = metadata.get("registration", {})
    anchor = registration.get("anchor_object")
    remain = registration.get("remaining_object")
    if anchor not in {"object_a", "object_b"} or remain not in {"object_a", "object_b"}:
        raise ValueError("registration anchor/remaining object is missing")

    anchor_mat = matrix(metadata, anchor)
    remain_mat = matrix(metadata, remain)
    anchor_to_identity = np.linalg.inv(anchor_mat)
    relative_remain = anchor_to_identity @ remain_mat

    scene = trimesh.Scene()
    for index, mesh in enumerate(transformed_meshes(input_dir / f"{anchor}.glb", np.eye(4))):
        scene.add_geometry(mesh, node_name=f"{anchor}_{index}")
    for index, mesh in enumerate(transformed_meshes(input_dir / f"{remain}.glb", relative_remain)):
        scene.add_geometry(mesh, node_name=f"{remain}_{index}")

    scene_path = input_dir / "registered_scene.glb"
    scene.export(scene_path)
    metadata.setdefault("multiobject_finalize", {})
    metadata["multiobject_finalize"].update(
        {
            "preserved_anchor_object": anchor,
            "transformed_remaining_object": remain,
            "anchor_to_identity_matrix": anchor_to_identity.tolist(),
            "remaining_relative_matrix": relative_remain.tolist(),
            "output": str(scene_path),
        }
    )
    (input_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(scene_path)


if __name__ == "__main__":
    main()
