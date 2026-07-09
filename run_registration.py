#!/usr/bin/env python3
"""Register generated objects to guidance meshes and compose a collision-aware scene."""

from __future__ import annotations

import argparse
import importlib
import gc
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from scipy.spatial import cKDTree

DEFAULT_GEOT_DIR = Path(os.getenv("INTERACT3D_GEOTRANSFORMER_DIR", "GeoTransformer"))
DEFAULT_GEOT_WEIGHTS = DEFAULT_GEOT_DIR / "weights" / "geotransformer-3dmatch.pth.tar"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--geotransformer-dir", default=DEFAULT_GEOT_DIR, type=Path)
    parser.add_argument("--geotransformer-weights", default=DEFAULT_GEOT_WEIGHTS, type=Path)
    parser.add_argument("--samples", default=4000, type=int)
    parser.add_argument("--geotransformer-samples", default=1024, type=int)
    parser.add_argument("--collision-samples", default=8000, type=int)
    parser.add_argument("--clearance", default=0, type=float)
    parser.add_argument("--anchor", choices=["auto", "object_a", "object_b"], default="auto")
    # Penetration severity gate for the stage-5 agentic refinement. If the
    # deepest residual penetration exceeds this fraction of the object scale,
    # the registration is flagged as a "failure" so the downstream VLM agent
    # can attempt a corrective re-composition.
    parser.add_argument("--failure-penetration-ratio", default=0.01, type=float)
    parser.add_argument("--no-geotransformer", action="store_true")
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[registration] {message}", file=sys.stderr, flush=True)


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh) or mesh.faces.size == 0:
        raise ValueError(f"Invalid mesh: {path}")
    return mesh


def sample_points(mesh: trimesh.Trimesh, count: int) -> np.ndarray:
    count = min(count, max(len(mesh.faces) * 8, 512))
    return trimesh.sample.sample_surface(mesh, count)[0].astype(np.float32)


def umeyama(src: np.ndarray, ref: np.ndarray, scale_bounds: tuple[float, float] | None = None) -> np.ndarray:
    """Least-squares similarity (scale + rotation + translation) aligning src onto ref."""
    src_mu, ref_mu = src.mean(0), ref.mean(0)
    src_c, ref_c = src - src_mu, ref - ref_mu
    cov = ref_c.T @ src_c / len(src)
    u, d, vt = np.linalg.svd(cov)
    sign = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        sign[-1, -1] = -1
    rot = u @ sign @ vt
    scale = float(np.trace(np.diag(d) @ sign) / (np.sum(src_c**2) / len(src)))
    if scale_bounds is not None:
        scale = float(np.clip(scale, scale_bounds[0], scale_bounds[1]))
    mat = np.eye(4)
    mat[:3, :3] = scale * rot
    mat[:3, 3] = ref_mu - scale * rot @ src_mu
    return mat


def apply(points: np.ndarray, mat: np.ndarray) -> np.ndarray:
    return points @ mat[:3, :3].T + mat[:3, 3]


def transform_scale(mat: np.ndarray) -> float:
    return float(np.cbrt(abs(np.linalg.det(mat[:3, :3]))))


def rotation_angle(mat: np.ndarray) -> float:
    scale = max(transform_scale(mat), 1e-8)
    rot = mat[:3, :3] / scale
    return float(np.arccos(np.clip((np.trace(rot) - 1.0) * 0.5, -1.0, 1.0)))


def obb_scale(src: trimesh.Trimesh, ref: trimesh.Trimesh) -> float:
    src_ext = np.maximum(src.bounding_box_oriented.extents, 1e-6)
    ref_ext = np.maximum(ref.bounding_box_oriented.extents, 1e-6)
    return float(np.median(np.sort(ref_ext) / np.sort(src_ext)))


def take_subset(points: np.ndarray, count: int) -> np.ndarray:
    if len(points) <= count:
        return points
    return points[np.linspace(0, len(points) - 1, count, dtype=np.int64)]


def icp_similarity(src: np.ndarray, ref: np.ndarray, init: np.ndarray, steps: int = 30, scale_bounds: tuple[float, float] | None = None) -> np.ndarray:
    mat, tree = init.copy(), cKDTree(ref)
    last = np.inf
    for _ in range(steps):
        moved = apply(src, mat)
        dist, idx = tree.query(moved, k=1)
        keep = dist <= np.percentile(dist, 80)
        if keep.sum() < 16:
            break
        mat = umeyama(src[keep], ref[idx[keep]], scale_bounds=scale_bounds)
        err = float(dist[keep].mean())
        if abs(last - err) < 1e-6:
            break
        last = err
    return mat


class GeoTransformerRunner:
    def __init__(self, repo: Path, weights: Path):
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        self.repo = repo.expanduser().resolve()
        self.weights = weights.expanduser().resolve()
        exp = self.repo / "experiments" / "geotransformer.3dmatch.stage4.gse.k3.max.oacl.stage2.sinkhorn"
        if not self.weights.exists() or not exp.exists():
            raise FileNotFoundError("GeoTransformer repo or 3DMatch weights are missing.")
        sys.path[:0] = [str(self.repo), str(exp)]
        self.torch = importlib.import_module("torch")
        self.cfg = importlib.import_module("config").make_cfg()
        checkpoint = self._load_checkpoint()
        model = importlib.import_module("model").create_model(self.cfg).eval()
        model.load_state_dict(checkpoint["model"])
        del checkpoint
        gc.collect()
        self.model = model.cuda().eval()
        self.torch.cuda.empty_cache()
        self.collate = importlib.import_module("geotransformer.utils.data").registration_collate_fn_stack_mode
        torch_utils = importlib.import_module("geotransformer.utils.torch")
        self.to_cuda, self.release_cuda = torch_utils.to_cuda, torch_utils.release_cuda

    def _load_checkpoint(self) -> dict[str, Any]:
        try:
            return self.torch.load(self.weights, map_location="cpu", weights_only=True)
        except TypeError:
            return self.torch.load(self.weights, map_location="cpu")

    def rigid(self, src: np.ndarray, ref: np.ndarray) -> np.ndarray:
        data = {
            "ref_points": ref.astype(np.float32),
            "src_points": src.astype(np.float32),
            "ref_feats": np.ones((len(ref), 1), dtype=np.float32),
            "src_feats": np.ones((len(src), 1), dtype=np.float32),
            "transform": np.eye(4, dtype=np.float32),
        }
        data = self.collate([data], self.cfg.backbone.num_stages, self.cfg.backbone.init_voxel_size, self.cfg.backbone.init_radius, [38, 36, 36, 38])
        with self.torch.no_grad():
            out = self.release_cuda(self.model(self.to_cuda(data)))
        mat = np.eye(4)
        mat[:3, :3] = out["estimated_transform"][:3, :3]
        mat[:3, 3] = out["estimated_transform"][:3, 3]
        return mat


def registration_cost(src: np.ndarray, ref: np.ndarray, mat: np.ndarray, scale_prior: float) -> float:
    moved = apply(src, mat)
    dist_src, _ = cKDTree(ref).query(moved, k=1)
    dist_ref, _ = cKDTree(moved).query(ref, k=1)
    trim_src = dist_src[dist_src <= np.percentile(dist_src, 80)]
    trim_ref = dist_ref[dist_ref <= np.percentile(dist_ref, 80)]
    geom = 0.5 * (float(trim_src.mean()) + float(trim_ref.mean())) / max(float(np.linalg.norm(np.ptp(ref, axis=0))), 1e-6)
    scale_penalty = abs(np.log(max(transform_scale(mat), 1e-8) / max(scale_prior, 1e-8)))
    rotation_penalty = rotation_angle(mat)
    return geom + 0.03 * scale_penalty + 0.01 * rotation_penalty


def register_pair(src_mesh: trimesh.Trimesh, ref_mesh: trimesh.Trimesh, geot: GeoTransformerRunner | None, samples: int, geot_samples: int) -> tuple[np.ndarray, str]:
    src, ref = sample_points(src_mesh, samples), sample_points(ref_mesh, samples)
    scale = obb_scale(src_mesh, ref_mesh)
    src_mu, ref_mu = src.mean(0), ref.mean(0)
    pre = np.eye(4)
    pre[:3, :3] *= scale
    pre[:3, 3] = ref_mu - scale * src_mu
    candidates: list[tuple[str, np.ndarray]] = [("obb_scale", pre)]

    if geot:
        try:
            src_g, ref_g = take_subset(src, geot_samples), take_subset(ref, geot_samples)
            candidates.append(("obb_scale+geotransformer", geot.rigid(apply(src_g, pre), ref_g) @ pre))
        except Exception as exc:
            candidates.append((f"obb_scale+geotransformer_failed:{type(exc).__name__}", pre))

    scale_bounds = (scale * 0.5, scale * 2.0)
    refined = []
    for method, init in candidates:
        mat = icp_similarity(src, ref, init, scale_bounds=scale_bounds)
        refined.append((registration_cost(src, ref, mat, scale), method, mat))
    cost, method, mat = min(refined, key=lambda item: item[0])
    return mat, f"{method}+scale_aware_icp(scale_clamped,cost={cost:.4f})"


def collision_points(mesh: trimesh.Trimesh, count: int) -> np.ndarray:
    return collision_points_with_normals(mesh, count)[0]


def collision_points_with_normals(mesh: trimesh.Trimesh, count: int) -> tuple[np.ndarray, np.ndarray]:
    vertex_count = min(len(mesh.vertices), max(256, count // 4), 4096)
    surface_count = max(512, count - vertex_count)
    points, face_ids = trimesh.sample.sample_surface(mesh, surface_count)
    normals = np.asarray(mesh.face_normals[face_ids], dtype=np.float32)
    chunks = [np.asarray(points, dtype=np.float32)]
    normal_chunks = [normals]
    if vertex_count > 0:
        ids = np.linspace(0, len(mesh.vertices) - 1, vertex_count, dtype=np.int64)
        chunks.append(np.asarray(mesh.vertices[ids], dtype=np.float32))
        normal_chunks.append(np.asarray(mesh.vertex_normals[ids], dtype=np.float32))
    points_np = np.vstack(chunks).astype(np.float32)
    normals_np = np.vstack(normal_chunks).astype(np.float32)
    normals_np /= np.maximum(np.linalg.norm(normals_np, axis=1, keepdims=True), 1e-8)
    return points_np, normals_np


class CollisionOracle:
    def __init__(self, anchor: trimesh.Trimesh, moving: trimesh.Trimesh, mat: np.ndarray, samples: int, clearance: float):
        self.anchor = anchor
        self.linear = mat[:3, :3].copy()
        self.translation = mat[:3, 3].copy()
        self.clearance = float(clearance)
        self.anchor_bounds = np.asarray(anchor.bounds, dtype=np.float32)
        self.moving_points = collision_points(moving, samples)
        self.anchor_points, self.anchor_normals = collision_points_with_normals(anchor, samples)
        self.anchor_tree = cKDTree(self.anchor_points)

        moving_surface, moving_normals = collision_points_with_normals(moving, samples)
        self.moving_surface = moving_surface @ self.linear.T + self.translation
        normal_linear = np.linalg.pinv(self.linear)
        self.moving_normals = moving_normals @ normal_linear
        self.moving_normals /= np.maximum(np.linalg.norm(self.moving_normals, axis=1, keepdims=True), 1e-8)
        self.moving_tree = cKDTree(self.moving_surface)
        self.band = max(self.clearance * 3.0, max(float(anchor.scale), float(np.linalg.norm(np.ptp(self.moving_surface, axis=0)))) * 0.012)

        self.torch = None
        self.device = None
        self.anchor_tensor = None
        self.backend = "cpu_kdtree_bidirectional"
        pair_count = len(self.moving_points) * len(self.anchor_points)
        try:
            torch = importlib.import_module("torch")
            if torch.cuda.is_available() and pair_count <= 250_000_000:
                self.torch = torch
                self.device = torch.device("cuda")
                self.anchor_tensor = torch.as_tensor(self.anchor_points, dtype=torch.float32, device=self.device)
                self.backend = "cuda_cdist_bidirectional"
        except Exception:
            pass

    def transformed_points(self, delta: np.ndarray) -> np.ndarray:
        return self.moving_points @ self.linear.T + self.translation + delta

    def nearest_anchor_points(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.torch is None or self.anchor_tensor is None or self.device is None:
            try:
                dist, idx = self.anchor_tree.query(points, k=1, workers=-1)
            except TypeError:
                dist, idx = self.anchor_tree.query(points, k=1)
            return self.anchor_points[idx], self.anchor_normals[idx], dist.astype(np.float32)

        torch = self.torch
        chunk = max(256, min(2048, 80_000_000 // max(1, len(self.anchor_points))))
        nearest_idx, nearest_dist = [], []
        with torch.no_grad():
            query = torch.as_tensor(points, dtype=torch.float32, device=self.device)
            for start in range(0, len(query), chunk):
                dist = torch.cdist(query[start : start + chunk], self.anchor_tensor)
                values, idx = dist.min(dim=1)
                nearest_dist.append(values.cpu().numpy())
                nearest_idx.append(idx.cpu().numpy())
        idx_np = np.concatenate(nearest_idx).astype(np.int64)
        return self.anchor_points[idx_np], self.anchor_normals[idx_np], np.concatenate(nearest_dist).astype(np.float32)

    def nearest_moving_points(self, delta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        try:
            dist, idx = self.moving_tree.query(self.anchor_points - delta, k=1, workers=-1)
        except TypeError:
            dist, idx = self.moving_tree.query(self.anchor_points - delta, k=1)
        return self.moving_surface[idx] + delta, self.moving_normals[idx], dist.astype(np.float32)

    def signed_violations(self, query: np.ndarray, nearest: np.ndarray, normals: np.ndarray, dist: np.ndarray, sign: float) -> tuple[np.ndarray, np.ndarray]:
        signed = np.einsum("ij,ij->i", query - nearest, normals)
        violating = (signed < self.clearance) & (dist < self.band)
        if not violating.any():
            return np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.float32)
        penetration = np.minimum(np.maximum(self.clearance - signed[violating], 0.0), self.band).astype(np.float32)
        vectors = sign * normals[violating] * penetration[:, None]
        return vectors.astype(np.float32), np.linalg.norm(vectors, axis=1)

    def evaluate(self, delta: np.ndarray) -> dict[str, Any]:
        moving_query = self.transformed_points(delta)
        moving_bounds = np.asarray([moving_query.min(axis=0), moving_query.max(axis=0)], dtype=np.float32)
        axis_gaps = np.maximum(self.anchor_bounds[0] - moving_bounds[1], moving_bounds[0] - self.anchor_bounds[1])
        bbox_gap = float(axis_gaps.max())
        if bbox_gap >= self.clearance:
            return {
                "violations": 0,
                "moving_violations": 0,
                "anchor_violations": 0,
                "min_surface_distance": bbox_gap,
                "max_penetration": 0.0,
                "mean_penetration": 0.0,
                "escape_vector": np.zeros(3, dtype=np.float64),
            }
        nearest_anchor, anchor_normals, dist_forward = self.nearest_anchor_points(moving_query)
        forward_vectors, forward_lengths = self.signed_violations(moving_query, nearest_anchor, anchor_normals, dist_forward, 1.0)

        nearest_moving, moving_normals, dist_reverse = self.nearest_moving_points(delta)
        reverse_vectors, reverse_lengths = self.signed_violations(self.anchor_points, nearest_moving, moving_normals, dist_reverse, -1.0)

        vectors = np.vstack([forward_vectors, reverse_vectors]) if len(forward_vectors) + len(reverse_vectors) else np.zeros((0, 3), dtype=np.float32)
        lengths = np.concatenate([forward_lengths, reverse_lengths]) if len(vectors) else np.zeros(0, dtype=np.float32)
        if len(vectors):
            direction = vectors.sum(axis=0)
            if np.linalg.norm(direction) < 1e-8:
                direction = vectors[int(np.argmax(lengths))]
            escape_vector = direction / max(np.linalg.norm(direction), 1e-8) * float(np.percentile(lengths, 90))
        else:
            escape_vector = np.zeros(3, dtype=np.float32)

        return {
            "violations": int(len(lengths)),
            "moving_violations": int(len(forward_lengths)),
            "anchor_violations": int(len(reverse_lengths)),
            "min_surface_distance": float(min(dist_forward.min(), dist_reverse.min())) if len(dist_forward) and len(dist_reverse) else float("inf"),
            "max_penetration": float(lengths.max()) if len(lengths) else 0.0,
            "mean_penetration": float(lengths.mean()) if len(lengths) else 0.0,
            "escape_vector": escape_vector.astype(np.float64),
        }


def summarize_collision(info: dict[str, Any], delta: np.ndarray, rotation_degrees: float = 0.0) -> dict[str, Any]:
    summary = {
        "violations": int(info["violations"]),
        "min_surface_distance": float(info["min_surface_distance"]),
        "max_penetration": float(info["max_penetration"]),
        "mean_penetration": float(info["mean_penetration"]),
        "translation_delta_norm": float(np.linalg.norm(delta)),
        "rotation_delta_degrees": float(rotation_degrees),
    }
    if "moving_violations" in info:
        summary["moving_violations"] = int(info["moving_violations"])
        summary["anchor_violations"] = int(info.get("anchor_violations", 0))
    return summary


def collision_score(info: dict[str, Any], delta: np.ndarray, max_shift: float) -> float:
    # Normalized drift relative to the allowed budget. Following the paper's
    # Stage-2 objective (alignment + collision), we must strongly discourage
    # solutions that wander away from the guidance pose, because a large pure
    # translation destroys the object-object relationship (e.g. an inserted
    # shovel/coin being pushed out of the pot/piggy bank).
    shift = float(np.linalg.norm(delta)) / max(max_shift, 1e-8)
    gap = max(float(info["min_surface_distance"]), 0.0) if int(info["violations"]) == 0 else 0.0
    # Penetration terms dominate, but the violation count now carries real
    # weight so that a candidate creating many more contacts can never win
    # just because its worst-case penetration dipped marginally.
    penetration_cost = float(info["max_penetration"]) + 0.6 * float(info["mean_penetration"])
    violation_cost = 4e-4 * int(info["violations"])
    drift_cost = 0.12 * shift * shift
    return penetration_cost + violation_cost + 0.02 * gap + drift_cost


def better_collision(info: dict[str, Any], delta: np.ndarray, best_info: dict[str, Any] | None, best_delta: np.ndarray) -> bool:
    if best_info is None:
        return True
    max_shift = max(float(np.linalg.norm(delta)), float(np.linalg.norm(best_delta)), 1e-8)
    return collision_score(info, delta, max_shift) < collision_score(best_info, best_delta, max_shift)


def binary_refine_clear_delta(oracle: CollisionOracle, delta: np.ndarray, steps: int = 16) -> tuple[np.ndarray, dict[str, Any]]:
    length = float(np.linalg.norm(delta))
    if length < 1e-10:
        info = oracle.evaluate(delta)
        return delta, info
    direction = delta / length
    low, high = 0.0, length
    best_delta, best_info = delta.copy(), oracle.evaluate(delta)
    for _ in range(steps):
        mid = (low + high) * 0.5
        trial_delta = direction * mid
        info = oracle.evaluate(trial_delta)
        if info["violations"] == 0:
            best_delta, best_info, high = trial_delta, info, mid
        else:
            low = mid
    return best_delta, best_info


def directional_collision_search(oracle: CollisionOracle, directions: list[np.ndarray], max_shift: float, best_delta: np.ndarray, best_info: dict[str, Any] | None) -> tuple[np.ndarray, dict[str, Any]]:
    if best_info is None:
        best_info = oracle.evaluate(best_delta)
    seen: list[np.ndarray] = []
    for direction in directions:
        direction = np.asarray(direction, dtype=np.float64)
        norm = np.linalg.norm(direction)
        if norm < 1e-8:
            continue
        direction /= norm
        if any(abs(float(direction @ old)) > 0.995 for old in seen):
            continue
        seen.append(direction)
        for length in np.linspace(0.0, max_shift, 13)[1:]:
            trial_delta = direction * float(length)
            info = oracle.evaluate(trial_delta)
            if info["violations"] == 0:
                trial_delta, info = binary_refine_clear_delta(oracle, trial_delta)
            if collision_score(info, trial_delta, max_shift) < collision_score(best_info, best_delta, max_shift):
                best_delta, best_info = trial_delta, info
    return best_delta, best_info


def torch_rodrigues(torch: Any, vector: Any) -> Any:
    angle = torch.linalg.norm(vector)
    zero = vector.new_tensor(0.0)
    k = torch.stack(
        [
            torch.stack([zero, -vector[2], vector[1]]),
            torch.stack([vector[2], zero, -vector[0]]),
            torch.stack([-vector[1], vector[0], zero]),
        ]
    )
    eye = torch.eye(3, dtype=vector.dtype, device=vector.device)
    angle2 = angle * angle
    small = angle < 1e-4
    a = torch.where(small, 1.0 - angle2 / 6.0, torch.sin(angle) / torch.clamp(angle, min=1e-8))
    b = torch.where(small, 0.5 - angle2 / 24.0, (1.0 - torch.cos(angle)) / torch.clamp(angle2, min=1e-8))
    return eye + a * k + b * (k @ k)


def signed_distance_torch(torch: Any, points: Any, anchor_points: Any, anchor_normals: Any) -> Any:
    chunk = max(256, min(2048, 80_000_000 // max(1, int(anchor_points.shape[0]))))
    signed_chunks = []
    band = float(torch.linalg.norm(anchor_points.max(dim=0).values - anchor_points.min(dim=0).values).detach().cpu()) * 0.012
    for start in range(0, int(points.shape[0]), chunk):
        query = points[start : start + chunk]
        dist = torch.cdist(query, anchor_points)
        values, idx = dist.min(dim=1)
        nearest = anchor_points[idx]
        normals = anchor_normals[idx]
        signed = ((query - nearest) * normals).sum(dim=1)
        signed_chunks.append(torch.where(values < band, signed, torch.clamp(signed, min=0.0)))
    return torch.cat(signed_chunks, dim=0)


def worst_mean(torch: Any, values: Any, count: int = 1024) -> Any:
    if int(values.numel()) <= count:
        return values.mean()
    return torch.topk(values, k=count).values.mean()


def optimize_collision_pose(
    anchor: trimesh.Trimesh,
    moving: trimesh.Trimesh,
    guidance: trimesh.Trimesh,
    mat: np.ndarray,
    samples: int,
    clearance: float,
    scale: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        torch = importlib.import_module("torch")
        if not torch.cuda.is_available():
            return mat, {"enabled": False, "reason": "cuda_unavailable"}
        device = torch.device("cuda")
        opt_anchor_count = min(max(2048, samples // 2), 8192)
        opt_moving_count = min(max(1024, samples // 3), 4096)
        align_count = min(max(512, samples // 6), 2048)

        anchor_points, anchor_normals = collision_points_with_normals(anchor, opt_anchor_count)
        moving_local, moving_normals = collision_points_with_normals(moving, opt_moving_count)
        align_local = sample_points(moving, align_count)
        guide_points = sample_points(guidance, max(align_count * 2, 1024))
        align_initial = apply(align_local, mat)
        dist, idx = cKDTree(guide_points).query(align_initial, k=1)
        keep = dist <= np.percentile(dist, 85)
        if keep.sum() >= 64:
            align_local, align_target = align_local[keep], guide_points[idx[keep]]
        else:
            align_target = guide_points[idx]

        anchor_tensor = torch.as_tensor(anchor_points, dtype=torch.float32, device=device)
        normal_tensor = torch.as_tensor(anchor_normals, dtype=torch.float32, device=device)
        moving_tensor = torch.as_tensor(moving_local, dtype=torch.float32, device=device)
        align_tensor = torch.as_tensor(align_local, dtype=torch.float32, device=device)
        target_tensor = torch.as_tensor(align_target, dtype=torch.float32, device=device)
        linear = torch.as_tensor(mat[:3, :3], dtype=torch.float32, device=device)
        translation = torch.as_tensor(mat[:3, 3], dtype=torch.float32, device=device)
        normal_linear = torch.as_tensor(np.linalg.pinv(mat[:3, :3]), dtype=torch.float32, device=device)
        moving_normal_tensor = torch.as_tensor(moving_normals, dtype=torch.float32, device=device)
        base_moving = moving_tensor @ linear.T + translation
        base_normals = moving_normal_tensor @ normal_linear
        base_normals = base_normals / torch.clamp(torch.linalg.norm(base_normals, dim=1, keepdim=True), min=1e-8)
        base_align = align_tensor @ linear.T + translation
        center = base_moving.mean(dim=0).detach()

        rot = torch.zeros(3, dtype=torch.float32, device=device, requires_grad=True)
        trans = torch.zeros(3, dtype=torch.float32, device=device, requires_grad=True)
        optimizer = torch.optim.Adam([rot, trans], lr=0.035 * scale)
        steps = 72
        max_angle = np.deg2rad(10.0)
        max_shift = scale * 0.18 + clearance
        clearance_t = torch.tensor(float(clearance), dtype=torch.float32, device=device)
        scale_t = torch.tensor(float(max(scale, 1e-6)), dtype=torch.float32, device=device)
        history = []

        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            delta_rot = torch_rodrigues(torch, rot)
            moved = (base_moving - center) @ delta_rot.T + center + trans
            moved_normals = base_normals @ delta_rot.T
            moved_normals = moved_normals / torch.clamp(torch.linalg.norm(moved_normals, dim=1, keepdim=True), min=1e-8)
            aligned = (base_align - center) @ delta_rot.T + center + trans
            signed_forward = signed_distance_torch(torch, moved, anchor_tensor, normal_tensor)
            signed_reverse = signed_distance_torch(torch, anchor_tensor, moved, moved_normals)
            hard = torch.cat([torch.relu(-signed_forward - clearance_t), torch.relu(-signed_reverse - clearance_t)])
            soft = torch.cat([torch.relu(clearance_t - signed_forward), torch.relu(clearance_t - signed_reverse)])
            beta = 3.0 * step / max(1, steps - 1)
            collision_loss = worst_mean(torch, hard.square()) + 0.003 * worst_mean(torch, soft.square())
            alignment_loss = (aligned - target_tensor).square().sum(dim=1).mean()
            regularizer = 0.02 * trans.square().sum() + 0.002 * (rot / max_angle).square().sum()
            loss = alignment_loss / scale_t.square() + beta * collision_loss / scale_t.square() + regularizer
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                angle = torch.linalg.norm(rot)
                if angle > max_angle:
                    rot.mul_(max_angle / angle)
                shift = torch.linalg.norm(trans)
                if shift > max_shift:
                    trans.mul_(max_shift / shift)
            if step in {0, steps // 2, steps - 1}:
                history.append(
                    {
                        "step": int(step),
                        "loss": float(loss.detach().cpu()),
                        "alignment_loss": float(alignment_loss.detach().cpu()),
                        "collision_loss": float(collision_loss.detach().cpu()),
                    }
                )

        with torch.no_grad():
            delta_rot = torch_rodrigues(torch, rot)
            rot_np = delta_rot.detach().cpu().numpy().astype(np.float64)
            trans_np = trans.detach().cpu().numpy().astype(np.float64)
            center_np = center.detach().cpu().numpy().astype(np.float64)
        out = mat.copy()
        out[:3, :3] = rot_np @ out[:3, :3]
        out[:3, 3] = rot_np @ (out[:3, 3] - center_np) + center_np + trans_np
        torch.cuda.empty_cache()
        return out, {
            "enabled": True,
            "backend": "cuda_signed_distance_pose",
            "steps": steps,
            "optimized_anchor_samples": int(len(anchor_points)),
            "optimized_moving_samples": int(len(moving_local)),
            "alignment_samples": int(len(align_local)),
            "rotation_delta_degrees": float(np.rad2deg(np.linalg.norm(rot.detach().cpu().numpy()))),
            "translation_delta": trans_np.tolist(),
            "translation_delta_norm": float(np.linalg.norm(trans_np)),
            "history": history,
        }
    except Exception as exc:
        try:
            torch = importlib.import_module("torch")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        return mat, {"enabled": False, "reason": f"{type(exc).__name__}: {exc}"}


def resolve_collision_translation(anchor: trimesh.Trimesh, moving: trimesh.Trimesh, mat: np.ndarray, samples: int, clearance: float) -> tuple[np.ndarray, dict[str, Any]]:
    mat = mat.copy()
    moved = moving.copy()
    moved.apply_transform(mat)
    scale = max(float(anchor.scale), float(moved.scale), 1e-6)
    effective_clearance = max(float(clearance), scale * 1e-4)
    oracle = CollisionOracle(anchor, moving, mat, samples, effective_clearance)

    center_direction = moved.bounding_box.centroid - anchor.bounding_box.centroid
    center_norm = np.linalg.norm(center_direction)
    center_direction = center_direction / center_norm if center_norm > 1e-8 else np.array([1.0, 0.0, 0.0])

    delta = np.zeros(3, dtype=np.float64)
    # Tighter budgets than before: the SDF pose optimization has already
    # placed the object close to a physically valid pose, so this polish is
    # only allowed to nudge it. Large shifts were the source of the visible
    # "drift" (e.g. the shovel sliding out of the pot).
    max_shift = scale * 0.04 + effective_clearance
    max_step = scale * 0.008 + effective_clearance
    history = []

    # Anchor state (no translation). Every accepted candidate must beat this,
    # and we keep it as a guaranteed fallback so a bad polish can never make a
    # already-good registration worse.
    base_info = oracle.evaluate(delta)
    base_score = collision_score(base_info, delta, max_shift)
    history.append(summarize_collision(base_info, delta))
    best_delta = delta.copy()
    best_info: dict[str, Any] | None = base_info
    if base_info["violations"] == 0:
        best_delta, best_info = binary_refine_clear_delta(oracle, delta)
        final_info = best_info
        mat[:3, 3] += best_delta
        history.append(summarize_collision(final_info, best_delta))
        return mat, {
            "backend": oracle.backend,
            "iterations": len(history),
            "translation_delta": best_delta.tolist(),
            "translation_delta_norm": float(np.linalg.norm(best_delta)),
            "final": summarize_collision(final_info, best_delta),
            "history": history[-8:],
        }

    for _ in range(24):
        info = oracle.evaluate(delta)
        history.append(summarize_collision(info, delta))
        if better_collision(info, delta, best_info, best_delta):
            best_delta, best_info = delta.copy(), info
        if info["violations"] == 0:
            cleared_delta, cleared_info = binary_refine_clear_delta(oracle, delta)
            if better_collision(cleared_info, cleared_delta, best_info, best_delta):
                best_delta, best_info = cleared_delta, cleared_info
            break

        step = np.asarray(info["escape_vector"], dtype=np.float64)
        if np.linalg.norm(step) < 1e-8:
            step = center_direction * (scale * 0.002 + effective_clearance)
        step_norm = np.linalg.norm(step)
        if step_norm > max_step:
            step = step / step_norm * max_step
        # No overshoot: apply the (clamped) escape step directly to stay close
        # to the guidance-aligned pose.
        delta = delta + step
        delta_norm = np.linalg.norm(delta)
        if delta_norm > max_shift:
            delta = delta / delta_norm * max_shift

    if best_info is None or best_info["violations"] != 0:
        directions = [center_direction]
        if best_info is not None and np.linalg.norm(best_info["escape_vector"]) > 1e-8:
            directions.insert(0, best_info["escape_vector"] / np.linalg.norm(best_info["escape_vector"]))
        directions.extend(np.eye(3))
        directions.extend(-np.eye(3))

        seen: list[np.ndarray] = []
        for direction in directions:
            direction = np.asarray(direction, dtype=np.float64)
            direction /= max(np.linalg.norm(direction), 1e-8)
            if any(abs(float(direction @ old)) > 0.995 for old in seen):
                continue
            seen.append(direction)

            low, high = 0.0, max(scale * 0.002 + effective_clearance, np.linalg.norm(best_delta))
            found_delta = None
            for _ in range(14):
                trial_delta = direction * high
                info = oracle.evaluate(trial_delta)
                if better_collision(info, trial_delta, best_info, best_delta):
                    best_delta, best_info = trial_delta.copy(), info
                if info["violations"] == 0:
                    found_delta = trial_delta
                    break
                low, high = high, min(high * 1.8, max_shift)
                if high >= max_shift and low >= max_shift:
                    break
            if found_delta is not None:
                refined_delta, refined_info = binary_refine_clear_delta(oracle, found_delta)
                if better_collision(refined_info, refined_delta, best_info, best_delta):
                    best_delta, best_info = refined_delta, refined_info

    search_dirs = [center_direction]
    if best_info is not None:
        search_dirs.append(best_info.get("escape_vector", np.zeros(3)))
    search_dirs.extend(np.eye(3))
    search_dirs.extend(-np.eye(3))
    best_delta, best_info = directional_collision_search(oracle, search_dirs, max_shift, best_delta, best_info)

    # Fallback gate: the polish is only allowed to move the object if it
    # yields a clearly better collision state than the anchor (no-translation)
    # pose. This prevents drift where a large shift trades a big alignment
    # sacrifice for a negligible penetration gain, and guarantees that an
    # already-good registration can never be degraded by this stage.
    best_score = collision_score(best_info, best_delta, max_shift) if best_info is not None else base_score
    accept = False
    if best_info is not None and np.linalg.norm(best_delta) > 1e-9:
        cleared_now = int(best_info["violations"]) == 0 and int(base_info["violations"]) > 0
        # Require a meaningful improvement margin to accept any drift.
        improved_enough = best_score <= base_score - max(1e-4, 0.02 * base_score)
        accept = cleared_now or improved_enough
    if not accept:
        best_delta = np.zeros(3, dtype=np.float64)
        best_info = base_info

    final_info = best_info or oracle.evaluate(best_delta)
    mat[:3, 3] += best_delta
    history.append(summarize_collision(final_info, best_delta))
    return mat, {
        "backend": oracle.backend,
        "iterations": len(history),
        "translation_delta": best_delta.tolist(),
        "translation_delta_norm": float(np.linalg.norm(best_delta)),
        "final": summarize_collision(final_info, best_delta),
        "history": history[-8:],
    }


def finalize_contact(
    anchor: trimesh.Trimesh,
    moving: trimesh.Trimesh,
    mat: np.ndarray,
    samples: int,
    clearance: float,
    scale: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """High-precision finishing pass on the already-selected pose.

    The standard-density collision oracle used for candidate scoring can leave
    two visible artifacts that this pass repairs, each with a denser and more
    stable oracle so the reported/final state stops flickering with sampling
    noise:

      1. Residual micro-penetration. Sparse surface sampling can skip a small
         overlapping patch (e.g. the penguin figurine barely poking the napkin
         box). We escape along the contact normal by a strictly bounded amount
         and bisect back to just-clear.
      2. Floating gap. The collision objective only penalizes overlap, never
         separation, so a case can be left visibly hovering (e.g. the shirt
         above the washing machine). We nudge it toward the anchor centroid
         until the surfaces are just about to touch.

    Every move is strictly bounded and gated against the denser oracle, so it
    can never introduce drift or new overlap, and an already-well-seated pose
    is returned untouched.
    """
    dense_samples = int(min(max(samples * 2, samples + 4000), 24000))
    oracle = CollisionOracle(anchor, moving, mat, dense_samples, clearance)
    base = oracle.evaluate(np.zeros(3, dtype=np.float64))
    payload: dict[str, Any] = {
        "dense_samples": dense_samples,
        "initial": summarize_collision(base, np.zeros(3, dtype=np.float64)),
    }

    moved = moving.copy()
    moved.apply_transform(mat)
    toward = anchor.bounding_box.centroid - moved.bounding_box.centroid
    tn = float(np.linalg.norm(toward))
    toward = toward / tn if tn > 1e-8 else np.array([1.0, 0.0, 0.0])

    # ---- 1. Residual micro-penetration removal ----
    # Guard against sampling noise: only act on a genuine (if small) overlap.
    pen_floor = scale * 1.5e-3
    if int(base["violations"]) > 0 and float(base["max_penetration"]) > pen_floor:
        base_pen = float(base["max_penetration"])
        # A single summed escape vector can point in a compromise direction
        # that never fully separates two interlocked parts (forward-vs-reverse
        # violation vectors partially cancel). We instead try several
        # physically meaningful separation directions and keep the one that
        # clears the overlap with the *least* travel:
        #   - the summed escape vector (its natural first guess),
        #   - pushing the object away from / toward the anchor centroid
        #     (the dominant mode for a part resting in/on another, e.g. a
        #     figurine sunk into a box top),
        #   - and the six canonical axes as a robust fallback basis.
        escape = np.asarray(base["escape_vector"], dtype=np.float64)
        en = float(np.linalg.norm(escape))
        directions: list[np.ndarray] = []
        if en > 1e-8:
            directions.append(escape / en)
        directions.append(-toward)  # push moving away from the anchor centroid
        directions.append(toward)
        directions.extend(list(np.eye(3)))
        directions.extend(list(-np.eye(3)))

        # Enough travel to exit a genuine overlap, still tightly bounded so it
        # can never drift the object out of its interaction relationship.
        max_push = float(min(max(scale * 0.06, base_pen * 4.0 + clearance), scale * 0.14))

        best_clear: tuple[float, np.ndarray, dict[str, Any]] | None = None
        best_reduce: tuple[float, np.ndarray, dict[str, Any]] | None = None
        reduce_cap = scale * 0.07  # tighter cap for the non-clearing fallback
        seen: list[np.ndarray] = []
        for raw in directions:
            d = np.asarray(raw, dtype=np.float64)
            nd = float(np.linalg.norm(d))
            if nd < 1e-8:
                continue
            d = d / nd
            if any(abs(float(d @ s)) > 0.985 for s in seen):
                continue
            seen.append(d)

            high = max(scale * 0.004 + clearance, base_pen)
            cleared = None
            for _ in range(16):
                trial = d * min(high, max_push)
                info = oracle.evaluate(trial)
                pen = float(info["max_penetration"])
                travel = float(np.linalg.norm(trial))
                # Track the best strictly-improving bounded fallback: a clear
                # penetration reduction with small travel.
                if pen <= base_pen * 0.7 and travel <= reduce_cap:
                    cand = pen / max(base_pen, 1e-8) + 0.5 * travel / max(reduce_cap, 1e-8)
                    if best_reduce is None or cand < best_reduce[0]:
                        best_reduce = (cand, trial.copy(), info)
                if int(info["violations"]) == 0:
                    cleared = trial
                    break
                if high >= max_push:
                    break
                high = min(high * 1.6, max_push)
            if cleared is not None:
                cd, ci = binary_refine_clear_delta(oracle, cleared)
                travel = float(np.linalg.norm(cd))
                if best_clear is None or travel < best_clear[0]:
                    best_clear = (travel, cd, ci)

        if best_clear is not None:
            _, cd, ci = best_clear
            out = mat.copy()
            out[:3, 3] += cd
            payload["mode"] = "micro_penetration_cleared"
            payload["push"] = float(np.linalg.norm(cd))
            payload["max_push"] = max_push
            payload["directions_tried"] = len(seen)
            payload["final"] = summarize_collision(ci, cd)
            return out, payload
        if best_reduce is not None:
            _, rd, ri = best_reduce
            out = mat.copy()
            out[:3, 3] += rd
            payload["mode"] = "micro_penetration_reduced"
            payload["push"] = float(np.linalg.norm(rd))
            payload["max_push"] = max_push
            payload["directions_tried"] = len(seen)
            payload["residual_max_penetration"] = float(ri["max_penetration"])
            payload["final"] = summarize_collision(ri, rd)
            return out, payload
        payload["mode"] = "micro_penetration_unresolved"
        payload["directions_tried"] = len(seen)
        payload["final"] = summarize_collision(base, np.zeros(3, dtype=np.float64))
        return mat, payload

    # ---- 2. Floating seating (no penetration) ----
    gap = float(base["min_surface_distance"])
    contact_threshold = max(scale * 0.01, clearance * 4.0)
    if int(base["violations"]) > 0 or not np.isfinite(gap) or gap <= contact_threshold:
        payload["mode"] = "contact_ok"
        payload["final"] = summarize_collision(base, np.zeros(3, dtype=np.float64))
        return mat, payload

    # Bound the seating travel so it can never fling the object across the scene.
    max_travel = min(gap * 1.5 + clearance, scale * 0.5)
    target_gap = max(clearance, scale * 5e-3)
    low, high = 0.0, max_travel
    best_delta = np.zeros(3, dtype=np.float64)
    best_info = base
    for _ in range(24):
        mid = 0.5 * (low + high)
        info = oracle.evaluate(toward * mid)
        if int(info["violations"]) == 0:
            # Still clear: accept and keep closing the gap toward the target.
            best_delta, best_info = toward * mid, info
            if float(info["min_surface_distance"]) > target_gap:
                low = mid
            else:
                high = mid
        else:
            # Contact/penetration: back off.
            high = mid

    out = mat.copy()
    out[:3, 3] += best_delta
    payload["mode"] = "seated" if float(np.linalg.norm(best_delta)) > 1e-9 else "contact_ok"
    payload["initial_gap"] = gap
    payload["final_gap"] = float(best_info["min_surface_distance"])
    payload["target_gap"] = float(target_gap)
    payload["contact_threshold"] = float(contact_threshold)
    payload["travel"] = float(np.linalg.norm(best_delta))
    payload["max_travel"] = float(max_travel)
    payload["final"] = summarize_collision(best_info, best_delta)
    return out, payload


def classify_penetration(final_summary: dict[str, Any], scale: float, failure_ratio: float) -> dict[str, Any]:
    """Grade the residual interpenetration of the composed scene.

    The severity is measured as the deepest residual penetration normalized by
    the object scale, which makes the threshold size-invariant. When it exceeds
    the configured budget the case is marked ``failure`` so the stage-5 agentic
    refinement knows it must intervene; otherwise it is ``success``.
    """
    max_pen = float(final_summary.get("max_penetration", 0.0))
    mean_pen = float(final_summary.get("mean_penetration", 0.0))
    denom = max(float(scale), 1e-8)
    max_ratio = max_pen / denom
    status = "failure" if max_ratio > float(failure_ratio) else "success"
    return {
        "status": status,
        "scale": float(scale),
        "max_penetration": max_pen,
        "max_penetration_ratio": float(max_ratio),
        "mean_penetration_ratio": float(mean_pen / denom),
        "failure_penetration_ratio": float(failure_ratio),
        "violations": int(final_summary.get("violations", 0)),
    }


def resolve_collision(anchor: trimesh.Trimesh, moving: trimesh.Trimesh, guidance: trimesh.Trimesh, mat: np.ndarray, samples: int, clearance: float, failure_ratio: float = 0.02) -> tuple[np.ndarray, dict[str, Any]]:
    moved = moving.copy()
    moved.apply_transform(mat)
    scale = max(float(anchor.scale), float(moved.scale), 1e-6)
    effective_clearance = max(float(clearance), scale * 1e-4)
    initial_info = CollisionOracle(anchor, moving, mat, samples, effective_clearance).evaluate(np.zeros(3, dtype=np.float64))
    initial_summary = summarize_collision(initial_info, np.zeros(3, dtype=np.float64))
    if initial_info["violations"] == 0:
        # Even a collision-free registration can be visibly floating or hide a
        # sub-sampling-scale overlap; run the high-precision finishing pass.
        seated_mat, finalize_payload = finalize_contact(anchor, moving, mat, samples, effective_clearance, scale)
        return seated_mat, {
            "mode": "already_clear",
            "requested_clearance": float(clearance),
            "effective_clearance": float(effective_clearance),
            "initial": initial_summary,
            "contact_finalization": finalize_payload,
            "final": finalize_payload["final"],
            "penetration_check": classify_penetration(finalize_payload["final"], scale, failure_ratio),
        }

    pose_mat, pose_payload = optimize_collision_pose(anchor, moving, guidance, mat, samples, effective_clearance, scale)
    final_mat, translation_payload = resolve_collision_translation(anchor, moving, pose_mat, samples, effective_clearance)

    # Global fallback gate. Any stage in the collision pipeline could in
    # principle degrade the pose, so we score the three candidate transforms
    # under a single, consistent collision oracle and keep the best one. This
    # is the hard guarantee that "already-good" cases can never be made worse:
    #   - mat        : registration only (no collision handling)
    #   - pose_mat   : after SDF-based pose optimization (paper Stage 2)
    #   - final_mat  : after the bounded translation polish
    def _score(candidate: np.ndarray) -> float:
        info = CollisionOracle(anchor, moving, candidate, samples, effective_clearance).evaluate(
            np.zeros(3, dtype=np.float64)
        )
        # max_shift here only normalizes the (zero) drift term, so it is inert;
        # the score reduces to the penetration/violation cost of the pose.
        return collision_score(info, np.zeros(3, dtype=np.float64), max(scale, 1e-6))

    candidates = [
        ("registration_only", mat),
        ("sdf_pose_optimization", pose_mat),
        ("translation_polish", final_mat),
    ]
    scored = [(name, cand, _score(cand)) for name, cand in candidates]
    best_name, best_mat, _best_score = min(scored, key=lambda item: item[2])

    # High-precision finishing pass on the selected pose: a denser, more stable
    # oracle removes any residual micro-penetration the sparse candidate scoring
    # missed (the source of the faint penguin/napkin-box overlap) and reports a
    # sampling-noise-free final state.
    final_mat, finalize_payload = finalize_contact(anchor, moving, best_mat, samples, effective_clearance, scale)

    return final_mat, {
        "mode": "cuda_sdf_pose_optimization_with_translation_polish",
        "requested_clearance": float(clearance),
        "effective_clearance": float(effective_clearance),
        "initial": initial_summary,
        "sdf_pose_optimization": pose_payload,
        "translation_polish": translation_payload,
        "selected_candidate": best_name,
        "candidate_scores": {name: float(score) for name, _cand, score in scored},
        "contact_finalization": finalize_payload,
        "final": finalize_payload["final"],
        "penetration_check": classify_penetration(finalize_payload["final"], scale, failure_ratio),
    }


def matrix_info(mat: np.ndarray, method: str) -> dict[str, Any]:
    scale = float(np.cbrt(abs(np.linalg.det(mat[:3, :3]))))
    rot = mat[:3, :3] / max(scale, 1e-8)
    return {"method": method, "scale": scale, "rotation": rot.tolist(), "translation": mat[:3, 3].tolist(), "matrix": mat.tolist()}


def silhouette_area(mesh: trimesh.Trimesh, axis: int) -> float:
    """Area of the mesh silhouette projected onto the plane orthogonal to axis."""
    keep = [i for i in range(3) if i != axis]
    pts = np.asarray(mesh.vertices[:, keep], dtype=np.float64)
    if len(pts) < 3:
        return 0.0
    try:
        from scipy.spatial import ConvexHull

        # For 2D points, ConvexHull.volume is the enclosed area.
        return float(ConvexHull(pts).volume)
    except Exception:
        # Degenerate/collinear fallback: axis-aligned projected bbox area.
        return float(np.ptp(pts[:, 0]) * np.ptp(pts[:, 1]))


def select_anchor(guide: dict[str, trimesh.Trimesh], override: str = "auto") -> str:
    """Pick the anchor object for Stage-1 registration.

    Following the paper, the anchor should be the object with the larger
    projected area in the scene image, which empirically gives the most stable
    "main-body" reference. Since we do not carry the scene-image projection
    here, we approximate it with the largest silhouette area of the guidance
    mesh over the three canonical view axes. This is far more faithful to the
    paper than raw AABB volume, which can be misleadingly large for thin,
    tilted parts (e.g. a shovel inserted diagonally into a pot) and cause the
    wrong object to become the anchor, leading to an unstable alignment and
    visible drift.
    """
    if override in guide:
        return override

    def projected_size(mesh: trimesh.Trimesh) -> float:
        return max(silhouette_area(mesh, axis) for axis in range(3))

    return max(guide, key=lambda k: projected_size(guide[k]))


def cleanup_final_artifacts(input_dir: Path) -> None:
    for name in ("registered_object_a.glb", "registered_object_b.glb"):
        (input_dir / name).unlink(missing_ok=True)
    for name in ("partfield_parts", "partfield_renders", "partfield_work"):
        shutil.rmtree(input_dir / name, ignore_errors=True)


def main() -> None:
    args = parse_args()
    paths = {name: args.input_dir / f"{name}.glb" for name in ["object_a", "object_b", "guidance_object_a", "guidance_object_b"]}
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)

    geot = None
    if args.no_geotransformer:
        log("GeoTransformer disabled")
    else:
        log("loading GeoTransformer")
        try:
            geot = GeoTransformerRunner(args.geotransformer_dir, args.geotransformer_weights)
        except Exception as exc:
            log(f"GeoTransformer unavailable ({type(exc).__name__}); falling back to OBB + ICP")
    log("loading meshes")
    obj = {k: load_mesh(paths[k]) for k in ["object_a", "object_b"]}
    guide = {k: load_mesh(paths[f"guidance_{k}"]) for k in ["object_a", "object_b"]}
    anchor_key = select_anchor(guide, args.anchor)
    remain_key = "object_b" if anchor_key == "object_a" else "object_a"
    log(f"anchor={anchor_key}, remaining={remain_key}")

    transforms, methods = {}, {}
    log(f"registering anchor {anchor_key}")
    transforms[anchor_key], methods[anchor_key] = register_pair(obj[anchor_key], guide[anchor_key], geot, args.samples, args.geotransformer_samples)
    log(f"registering remaining {remain_key}")
    transforms[remain_key], methods[remain_key] = register_pair(obj[remain_key], guide[remain_key], geot, args.samples, args.geotransformer_samples)
    if geot is not None:
        torch = geot.torch
        del geot
        gc.collect()
        torch.cuda.empty_cache()

    log("running collision-aware pose optimization")
    anchor_world = obj[anchor_key].copy(); anchor_world.apply_transform(transforms[anchor_key])
    transforms[remain_key], collision = resolve_collision(anchor_world, obj[remain_key], guide[remain_key], transforms[remain_key], args.collision_samples, args.clearance, args.failure_penetration_ratio)

    log("exporting registered scene")
    registered = {}
    for key in ["object_a", "object_b"]:
        mesh = obj[key].copy(); mesh.apply_transform(transforms[key]); registered[key] = mesh

    scene = trimesh.Scene()
    scene.add_geometry(registered["object_a"], node_name="object_a")
    scene.add_geometry(registered["object_b"], node_name="object_b")
    scene_path = args.input_dir / "registered_scene.glb"
    scene.export(scene_path)
    cleanup_final_artifacts(args.input_dir)

    penetration_check = collision.get("penetration_check", {})
    status = penetration_check.get("status", "success")

    meta_path = args.input_dir / "metadata.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    metadata["registration"] = {
        "anchor_object": anchor_key,
        "remaining_object": remain_key,
        "geotransformer_dir": str(args.geotransformer_dir),
        "geotransformer_weights": str(args.geotransformer_weights),
        "samples": args.samples,
        "geotransformer_samples": args.geotransformer_samples,
        "collision_samples": args.collision_samples,
        "clearance": args.clearance,
        "failure_penetration_ratio": args.failure_penetration_ratio,
        # Top-level verdict consumed by the stage-5 agentic refinement: a
        # "failure" here means the residual interpenetration is severe enough to
        # warrant a corrective re-composition; "success" means the scene is
        # physically acceptable.
        "status": status,
        "penetration_check": penetration_check,
        "transforms": {key: matrix_info(transforms[key], methods[key]) for key in ["object_a", "object_b"]},
        "collision_refinement": collision,
        "outputs": {"registered_scene": str(scene_path)},
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    log(f"registration status={status} (max_penetration_ratio={penetration_check.get('max_penetration_ratio', 0.0):.4f}, threshold={args.failure_penetration_ratio})")
    print(json.dumps(metadata["registration"], indent=2))


if __name__ == "__main__":
    main()
