#!/usr/bin/env python3
"""Remove thin, supported floor overlays from a collision USD.

The v1 policy removes thin horizontal prims supported by SM_floorplan. Opt-in
v2 preserves those removals and adds source-bound, explicitly approved prims
passing ground-relative thickness, contact and support checks. Shape ratios
are diagnostic for approved v2 instances. Raw USD is never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from pxr import Sdf, Usd, UsdGeom

POLICY_ID = "sage3d.thin_floor_overlay.v1"
POLICY_V2 = "sage3d.thin_floor_overlay.v2"
V2_THRESHOLDS = {"max_overlay_thickness_m": .020, "max_top_above_ground_m": .025,
                 "max_contact_gap_m": .005, "max_support_gap_m": .025,
                 "min_support_coverage": .95, "review_thickness_over_short_side": .05}
DEFAULTS = {
    "max_overlay_thickness_m": 0.005,
    "min_horizontal_area_ratio": 0.99,
    "max_support_gap_m": 0.005,
    "min_support_coverage": 0.95,
    "max_floor_height_m": 0.25,
}
_EPS = 1.0e-7
_SUPPORT_BARYCENTRIC_SAMPLES = np.asarray(
    [
        (i / 3.0, j / 3.0, 1.0 - (i + j) / 3.0)
        for i in range(4)
        for j in range(4 - i)
    ],
    dtype=np.float64,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _world_points(mesh: UsdGeom.Mesh, cache: UsdGeom.XformCache) -> np.ndarray:
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    matrix = np.asarray(cache.GetLocalToWorldTransform(mesh.GetPrim()), dtype=np.float64)
    return (np.column_stack((points, np.ones(len(points)))) @ matrix)[:, :3]


def _triangles(mesh: UsdGeom.Mesh, points: np.ndarray) -> np.ndarray | None:
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
    if not len(counts) or len(indices) != int(counts.sum()) or not np.all(counts == 3):
        return None
    if len(indices) and (indices.min() < 0 or indices.max() >= len(points)):
        return None
    return points[indices.reshape(-1, 3)]


def _geometry_metrics(mesh: UsdGeom.Mesh, points: np.ndarray) -> dict:
    triangles = _triangles(mesh, points)
    metrics = {
        "vertex_count": int(len(points)),
        "face_count": int(len(mesh.GetFaceVertexCountsAttr().Get())),
        "geometry_valid": triangles is not None,
        "thickness_m": float(np.ptp(points[:, 2])) if len(points) else None,
    }
    if triangles is None or not len(triangles):
        metrics.update({"area_m2": 0.0, "horizontal_area_ratio": 0.0})
        return metrics
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(cross, axis=1)
    areas = 0.5 * lengths
    area = float(areas.sum())
    horizontal = np.divide(
        np.abs(cross[:, 2]), lengths, out=np.zeros_like(lengths), where=lengths > _EPS
    )
    metrics.update(
        {
            "area_m2": area,
            "horizontal_area_ratio": float(np.dot(areas, horizontal) / area)
            if area > _EPS
            else 0.0,
        }
    )
    return metrics


class _FloorSupportIndex:
    """Small XY grid over horizontal floorplan triangles."""

    def __init__(self, triangles: np.ndarray) -> None:
        cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        lengths = np.linalg.norm(cross, axis=1)
        areas = 0.5 * lengths
        horizontal = np.divide(
            np.abs(cross[:, 2]), lengths, out=np.zeros_like(lengths), where=lengths > _EPS
        )
        keep = (areas > _EPS) & (horizontal >= 0.99)
        self.triangles = triangles[keep]
        if not len(self.triangles):
            self.cell_size = 1.0
            self.cells = {}
            return
        mins = self.triangles[:, :, :2].min(axis=1)
        maxs = self.triangles[:, :, :2].max(axis=1)
        extent = maxs.max(axis=0) - mins.min(axis=0)
        self.cell_size = max(float(extent.max()) / 128.0, 0.25)
        self.origin = mins.min(axis=0)
        self.cells: dict[tuple[int, int], list[int]] = {}
        for index, (lower, upper) in enumerate(zip(mins, maxs)):
            lower_cell = np.floor((lower - self.origin) / self.cell_size).astype(int)
            upper_cell = np.floor((upper - self.origin) / self.cell_size).astype(int)
            for ix in range(int(lower_cell[0]), int(upper_cell[0]) + 1):
                for iy in range(int(lower_cell[1]), int(upper_cell[1]) + 1):
                    self.cells.setdefault((ix, iy), []).append(index)

    def _candidate_indices(self, point: np.ndarray) -> list[int]:
        if not len(self.triangles):
            return []
        cell = tuple(np.floor((point[:2] - self.origin) / self.cell_size).astype(int))
        return self.cells.get(cell, [])

    def floor_z_below(self, point: np.ndarray) -> float | None:
        z = self.floor_z_below_many(np.asarray([point], dtype=np.float64))[0]
        return None if np.isnan(z) else float(z)

    def floor_z_below_many(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        result = np.full(len(points), np.nan, dtype=np.float64)
        if not len(self.triangles):
            return result
        positions: dict[tuple[int, int], list[int]] = {}
        for position, point in enumerate(points):
            cell = tuple(np.floor((point[:2] - self.origin) / self.cell_size).astype(int))
            positions.setdefault(cell, []).append(position)
        for cell, point_positions in positions.items():
            candidates = self.cells.get(cell, [])
            if not candidates:
                continue
            triangles = self.triangles[candidates]
            queries = points[point_positions]
            a = triangles[:, 0, :2]
            ab = triangles[:, 1, :2] - a
            ac = triangles[:, 2, :2] - a
            ap = queries[:, None, :2] - a[None, :, :]
            denominator = ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]
            u = np.divide(
                ap[..., 0] * ac[None, :, 1] - ap[..., 1] * ac[None, :, 0],
                denominator[None, :],
                out=np.full((len(queries), len(triangles)), np.nan),
                where=np.abs(denominator)[None, :] > _EPS,
            )
            v = np.divide(
                ab[None, :, 0] * ap[..., 1] - ab[None, :, 1] * ap[..., 0],
                denominator[None, :],
                out=np.full((len(queries), len(triangles)), np.nan),
                where=np.abs(denominator)[None, :] > _EPS,
            )
            inside = (u >= -_EPS) & (v >= -_EPS) & (u + v <= 1.0 + _EPS)
            z = (
                triangles[None, :, 0, 2]
                + u * (triangles[None, :, 1, 2] - triangles[None, :, 0, 2])
                + v * (triangles[None, :, 2, 2] - triangles[None, :, 0, 2])
            )
            valid = inside & (z <= queries[:, None, 2] + _EPS)
            z = np.where(valid, z, -np.inf)
            best = z.max(axis=1)
            best[~np.isfinite(best)] = np.nan
            result[point_positions] = best
        return result


def _support_metrics(triangles: np.ndarray, index: _FloorSupportIndex, max_gap: float) -> dict:
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    total_area = float(areas.sum())
    supported_area = 0.0
    samples = np.einsum("sj,tjk->tsk", _SUPPORT_BARYCENTRIC_SAMPLES, triangles)
    flat_samples = samples.reshape(-1, 3)
    floor_z = index.floor_z_below_many(flat_samples).reshape(len(triangles), -1)
    gaps = flat_samples[:, 2].reshape(len(triangles), -1) - floor_z
    supported = np.isfinite(floor_z) & (gaps >= -_EPS) & (gaps <= max_gap)
    supported_area = float(np.dot(areas, supported.mean(axis=1)))
    support_sample_count = int(supported.sum())
    support_sample_total = int(supported.size)
    supported_faces = int(np.sum(supported.all(axis=1)))
    support_gaps = np.maximum(gaps[supported], 0.0)
    support_heights = floor_z[supported]
    return {
        "support_coverage": supported_area / total_area if total_area > _EPS else 0.0,
        "supported_area_m2": supported_area,
        "supported_face_count": supported_faces,
        "support_sample_count": support_sample_count,
        "support_sample_total": support_sample_total,
        "support_gap_max_m": float(support_gaps.max()) if support_gaps.size else None,
        "support_floor_height_max_m": float(support_heights.max())
        if support_heights.size
        else None,
        "minimum_contact_gap_m": float(np.maximum(gaps[np.isfinite(gaps) & (gaps >= -_EPS)], 0).min())
        if np.any(np.isfinite(gaps) & (gaps >= -_EPS)) else None,
    }


def _geometry_sha256(points, faces):
    return hashlib.sha256(points.astype("<f8").tobytes() + faces.astype("<i8").tobytes()).hexdigest()


def _v2_reasons(mesh, points, metrics, support_index, ground_z):
    import trimesh
    from scipy.spatial import QhullError

    triangles = _triangles(mesh, points)
    if triangles is None or not len(triangles) or not np.isfinite(points).all() or metrics["area_m2"] <= _EPS:
        return ["invalid_or_nontriangular_geometry"]
    try:
        _, extents = trimesh.bounds.oriented_bounds_2D(points[:, :2])
    except QhullError:
        return ["degenerate_xy_footprint"]
    short_side = float(min(extents))
    metrics.update(_support_metrics(triangles, support_index, V2_THRESHOLDS["max_support_gap_m"]))
    metrics.update(short_xy_side_m=short_side,
                   thickness_over_short_side=float(metrics["thickness_m"] / short_side),
                   sheet_shape_warning=bool(metrics["thickness_m"] / short_side > V2_THRESHOLDS["review_thickness_over_short_side"] + _EPS),
                   bottom_above_ground_m=float(points[:, 2].min() - ground_z),
                   top_above_ground_m=float(points[:, 2].max() - ground_z))
    contact = metrics["minimum_contact_gap_m"]
    checks = {
        "too_thick": metrics["thickness_m"] <= V2_THRESHOLDS["max_overlay_thickness_m"] + _EPS,
        "below_ground": metrics["bottom_above_ground_m"] >= -_EPS,
        "too_high": metrics["top_above_ground_m"] <= V2_THRESHOLDS["max_top_above_ground_m"] + _EPS,
        "support_coverage_below_threshold": metrics["support_coverage"] >= V2_THRESHOLDS["min_support_coverage"],
        "support_contact_gap_above_limit": contact is not None and contact <= V2_THRESHOLDS["max_contact_gap_m"] + _EPS,
        "contains_child_prims": not list(mesh.GetPrim().GetAllChildren()),
    }
    return [name for name, passed in checks.items() if not passed]


def _validate_thresholds(thresholds: dict[str, float]) -> None:
    for name, value in thresholds.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    for name in (
        "max_overlay_thickness_m",
        "max_support_gap_m",
        "max_floor_height_m",
    ):
        if thresholds[name] < 0.0:
            raise ValueError(f"{name} must be nonnegative")
    for name in ("min_horizontal_area_ratio", "min_support_coverage"):
        if not 0.0 <= thresholds[name] <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")


def _find_floorplan(stage: Usd.Stage) -> Usd.Prim:
    matches = [prim for prim in stage.Traverse() if prim.GetName() == "SM_floorplan"]
    if len(matches) != 1 or str(matches[0].GetPath()) != "/Root/SM_floorplan":
        raise RuntimeError(
            "Expected exactly one /Root/SM_floorplan, "
            f"found {[str(prim.GetPath()) for prim in matches]}"
        )
    floor = matches[0]
    if not floor.IsA(UsdGeom.Mesh):
        raise RuntimeError("/Root/SM_floorplan must be a Mesh")
    return floor


def sanitize_collision_usd(
    input_path: Path,
    output_path: Path,
    report_path: Path,
    *,
    max_overlay_thickness_m: float = DEFAULTS["max_overlay_thickness_m"],
    min_horizontal_area_ratio: float = DEFAULTS["min_horizontal_area_ratio"],
    max_support_gap_m: float = DEFAULTS["max_support_gap_m"],
    min_support_coverage: float = DEFAULTS["min_support_coverage"],
    max_floor_height_m: float = DEFAULTS["max_floor_height_m"],
    policy: str = "v1",
    approved_manifest: Path | None = None,
    dry_run: bool = False,
) -> dict:
    input_path = Path(input_path)
    output_path = Path(output_path)
    report_path = Path(report_path)
    thresholds = dict(
        max_overlay_thickness_m=float(max_overlay_thickness_m),
        min_horizontal_area_ratio=float(min_horizontal_area_ratio),
        max_support_gap_m=float(max_support_gap_m),
        min_support_coverage=float(min_support_coverage),
        max_floor_height_m=float(max_floor_height_m),
    )
    _validate_thresholds(thresholds)
    resolved_paths = {path.resolve() for path in (input_path, output_path, report_path)}
    if len(resolved_paths) != 3:
        raise ValueError("input, output, and report paths must be distinct")
    if policy not in ("v1", "v2") or (policy == "v1" and approved_manifest is not None):
        raise ValueError("approved_manifest requires policy v2")
    source_sha256 = _sha256(input_path)
    approval_record = None
    approvals = {}
    if policy == "v2":
        if thresholds != DEFAULTS:
            raise ValueError("policy v2 preserves the default legacy thresholds")
        if approved_manifest is None:
            raise ValueError("policy v2 requires an approved manifest")
        if Path(approved_manifest).resolve() in resolved_paths:
            raise ValueError("approved manifest must be distinct from input, output and report")
        manifest = json.loads(Path(approved_manifest).read_text())
        matching = [record for record in manifest["scenes"].values()
                    if record["source_sha256"] == source_sha256]
        if manifest["policy"] != POLICY_V2 or len(matching) != 1:
            raise ValueError("manifest policy or source SHA256 mismatch")
        approval_record = matching[0]
        ground_z = approval_record["ground_z_m"]
        if ground_z is not None and not math.isfinite(ground_z):
            raise ValueError("ground datum must be finite")
        approvals = {entry["path"]: entry for entry in approval_record["approvals"]}
        if len(approvals) != len(approval_record["approvals"]):
            raise ValueError("duplicate approval paths")
        if approvals and ground_z is None:
            raise ValueError("approved removals require a reliable ground datum")
    stage = Usd.Stage.Open(str(input_path))
    if stage is None:
        raise RuntimeError(f"Could not open collision USD: {input_path}")
    if policy == "v2" or dry_run:
        if stage.GetRootLayer().GetExternalReferences():
            raise ValueError("v2/dry-run requires a self-contained collision USD")
        # USD caches layers by identifier; edits must not alter another reader's input stage.
        stage = Usd.Stage.Open(Sdf.Layer.OpenAsAnonymous(str(input_path)))
    if policy == "v2" and (UsdGeom.GetStageUpAxis(stage) != "Z" or UsdGeom.GetStageMetersPerUnit(stage) != 1.):
        raise ValueError("policy v2 requires Z-up collision geometry in metres")
    floor_prim = _find_floorplan(stage)
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    floor_mesh = UsdGeom.Mesh(floor_prim)
    floor_points = _world_points(floor_mesh, cache)
    floor_triangles = _triangles(floor_mesh, floor_points)
    if floor_triangles is None:
        raise RuntimeError("/Root/SM_floorplan must be fully triangulated")
    support_index = _FloorSupportIndex(floor_triangles)
    v2_support = None
    if approval_record is not None and approval_record["ground_z_m"] is not None:
        ground_z = approval_record["ground_z_m"]
        near = np.abs(floor_triangles[:, :, 2] - ground_z).max(axis=1) <= .01 + _EPS
        v2_support = _FloorSupportIndex(floor_triangles[near])
    for path, approval in approvals.items():
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsA(UsdGeom.Mesh) or prim == floor_prim:
            raise ValueError(f"invalid approved Mesh path: {path}")
        mesh = UsdGeom.Mesh(prim)
        triangles = _triangles(mesh, _world_points(mesh, cache))
        if triangles is None:
            raise ValueError(f"invalid approved geometry: {path}")
        faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64).reshape(-1, 3)
        if _geometry_sha256(_world_points(mesh, cache), faces) != approval["world_geometry_sha256"]:
            raise ValueError(f"approved geometry SHA256 mismatch: {path}")

    removed = []
    skipped = []
    mesh_records = []
    remove_paths = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = _world_points(mesh, cache)
        metrics = _geometry_metrics(mesh, points)
        mesh_records.append(metrics)
        if prim == floor_prim or metrics["thickness_m"] is None:
            continue
        v2_candidate = (v2_support is not None and metrics["thickness_m"] <= V2_THRESHOLDS["max_overlay_thickness_m"] + _EPS
                        and points[:, 2].min() >= ground_z - _EPS
                        and points[:, 2].max() <= ground_z + V2_THRESHOLDS["max_top_above_ground_m"] + _EPS)
        if metrics["thickness_m"] > max_overlay_thickness_m and not v2_candidate and str(prim.GetPath()) not in approvals:
            continue
        triangles = _triangles(mesh, points)
        reasons = []
        if metrics["thickness_m"] > max_overlay_thickness_m:
            reasons.append("thickness_above_legacy_limit")
        if not metrics["geometry_valid"] or triangles is None:
            reasons.append("invalid_or_nontriangular_geometry")
        if metrics["horizontal_area_ratio"] < min_horizontal_area_ratio:
            reasons.append("horizontal_area_ratio_below_threshold")
        support = (
            _support_metrics(triangles, support_index, max_support_gap_m)
            if triangles is not None and len(triangles)
            else {
                "support_coverage": 0.0,
                "supported_area_m2": 0.0,
                "supported_face_count": 0,
                "support_sample_count": 0,
                "support_sample_total": 0,
                "support_gap_max_m": None,
                "support_floor_height_max_m": None,
            }
        )
        metrics.update(support)
        if support["support_coverage"] < min_support_coverage:
            reasons.append("support_coverage_below_threshold")
        if support["support_floor_height_max_m"] is None:
            reasons.append("supporting_floor_not_established")
        elif support["support_floor_height_max_m"] > max_floor_height_m:
            reasons.append("supporting_floor_above_height_threshold")
        entry = {"path": str(prim.GetPath()), "metrics": metrics}
        if reasons and policy == "v2" and (v2_candidate or entry["path"] in approvals):
            reasons = _v2_reasons(mesh, points, metrics, v2_support, ground_z)
            if entry["path"] not in approvals:
                reasons.append("instance_not_approved")
            elif reasons:
                raise ValueError(f"approved instance failed v2 geometry: {entry['path']}: {reasons}")
            entry["deletion_rule"] = "approved_v2"
        elif not reasons and policy == "v2":
            if list(prim.GetAllChildren()):
                raise ValueError(f"legacy removal contains child prims: {entry['path']}")
            entry["deletion_rule"] = "legacy_v1"
        if reasons:
            entry["reasons"] = reasons
            skipped.append(entry)
        else:
            remove_paths.append(prim.GetPath())
            removed.append(entry)

    if not set(approvals) <= {str(path) for path in remove_paths}:
        raise ValueError("not every approved instance was selected")
    if _sha256(input_path) != source_sha256:
        raise ValueError("input USD changed during sanitization")
    for path in remove_paths:
        stage.RemovePrim(path)

    input_mesh_count = len(mesh_records)
    input_face_count = sum(record["face_count"] for record in mesh_records)
    output_meshes = [prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)]
    output_face_count = sum(
        len(UsdGeom.Mesh(prim).GetFaceVertexCountsAttr().Get()) for prim in output_meshes
    )
    report = {
        "policy": POLICY_V2 if policy == "v2" else POLICY_ID,
        "source_path": str(input_path),
        "source_sha256": source_sha256,
        "thresholds": thresholds,
        "removed_prims": removed,
        "skipped_candidates": skipped,
        "input_mesh_count": input_mesh_count,
        "output_mesh_count": len(output_meshes),
        "input_face_count": input_face_count,
        "output_face_count": output_face_count,
    }
    if policy == "v2":
        report.update(v2_thresholds=V2_THRESHOLDS, ground_z_m=approval_record["ground_z_m"],
                      approved_manifest_sha256=_sha256(Path(approved_manifest)),
                      v2_shape_ratios_are_diagnostics=True)
    report["dry_run"] = dry_run
    report["sanitizer_sha256"] = _sha256(Path(__file__))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output_tmp = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    report_tmp = report_path.with_name(f".{report_path.name}.tmp-{os.getpid()}")
    try:
        if not dry_run:
            stage.Export(str(output_tmp), args={"format": "usdc"} if policy == "v2" else {})
        with report_tmp.open("w", encoding="utf-8") as file:
            json.dump(report, file, indent=2, sort_keys=True)
            file.write("\n")
        if not dry_run:
            os.replace(output_tmp, output_path)
        os.replace(report_tmp, report_path)
    finally:
        output_tmp.unlink(missing_ok=True)
        report_tmp.unlink(missing_ok=True)
    print(
        f"[SUMMARY] {input_path.name}: meshes={input_mesh_count}, "
        f"faces={input_face_count}, removed={len(removed)}, "
        f"output_meshes={len(output_meshes)}"
    )
    if not dry_run:
        print(f"[OK] Wrote sanitized collision USD: {output_path}")
    print(f"[OK] Wrote sanitation report: {report_path}")
    return report


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Raw collision USD")
    parser.add_argument("--output", type=Path, required=True, help="Sanitized collision USD")
    parser.add_argument("--report", type=Path, required=True, help="JSON sanitation report")
    parser.add_argument("--policy", choices=("v1", "v2"), default="v1")
    parser.add_argument("--approved-manifest", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    for name, default in DEFAULTS.items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=float, default=default)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    sanitize_collision_usd(
        args.input,
        args.output,
        args.report,
        max_overlay_thickness_m=args.max_overlay_thickness_m,
        min_horizontal_area_ratio=args.min_horizontal_area_ratio,
        max_support_gap_m=args.max_support_gap_m,
        min_support_coverage=args.min_support_coverage,
        max_floor_height_m=args.max_floor_height_m,
        policy=args.policy,
        approved_manifest=args.approved_manifest,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
