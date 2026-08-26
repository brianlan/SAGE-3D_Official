#!/usr/bin/env python3
"""
Filter a collision-mesh USD file by removing mesh geometry with Z > threshold.

For every UsdGeom.Mesh prim:
  - Keep vertices where Z <= threshold.
  - Keep faces where ALL vertices are kept.
  - Remap face-vertex indices to the new vertex list.
  - If a mesh ends up with 0 points, the prim is removed.
  - If a mesh ends up with 0 faces but still has points, faces/faces counts are cleared.
"""

import argparse
from pathlib import Path

from pxr import Usd, UsdGeom, Vt


def filter_mesh_by_z(mesh: UsdGeom.Mesh, z_threshold: float) -> bool:
    """
    Filter a mesh prim in-place by Z threshold.

    Returns True if the mesh still has geometry after filtering,
    False if it became empty (caller may remove the prim).
    """
    points_attr = mesh.GetPointsAttr()
    points = points_attr.Get()
    if not points:
        return False

    # Build keep mask and index mapping
    keep_mask = [p[2] <= z_threshold for p in points]
    if all(keep_mask):
        return True  # nothing to do

    if not any(keep_mask):
        return False  # all vertices removed

    # Mapping: old_index -> new_index
    index_map = {}
    new_points = []
    for old_idx, p in enumerate(points):
        if keep_mask[old_idx]:
            index_map[old_idx] = len(new_points)
            new_points.append(p)

    # Filter faces: keep faces where ALL vertices are kept
    face_counts = mesh.GetFaceVertexCountsAttr().Get()
    face_indices = mesh.GetFaceVertexIndicesAttr().Get()
    if not face_counts or not face_indices:
        # No faces, just update points
        points_attr.Set(Vt.Vec3fArray(new_points))
        return len(new_points) > 0

    new_face_counts = []
    new_face_indices = []
    idx_ptr = 0
    for count in face_counts:
        face_old_indices = list(face_indices[idx_ptr:idx_ptr + count])
        idx_ptr += count

        if all(idx in index_map for idx in face_old_indices):
            new_face_counts.append(count)
            new_face_indices.extend(index_map[idx] for idx in face_old_indices)

    # Update attributes
    points_attr.Set(Vt.Vec3fArray(new_points))
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray(new_face_counts))
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(new_face_indices))

    # Recompute extent
    if new_points:
        zs = [p[2] for p in new_points]
        xs = [p[0] for p in new_points]
        ys = [p[1] for p in new_points]
        bbox = Vt.Vec3fArray([
            (min(xs), min(ys), min(zs)),
            (max(xs), max(ys), max(zs))
        ])
        mesh.GetExtentAttr().Set(bbox)

    return len(new_points) > 0 and len(new_face_counts) > 0


def process_collision_usd(input_path: Path, output_path: Path, z_threshold: float):
    """Open a collision USD, filter meshes by Z, and export."""
    stage = Usd.Stage.Open(str(input_path))
    if not stage:
        raise RuntimeError(f"Failed to open USD stage: {input_path}")

    prims_to_remove = []
    mesh_count = 0
    removed_count = 0

    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            mesh_count += 1
            has_geometry = filter_mesh_by_z(mesh, z_threshold)
            if not has_geometry:
                prims_to_remove.append(prim)

    # Remove empty prims
    for prim in prims_to_remove:
        stage.RemovePrim(prim.GetPath())
        removed_count += 1

    print(
        f"[INFO] {input_path.name}: meshes={mesh_count}, "
        f"removed_empty={removed_count}, kept={mesh_count - removed_count}"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stage.Export(str(output_path))
    print(f"[OK] Wrote filtered collision USD to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Remove collision-mesh geometry with Z > threshold from a USD file"
    )
    parser.add_argument("input", type=Path, help="Input collision USD file")
    parser.add_argument("--output", "-o", type=Path, help="Output USD file (default: overwrite input)")
    parser.add_argument("--z-threshold", "-z", type=float, required=True, help="Keep vertices with Z <= threshold")
    args = parser.parse_args()

    output = args.output if args.output else args.input
    process_collision_usd(args.input, output, args.z_threshold)


if __name__ == "__main__":
    main()
