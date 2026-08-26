#!/usr/bin/env python3
"""
Build a complete SAGE-3D scene from compressed PLY to final Z-filtered USDA.

Sequential, fail-fast pipeline:
  1. splat-transform        InteriorGS compressed PLY -> temporary PLY
  2. sage_ply_to_usdz.py    temporary PLY -> USDZ
  3. filter_usdz_by_z.py    USDZ -> Z-filtered USDZ
  4. sage3d_usda_builder.py filtered USDZ + original collision -> final USDA

The final USDA references the filtered visual USDZ and the original,
unfiltered collision mesh. Existing final USDA files are preserved unless
--overwrite is given.
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
BUILDER = REPO_ROOT / "Code" / "benchmark" / "scene_data" / "sage3d_usda_builder.py"
TEMPLATE_PATH = REPO_ROOT / "Data" / "template.usda"

# Interpreter for data-prep subprocesses (msgpack / numpy / pxr live here).
DATA_PY = "/ssd4/envs/vln_data_prep_py311/bin/python"

INTERIORGS_ROOT = Path("/ssd5/datasets/SAGE3D/InteriorGS")
USDZ_DIR = Path("/ssd5/datasets/SAGE3D/InteriorGS_usdz_from_ply")
FILTERED_ROOT = Path("/ssd5/datasets/SAGE3D/Filtered_By_Z")
COLLISION_DIR = Path("/ssd5/datasets/SAGE3D/Collision_Mesh/Collision_Mesh")
OUT_DIR = Path("/ssd5/datasets/SAGE3D/InteriorGS_CollisionMesh_usda")
TMP_PLY_DIR = Path("/tmp/sage_ply")


def find_source_ply(scene_id: str, input_root: Path) -> Path:
    """Locate <input_root>/<prefix>_<scene_id>/3dgs_compressed.ply."""
    matches = sorted(input_root.glob(f"*_{scene_id}/3dgs_compressed.ply"))
    if not matches:
        sys.exit(f"[ERROR] No InteriorGS folder matching '*_{scene_id}' under {input_root}")
    if len(matches) > 1:
        sys.exit(f"[ERROR] Multiple InteriorGS matches for '*_{scene_id}': {matches}")
    return matches[0]


def compute_paths(scene_id: str, max_z: float) -> dict:
    """All input/output paths for one scene build (no filesystem access)."""
    work_dir = FILTERED_ROOT / f"{scene_id}_z{max_z}"
    return {
        "tmp_ply": TMP_PLY_DIR / f"{scene_id}.ply",
        "usdz": USDZ_DIR / f"{scene_id}.usdz",
        "work_dir": work_dir,
        "filtered_usdz": work_dir / f"{scene_id}.usdz",
        "collision": COLLISION_DIR / scene_id / f"{scene_id}_collision.usd",
        "generated_usda": work_dir / f"{scene_id}.usda",
        "final_usda": OUT_DIR / f"{scene_id}_z{max_z}.usda",
    }


def run(cmd: list, desc: str) -> None:
    print(f"[RUN] {desc}\n      {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Z-filtered SAGE-3D scene from compressed PLY to final USDA"
    )
    parser.add_argument("--scene-id", default="840133", help="Numeric scene ID (default: 840133)")
    parser.add_argument(
        "--source-ply",
        type=Path,
        help="Override compressed PLY path (default: <input-root>/*_<scene-id>/3dgs_compressed.ply)",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=INTERIORGS_ROOT,
        help="Root of InteriorGS scene folders (default: /ssd5/datasets/SAGE3D/InteriorGS)",
    )
    parser.add_argument("--max-z", type=float, default=2.75, help="Z threshold (keep Z <= max-z)")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing final USDA")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    scene_id = args.scene_id
    max_z = args.max_z
    source_ply = args.source_ply or find_source_ply(scene_id, args.input_root)
    paths = compute_paths(scene_id, max_z)

    if not source_ply.exists():
        sys.exit(f"[ERROR] Source PLY not found: {source_ply}")
    if not paths["collision"].exists():
        sys.exit(f"[ERROR] Collision mesh not found: {paths['collision']}")
    if paths["final_usda"].exists() and not args.overwrite:
        print(f"[SKIP] {paths['final_usda']} already exists (use --overwrite to replace)")
        return

    for d in (paths["tmp_ply"].parent, paths["usdz"].parent, paths["work_dir"]):
        d.mkdir(parents=True, exist_ok=True)

    run(
        ["splat-transform", str(source_ply), str(paths["tmp_ply"])],
        "Decompress compressed PLY",
    )
    run(
        [DATA_PY, str(SCRIPT_DIR / "sage_ply_to_usdz.py"),
         str(paths["tmp_ply"]), "--output_file", str(paths["usdz"])],
        "Convert PLY to USDZ",
    )
    paths["tmp_ply"].unlink(missing_ok=True)
    run(
        [DATA_PY, str(SCRIPT_DIR / "filter_usdz_by_z.py"),
         str(paths["usdz"]), "--output", str(paths["filtered_usdz"]),
         "--z-threshold", str(max_z)],
        f"Filter USDZ gaussians by Z <= {max_z}",
    )
    run(
        [DATA_PY, str(BUILDER),
         "--usdz-dir", str(paths["work_dir"]),
         "--out-dir", str(paths["work_dir"]),
         "--template", str(TEMPLATE_PATH),
         "--usdz-placeholder", "@usdz_root[gauss.usda]@",
         "--usdz-path-template", f"{paths['filtered_usdz']}[gauss.usda]",
         "--collision-placeholder", "@collision_root@",
         "--collision-path-template", str(paths["collision"]),
         "--overwrite"],
        "Build USDA wrapper",
    )

    generated = paths["generated_usda"]
    if not generated.exists():
        sys.exit(f"[ERROR] Expected generated USDA not found: {generated}")
    content = generated.read_text(encoding="utf-8")
    old_token = f'string authoring_layer = "./{scene_id}.usda"'
    new_token = f'string authoring_layer = "./{scene_id}_z{max_z}.usda"'
    if old_token not in content:
        sys.exit(f"[ERROR] Expected authoring_layer token not found in {generated}: {old_token}")
    content = content.replace(old_token, new_token)
    generated.write_text(content, encoding="utf-8")
    paths["final_usda"].parent.mkdir(parents=True, exist_ok=True)
    generated.rename(paths["final_usda"])
    print(f"[OK] Final USDA: {paths['final_usda']}")


if __name__ == "__main__":
    main()
