#!/usr/bin/env python3
"""
Build a complete SAGE-3D scene from compressed PLY to final Z-filtered USDA.

Sequential, fail-fast pipeline:
  1. splat-transform        InteriorGS compressed PLY -> temporary PLY
  2. sage_ply_to_usdz.py    temporary PLY -> USDZ
  3. filter_usdz_by_z.py    USDZ -> Z-filtered USDZ
  4. sanitize_collision_usd.py raw collision -> sanitized collision + report
  5. sage3d_usda_builder.py filtered visual + sanitized collision -> final USDA

The final USDA references the filtered visual USDZ and the sanitized collision
mesh. Existing final USDA files are preserved unless
--overwrite is given.
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
BUILDER = REPO_ROOT / "Code" / "benchmark" / "scene_data" / "sage3d_usda_builder.py"
TEMPLATE_PATH = REPO_ROOT / "Data" / "template.usda"

DEFAULT_DATA_ROOT = Path("/ssd5/datasets/SAGE3D")
TMP_PLY_DIR = Path("/tmp/sage_ply")


def find_source_ply(scene_id: str, input_root: Path) -> Path:
    """Locate <input_root>/<prefix>_<scene_id>/3dgs_compressed.ply."""
    matches = sorted(input_root.glob(f"*_{scene_id}/3dgs_compressed.ply"))
    if not matches:
        sys.exit(f"[ERROR] No InteriorGS folder matching '*_{scene_id}' under {input_root}")
    if len(matches) > 1:
        sys.exit(f"[ERROR] Multiple InteriorGS matches for '*_{scene_id}': {matches}")
    return matches[0]


def compute_paths(scene_id: str, max_z: float, data_root: Path, *,
                  sanitized_collision_root: Path | None = None,
                  final_usda_root: Path | None = None) -> dict:
    """Absolute input/output paths, keeping asset references valid after relocation."""
    data_root = data_root.resolve()
    work_dir = data_root / "Filtered_By_Z" / f"{scene_id}_z{max_z}"
    collision_dir = data_root / "Collision_Mesh" / "Collision_Mesh"
    sanitized_dir = (sanitized_collision_root or data_root / "Collision_Mesh_Sanitized").resolve()
    final_dir = (final_usda_root or data_root / "InteriorGS_CollisionMesh_usda").resolve()
    return {
        "tmp_ply": TMP_PLY_DIR / f"{scene_id}.ply",
        "usdz": data_root / "InteriorGS_usdz_from_ply" / f"{scene_id}.usdz",
        "work_dir": work_dir,
        "filtered_usdz": work_dir / f"{scene_id}.usdz",
        "collision": collision_dir / scene_id / f"{scene_id}_collision.usd",
        "sanitized_collision": sanitized_dir / scene_id / f"{scene_id}_collision.usd",
        "sanitize_report": sanitized_dir / scene_id / f"{scene_id}_collision.sanitize.json",
        "generated_usda": work_dir / f"{scene_id}.usda",
        "final_usda": final_dir / f"{scene_id}_z{max_z}.usda",
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
        default=None,
        help="Root of InteriorGS scene folders (default: <data-root>/InteriorGS)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Root of all SAGE3D data (default: {DEFAULT_DATA_ROOT})",
    )
    parser.add_argument(
        "--data-py",
        default=sys.executable,
        help="Python interpreter for subprocesses with msgpack/numpy/pxr (default: current interpreter)",
    )
    parser.add_argument("--max-z", type=float, default=2.75, help="Z threshold (keep Z <= max-z)")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing final USDA")
    parser.add_argument("--collision-policy", choices=("v1", "v2"), default="v1")
    parser.add_argument("--approved-manifest", type=Path, help="Reviewed collision-removal manifest for v2")
    parser.add_argument("--sanitized-collision-root", type=Path, help="Override sanitized collision output root")
    parser.add_argument("--final-usda-root", type=Path, help="Override final scene USDA output root")
    args = parser.parse_args(argv)
    if args.collision_policy == "v2" and args.approved_manifest is None:
        parser.error("--collision-policy v2 requires --approved-manifest")
    if args.collision_policy == "v1" and args.approved_manifest is not None:
        parser.error("--approved-manifest requires --collision-policy v2")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.approved_manifest is not None and not args.approved_manifest.is_file():
        sys.exit(f"[ERROR] Approved collision manifest not found: {args.approved_manifest}")
    scene_id = args.scene_id
    max_z = args.max_z
    input_root = args.input_root or args.data_root / "InteriorGS"
    source_ply = args.source_ply or find_source_ply(scene_id, input_root)
    paths = compute_paths(scene_id, max_z, args.data_root,
                          sanitized_collision_root=args.sanitized_collision_root,
                          final_usda_root=args.final_usda_root)

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
        [args.data_py, str(SCRIPT_DIR / "sage_ply_to_usdz.py"),
         str(paths["tmp_ply"]), "--output_file", str(paths["usdz"])],
        "Convert PLY to USDZ",
    )
    paths["tmp_ply"].unlink(missing_ok=True)
    run(
        [args.data_py, str(SCRIPT_DIR / "filter_usdz_by_z.py"),
         str(paths["usdz"]), "--output", str(paths["filtered_usdz"]),
         "--z-threshold", str(max_z)],
        f"Filter USDZ gaussians by Z <= {max_z}",
    )
    sanitize_command = [args.data_py, str(SCRIPT_DIR / "sanitize_collision_usd.py"),
                        str(paths["collision"]), "--output", str(paths["sanitized_collision"]),
                        "--report", str(paths["sanitize_report"])]
    if args.collision_policy == "v2":
        sanitize_command += ["--policy", "v2", "--approved-manifest", str(args.approved_manifest)]
    run(sanitize_command, "Remove supported thin collision overlays")
    run(
        [args.data_py, str(BUILDER),
         "--usdz-dir", str(paths["work_dir"]),
         "--out-dir", str(paths["work_dir"]),
         "--template", str(TEMPLATE_PATH),
         "--usdz-placeholder", "@usdz_root[gauss.usda]@",
         "--usdz-path-template", f"{paths['filtered_usdz']}[gauss.usda]",
         "--collision-placeholder", "@collision_root@",
         "--collision-path-template", str(paths["sanitized_collision"]),
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
    final = paths["final_usda"]
    final.parent.mkdir(parents=True, exist_ok=True)
    # Stage on the destination disk so replacement stays atomic across output disks.
    with tempfile.TemporaryDirectory(dir=final.parent) as td:
        staged = Path(td) / final.name
        staged.write_text(content, encoding="utf-8")
        staged.replace(final)
    generated.unlink()
    print(f"[OK] Final USDA: {final}")


if __name__ == "__main__":
    main()
