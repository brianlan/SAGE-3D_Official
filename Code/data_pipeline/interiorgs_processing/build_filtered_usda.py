#!/usr/bin/env python3
"""
Build a Z-filtered USDA for a SAGE-3D scene.

Orchestrates three existing tools:
  1. filter_usdz_by_z.py      — filters gaussians in the USDZ
  2. filter_collision_usd_by_z.py — filters the collision mesh
  3. sage3d_usda_builder.py   — generates the USDA wrapper

The output is saved as {scene_id}_z{max_z}.usda so the original unfiltered
USDA is never overwritten.
"""

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ORIG_USDZ_DIR = Path("/ssd5/datasets/SAGE3D/InteriorGS_usdz")
ORIG_COLLISION_DIR = Path("/ssd5/datasets/SAGE3D/Collision_Mesh/Collision_Mesh")
DEFAULT_WORK_DIR = Path("/ssd5/datasets/SAGE3D/Filtered_By_Z")
DEFAULT_OUT_DIR = Path("/ssd5/datasets/SAGE3D/InteriorGS_CollisionMesh_usda")
TEMPLATE_PATH = REPO_ROOT / "Data" / "template.usda"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_cmd(cmd: list[str], description: str):
    """Run a subprocess and exit on failure."""
    print(f"[RUN] {description}")
    print(f"      {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"[ERROR] {description} failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="Build a Z-filtered USDA variant for a SAGE-3D scene"
    )
    parser.add_argument("--scene-id", required=True, help="Numeric scene ID (e.g., 840040)")
    parser.add_argument("--max-z", type=float, required=True, help="Z threshold (keep Z <= max_z)")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=DEFAULT_WORK_DIR,
        help="Directory to store filtered intermediate assets",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for the final USDA output",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output")
    args = parser.parse_args()

    scene_id = args.scene_id
    max_z = args.max_z
    work_dir = args.work_dir / f"{scene_id}_z{max_z}"
    final_usda = args.out_dir / f"{scene_id}_z{max_z}.usda"

    # Validate originals exist
    orig_usdz = ORIG_USDZ_DIR / f"{scene_id}.usdz"
    orig_collision = ORIG_COLLISION_DIR / scene_id / f"{scene_id}_collision.usd"

    if not orig_usdz.exists():
        print(f"[ERROR] Original USDZ not found: {orig_usdz}")
        sys.exit(1)
    if not orig_collision.exists():
        print(f"[ERROR] Original collision mesh not found: {orig_collision}")
        sys.exit(1)

    # Guard against overwriting original USDA
    original_usda = args.out_dir / f"{scene_id}.usda"
    if final_usda == original_usda:
        print(f"[ERROR] Output path would overwrite original USDA: {final_usda}")
        sys.exit(1)

    if final_usda.exists() and not args.overwrite:
        print(f"[SKIP] Output already exists: {final_usda} (use --overwrite to regenerate)")
        sys.exit(0)

    # Clean and recreate working directory
    if work_dir.exists():
        import shutil
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Filter USDZ
    # ------------------------------------------------------------------
    filtered_usdz = work_dir / f"{scene_id}.usdz"
    run_cmd(
        [
            sys.executable,
            str(REPO_ROOT / "Code" / "data_pipeline" / "interiorgs_processing" / "filter_usdz_by_z.py"),
            str(orig_usdz),
            "--output", str(filtered_usdz),
            "--z-threshold", str(max_z),
        ],
        "Filter USDZ gaussians by Z",
    )

    # ------------------------------------------------------------------
    # Step 2: Filter collision mesh
    # ------------------------------------------------------------------
    filtered_collision = work_dir / f"{scene_id}_collision.usd"
    run_cmd(
        [
            sys.executable,
            str(REPO_ROOT / "Code" / "data_pipeline" / "interiorgs_processing" / "filter_collision_usd_by_z.py"),
            str(orig_collision),
            "--output", str(filtered_collision),
            "--z-threshold", str(max_z),
        ],
        "Filter collision mesh by Z",
    )

    # ------------------------------------------------------------------
    # Step 3: Build USDA
    # ------------------------------------------------------------------
    run_cmd(
        [
            sys.executable,
            str(REPO_ROOT / "Code" / "benchmark" / "scene_data" / "sage3d_usda_builder.py"),
            "--usdz-dir", str(work_dir),
            "--out-dir", str(work_dir),
            "--template", str(TEMPLATE_PATH),
            "--base-id", "839920",
            "--expected-count", "1",
            "--usdz-placeholder", "@usdz_root[gauss.usda]@",
            "--usdz-path-template", f"{work_dir}/{scene_id}.usdz[gauss.usda]",
            "--collision-placeholder", "@collision_root@",
            "--collision-path-template", f"{work_dir}/{scene_id}_collision.usd",
            "--overwrite",
        ],
        "Build USDA from filtered assets",
    )

    # ------------------------------------------------------------------
    # Step 4: Fix authoring layer and move to final output
    # ------------------------------------------------------------------
    generated_usda = work_dir / f"{scene_id}.usda"
    if not generated_usda.exists():
        print(f"[ERROR] Expected generated USDA not found: {generated_usda}")
        sys.exit(1)

    content = generated_usda.read_text(encoding="utf-8")
    content = content.replace(
        f'string authoring_layer = "./{scene_id}.usda"',
        f'string authoring_layer = "./{scene_id}_z{max_z}.usda"',
    )
    generated_usda.write_text(content, encoding="utf-8")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    generated_usda.rename(final_usda)

    print(f"[OK] Filtered USDA saved to: {final_usda}")


if __name__ == "__main__":
    main()
