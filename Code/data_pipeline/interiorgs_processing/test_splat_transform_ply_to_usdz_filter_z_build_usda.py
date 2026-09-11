#!/usr/bin/env python3
"""Offline self-checks for splat_transform_ply_to_usdz_filter_z_build_usda.py: command ordering, default
paths, collision reference, final USDA naming, preserve/overwrite, temp PLY
cleanup. Runs no real tools and touches no production data."""

import importlib.util
import tempfile
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "splat_transform_ply_to_usdz_filter_z_build_usda",
    Path(__file__).with_name("splat_transform_ply_to_usdz_filter_z_build_usda.py"),
)
prep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prep)

DATA_PY = "/ssd4/envs/vln_data_prep_py311/bin/python"
CONSTANTS = (
    "INTERIORGS_ROOT", "USDZ_DIR", "FILTERED_ROOT", "COLLISION_DIR",
    "SANITIZED_COLLISION_DIR", "OUT_DIR", "TMP_PLY_DIR",
)


def test_default_paths():
    paths = prep.compute_paths("840133", 2.75)
    assert str(paths["tmp_ply"]) == "/tmp/sage_ply/840133.ply"
    assert str(paths["usdz"]) == "/ssd5/datasets/SAGE3D/InteriorGS_usdz_from_ply/840133.usdz"
    assert str(paths["filtered_usdz"]) == "/ssd5/datasets/SAGE3D/Filtered_By_Z/840133_z2.75/840133.usdz"
    assert str(paths["collision"]) == "/ssd5/datasets/SAGE3D/Collision_Mesh/Collision_Mesh/840133/840133_collision.usd"
    assert str(paths["sanitized_collision"]) == "/ssd5/datasets/SAGE3D/Collision_Mesh_Sanitized/840133/840133_collision.usd"
    assert str(paths["sanitize_report"]) == "/ssd5/datasets/SAGE3D/Collision_Mesh_Sanitized/840133/840133_collision.sanitize.json"
    assert str(paths["final_usda"]) == "/ssd5/datasets/SAGE3D/InteriorGS_CollisionMesh_usda/840133_z2.75.usda"
    print("PASS default paths")


def test_find_source_ply():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        ply = root / "0403_840133" / "3dgs_compressed.ply"
        ply.parent.mkdir(parents=True)
        ply.write_bytes(b"ply")
        assert prep.find_source_ply("840133", root) == ply
        try:
            prep.find_source_ply("999999", root)
        except SystemExit:
            print("PASS find_source_ply")
            return
        raise AssertionError("expected SystemExit for unknown scene")


def test_pipeline():
    saved = {name: getattr(prep, name) for name in CONSTANTS}
    real_run = prep.run
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        try:
            prep.INTERIORGS_ROOT = tmp / "interiorgs"
            prep.USDZ_DIR = tmp / "usdz_from_ply"
            prep.FILTERED_ROOT = tmp / "filtered"
            prep.COLLISION_DIR = tmp / "collision"
            prep.SANITIZED_COLLISION_DIR = tmp / "sanitized"
            prep.OUT_DIR = tmp / "out"
            prep.TMP_PLY_DIR = tmp / "sage_ply"

            src = prep.INTERIORGS_ROOT / "0403_840133" / "3dgs_compressed.ply"
            src.parent.mkdir(parents=True)
            src.write_bytes(b"ply")
            collision = prep.COLLISION_DIR / "840133" / "840133_collision.usd"
            collision.parent.mkdir(parents=True)
            collision.write_bytes(b"usd")

            record = []
            bad_usda = False

            def fake_run(cmd, desc):
                record.append(list(cmd))
                if cmd[0] == "splat-transform":
                    Path(cmd[2]).write_bytes(b"ply")
                script = Path(cmd[1]).name if len(cmd) > 1 else ""
                if script == "filter_usdz_by_z.py":
                    Path(cmd[cmd.index("--output") + 1]).write_bytes(b"usdz")
                elif script == "sanitize_collision_usd.py":
                    output = Path(cmd[cmd.index("--output") + 1])
                    report = Path(cmd[cmd.index("--report") + 1])
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_bytes(b"usd")
                    report.write_text("{}")
                elif script == "sage3d_usda_builder.py":
                    usdz_dir = Path(cmd[cmd.index("--usdz-dir") + 1])
                    scene_id = next(usdz_dir.glob("*.usdz")).stem
                    token = "BAD" if bad_usda else f'string authoring_layer = "./{scene_id}.usda"'
                    (usdz_dir / f"{scene_id}.usda").write_text(token + "\n")

            prep.run = fake_run

            # First build
            prep.main(["--scene-id", "840133"])
            assert len(record) == 5, "expected exactly five commands"
            splat, convert, filt, sanitize, build = record
            assert splat[0] == "splat-transform"
            assert splat[1:] == [str(src), str(prep.TMP_PLY_DIR / "840133.ply")]
            assert Path(convert[1]).name == "sage_ply_to_usdz.py"
            assert Path(filt[1]).name == "filter_usdz_by_z.py"
            assert Path(sanitize[1]).name == "sanitize_collision_usd.py"
            assert Path(build[1]).name == "sage3d_usda_builder.py"
            assert convert[0] == filt[0] == sanitize[0] == build[0] == DATA_PY
            assert str(prep.USDZ_DIR / "840133.usdz") in convert
            assert filt[filt.index("--z-threshold") + 1] == "2.75"
            assert str(prep.FILTERED_ROOT / "840133_z2.75" / "840133.usdz") in filt
            assert sanitize[2] == str(collision)
            assert sanitize[sanitize.index("--output") + 1] == str(
                prep.SANITIZED_COLLISION_DIR / "840133" / "840133_collision.usd"
            )
            assert sanitize[sanitize.index("--report") + 1] == str(
                prep.SANITIZED_COLLISION_DIR / "840133" / "840133_collision.sanitize.json"
            )
            # USDA must reference filtered visual USDZ and SANITIZED collision
            assert build[build.index("--usdz-path-template") + 1] == str(
                prep.FILTERED_ROOT / "840133_z2.75" / "840133.usdz") + "[gauss.usda]"
            assert build[build.index("--collision-path-template") + 1] == str(
                prep.SANITIZED_COLLISION_DIR / "840133" / "840133_collision.usd"
            )

            final = prep.OUT_DIR / "840133_z2.75.usda"
            assert final.exists(), "final USDA missing"
            assert 'string authoring_layer = "./840133_z2.75.usda"' in final.read_text()
            assert not (prep.TMP_PLY_DIR / "840133.ply").exists(), "temp PLY not cleaned up"

            # Second build without --overwrite: preserved, nothing runs
            prep.main(["--scene-id", "840133"])
            assert len(record) == 5, "existing USDA must not be rebuilt without --overwrite"

            # Third build with --overwrite: rebuilt
            prep.main(["--scene-id", "840133", "--overwrite"])
            assert len(record) == 10

            # Guard: missing authoring_layer token must fail loudly
            bad_usda = True
            try:
                prep.main(["--scene-id", "840133", "--overwrite"])
            except SystemExit:
                print("PASS authoring_layer guard")
            else:
                raise AssertionError("expected SystemExit on missing authoring_layer token")
            print("PASS pipeline ordering, sanitized collision ref, naming, preserve/overwrite")
        finally:
            prep.run = real_run
            for name, value in saved.items():
                setattr(prep, name, value)


if __name__ == "__main__":
    test_default_paths()
    test_find_source_ply()
    test_pipeline()
