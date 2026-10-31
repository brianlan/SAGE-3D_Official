#!/usr/bin/env python3
"""Offline self-checks for splat_transform_ply_to_usdz_filter_z_build_usda.py: command ordering, default
paths, collision reference, final USDA naming, preserve/overwrite, temp PLY
cleanup. Runs no real tools and touches no production data."""

import errno
import importlib.util
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "splat_transform_ply_to_usdz_filter_z_build_usda",
    Path(__file__).with_name("splat_transform_ply_to_usdz_filter_z_build_usda.py"),
)
prep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prep)

DATA_PY = sys.executable


def test_default_paths():
    paths = prep.compute_paths("840133", 2.75, prep.DEFAULT_DATA_ROOT)
    assert str(paths["tmp_ply"]) == "/tmp/sage_ply/840133.ply"
    assert str(paths["usdz"]) == "/ssd5/datasets/SAGE3D/InteriorGS_usdz_from_ply/840133.usdz"
    assert str(paths["filtered_usdz"]) == "/ssd5/datasets/SAGE3D/Filtered_By_Z/840133_z2.75/840133.usdz"
    assert str(paths["collision"]) == "/ssd5/datasets/SAGE3D/Collision_Mesh/Collision_Mesh/840133/840133_collision.usd"
    assert str(paths["sanitized_collision"]) == "/ssd5/datasets/SAGE3D/Collision_Mesh_Sanitized/840133/840133_collision.usd"
    assert str(paths["sanitize_report"]) == "/ssd5/datasets/SAGE3D/Collision_Mesh_Sanitized/840133/840133_collision.sanitize.json"
    assert str(paths["final_usda"]) == "/ssd5/datasets/SAGE3D/InteriorGS_CollisionMesh_usda/840133_z2.75.usda"
    versioned = prep.compute_paths("840133", 2.75, prep.DEFAULT_DATA_ROOT,
                                   sanitized_collision_root=Path("/new/collision"),
                                   final_usda_root=Path("/new/scenes"))
    assert versioned["sanitized_collision"] == Path("/new/collision/840133/840133_collision.usd")
    assert versioned["sanitize_report"] == Path("/new/collision/840133/840133_collision.sanitize.json")
    assert versioned["final_usda"] == Path("/new/scenes/840133_z2.75.usda")
    assert versioned["collision"] == paths["collision"]
    assert versioned["filtered_usdz"] == paths["filtered_usdz"]
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


def test_relative_paths():
    paths = prep.compute_paths("840133", 2.75, Path("data"))
    assert paths["filtered_usdz"] == Path.cwd() / "data/Filtered_By_Z/840133_z2.75/840133.usdz"
    assert paths["sanitized_collision"] == Path.cwd() / "data/Collision_Mesh_Sanitized/840133/840133_collision.usd"
    versioned = prep.compute_paths("840133", 2.75, Path("data"),
                                   sanitized_collision_root=Path("collision-v2"),
                                   final_usda_root=Path("scenes-v2"))
    assert versioned["sanitized_collision"] == Path.cwd() / "collision-v2/840133/840133_collision.usd"
    assert versioned["sanitize_report"] == Path.cwd() / "collision-v2/840133/840133_collision.sanitize.json"
    assert versioned["final_usda"] == Path.cwd() / "scenes-v2/840133_z2.75.usda"
    print("PASS relative paths become absolute asset references")


def test_pipeline():
    real_run = prep.run
    real_tmp_ply_dir = prep.TMP_PLY_DIR
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        try:
            prep.TMP_PLY_DIR = root / "sage_ply"
            paths = prep.compute_paths("840133", 2.75, root)

            src = root / "InteriorGS" / "0403_840133" / "3dgs_compressed.ply"
            src.parent.mkdir(parents=True)
            src.write_bytes(b"ply")
            collision = paths["collision"]
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

            argv = ["--scene-id", "840133", "--data-root", str(root), "--data-py", DATA_PY]

            # First build
            prep.main(argv)
            assert len(record) == 5, "expected exactly five commands"
            splat, convert, filt, sanitize, build = record
            assert splat[0] == "splat-transform"
            assert splat[1:] == [str(src), str(paths["tmp_ply"])]
            assert Path(convert[1]).name == "sage_ply_to_usdz.py"
            assert Path(filt[1]).name == "filter_usdz_by_z.py"
            assert Path(sanitize[1]).name == "sanitize_collision_usd.py"
            assert Path(build[1]).name == "sage3d_usda_builder.py"
            assert convert[0] == filt[0] == sanitize[0] == build[0] == DATA_PY
            assert str(paths["usdz"]) in convert
            assert filt[filt.index("--z-threshold") + 1] == "2.75"
            assert str(paths["filtered_usdz"]) in filt
            assert sanitize[2] == str(collision)
            assert sanitize[sanitize.index("--output") + 1] == str(paths["sanitized_collision"])
            assert sanitize[sanitize.index("--report") + 1] == str(paths["sanitize_report"])
            assert "--policy" not in sanitize
            # USDA must reference filtered visual USDZ and SANITIZED collision
            assert build[build.index("--usdz-path-template") + 1] == str(
                paths["filtered_usdz"]) + "[gauss.usda]"
            assert build[build.index("--collision-path-template") + 1] == str(
                paths["sanitized_collision"]
            )

            final = paths["final_usda"]
            assert final.exists(), "final USDA missing"
            assert 'string authoring_layer = "./840133_z2.75.usda"' in final.read_text()
            assert not paths["tmp_ply"].exists(), "temp PLY not cleaned up"

            # Second build without --overwrite: preserved, nothing runs
            prep.main(argv)
            assert len(record) == 5, "existing USDA must not be rebuilt without --overwrite"

            # Third build with --overwrite: rebuilt
            prep.main(argv + ["--overwrite"])
            assert len(record) == 10

            # v2 forwards reviewed approvals and isolates collision/final USDA outputs.
            manifest = root / "approved.json"
            manifest.write_text("{}")
            collision_root, final_root = root / "collision-v2", root / "scenes-v2"
            prep.main(argv + ["--collision-policy", "v2", "--approved-manifest", str(manifest),
                              "--sanitized-collision-root", str(collision_root),
                              "--final-usda-root", str(final_root)])
            assert len(record) == 15
            sanitize, build = record[-2:]
            assert sanitize[sanitize.index("--policy") + 1] == "v2"
            assert sanitize[sanitize.index("--approved-manifest") + 1] == str(manifest)
            collision_v2 = collision_root / "840133/840133_collision.usd"
            assert sanitize[sanitize.index("--output") + 1] == str(collision_v2)
            assert build[build.index("--collision-path-template") + 1] == str(collision_v2)
            assert (final_root / "840133_z2.75.usda").exists()
            assert final.exists()

            # A cross-device rename must not be needed to publish the final file.
            with patch("os.rename", side_effect=OSError(errno.EXDEV, "cross-device link")):
                prep.main(argv + ["--overwrite"])
            assert 'string authoring_layer = "./840133_z2.75.usda"' in final.read_text()
            assert not paths["generated_usda"].exists()

            # Failed publication preserves both the previous final and the source.
            previous = final.read_bytes()
            with patch.object(Path, "replace", side_effect=OSError(errno.ENOSPC, "disk full")):
                try:
                    prep.main(argv + ["--overwrite"])
                except OSError as error:
                    assert error.errno == errno.ENOSPC
                else:
                    raise AssertionError("expected publication failure")
            assert final.read_bytes() == previous
            assert paths["generated_usda"].exists()
            assert list(final.parent.iterdir()) == [final], "staging files were not cleaned up"
            print("PASS cross-device publication and failure preserve existing output")

            # Guard: missing authoring_layer token must fail loudly
            bad_usda = True
            try:
                prep.main(argv + ["--overwrite"])
            except SystemExit:
                print("PASS authoring_layer guard")
            else:
                raise AssertionError("expected SystemExit on missing authoring_layer token")
            print("PASS pipeline ordering, sanitized collision ref, naming, preserve/overwrite")
        finally:
            prep.run = real_run
            prep.TMP_PLY_DIR = real_tmp_ply_dir


def test_v2_manifest_required_before_conversion():
    real_run = prep.run
    calls = []
    prep.run = lambda *args: calls.append(args)
    try:
        for argv in (["--collision-policy", "v2"],
                     ["--collision-policy", "v2", "--approved-manifest", "/nonexistent/approved.json"],
                     ["--approved-manifest", __file__]):
            try:
                prep.main(argv)
            except SystemExit as error:
                assert error.code != 0
            else:
                raise AssertionError("v2 without an existing manifest was accepted")
        assert not calls
    finally:
        prep.run = real_run
    print("PASS v2 manifest validation before conversion")


if __name__ == "__main__":
    test_default_paths()
    test_find_source_ply()
    test_relative_paths()
    test_pipeline()
    test_v2_manifest_required_before_conversion()
