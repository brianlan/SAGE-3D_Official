#!/usr/bin/env python3
"""Focused executable checks for sanitize_collision_usd.py."""

import importlib.util
import json
import tempfile
from pathlib import Path

from pxr import Gf, Usd, UsdGeom, Vt

SPEC = importlib.util.spec_from_file_location(
    "sanitize_collision_usd", Path(__file__).with_name("sanitize_collision_usd.py")
)
sanitizer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sanitizer)


def add_mesh(stage, path, points, faces, transform=None):
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.GetPointsAttr().Set(Vt.Vec3dArray(points))
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([len(face) for face in faces]))
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray([index for face in faces for index in face]))
    if transform is not None:
        xform = UsdGeom.Xformable(mesh)
        translation, rotation_z, scale = transform
        xform.AddTranslateOp().Set(Gf.Vec3d(*translation))
        xform.AddRotateZOp().Set(rotation_z)
        xform.AddScaleOp().Set(Gf.Vec3d(*scale))
    return mesh


def make_stage(
    path, candidates=(), *, high_floor=False, multiple_floorplans=False, include_floor=True
):
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.Xform.Define(stage, "/Root")
    floor_points = [(-2, -2, 0), (2, -2, 0), (2, 2, 0), (-2, 2, 0)]
    floor_faces = [(0, 1, 2), (0, 2, 3)]
    if high_floor:
        floor_points += [(-2, -2, 3), (2, -2, 3), (2, 2, 3), (-2, 2, 3)]
        floor_faces += [(4, 6, 5), (4, 7, 6)]
    if include_floor:
        add_mesh(stage, "/Root/SM_floorplan", floor_points, floor_faces)
    if multiple_floorplans:
        UsdGeom.Xform.Define(stage, "/Root/Other")
        add_mesh(stage, "/Root/Other/SM_floorplan", floor_points, floor_faces)
    for path_name, points, faces, transform in candidates:
        add_mesh(stage, path_name, points, faces, transform)
    stage.GetRootLayer().Save()


def box(z, thickness=0.001, x0=-1, x1=1, y0=-1, y1=1):
    points = [
        (x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z),
        (x0, y0, z + thickness), (x1, y0, z + thickness),
        (x1, y1, z + thickness), (x0, y1, z + thickness),
    ]
    faces = [
        (0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
        (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
        (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7),
    ]
    return points, faces


def run(input_path, output_path, report_path):
    return sanitizer.sanitize_collision_usd(input_path, output_path, report_path)


def assert_fails(path, *, multiple=False):
    make_stage(path, multiple_floorplans=multiple, include_floor=multiple)
    try:
        run(path, path.with_name("out.usd"), path.with_name("out.json"))
    except RuntimeError:
        return
    raise AssertionError("expected floorplan validation failure")


def assert_rejects_invalid_contract(source):
    output = source.with_name("contract_out.usd")
    report = source.with_name("contract.json")
    original = source.read_bytes()
    for kwargs in (
        {"output_path": source},
        {"report_path": source},
        {"output_path": report},
        {"max_overlay_thickness_m": -1},
        {"max_support_gap_m": -1},
        {"max_floor_height_m": -1},
        {"min_horizontal_area_ratio": 1.1},
        {"min_support_coverage": -0.1},
    ):
        args = {"output_path": output, "report_path": report}
        args.update(kwargs)
        try:
            sanitizer.sanitize_collision_usd(source, **args)
        except ValueError:
            assert source.read_bytes() == original
            continue
        raise AssertionError(f"expected contract validation failure: {kwargs}")


def main():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)

        # World transform: local z=0 is translated above the floor, rotated,
        # and scaled in separate USD xform ops.
        points, faces = box(0.0)
        transform = ((0.2, 0.1, 0.001), 25, (0.7, 0.7, 1))
        source = root / "supported.usd"
        make_stage(source, [("/Root/Overlay", points, faces, transform)])
        report = run(source, root / "supported_out.usd", root / "supported.json")
        assert [entry["path"] for entry in report["removed_prims"]] == ["/Root/Overlay"]
        metrics = report["removed_prims"][0]["metrics"]
        assert 0.0009 < metrics["thickness_m"] < 0.0011
        assert 3.8 < metrics["area_m2"] < 4.1
        assert metrics["support_gap_max_m"] > 0.0011
        assert metrics["support_coverage"] > 0.99
        assert report["output_mesh_count"] == report["input_mesh_count"] - 1
        assert report["output_face_count"] == report["input_face_count"] - len(faces)
        output = Usd.Stage.Open(str(root / "supported_out.usd"))
        assert not output.GetPrimAtPath("/Root/Overlay")
        assert json.loads((root / "supported.json").read_text())["policy"] == sanitizer.POLICY_ID

        cases = {
            "unsupported": box(0.5),
            "ceiling": box(3.001),
            "step": box(0.001, thickness=0.01),
            "partial": box(0.001, x0=-1, x1=4),
        }
        for name, geometry in cases.items():
            source = root / f"{name}.usd"
            make_stage(
                source,
                [(f"/Root/{name}", geometry[0], geometry[1], None)],
                high_floor=name == "ceiling",
            )
            report = run(source, root / f"{name}_out.usd", root / f"{name}.json")
            assert not report["removed_prims"], name
            assert report["output_mesh_count"] == report["input_mesh_count"], name

        # The centroid is over the floor, but most of this huge triangle is not.
        source = root / "adversarial_large_triangle.usd"
        large_points = [(-100, -100, 0.001), (100, -100, 0.001), (0, 200, 0.001)]
        make_stage(source, [("/Root/large_triangle", large_points, [(0, 1, 2)], None)])
        report = run(
            source,
            root / "adversarial_large_triangle_out.usd",
            root / "adversarial_large_triangle.json",
        )
        assert not report["removed_prims"]
        assert report["skipped_candidates"][0]["metrics"]["support_coverage"] < 0.95

        # A rotated thin sheet is not predominantly horizontal in world space.
        source = root / "ramp.usd"
        ramp_points, ramp_faces = box(0.001)
        ramp_points = [(x, y, z + 0.3 * x) for x, y, z in ramp_points]
        make_stage(source, [("/Root/ramp", ramp_points, ramp_faces, None)])
        assert not run(source, root / "ramp_out.usd", root / "ramp.json")["removed_prims"]

        assert_fails(root / "missing.usd")
        assert_fails(root / "multiple.usd", multiple=True)
        assert_rejects_invalid_contract(source)
    print("PASS sanitizer synthetic cases")


if __name__ == "__main__":
    main()
