#!/usr/bin/env python3
"""Focused executable checks for sanitize_collision_usd.py."""

import importlib.util
import json
import tempfile
from pathlib import Path

import numpy as np

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


def test_v2():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "raw.usd"
        geometries = {"Carpet": box(.0015, .015),
                      "Checker": box(.0015, .002, x0=.9, x1=.95, y0=.9, y1=.95),
                      "Drain": box(.0015, .003, x0=-.2, x1=-.102, y0=-.2, y1=-.102),
                      "Unknown": box(.0015, .015), "Legacy": box(.001, .001)}
        make_stage(source, [(f"/Root/{name}", p, f, None) for name, (p, f) in geometries.items()])
        approvals = []
        source_stage = Usd.Stage.Open(str(source))
        UsdGeom.SetStageUpAxis(source_stage, "Z")
        UsdGeom.SetStageMetersPerUnit(source_stage, 1.)
        source_stage.GetRootLayer().Save()
        for name in ("Carpet", "Checker", "Drain"):
            mesh = UsdGeom.Mesh(source_stage.GetPrimAtPath(f"/Root/{name}"))
            p = sanitizer._world_points(mesh, UsdGeom.XformCache())
            f = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
            approvals.append({"path": f"/Root/{name}", "world_geometry_sha256":
                              sanitizer._geometry_sha256(np.array(p), np.array(f))})
        manifest = {"policy": sanitizer.POLICY_V2, "scenes": {"test": {
            "source_sha256": sanitizer._sha256(source), "ground_z_m": 0., "approvals": approvals}}}
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        kwargs = dict(policy="v2", approved_manifest=manifest_path)
        dry = sanitizer.sanitize_collision_usd(source, root / "dry.usd", root / "dry.json", dry_run=True, **kwargs)
        assert not (root / "dry.usd").exists()
        actual = sanitizer.sanitize_collision_usd(source, root / "out.usd", root / "out.json", **kwargs)
        assert actual["removed_prims"] == dry["removed_prims"]
        assert {entry["path"] for entry in actual["removed_prims"]} == {
            "/Root/Carpet", "/Root/Checker", "/Root/Drain", "/Root/Legacy"}
        assert any(entry["path"] == "/Root/Unknown" and "instance_not_approved" in entry["reasons"]
                   for entry in actual["skipped_candidates"])
        assert actual["policy"] == sanitizer.POLICY_V2
        saved = source.read_bytes()
        for change in ("source", "geometry", "floor", "missing", "ground", "child"):
            value = json.loads(json.dumps(manifest))
            record = value["scenes"]["test"]
            if change == "source":
                record["source_sha256"] = "0" * 64
            elif change == "geometry":
                record["approvals"][0]["world_geometry_sha256"] = "0" * 64
            elif change == "floor":
                record["approvals"][0]["path"] = "/Root/SM_floorplan"
            elif change == "missing":
                record["approvals"][0]["path"] = "/Root/Missing"
            elif change == "ground":
                record["ground_z_m"] = None
            else:
                stage = Usd.Stage.Open(str(source))
                UsdGeom.Xform.Define(stage, "/Root/Carpet/Child")
                stage.GetRootLayer().Save()
                record["source_sha256"] = sanitizer._sha256(source)
            manifest_path.write_text(json.dumps(value))
            try:
                sanitizer.sanitize_collision_usd(source, root / f"bad-{change}.usd", root / "bad.json", **kwargs)
            except ValueError:
                assert not (root / f"bad-{change}.usd").exists()
            else:
                raise AssertionError(f"expected v2 identity/approval rejection: {change}")
            if change != "child":
                assert source.read_bytes() == saved
        # Height/contact boundaries stay strict; narrowness is diagnostic only.
        for thickness, short_side, bottom, expected in [
            (.02, .5, .0015, True), (.02001, .5, .0015, False),
            (.015, .5, .005, True), (.015, .5, .00501, False),
            (.02, .4, .0015, True), (.02, .399, .0015, True),
            (.02, .5, .005, True), (.02, .5, .00501, False),
        ]:
            stage = Usd.Stage.CreateInMemory()
            p, f = box(bottom, thickness, x0=0, x1=short_side, y0=0, y1=1)
            mesh = add_mesh(stage, "/Candidate", p, f)
            points = np.array(p)
            floor = np.array([[[-2., -2., 0.], [2., -2., 0.], [2., 2., 0.]],
                              [[-2., -2., 0.], [2., 2., 0.], [-2., 2., 0.]]])
            rejected = sanitizer._v2_reasons(mesh, points, sanitizer._geometry_metrics(mesh, points),
                                             sanitizer._FloorSupportIndex(floor), 0.)
            assert (not rejected) == expected, (thickness, short_side, bottom, rejected)
    print("PASS v2 approved instances, low-ratio marking, dry-run, identity and boundary checks")


def v2_manifest(source, names):
    stage = Usd.Stage.Open(str(source))
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    stage.GetRootLayer().Save()
    approvals = []
    for name in names:
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/Root/" + name))
        points = sanitizer._world_points(mesh, UsdGeom.XformCache())
        faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64).reshape(-1, 3)
        approvals.append({"path": "/Root/" + name,
                          "world_geometry_sha256": sanitizer._geometry_sha256(points, faces)})
    manifest = {"policy": sanitizer.POLICY_V2, "scenes": {"test": {
        "source_sha256": sanitizer._sha256(source), "ground_z_m": 0., "approvals": approvals}}}
    path = source.with_suffix(".manifest.json")
    path.write_text(json.dumps(manifest))
    return path, manifest


def test_v2_world_transform_and_approval_guard():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "transformed.usd"
        points, faces = box(0., .009)
        transform = ((.1, .1, .0015), 37., (.05, .8, 2.))
        make_stage(source, [("/Root/Approved", points, faces, transform),
                            ("/Root/Unknown", points, faces, transform)])
        manifest, _ = v2_manifest(source, ["Approved"])
        held = Usd.Stage.Open(str(source))
        before_layer, before_file = held.GetRootLayer().ExportToString(), source.read_bytes()
        kwargs = dict(policy="v2", approved_manifest=manifest)
        dry = sanitizer.sanitize_collision_usd(source, root / "dry.usd", root / "dry.json", dry_run=True, **kwargs)
        actual = sanitizer.sanitize_collision_usd(source, root / "out.usd", root / "out.json", **kwargs)
        assert actual["removed_prims"] == dry["removed_prims"]
        assert [entry["path"] for entry in actual["removed_prims"]] == ["/Root/Approved"]
        metrics = actual["removed_prims"][0]["metrics"]
        assert np.isclose(metrics["thickness_m"], .018)
        assert np.isclose(metrics["short_xy_side_m"], .1)
        assert metrics["sheet_shape_warning"] and metrics["thickness_over_short_side"] > .05
        assert any(entry["path"] == "/Root/Unknown" and "instance_not_approved" in entry["reasons"]
                   for entry in actual["skipped_candidates"])
        assert held.GetRootLayer().ExportToString() == before_layer
        assert source.read_bytes() == before_file and not (root / "dry.usd").exists()
        existing = root / "existing.usd"
        existing.write_bytes(b"existing output must survive a dry run")
        sanitizer.sanitize_collision_usd(source, existing, root / "existing-dry.json", dry_run=True, **kwargs)
        assert existing.read_bytes() == b"existing output must survive a dry run"
        result = Usd.Stage.Open(str(root / "out.usd"))
        assert result.GetPrimAtPath("/Root/Unknown") and not result.GetPrimAtPath("/Root/Approved")
    print("PASS v2 world transforms, narrow approved covers, unknown solids and isolated input layer")


def test_v2_support_and_invalid_geometry():
    class Hits:
        def __init__(self, missing):
            self.missing = missing

        def floor_z_below_many(self, points):
            result = np.zeros(len(points))
            result[:self.missing] = np.nan
            return result

    # One missing sample among two equal-area triangles is exactly 95% support.
    for scale, missing, accepted in [(1., 1, True), (1., 2, False),
                                     (1.0001, 1, False), (.9999, 1, True)]:
        stage = Usd.Stage.CreateInMemory()
        points = np.array([(0., 0., .0015), (scale, 0., .0015), (0., 1., .0015),
                           (0., 0., .0015), (1., 0., .0015), (1., 1., .0015)])
        mesh = add_mesh(stage, "/Candidate", points.tolist(), [(0, 1, 2), (3, 4, 5)])
        metrics = sanitizer._geometry_metrics(mesh, points)
        rejected = sanitizer._v2_reasons(mesh, points, metrics, Hits(missing), 0.)
        assert (not rejected) == accepted, (metrics["support_coverage"], rejected)
        if scale == 1. and missing == 1:
            assert metrics["support_coverage"] == .95
    for points, expected in [
        ([(0., 0., .001), (float("nan"), 1., .001), (1., 0., .001)], "invalid_or_nontriangular_geometry"),
        ([(0., 0., .001), (float("inf"), 1., .001), (1., 0., .001)], "invalid_or_nontriangular_geometry"),
        ([(0., 0., .001), (1., 0., .001), (1., 0., .015)], "degenerate_xy_footprint"),
        ([(0., 0., .001)] * 3, "invalid_or_nontriangular_geometry"),
    ]:
        stage = Usd.Stage.CreateInMemory()
        points = np.array(points)
        mesh = add_mesh(stage, "/Candidate", points.tolist(), [(0, 1, 2)])
        with np.errstate(invalid="ignore"):
            reasons = sanitizer._v2_reasons(mesh, points, sanitizer._geometry_metrics(mesh, points), Hits(0), 0.)
        assert expected in reasons
    print("PASS v2 exact area-weighted support boundary and invalid/degenerate geometry")


def test_v2_contract_and_concurrent_source_change():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "raw.usd"
        points, faces = box(.0015, .015)
        make_stage(source, [("/Root/Cover", points, faces, None)])
        manifest_path, manifest = v2_manifest(source, ["Cover"])
        kwargs = dict(policy="v2", approved_manifest=manifest_path)
        manifest_before = manifest_path.read_bytes()
        for output, report in [(manifest_path, root / "alias.json"), (root / "alias.usd", manifest_path)]:
            try:
                sanitizer.sanitize_collision_usd(source, output, report, **kwargs)
            except ValueError:
                assert manifest_path.read_bytes() == manifest_before
            else:
                raise AssertionError("v2 permitted overwriting its approval manifest")
        for index, changed in enumerate([
            {"max_overlay_thickness_m": .03}, {"min_horizontal_area_ratio": 0.},
            {"max_support_gap_m": 1.}, {"min_support_coverage": 0.}, {"max_floor_height_m": 10.},
        ]):
            output = root / f"contract-{index}.usd"
            try:
                sanitizer.sanitize_collision_usd(source, output, root / "contract.json", **kwargs, **changed)
            except ValueError:
                assert not output.exists()
            else:
                raise AssertionError("v2 allowed custom thresholds to expand legacy deletion")
        # Source-matching manifests do not make an incompatible unit system safe.
        for axis, units in [("Y", 1.), ("Z", .01)]:
            stage = Usd.Stage.Open(str(source))
            UsdGeom.SetStageUpAxis(stage, axis)
            UsdGeom.SetStageMetersPerUnit(stage, units)
            stage.GetRootLayer().Save()
            manifest["scenes"]["test"]["source_sha256"] = sanitizer._sha256(source)
            manifest_path.write_text(json.dumps(manifest))
            try:
                sanitizer.sanitize_collision_usd(source, root / "units.usd", root / "units.json", **kwargs)
            except ValueError:
                assert not (root / "units.usd").exists()
            else:
                raise AssertionError("v2 accepted non-Z-up or non-metre geometry")
        manifest_path, _ = v2_manifest(source, ["Cover"])
        before = source.read_bytes()
        original = sanitizer._v2_reasons

        def changed_source(*args):
            result = original(*args)
            source.write_bytes(before + b"\n# concurrent input edit\n")
            return result

        sanitizer._v2_reasons = changed_source
        try:
            try:
                sanitizer.sanitize_collision_usd(source, root / "raced.usd", root / "raced.json", **kwargs)
            except ValueError:
                assert not (root / "raced.usd").exists()
            else:
                raise AssertionError("Source changed during validation without rejection")
        finally:
            sanitizer._v2_reasons = original
            source.write_bytes(before)
        legacy = root / "legacy.usd"
        points, faces = box(.001, .001)
        make_stage(legacy, [("/Root/Legacy", points, faces, None)])
        held = Usd.Stage.Open(str(legacy))
        before = held.GetRootLayer().ExportToString()
        result = sanitizer.sanitize_collision_usd(legacy, root / "legacy-dry.usd", root / "legacy-dry.json", dry_run=True)
        assert len(result["removed_prims"]) == 1 and not (root / "legacy-dry.usd").exists()
        assert held.GetRootLayer().ExportToString() == before
    print("PASS immutable v2 legacy thresholds, unit contract, source race rejection and v1 dry-run isolation")


if __name__ == "__main__":
    main()
    test_v2()
    test_v2_world_transform_and_approval_guard()
    test_v2_support_and_invalid_geometry()
    test_v2_contract_and_concurrent_source_change()
