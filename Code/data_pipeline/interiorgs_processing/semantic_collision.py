"""Match InteriorGS openings and carpet instances to separable source mesh faces."""

import hashlib
import json
from collections import Counter, defaultdict

import cv2
import numpy as np
from pxr import UsdGeom
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from sanitize_collision_usd import _geometry_sha256, _support_metrics, _world_points

LIMITS = {
    "door": {"max_thickness_m": .005, "max_top_m": .005, "min_obb_iou": .80},
    "carpet": {"max_thickness_m": .025, "max_top_m": .025, "min_obb_iou": .90},
    "max_contact_gap_m": .005, "min_support": .95, "min_horizontal_area_ratio": .95,
}
EPS = 1e-7


def _targets(directory, scene):
    if directory is None or directory.name.rsplit("_", 1)[-1] != scene:
        raise ValueError(f"V3 requires the InteriorGS directory for source scene {scene}")
    data, hashes = {}, {}
    for name in ("structure", "labels"):
        raw = (directory / f"{name}.json").read_bytes()
        data[name] = json.loads(raw)
        hashes[name] = hashlib.sha256(raw).hexdigest()

    targets = []
    for index, hole in enumerate(data["structure"]["holes"]):
        if hole["type"] != "DOOR":
            continue
        profile = np.asarray(hole["profile"], dtype=float)
        depth = hole["thickness"]
        target = dict(rule="door", id=index)
        targets.append(target)
        if profile.ndim != 2 or profile.shape[1] != 3 or len(profile) < 3 or depth <= 0:
            continue
        # Profile winding gives the bottom edge direction and the wall extrusion side.
        end = np.roll(profile, -1, axis=0)
        bottom = np.maximum(profile[:, 2], end[:, 2]) <= profile[:, 2].min() + .001
        edges = np.flatnonzero(bottom & (np.linalg.norm(end[:, :2]-profile[:, :2], axis=1) > .01))
        if len(edges) != 1:
            continue
        a, b = profile[edges[0], :2], end[edges[0], :2]
        dx, dy = b - a
        offset = depth * np.array([-dy, dx]) / np.linalg.norm(b-a)
        target.update(polygon=np.array([a, b, b+offset, a+offset], dtype=np.float32), z=float(profile[:, 2].min()))

    carpets = [v for v in data["labels"] if v["label"] == "carpet"]
    counts = Counter(str(v["ins_id"]) for v in carpets)
    for label in carpets:
        key = str(label["ins_id"])
        target = dict(rule="carpet", id=key)
        targets.append(target)
        if counts[key] != 1:
            target["reason"] = "duplicate_instance_id"
            continue
        corners = np.array([[p[k] for k in "xyz"] for p in label["bounding_box"]])
        if corners.shape == (8, 3) and np.isfinite(corners).all():
            obb = cv2.convexHull(corners[:, :2].astype(np.float32)).reshape(-1, 2)
            if cv2.contourArea(obb) > EPS:
                target.update(polygon=obb, z=float(corners[:, 2].min()), top=float(corners[:, 2].max()))
    return targets, hashes


def _ground_components(points, faces, ground):
    # Every neighbour of a low triangle shares low vertices. Including straddling
    # triangles prevents cutting a sill welded to a taller frame at a shared edge.
    low = points[:, 2] <= ground + .05 + EPS
    ids = np.flatnonzero(low[faces].any(axis=1))
    if not len(ids):
        return
    indices = faces[ids]
    vertices, inverse = np.unique(indices, return_inverse=True)
    _, welded = np.unique(points[vertices], axis=0, return_inverse=True)
    tri = welded[inverse].reshape(-1, 3)
    edges = np.sort(tri[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    linked = (edges[order[1:]] == edges[order[:-1]]).all(axis=1)
    graph = coo_matrix((np.ones(linked.sum(), dtype=bool),
                        (order[1:][linked] // 3, order[:-1][linked] // 3)), shape=(len(ids), len(ids)))
    _, groups = connected_components(graph, directed=False)
    order = np.argsort(groups, kind="stable")
    for group in np.split(order, np.flatnonzero(np.diff(groups[order])) + 1):
        bottom = low[indices[group]].all(axis=1)
        if bottom.any():
            yield ids[group[bottom]], not bottom.all()


def _topology_edits(mesh, removed):
    """Return attribute edits, or reject a representation we cannot safely remap."""
    prim = mesh.GetPrim()
    if prim.GetAllChildren() or any(a.GetNumTimeSamples() or a.GetConnections() for a in prim.GetAuthoredAttributes()):
        raise ValueError("children_or_dynamic_attributes")
    faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
    keep = np.ones(len(faces), dtype=bool)
    keep[removed] = False
    points = np.asarray(mesh.GetPointsAttr().Get())
    used, remapped = np.unique(faces[keep], return_inverse=True)
    edits = {"points": points[used], "faceVertexIndices": remapped.ravel().astype(np.int32),
             "faceVertexCounts": np.full(keep.sum(), 3, dtype=np.int32)}
    if mesh.GetExtentAttr().HasAuthoredValueOpinion():
        extent = np.array([points[used].min(axis=0), points[used].max(axis=0)])
        extent[extent == 0] = 0.  # Canonical bounds across USDA and USDC signed zero.
        edits["extent"] = extent
    masks = {"vertex": used, "varying": used, "uniform": np.flatnonzero(keep),
             "faceVarying": np.flatnonzero(keep.repeat(3))}
    lengths = {"vertex": len(points), "varying": len(points), "uniform": len(faces), "faceVarying": faces.size}
    for attr in prim.GetAuthoredAttributes():
        name = attr.GetName()
        if name in edits or not attr.GetTypeName().isArray or name.startswith("xformOp"):
            continue
        if name == "normals":
            interp = mesh.GetNormalsInterpolation()
            size = 1
        elif name.startswith("primvars:"):
            if name.endswith(":indices"):
                continue
            pv = UsdGeom.Primvar(attr)
            interp, size = pv.GetInterpolation(), pv.GetElementSize()
            if pv.IsIndexed():
                attr, size = pv.GetIndicesAttr(), 1
        else:
            raise ValueError("unsupported_topology_attribute:" + name)
        if interp == "constant":
            continue
        value = np.asarray(attr.Get())
        if interp not in masks or size < 1 or len(value) != lengths[interp] * size:
            raise ValueError("invalid_interpolated_attribute:" + name)
        indices = (masks[interp][:, None] * size + np.arange(size)).ravel()
        authored = attr.Get()
        edits[attr.GetName()] = [authored[int(index)] for index in indices]
    return edits


def _check_sheet(target, iou, bottom, top, ground, triangles, support_index):
    limits = LIMITS[target["rule"]]
    thickness = top - bottom
    metrics = dict(obb_iou=iou, thickness_m=thickness, top_above_ground_m=top-ground)
    if iou < limits["min_obb_iou"]:
        return "footprint_mismatch", metrics
    if bottom < ground-EPS or thickness > limits["max_thickness_m"]+EPS or top-ground > limits["max_top_m"]+EPS:
        return "outside_height_limits", metrics
    if target["rule"] == "carpet" and abs(top - target["top"]) > .005+EPS:
        return "annotation_height_mismatch", metrics
    cross = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    projected = np.abs(cross[:, 2]).sum()
    ratio = projected / max(EPS, np.linalg.norm(cross, axis=1).sum())
    # A closed slab has two broad faces. Count each winding separately so a
    # thin rectangular frame cannot qualify by counting its upper and lower rim twice.
    plan_area = max(np.maximum(cross[:, 2], 0).sum(), np.maximum(-cross[:, 2], 0).sum()) / 2
    if ratio < LIMITS["min_horizontal_area_ratio"] or plan_area < target["area"] * .75:
        return "not_horizontal_sheet", metrics
    metrics.update(_support_metrics(triangles, support_index, limits["max_top_m"]))
    gap = metrics["minimum_contact_gap_m"]
    if gap is None or gap > LIMITS["max_contact_gap_m"]+EPS or metrics["support_coverage"] < LIMITS["min_support"]:
        return "insufficient_floor_contact", metrics
    return None, metrics


def semantic_decisions(stage, cache, scene, directory, ground, support_index, previously_removed):
    targets, hashes = _targets(directory, scene)
    matches = defaultdict(list)
    nearby = defaultdict(Counter)
    rejected = defaultdict(list)
    active = []
    for i, target in enumerate(targets):
        if ground is None:
            target["reason"] = "unknown_ground"
        elif "polygon" not in target:
            target.setdefault("reason", "invalid_annotation")
        elif abs(target["z"] - ground) > .05:
            target["reason"] = "annotation_above_or_below_ground"
        else:
            polygon = target["polygon"]
            target["area"] = cv2.contourArea(polygon)
            target["bounds"] = [*polygon.min(axis=0), *polygon.max(axis=0)]
            active.append(i)
    bounds = np.asarray([targets[i]["bounds"] for i in active]).reshape(-1, 4)
    for prim in stage.Traverse():
        if not active or not prim.IsA(UsdGeom.Mesh) or prim.GetPath() == "/Root/SM_floorplan":
            continue
        mesh = UsdGeom.Mesh(prim)
        points = _world_points(mesh, cache)
        if not len(points) or not np.isfinite(points).all() or points[:, 2].min() > ground + .05:
            continue
        lo, hi = points[:, :2].min(axis=0), points[:, :2].max(axis=0)
        hits = np.flatnonzero((bounds[:, :2] < hi).all(axis=1) & (bounds[:, 2:] > lo).all(axis=1))
        if not len(hits):
            continue
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
        flat = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
        if not len(counts) or not (counts == 3).all() or len(flat) != len(counts)*3 or flat.min() < 0 or flat.max() >= len(points):
            continue
        faces = flat.reshape(-1, 3)
        for ids, attached in _ground_components(points, faces, ground):
            vertices = points[np.unique(faces[ids])]
            lo, hi = vertices.min(axis=0), vertices.max(axis=0)
            # Bounding-box intersection is a cheap upper bound for the expensive OBB comparison.
            common = np.maximum(0, np.minimum(bounds[hits, 2:], hi[:2])-np.maximum(bounds[hits, :2], lo[:2])).prod(axis=1)
            possible = [active[h] for h, area in zip(hits, common)
                        if area >= targets[active[h]]["area"] * .80]
            if not possible:
                continue
            obb = cv2.boxPoints(cv2.minAreaRect(vertices[:, :2].astype(np.float32)))
            for i in possible:
                target = targets[i]
                polygon = target["polygon"]
                inter, _ = cv2.intersectConvexConvex(polygon, obb)
                iou = min(1., inter / max(EPS, target["area"] + cv2.contourArea(obb) - inter))
                if attached:
                    if iou >= LIMITS[target["rule"]]["min_obb_iou"]:
                        nearby[i]["attached_to_taller_geometry"] += 1
                    continue
                reason, metrics = _check_sheet(target, iou, lo[2], hi[2], ground,
                                               points[faces[ids]], support_index)
                if reason:
                    nearby[i][reason] += 1
                    if reason in ("outside_height_limits", "insufficient_floor_contact"):
                        rejected[i].append(dict(path=str(prim.GetPath()), **metrics, reason=reason))
                    continue
                path = str(prim.GetPath())
                matches[i].append(dict(path=path, faces=ids.tolist(), source_face_count=len(faces),
                                       world_geometry_sha256=_geometry_sha256(points, faces), metrics=metrics))
    records, edits, whole = _resolve_matches(stage, targets, matches, nearby, rejected, previously_removed)
    return dict(inputs_sha256=hashes, thresholds=LIMITS, decisions=records), edits, whole


def _resolve_matches(stage, targets, matches, nearby, rejected, previously_removed):
    claims = Counter((m["path"], m["faces"][0]) for options in matches.values() for m in options)
    by_path = defaultdict(list)
    records = []
    for i, target in enumerate(targets):
        options = matches[i]
        row = dict(rule=target["rule"], id=target["id"], nearby_rejections=dict(nearby[i]))
        if rejected[i]:
            row["rejected_candidates"] = rejected[i]
        records.append(row)
        if len(options) == 1 and claims[(options[0]["path"], options[0]["faces"][0])] == 1:
            row.update(options[0])
            if row["path"] in previously_removed:
                row["reason"] = "already_removed_v2"
            else:
                by_path[row["path"]].append(row)
        else:
            row["reason"] = ("ambiguous_match" if options else target.get("reason", "no_matching_separable_sheet"))
            if options:
                row["candidate_paths"] = [m["path"] for m in options]
    edits, whole = {}, []
    for path, rows in by_path.items():
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath(path))
        ids = sorted(f for row in rows for f in row["faces"])
        try:
            parent = mesh.GetPrim().GetParent()
            while parent and not parent.IsPseudoRoot():
                if UsdGeom.Xformable(parent) and UsdGeom.Xformable(parent).TransformMightBeTimeVarying():
                    raise ValueError("animated_parent_transform")
                parent = parent.GetParent()
            if len(ids) == rows[0]["source_face_count"]:
                if mesh.GetPrim().GetAllChildren() or any(a.GetNumTimeSamples() for a in mesh.GetPrim().GetAuthoredAttributes()):
                    raise ValueError("children_or_dynamic_attributes")
                whole.append(path)
            else:
                edits[path] = _topology_edits(mesh, ids)
        except ValueError as exc:
            for row in rows:
                row["reason"] = str(exc)
            continue
        for row in rows:
            row["action"] = "remove_prim" if path in whole else "remove_faces"
    return records, edits, whole
