#!/usr/bin/env python3
"""
Filter a USDZ (NuRec) file by removing gaussians whose Z position exceeds a threshold.

This operates directly on the .nurec msgpack payload inside the USDZ zip, updates
all gaussian attribute arrays, and refreshes the bounding-box metadata in gauss.usda.
"""

import argparse
import gzip
import io
import re
import zipfile
from pathlib import Path

import msgpack
import numpy as np


def read_nurec_state_dict(zip_file: zipfile.ZipFile, nurec_name: str) -> dict:
    """Extract and unpack the nurec state_dict from inside a USDZ zip."""
    data = zip_file.read(nurec_name)
    decompressed = gzip.decompress(data)
    unpacked = msgpack.unpackb(decompressed, raw=False)
    return unpacked["nre_data"]["state_dict"]


def read_nurec_full(zip_file: zipfile.ZipFile, nurec_name: str) -> dict:
    """Extract the full nre_data dict (preserving version, model, config)."""
    data = zip_file.read(nurec_name)
    decompressed = gzip.decompress(data)
    unpacked = msgpack.unpackb(decompressed, raw=False)
    return unpacked["nre_data"]


def pack_nurec(nre_data: dict) -> bytes:
    """Re-pack a full nre_data dict into gzip-compressed msgpack bytes."""
    payload = {"nre_data": nre_data}
    packed = msgpack.packb(payload)
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", compresslevel=0) as f:
        f.write(packed)
    return buffer.getvalue()


def filter_gaussians(state_dict: dict, z_threshold: float):
    """Remove gaussians with Z > threshold. Works in-place on state_dict."""
    sd = state_dict
    keys = [
        ".gaussians_nodes.gaussians.positions",
        ".gaussians_nodes.gaussians.rotations",
        ".gaussians_nodes.gaussians.scales",
        ".gaussians_nodes.gaussians.densities",
        ".gaussians_nodes.gaussians.features_albedo",
        ".gaussians_nodes.gaussians.features_specular",
        ".gaussians_nodes.gaussians.extra_signal",
    ]

    # Read positions to build mask
    pos = np.frombuffer(sd[".gaussians_nodes.gaussians.positions"], dtype=np.float16)
    pos_shape = sd[".gaussians_nodes.gaussians.positions.shape"]
    pos = pos.reshape(pos_shape)
    mask = pos[:, 2] <= z_threshold
    keep = int(mask.sum())
    original = pos_shape[0]

    if keep == original:
        print(f"[INFO] No gaussians above Z={z_threshold}; nothing to do.")
        return original, keep, pos.min(axis=0), pos.max(axis=0)

    if keep == 0:
        raise ValueError(f"Z threshold {z_threshold} removes all gaussians!")

    # Filter every array
    for key in keys:
        if key not in sd:
            continue
        arr = np.frombuffer(sd[key], dtype=np.float16)
        shape = sd[f"{key}.shape"]
        arr = arr.reshape(shape)
        arr = arr[mask]
        sd[key] = arr.tobytes()
        sd[f"{key}.shape"] = list(arr.shape)

    # Update the n_active_features shape (scalar, unchanged value)
    sd[".gaussians_nodes.gaussians.n_active_features.shape"] = []

    new_pos = pos[mask]
    return original, keep, new_pos.min(axis=0), new_pos.max(axis=0)


def update_gauss_usda_text(text: str, min_coords: np.ndarray, max_coords: np.ndarray) -> str:
    """Update extent and crop bounds in gauss.usda via regex (no pxr needed)."""
    min_x, min_y, min_z = map(float, min_coords)
    max_x, max_y, max_z = map(float, max_coords)

    # extent = [(min_x, min_y, min_z), (max_x, max_y, max_z)]
    text = re.sub(
        r'extent\s*=\s*\[\([^\)]+\),\s*\([^\)]+\)\]',
        f'extent = [({min_x}, {min_y}, {min_z}), ({max_x}, {max_y}, {max_z})]',
        text,
    )

    # omni:nurec:crop:minBounds
    text = re.sub(
        r'omni:nurec:crop:minBounds\s*=\s*[^\n]+',
        f'omni:nurec:crop:minBounds = ({min_x}, {min_y}, {min_z})',
        text,
    )

    # omni:nurec:crop:maxBounds
    text = re.sub(
        r'omni:nurec:crop:maxBounds\s*=\s*[^\n]+',
        f'omni:nurec:crop:maxBounds = ({max_x}, {max_y}, {max_z})',
        text,
    )

    return text


def process_usdz(input_path: Path, output_path: Path, z_threshold: float):
    """Filter gaussians by Z and write a new USDZ."""
    with zipfile.ZipFile(input_path, "r") as zin:
        nurec_name = None
        for name in zin.namelist():
            if name.endswith(".nurec"):
                nurec_name = name
                break
        if nurec_name is None:
            raise ValueError("No .nurec file found inside USDZ")

        nre_data = read_nurec_full(zin, nurec_name)
        sd = nre_data["state_dict"]
        original, keep, min_c, max_c = filter_gaussians(sd, z_threshold)
        print(f"[INFO] {input_path.name}: {original} -> {keep} gaussians (removed {original - keep})")

        new_nurec_bytes = pack_nurec(nre_data)

        gauss_text = zin.read("gauss.usda").decode("utf-8")
        gauss_text = update_gauss_usda_text(gauss_text, min_c, max_c)

        default_text = zin.read("default.usda").decode("utf-8")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_STORED) as zout:
        zout.writestr("default.usda", default_text)
        zout.writestr(nurec_name, new_nurec_bytes)
        zout.writestr("gauss.usda", gauss_text)

    print(f"[OK] Wrote filtered USDZ to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Remove gaussians with Z > threshold from a NuRec USDZ")
    parser.add_argument("input", type=Path, help="Input USDZ file")
    parser.add_argument("--output", "-o", type=Path, help="Output USDZ file (default: overwrite input)")
    parser.add_argument("--z-threshold", "-z", type=float, required=True, help="Keep gaussians with Z <= threshold")
    args = parser.parse_args()

    output = args.output if args.output else args.input
    process_usdz(args.input, output, args.z_threshold)


if __name__ == "__main__":
    main()
