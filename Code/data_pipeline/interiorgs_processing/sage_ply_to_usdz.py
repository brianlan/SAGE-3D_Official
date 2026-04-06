#!/usr/bin/env python3
"""
Standalone PLY to USDZ converter for SAGE-3D.

Converts decompressed 3D Gaussian Splatting PLY files to USDZ (NuRec format)
compatible with NVIDIA Isaac Sim 5.0+.

This script bypasses the heavy 3dgrut dependencies (ncore, tracers, etc.) by
directly implementing the core NuRec export logic.
"""

import argparse
import gzip
import io
import logging
import os
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import msgpack
import numpy as np
from plyfile import PlyData
from pxr import Gf, Sdf, Usd, UsdGeom, UsdUtils, UsdVol

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers (from threedgrut.export.usd.stage_utils)
# ---------------------------------------------------------------------------

@dataclass(kw_only=True)
class NamedUSDStage:
    filename: str
    stage: Usd.Stage

    def save_to_zip(self, zip_file: zipfile.ZipFile):
        with tempfile.NamedTemporaryFile(mode="wb", suffix=self.filename, delete=False) as temp_file:
            temp_file_path = temp_file.name
        self.stage.GetRootLayer().Export(temp_file_path)
        with open(temp_file_path, "rb") as file:
            usd_data = file.read()
        zip_file.writestr(self.filename, usd_data)
        os.unlink(temp_file_path)


@dataclass(kw_only=True)
class NamedSerialized:
    filename: str
    serialized: Union[str, bytes]

    def save_to_zip(self, zip_file: zipfile.ZipFile):
        zip_file.writestr(self.filename, self.serialized)


# ---------------------------------------------------------------------------
# USD stage utilities
# ---------------------------------------------------------------------------

def initialize_usd_stage(up_axis: str = "Y") -> Usd.Stage:
    stage = Usd.Stage.CreateInMemory()
    stage.SetMetadata("metersPerUnit", 1.0)
    stage.SetMetadata("upAxis", up_axis)
    stage.SetTimeCodesPerSecond(24.0)
    UsdGeom.Xform.Define(stage, "/World")
    stage.SetMetadata("defaultPrim", "World")
    return stage


# ---------------------------------------------------------------------------
# NuRec template generation
# ---------------------------------------------------------------------------

def fill_3dgut_template(
    positions: np.ndarray,
    rotations: np.ndarray,
    scales: np.ndarray,
    densities: np.ndarray,
    features_albedo: np.ndarray,
    features_specular: np.ndarray,
    n_active_features: int,
    density_activation: str = "sigmoid",
    scale_activation: str = "exp",
    rotation_activation: str = "normalize",
    density_kernel_degree: int = 2,
    density_kernel_density_clamping: bool = False,
    density_kernel_min_response: float = 0.0113,
    radiance_sph_degree: int = 3,
    transmittance_threshold: float = 0.001,
    global_z_order: bool = False,
    n_rolling_shutter_iterations: int = 5,
    ut_alpha: float = 1.0,
    ut_beta: float = 2.0,
    ut_kappa: float = 0.0,
    ut_require_all_sigma_points: bool = False,
    image_margin_factor: float = 0.1,
    rect_bounding: bool = True,
    tight_opacity_bounding: bool = True,
    tile_based_culling: bool = True,
    k_buffer_size: int = 0,
) -> Dict[str, Any]:
    template = {
        "nre_data": {
            "version": "0.2.576",
            "model": "nre",
            "config": {
                "layers": {
                    "gaussians": {
                        "name": "sh-gaussians",
                        "device": "cuda",
                        "density_activation": density_activation,
                        "scale_activation": scale_activation,
                        "rotation_activation": rotation_activation,
                        "precision": 16,
                        "particle": {
                            "density_kernel_planar": False,
                            "density_kernel_degree": density_kernel_degree,
                            "density_kernel_density_clamping": density_kernel_density_clamping,
                            "density_kernel_min_response": density_kernel_min_response,
                            "radiance_sph_degree": radiance_sph_degree,
                        },
                        "transmittance_threshold": transmittance_threshold,
                    }
                },
                "renderer": {
                    "name": "3dgut-nrend",
                    "log_level": 3,
                    "force_update": False,
                    "update_step_train_batch_end": False,
                    "per_ray_features": False,
                    "global_z_order": global_z_order,
                    "projection": {
                        "n_rolling_shutter_iterations": n_rolling_shutter_iterations,
                        "ut_dim": 3,
                        "ut_alpha": ut_alpha,
                        "ut_beta": ut_beta,
                        "ut_kappa": ut_kappa,
                        "ut_require_all_sigma_points": ut_require_all_sigma_points,
                        "image_margin_factor": image_margin_factor,
                        "min_projected_ray_radius": 0.5477225575051661,
                    },
                    "culling": {
                        "rect_bounding": rect_bounding,
                        "tight_opacity_bounding": tight_opacity_bounding,
                        "tile_based": tile_based_culling,
                        "near_clip_distance": 1e-8,
                        "far_clip_distance": 3.402823466e38,
                    },
                    "render": {"mode": "kbuffer", "k_buffer_size": k_buffer_size},
                },
                "name": "gaussians_primitive",
                "appearance_embedding": {"name": "skip-appearance", "embedding_dim": 0, "device": "cuda"},
                "background": {"name": "skip-background", "device": "cuda", "composite_in_linear_space": False},
            },
            "state_dict": {
                "._extra_state": {"obj_track_ids": {"gaussians": []}},
                ".gaussians_nodes.gaussians.positions": None,
                ".gaussians_nodes.gaussians.rotations": None,
                ".gaussians_nodes.gaussians.scales": None,
                ".gaussians_nodes.gaussians.densities": None,
                ".gaussians_nodes.gaussians.extra_signal": None,
                ".gaussians_nodes.gaussians.features_albedo": None,
                ".gaussians_nodes.gaussians.features_specular": None,
                ".gaussians_nodes.gaussians.n_active_features": None,
                ".gaussians_nodes.gaussians.positions.shape": None,
                ".gaussians_nodes.gaussians.rotations.shape": None,
                ".gaussians_nodes.gaussians.scales.shape": None,
                ".gaussians_nodes.gaussians.densities.shape": None,
                ".gaussians_nodes.gaussians.extra_signal.shape": None,
                ".gaussians_nodes.gaussians.features_albedo.shape": None,
                ".gaussians_nodes.gaussians.features_specular.shape": None,
                ".gaussians_nodes.gaussians.n_active_features.shape": None,
            },
        }
    }

    dtype = np.float16
    sd = template["nre_data"]["state_dict"]
    sd[".gaussians_nodes.gaussians.positions"] = positions.astype(dtype).tobytes()
    sd[".gaussians_nodes.gaussians.rotations"] = rotations.astype(dtype).tobytes()
    sd[".gaussians_nodes.gaussians.scales"] = scales.astype(dtype).tobytes()
    sd[".gaussians_nodes.gaussians.densities"] = densities.astype(dtype).tobytes()
    sd[".gaussians_nodes.gaussians.features_albedo"] = features_albedo.astype(dtype).tobytes()
    sd[".gaussians_nodes.gaussians.features_specular"] = features_specular.astype(dtype).tobytes()
    extra_signal = np.zeros((positions.shape[0], 0), dtype=dtype)
    sd[".gaussians_nodes.gaussians.extra_signal"] = extra_signal.tobytes()
    sd[".gaussians_nodes.gaussians.n_active_features"] = np.array([n_active_features], dtype=np.int64).tobytes()
    sd[".gaussians_nodes.gaussians.positions.shape"] = list(positions.shape)
    sd[".gaussians_nodes.gaussians.rotations.shape"] = list(rotations.shape)
    sd[".gaussians_nodes.gaussians.scales.shape"] = list(scales.shape)
    sd[".gaussians_nodes.gaussians.densities.shape"] = list(densities.shape)
    sd[".gaussians_nodes.gaussians.features_albedo.shape"] = list(features_albedo.shape)
    sd[".gaussians_nodes.gaussians.features_specular.shape"] = list(features_specular.shape)
    sd[".gaussians_nodes.gaussians.extra_signal.shape"] = list(extra_signal.shape)
    sd[".gaussians_nodes.gaussians.n_active_features.shape"] = []

    return template


# ---------------------------------------------------------------------------
# NuRec USD serialization
# ---------------------------------------------------------------------------

def serialize_nurec_usd(model_file: NamedSerialized, positions: np.ndarray) -> NamedUSDStage:
    min_coord = np.min(positions, axis=0)
    max_coord = np.max(positions, axis=0)
    min_x, min_y, min_z = float(min_coord[0]), float(min_coord[1]), float(min_coord[2])
    max_x, max_y, max_z = float(max_coord[0]), float(max_coord[1]), float(max_coord[2])

    stage = initialize_usd_stage(up_axis="Z")

    render_settings = {
        "rtx:rendermode": "RaytracedLighting",
        "rtx:directLighting:sampledLighting:samplesPerPixel": 8,
        "rtx:post:histogram:enabled": False,
        "rtx:post:registeredCompositing:invertToneMap": True,
        "rtx:post:registeredCompositing:invertColorCorrection": True,
        "rtx:material:enableRefraction": False,
        "rtx:post:tonemap:op": 2,
        "rtx:raytracing:fractionalCutoutOpacity": False,
        "rtx:matteObject:visibility:secondaryRays": True,
    }
    stage.SetMetadataByDictKey("customLayerData", "renderSettings", render_settings)

    gauss_path = "/World/gauss"
    gauss_volume = UsdVol.Volume.Define(stage, gauss_path)
    gauss_prim = gauss_volume.GetPrim()

    matrix_op = gauss_volume.AddTransformOp()
    matrix_op.Set(Gf.Matrix4d(1.0))

    gauss_prim.CreateAttribute("omni:nurec:isNuRecVolume", Sdf.ValueTypeNames.Bool).Set(True)
    gauss_prim.CreateAttribute("omni:nurec:useProxyTransform", Sdf.ValueTypeNames.Bool).Set(False)

    density_field_path = gauss_path + "/density_field"
    density_field = stage.DefinePrim(density_field_path, "OmniNuRecFieldAsset")
    gauss_volume.CreateFieldRelationship("density", density_field_path)

    emissive_color_field_path = gauss_path + "/emissive_color_field"
    emissive_color_field = stage.DefinePrim(emissive_color_field_path, "OmniNuRecFieldAsset")
    gauss_volume.CreateFieldRelationship("emissiveColor", emissive_color_field_path)

    nurec_relative_path = "./" + model_file.filename
    density_field.CreateAttribute("filePath", Sdf.ValueTypeNames.Asset).Set(nurec_relative_path)
    density_field.CreateAttribute("fieldName", Sdf.ValueTypeNames.Token).Set("density")
    density_field.CreateAttribute("fieldDataType", Sdf.ValueTypeNames.Token).Set("float")
    density_field.CreateAttribute("fieldRole", Sdf.ValueTypeNames.Token).Set("density")

    emissive_color_field.CreateAttribute("filePath", Sdf.ValueTypeNames.Asset).Set(nurec_relative_path)
    emissive_color_field.CreateAttribute("fieldName", Sdf.ValueTypeNames.Token).Set("emissiveColor")
    emissive_color_field.CreateAttribute("fieldDataType", Sdf.ValueTypeNames.Token).Set("float3")
    emissive_color_field.CreateAttribute("fieldRole", Sdf.ValueTypeNames.Token).Set("emissiveColor")

    emissive_color_field.CreateAttribute("omni:nurec:ccmR", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f([1.0, 0.0, 0.0, 0.0]))
    emissive_color_field.CreateAttribute("omni:nurec:ccmG", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f([0.0, 1.0, 0.0, 0.0]))
    emissive_color_field.CreateAttribute("omni:nurec:ccmB", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f([0.0, 0.0, 1.0, 0.0]))

    gauss_prim.GetAttribute("extent").Set([[min_x, min_y, min_z], [max_x, max_y, max_z]])
    gauss_prim.CreateAttribute("omni:nurec:offset", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3d(0.0, 0.0, 0.0))
    gauss_prim.CreateAttribute("omni:nurec:crop:minBounds", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3d(min_x, min_y, min_z))
    gauss_prim.CreateAttribute("omni:nurec:crop:maxBounds", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3d(max_x, max_y, max_z))
    gauss_prim.CreateRelationship("proxy")

    return NamedUSDStage(filename="gauss.usda", stage=stage)


def serialize_usd_default_layer(gauss_stage: NamedUSDStage) -> NamedUSDStage:
    stage = initialize_usd_stage(up_axis="Z")
    delegate = UsdUtils.CoalescingDiagnosticDelegate()
    prim = stage.OverridePrim(f"/World/{Path(gauss_stage.filename).stem}")
    prim.GetReferences().AddReference(gauss_stage.filename)

    gauss_layer = gauss_stage.stage.GetRootLayer()
    if "renderSettings" in gauss_layer.customLayerData:
        new_settings = gauss_layer.customLayerData["renderSettings"]
        current = stage.GetRootLayer().customLayerData.get("renderSettings", {})
        if current is None:
            current = {}
        current.update(new_settings)
        stage.SetMetadataByDictKey("customLayerData", "renderSettings", current)

    return NamedUSDStage(filename="default.usda", stage=stage)


def write_to_usdz(file_path: Path, model_file: NamedSerialized, gauss_usd: NamedUSDStage, default_usd: NamedUSDStage):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(file_path, "w", compression=zipfile.ZIP_STORED) as zip_file:
        default_usd.save_to_zip(zip_file)
        model_file.save_to_zip(zip_file)
        gauss_usd.save_to_zip(zip_file)
    logger.info(f"USDZ file created: {file_path}")


# ---------------------------------------------------------------------------
# PLY reader
# ---------------------------------------------------------------------------

def read_ply_as_gaussians(ply_path: str, max_sh_degree: int = 3):
    """Read a standard 3DGS PLY file and return Gaussian data arrays."""
    plydata = PlyData.read(ply_path)
    vertex = plydata.elements[0]
    num_gaussians = len(vertex)

    positions = np.stack(
        (np.asarray(vertex["x"]), np.asarray(vertex["y"]), np.asarray(vertex["z"])),
        axis=1,
    ).astype(np.float32)

    # Opacity (pre-activation, raw from PLY)
    densities = np.asarray(vertex["opacity"], dtype=np.float32)[..., np.newaxis]

    # DC features (albedo) - SH degree 0
    features_albedo = np.zeros((num_gaussians, 3), dtype=np.float32)
    features_albedo[:, 0] = np.asarray(vertex["f_dc_0"])
    features_albedo[:, 1] = np.asarray(vertex["f_dc_1"])
    features_albedo[:, 2] = np.asarray(vertex["f_dc_2"])

    # Higher-order SH features (specular)
    extra_f_names = sorted(
        [p.name for p in vertex.properties if p.name.startswith("f_rest_")],
        key=lambda x: int(x.split("_")[-1]),
    )
    num_specular = 3 * (((max_sh_degree + 1) ** 2) - 1)
    features_specular = np.zeros((num_gaussians, num_specular), dtype=np.float32)
    if len(extra_f_names) > 0:
        for idx, attr_name in enumerate(extra_f_names[:num_specular]):
            features_specular[:, idx] = np.asarray(vertex[attr_name])

    # Rotation (quaternion, wxyz order in PLY: rot_0=w, rot_1=x, rot_2=y, rot_3=z)
    rotations = np.zeros((num_gaussians, 4), dtype=np.float32)
    rotations[:, 0] = np.asarray(vertex["rot_0"])
    rotations[:, 1] = np.asarray(vertex["rot_1"])
    rotations[:, 2] = np.asarray(vertex["rot_2"])
    rotations[:, 3] = np.asarray(vertex["rot_3"])

    # Scale (pre-activation, raw log-scale from PLY)
    scales = np.zeros((num_gaussians, 3), dtype=np.float32)
    scales[:, 0] = np.asarray(vertex["scale_0"])
    scales[:, 1] = np.asarray(vertex["scale_1"])
    scales[:, 2] = np.asarray(vertex["scale_2"])

    n_active_features = max_sh_degree

    logger.info(f"Loaded {num_gaussians} gaussians from {ply_path}")
    logger.info(f"  SH degree: {max_sh_degree}, specular features: {num_specular}")
    logger.info(f"  Bounding box: min={positions.min(axis=0)}, max={positions.max(axis=0)}")

    return positions, rotations, scales, densities, features_albedo, features_specular, n_active_features


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert_ply_to_usdz(input_path: str, output_path: str, max_sh_degree: int = 3):
    """Convert a PLY file to USDZ (NuRec format)."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    logger.info(f"Converting {input_path} to {output_path}")

    # 1. Read PLY
    positions, rotations, scales, densities, albedo, specular, n_active = read_ply_as_gaussians(
        str(input_path), max_sh_degree=max_sh_degree
    )

    # 2. Fill NuRec template
    template = fill_3dgut_template(
        positions=positions,
        rotations=rotations,
        scales=scales,
        densities=densities,
        features_albedo=albedo,
        features_specular=specular,
        n_active_features=n_active,
    )

    # 3. Compress with gzip + msgpack
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", compresslevel=0) as f:
        packed = msgpack.packb(template)
        f.write(packed)

    model_file = NamedSerialized(filename=output_path.stem + ".nurec", serialized=buffer.getvalue())

    # 4. Create USD stages
    gauss_usd = serialize_nurec_usd(model_file, positions)
    default_usd = serialize_usd_default_layer(gauss_usd)

    # 5. Write USDZ
    write_to_usdz(output_path, model_file, gauss_usd, default_usd)
    logger.info(f"Successfully converted to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert 3DGS PLY to USDZ (NuRec format)")
    parser.add_argument("input_file", type=str, help="Input PLY file path")
    parser.add_argument("--output_file", type=str, help="Output USDZ file path")
    parser.add_argument("--max_sh_degree", type=int, default=3, help="Max SH degree (default: 3)")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if args.output_file:
        output_path = Path(args.output_file)
    else:
        output_path = input_path.with_suffix(".usdz")

    convert_ply_to_usdz(str(input_path), str(output_path), max_sh_degree=args.max_sh_degree)


if __name__ == "__main__":
    main()
