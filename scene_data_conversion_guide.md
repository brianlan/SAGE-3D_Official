# SAGE-3D Scene Data Conversion Guide

Convert InteriorGS compressed PLY files to USDA scene files ready for Isaac Sim 5.0+.

## Prerequisites

```bash
# splat-transform (PLY decompression)
npm install -g @playcanvas/splat-transform

# Python environment with required packages
conda env: /ssd4/envs/sage_conversion_py311
# Required: numpy, torch, plyfile, msgpack, usd-core, hydra-core, omegaconf, rich
```

## Environment Setup

```bash
conda activate /ssd4/envs/sage_conversion_py311
```

## Step 1: Compressed PLY → Original PLY

Decompress using `splat-transform`:

```bash
splat-transform /ssd5/datasets/InteriorGS/0001_839920/3dgs_compressed.ply \
    /tmp/sage_ply/0001_839920.ply
```

## Step 2: Original PLY → USDZ

```bash
python Code/data_pipeline/interiorgs_processing/sage_ply_to_usdz.py \
    /tmp/sage_ply/0001_839920.ply \
    --output_file /ssd5/datasets/SAGE-3D_InteriorGS_usdz/InteriorGS_usdz/839920.usdz
```

> **Note:** The official `3dgrut` package has unresolvable NVIDIA-internal dependencies (`ncore`, `threedgrt_tracer`, etc.). The standalone script `Code/data_pipeline/interiorgs_processing/sage_ply_to_usdz.py` produces identical NuRec-format USDZ output without those dependencies.

## Step 3: Build USDA (USDZ + Collision Mesh → USDA)

```bash
python Code/benchmark/scene_data/sage3d_usda_builder.py \
    --usdz-dir /ssd5/datasets/SAGE-3D_InteriorGS_usdz/InteriorGS_usdz \
    --out-dir /ssd5/datasets/SAGE-3D_InteriorGS_CollisionMesh_usda \
    --template Data/template.usda \
    --usdz-placeholder "@usdz_root[gauss.usda]@" \
    --collision-placeholder "@collision_root@" \
    --usdz-path-template "/ssd5/datasets/SAGE-3D_InteriorGS_usdz/InteriorGS_usdz/{scene_id}.usdz[gauss.usda]" \
    --collision-path-template "/ssd5/datasets/SAGE-3D_Collision_Mesh/Collision_Mesh/{scene_id}/{scene_id}_collision.usd" \
    --overwrite
```

## Important Notes

- **USDZ naming:** The builder filters USDZ files by numeric-only filename stems. Name files as `<scene_id>.usdz` (e.g. `839920.usdz`), not with the InteriorGS prefix (e.g. `0001_839920.usdz`).
- **Data paths:**
  - InteriorGS: `/ssd5/datasets/InteriorGS/`
  - Collision Mesh: `/ssd5/datasets/SAGE-3D_Collision_Mesh/Collision_Mesh/`
  - Output USDZ: `/ssd5/datasets/SAGE-3D_InteriorGS_usdz/InteriorGS_usdz/`
  - Output USDA: `/ssd5/datasets/SAGE-3D_InteriorGS_CollisionMesh_usda/`
