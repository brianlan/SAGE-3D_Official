# SAGE-3D Scene Data Conversion Guide

Convert InteriorGS compressed PLY files into USDA scene files ready for Isaac Sim 5.0+.

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

## Scene ID Rule (Important)

InteriorGS scene folders contain a 4-digit prefix plus the actual asset ID:

- InteriorGS folder: `0002_839955`
- Asset ID used by USDZ / collision mesh / USDA: `839955`

Always derive the downstream scene ID from the part after the underscore. The USDA builder only processes flat `.usdz` files whose stems are purely numeric.

## Step 1: Compressed PLY → Original PLY

Decompress the InteriorGS scene folder into a temporary PLY file.

```bash
splat-transform /ssd5/datasets/SAGE3D/InteriorGS/0002_839955/3dgs_compressed.ply \
    /tmp/sage_ply/839955.ply
```

The temporary PLY filename is not important, but using the numeric asset ID avoids confusion.

## Step 2: Original PLY → USDZ

Convert the temporary PLY into a flat numeric USDZ filename under `InteriorGS_usdz/`.

```bash
python Code/data_pipeline/interiorgs_processing/sage_ply_to_usdz.py \
    /tmp/sage_ply/839955.ply \
    --output_file /ssd5/datasets/SAGE3D/InteriorGS_usdz/839955.usdz
```

After the USDZ file is written successfully, delete the temporary PLY.

```bash
rm /tmp/sage_ply/839955.ply
```

> **Note:** The official `3dgrut` package has NVIDIA-internal dependencies that are not available in a normal environment. `Code/data_pipeline/interiorgs_processing/sage_ply_to_usdz.py` is the standalone replacement used here.

## Step 3: Build USDA (USDZ + Collision Mesh → USDA)

Run the USDA builder against the flat USDZ directory and the collision-mesh directory.

```bash
python Code/benchmark/scene_data/sage3d_usda_builder.py \
    --usdz-dir /ssd5/datasets/SAGE3D/InteriorGS_usdz \
    --out-dir /ssd5/datasets/SAGE3D/InteriorGS_CollisionMesh_usda \
    --template Data/template.usda \
    --usdz-placeholder "@usdz_root[gauss.usda]@" \
    --collision-placeholder "@collision_root@" \
    --usdz-path-template "/ssd5/datasets/SAGE3D/InteriorGS_usdz/{scene_id}.usdz[gauss.usda]" \
    --collision-path-template "/ssd5/datasets/SAGE3D/Collision_Mesh/Collision_Mesh/{scene_id}/{scene_id}_collision.usd" \
    --overwrite
```

## Important Notes

- **InteriorGS naming is not downstream naming:** `0002_839955` is an InteriorGS folder name, but downstream assets must use the numeric scene ID `839955`.
- **USDZ layout must be flat:** put files directly under `/ssd5/datasets/SAGE3D/InteriorGS_usdz/` as `<scene_id>.usdz`. Do not use nested directories such as `/.../InteriorGS_usdz/839955/839955.usdz`.
- **Builder discovery rule:** `Code/benchmark/scene_data/sage3d_usda_builder.py` only iterates over flat `*.usdz` files whose stems are numeric.
- **Collision mesh layout:** collision meshes live under `/ssd5/datasets/SAGE3D/Collision_Mesh/Collision_Mesh/{scene_id}/{scene_id}_collision.usd`.
- **Recommended outputs:**
  - InteriorGS input: `/ssd5/datasets/SAGE3D/InteriorGS/`
  - Flat USDZ output: `/ssd5/datasets/SAGE3D/InteriorGS_usdz/`
  - USDA output: `/ssd5/datasets/SAGE3D/InteriorGS_CollisionMesh_usda/`
