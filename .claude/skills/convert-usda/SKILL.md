---
name: convert-usda
description: Convert SAGE-3D / InteriorGS scene assets from compressed PLY to USDA, including the intermediate PLY and USDZ steps. Use when working with InteriorGS folders like 0002_839955, building flat numeric USDZ files like 839955.usdz, regenerating USDA scene files, or resolving scene-ID mismatches between InteriorGS, USDZ, and collision-mesh assets.
---

# Convert USDA

Use this workflow for InteriorGS scene conversion:

1. Decompress `3dgs_compressed.ply` with `splat-transform`.
2. Convert the temporary PLY to USDZ with `Code/data_pipeline/interiorgs_processing/sage_ply_to_usdz.py`.
3. Build USDA files with `Code/benchmark/scene_data/sage3d_usda_builder.py`.

## ID Mapping Rule

Convert the InteriorGS folder name to the downstream asset ID by dropping the 4-digit prefix before the underscore.

- InteriorGS folder: `0002_839955`
- USDZ / collision / USDA scene ID: `839955`

Always use the numeric scene ID for:

- USDZ filename: `839955.usdz`
- Collision mesh directory: `.../839955/839955_collision.usd`
- USDA filename: `839955.usda`

Do not name downstream assets with the InteriorGS prefix.

## Commands

### Step 1: Compressed PLY → Original PLY

```bash
splat-transform /ssd5/datasets/SAGE3D/InteriorGS/<interiorgs_folder>/3dgs_compressed.ply \
    /tmp/sage_ply/<scene_id>.ply
```

Example:

```bash
splat-transform /ssd5/datasets/SAGE3D/InteriorGS/0002_839955/3dgs_compressed.ply \
    /tmp/sage_ply/839955.ply
```

### Step 2: Original PLY → USDZ

```bash
cd /ssd4/github-knowledge-base/3dgrut && \
    PATH="/ssd4/github-knowledge-base/3dgrut/.venv/bin:$PATH" \
    .venv/bin/python -m threedgrut.export.scripts.ply_to_usd \
        /tmp/sage_ply/<scene_id>.ply \
        --output_file /ssd5/datasets/SAGE3D/InteriorGS_usdz/<scene_id>.usdz
```

If conversion succeeds, remove the temporary PLY:

```bash
rm /tmp/sage_ply/<scene_id>.ply
```

### Step 3: Build USDA

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

## Guardrails

- Keep USDZ files flat under `/ssd5/datasets/SAGE3D/InteriorGS_usdz/`.
- Ensure USDZ stems are numeric only; the builder skips non-numeric names.
- Use `/ssd5/datasets/SAGE3D/Collision_Mesh/Collision_Mesh/{scene_id}/{scene_id}_collision.usd` for collision payloads.
- If a user gives an InteriorGS folder name, derive `<scene_id>` from the suffix after `_` before writing USDZ or USDA outputs.
