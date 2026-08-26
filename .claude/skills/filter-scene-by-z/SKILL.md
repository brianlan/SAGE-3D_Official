---
name: filter-scene-by-z
description: Filter a SAGE-3D / InteriorGS scene by a Z height threshold, producing a new USDA file that references filtered gaussians and collision geometry. Use whenever the user wants to remove geometry above a certain Z height, create a height-clipped variant of a scene, generate a USDA like {scene_id}_z{max_z}.usda, or filter a scene by max Z. Also use when the user mentions clipping, culling, or slicing a 3DGS scene by height.
---

# Filter Scene by Z

This workflow creates a Z-clipped variant of a SAGE-3D scene. It filters both the USDZ gaussians and the collision mesh by a Z threshold, then builds a new USDA file with a suffixed name so the original unfiltered USDA is preserved.

## Inputs

- `scene_id` — numeric scene ID (e.g., `840040`)
- `max_z` — Z threshold; geometry with Z > max_z is removed

## Asset Paths

| Asset | Path |
|-------|------|
| Original USDZ | `/ssd5/datasets/SAGE3D/InteriorGS_usdz/{scene_id}.usdz` |
| Original Collision | `/ssd5/datasets/SAGE3D/Collision_Mesh/Collision_Mesh/{scene_id}/{scene_id}_collision.usd` |
| Filtered Assets | `/ssd5/datasets/SAGE3D/Filtered_By_Z/{scene_id}_z{max_z}/` |
| Output USDA | `/ssd5/datasets/SAGE3D/InteriorGS_CollisionMesh_usda/{scene_id}_z{max_z}.usda` |

## Quick Run (Wrapper Script)

```bash
python Code/data_pipeline/interiorgs_processing/build_filtered_usda.py \
    --scene-id <scene_id> \
    --max-z <max_z>
```

Add `--overwrite` to regenerate an existing variant.

## Manual Workflow

If you need to run steps individually, use the following commands from the repo root.

### 1. Prepare working directory

```bash
SCENE_ID=<scene_id>
MAX_Z=<max_z>
WORK_DIR=/ssd5/datasets/SAGE3D/Filtered_By_Z/${SCENE_ID}_z${MAX_Z}
rm -rf $WORK_DIR && mkdir -p $WORK_DIR
```

### 2. Filter USDZ

```bash
python Code/data_pipeline/interiorgs_processing/filter_usdz_by_z.py \
    /ssd5/datasets/SAGE3D/InteriorGS_usdz/${SCENE_ID}.usdz \
    --output $WORK_DIR/${SCENE_ID}.usdz \
    --z-threshold $MAX_Z
```

### 3. Filter collision mesh

```bash
python Code/data_pipeline/interiorgs_processing/filter_collision_usd_by_z.py \
    /ssd5/datasets/SAGE3D/Collision_Mesh/Collision_Mesh/${SCENE_ID}/${SCENE_ID}_collision.usd \
    --output $WORK_DIR/${SCENE_ID}_collision.usd \
    --z-threshold $MAX_Z
```

### 4. Build USDA

```bash
python Code/benchmark/scene_data/sage3d_usda_builder.py \
    --usdz-dir $WORK_DIR \
    --out-dir $WORK_DIR \
    --template Data/template.usda \
    --base-id 839920 \
    --expected-count 1 \
    --usdz-placeholder "@usdz_root[gauss.usda]@" \
    --usdz-path-template "$WORK_DIR/${SCENE_ID}.usdz[gauss.usda]" \
    --collision-placeholder "@collision_root@" \
    --collision-path-template "$WORK_DIR/${SCENE_ID}_collision.usd" \
    --overwrite
```

This generates `$WORK_DIR/${SCENE_ID}.usda`.

### 5. Fix authoring layer and move to final location

```bash
sed -i "s|string authoring_layer = \"./${SCENE_ID}.usda\"|string authoring_layer = \"./${SCENE_ID}_z${MAX_Z}.usda\"|" \
    $WORK_DIR/${SCENE_ID}.usda

mv $WORK_DIR/${SCENE_ID}.usda \
    /ssd5/datasets/SAGE3D/InteriorGS_CollisionMesh_usda/${SCENE_ID}_z${MAX_Z}.usda
```

## Guardrails

- Do **not** delete the working directory (`Filtered_By_Z/{scene_id}_z{max_z}/`) after building; the output USDA references the filtered assets inside it by absolute path.
- The wrapper script will refuse to overwrite the original `{scene_id}.usda`; the output is always named `{scene_id}_z{max_z}.usda`.
- If a filtered variant already exists, use `--overwrite` to regenerate it.
