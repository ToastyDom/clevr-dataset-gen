# Single-object VLM dataset

Controlled CLEVR images for analysing how LLaVA-1.5 processes object color, shape and position.
Each image contains exactly one object; color, shape and position vary independently:

- 36 positions (6 x 6 grid) x 3 shapes (sphere, cube, cylinder) x 8 colors (gray, red, blue, green, brown, purple, cyan, yellow) = **864 images**
- Rendered directly at **336 x 336** with a square camera frame (LLaVA-1.5 input size; 14 px patches, 24 x 24 patch grid)
- Fixed for all images: camera, lights, background, material (CLEVR `rubber`, i.e. matte), size (CLEVR `small`), object orientation

`placements.json` in this folder is a reference copy of the calibrated camera and object placements of the released dataset (the working copy is `output/CLEVR_SINGLE_OBJECT/placements.json`). `../vlm_two_object/layout.json` is the same for the two-object dataset.

## Generating

Each dataset folder (`output/CLEVR_SINGLE_OBJECT/`, `output/CLEVR_DUAL_OBJECT/`) has a `settings.json` with three sections: `layout` (anything that moves objects: `image_size`, `patch_size`, `grid_size`, `grid_spacing_patches`, `size`, `shapes`, `camera_distance`, `min_margin_px`, ...; for two objects `grid_size`, `grid_step`, `shapes`, `min_gap_px`, `min_world_gap`), `render` (`colors`, `cast_shadows`, `render_num_samples`, `denoise`, `use_gpu`, `skip_existing`) and `paths` (relative to the settings file). All options are listed at the top of the two generator scripts. Command-line options override the settings file, e.g. `--render_num_samples 64` for quick previews, or `--positions 0 14 35 --output_dir <folder>` to render a subset into another folder.

From the `image_generation` directory (`blender` = `/Applications/Blender.app/Contents/MacOS/Blender`):

```bash
# Single-object dataset: layout (~2 min; only recomputed if the layout settings changed),
# render (~25 min on an Apple Silicon GPU; an interrupted run continues), CLEVR export
blender --background --python-exit-code 1 --python render_single_object_dataset.py -- --config ../output/CLEVR_SINGLE_OBJECT/settings.json --stage layout
blender --background --python-exit-code 1 --python render_single_object_dataset.py -- --config ../output/CLEVR_SINGLE_OBJECT/settings.json --stage render
python export_clevr_format.py --dataset single --source_dir ../output/CLEVR_SINGLE_OBJECT/generation --target_dir ../output/CLEVR_SINGLE_OBJECT --overwrite 1

# Two-object dataset: built on the single-object placements (output/CLEVR_DUAL_OBJECT/grid_placements.json)
blender --background --python-exit-code 1 --python render_two_object_dataset.py -- --config ../output/CLEVR_DUAL_OBJECT/settings.json --stage layout
blender --background --python-exit-code 1 --python render_two_object_dataset.py -- --config ../output/CLEVR_DUAL_OBJECT/settings.json --stage render verify preview
python export_clevr_format.py --dataset dual --source_dir ../output/CLEVR_DUAL_OBJECT/generation --target_dir ../output/CLEVR_DUAL_OBJECT --overwrite 1
```

`--python-exit-code 1` makes Blender exit with an error code if the script fails. Rendering stops with an error if images already in the output folder were made with other `cast_shadows`/`render_num_samples`/`denoise` values (use `--skip_existing 0` or a new folder), and the render stage stops if the placements or layout were computed with other layout settings (run the layout stage first).

## Design

**Camera.** The camera keeps the CLEVR viewing direction (looking at the origin from 28.3° elevation, 139° azimuth). It is moved back from 11.3 to 200 units and uses a 1497 mm lens.
With the original camera, moving an object from the top to the bottom of the image would change its size by about 2.5x. Here the projected area varies by at most 6% across the grid (about 3% in width/height).

**Size.** CLEVR `small` objects (scale 0.35; cubes 0.35/√2 as in CLEVR) are used with a correspondingly longer lens. On screen they are as large as `large` objects would be with half the zoom.
This keeps all 36 placements inside the region CLEVR places objects in (|x|, |y| ≲ 3.3). There the floor is flat (at most 0.24 px of tilt under any object) and the lights work as designed.
With `large` objects the grid would cover four times the floor area, reaching onto the curved backdrop and past the key light.

**Positions.** Positions are chosen in projected image coordinates.
- *Projected center:* the projection of the object's 3D center.
- *Grid:* the 36 projected centers form a grid with a spacing of exactly 3 patches (42 px), at x, y ∈ {56, 98, 140, 182, 224, 266} px. Every object has the same offset relative to the patch boundaries; it is centered exactly on a patch corner.
- *World placement:* for each shape and position, a ray is cast through the target pixel and intersected with the height of the object's center when it rests on the ground. The three shapes therefore have slightly different world locations for the same projected center.

**Patch coverage calibration.** Object masks are rendered at 4x4 supersampling. Each patch's coverage is the fraction of its area covered by the object silhouette (shadows excluded).
The calibration minimizes, summed over all 108 shape-position masks:
- the number of *fragment* patches (coverage between 0 and 10%), plus
- half the number of patches with 10–25% coverage.

It searches jointly over the camera zoom (sphere diameter 50–60 px) and the shared sub-patch offset (0–14 px in 0.25 px steps).
The result is a sphere diameter of 55 px with centers on patch corners:

| shape    | patches touched | fragment patches (<10%) | patches 10–25% | silhouette (w x h px) | min. margin |
|----------|-----------------|-------------------------|----------------|-----------------------|-------------|
| sphere   | 16 (4x4)        | 0                       | 0              | 55 x 55               | 28 px       |
| cube     | 16 (4x4)        | 0                       | 0              | 39 x 50               | 31 px       |
| cylinder | 24 (4x6)        | 0                       | 1.3 on average | 55 x 74               | 20 px       |

These numbers hold at every position (see `calibration.summary` in `placements.json`).

**Orientation.** All objects have the same yaw (49°), so the cube shows one face square-on to the camera. Its silhouette edges then run along the patch grid.

**Color.** Colors are the CLEVR values from `properties.json`. As in CLEVR, the 0–255 values divided by 255 are used directly as the material's Color input, rendered with the `Standard` view transform.
For each shape and position, the object and material are created once and only the material's Color input changes between colors.
The rendered background is therefore almost identical across colors. Far from the object it differs by 0.2/255 on average (at most about 7/255) due to path-tracing noise and denoising. Near the object, physically correct color bleeding into the floor and shadow causes larger differences.

## Known covariates

- **Lighting varies with position.** The CLEVR lights stay fixed, so objects closer to the key light (bottom right) are brighter. A red sphere's mean rendered color ranges from about (114, 54, 54) at the top left to (133, 63, 60) at the bottom right. The measured mean color of every image is stored in the metadata.
- Shadows fall mostly to the left of the objects and change shape slightly with position.
- Perspective remains but is weak: projected size changes by up to ±1.5% (linear) across the grid.
- Cubes are smaller on screen than spheres and cylinders (CLEVR scales cubes by 1/√2), and cylinders are taller.

## Output files

The CLEVR export (`images/train/CLEVR_train_*.png`, `scenes/CLEVR_train_scenes.json`) is in `output/CLEVR_SINGLE_OBJECT/`; the generator output below is in `output/CLEVR_SINGLE_OBJECT/generation/` (git-ignored). The CLEVR images are hard links to the files in `generation/images/`:

| file | content |
|------|---------|
| `images/{shape}_{color}_pos{NN}.png` | RGB renders, 336 x 336. Position index `NN = row * 6 + col`, row 0 at the top, col 0 at the left |
| `masks/{shape}_pos{NN}.png` | per-pixel object coverage (0–255), identical for all colors |
| `coverage.npz` | `pixel_coverage` (108 x 336 x 336) and `patch_coverage` (108 x 24 x 24) as float arrays, with `shapes` and `position_index` |
| `metadata.json`, `metadata.csv` | one record per image: shape, color, position, projected center, world location, mask area/bbox/centroid, patch statistics, mean rendered object color |
| `placements.json` | copy of the placements used |

Pixel coordinates have x to the right and y down, with (0, 0) the top-left corner of the image. Patch `(r, c)` covers pixels `[14c, 14c+14) x [14r, 14r+14)`.
