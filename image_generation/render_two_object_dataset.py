"""
Renders a controlled two-object CLEVR dataset for VLM (LLaVA-1.5) analysis.

Every scene contains two objects with persistent IDs A and B. The dataset is
every combination of

  9 positions for A x 8 remaining positions for B x 3 shapes for A
  x 3 shapes for B x 2 colors (red, blue) for A x 2 colors for B = 2592

labeled assignments. Exchanging the labels A and B of an assignment gives a
different assignment but the same rendered scene, so each unique scene
(1296 when no layouts are excluded) is rendered once and the metadata lists
both assignments that it corresponds to.

Everything else is shared with the single-object dataset
(render_single_object_dataset.py): the camera, lights, background, material
(CLEVR "rubber"), size, object orientation, Cycles settings and the world
locations of the objects. The nine positions are a 3x3 subset of the
single-object 6x6 grid, so projected object centers stay on patch corners and
are spaced by 6 patches (84 px).

Controlled comparisons: a scene is built from scratch from (position, shape,
color) for each object, using the same per-shape world location for a
position in every scene. Changing one object's color, shape or position
therefore leaves the other object and all remaining scene parameters
unchanged. (The rendered pixels of the other object can still change slightly
through physically correct shadows and inter-reflections; the verify stage
measures this.)

Settings can be given on the command line or in a settings file (--config,
e.g. output/CLEVR_DUAL_OBJECT/settings.json) with "layout", "render" and
"paths" sections; command-line arguments override the file. The two-object
dataset is built on the single-object placements (--single_placements_json;
in the settings file, a copy named grid_placements.json next to it).

Stages (from the image_generation directory; several can be given, e.g.
--stage render verify preview):

blender --background --python render_two_object_dataset.py -- --stage all

- layout:  choose the positions and check every position pair for every
           shape pair with object masks; writes the layout JSON. Only runs if
           the layout file is missing, was made with other layout settings or
           another single-object placements file, or --relayout 1 is given
- render:  render images, masks, metadata, questions and groupings
- verify:  check the outputs and measure controlled comparisons
- preview: write a contact sheet with patch-grid and mask overlays
- all:     layout, render, verify, preview
"""

import argparse, csv, itertools, json, math, os, sys, time
from collections import defaultdict
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import bpy
import numpy as np
import utils
import render_single_object_dataset as single


# Shapes and colors of the dataset; set from the settings / layout in main()
SHAPES = ['sphere', 'cube', 'cylinder']
COLORS = ['red', 'blue']
OBJECT_IDS = ['A', 'B']
MATERIAL = single.MATERIAL
# Pixels further than this from both objects' bounding boxes count as
# background when measuring controlled comparisons
BACKGROUND_DISTANCE_PX = 20

# Image size, patch size and the single-object grid are taken from the
# single-object placements file (single.apply_geometry_from_placements), and
# read from the single module as single.IMAGE_SIZE etc.

# Settings that determine the layout; stored in the layout file
LAYOUT_SETTINGS = ['grid_size', 'grid_step', 'shapes', 'min_gap_px',
                   'min_world_gap']
PATH_SETTINGS = ['single_placements_json', 'layout_json', 'output_dir',
                 'base_scene_blendfile', 'properties_json', 'shape_dir',
                 'material_dir']


def path_arg(p):
  return os.path.join(SCRIPT_DIR, p)


parser = argparse.ArgumentParser()
parser.add_argument('--config', default=None,
    help="Settings file (JSON) with layout, render and paths sections.")
parser.add_argument('--stage', default=['all'], nargs='+',
    choices=['all', 'layout', 'render', 'verify', 'preview'])
parser.add_argument('--single_placements_json',
    default=path_arg('../output/CLEVR_SINGLE_OBJECT/placements.json'),
    help="Placements of the single-object dataset; camera, lights, grid and " +
         "world locations are reused from it.")
parser.add_argument('--layout_json',
    default=path_arg('../output/CLEVR_DUAL_OBJECT/layout.json'),
    help="Where the chosen positions and the pair checks are stored.")
parser.add_argument('--relayout', default=0, type=int,
    help="With --stage layout or all, recompute the layout even if it is up " +
         "to date.")
parser.add_argument('--output_dir',
    default=path_arg('../output/CLEVR_DUAL_OBJECT/generation'),
    help="Directory for the rendered images, masks and metadata.")

# Input assets
parser.add_argument('--base_scene_blendfile',
    default=path_arg('data/base_scene.blend'))
parser.add_argument('--properties_json',
    default=path_arg('data/properties.json'))
parser.add_argument('--shape_dir', default=path_arg('data/shapes'))
parser.add_argument('--material_dir', default=path_arg('data/materials'))

# Layout options (changing any of them requires a new layout)
parser.add_argument('--grid_size', default=3, type=int,
    help="Number of rows and columns of the two-object grid.")
parser.add_argument('--grid_step', default=2, type=int,
    help="The two-object grid uses every grid_step-th row and column of the " +
         "single-object grid (2: every other one, 84 px apart by default).")
parser.add_argument('--shapes', nargs='+', default=['sphere', 'cube', 'cylinder'],
    help="Shapes of the dataset; must be calibrated in the single-object " +
         "placements.")
parser.add_argument('--min_gap_px', default=4.0, type=float,
    help="Minimum distance between the two object silhouettes in pixels.")
parser.add_argument('--min_world_gap', default=0.1, type=float,
    help="Minimum distance between the two objects' footprints on the " +
         "ground (world units; objects are 0.7 wide).")

# Rendering options (same defaults as the single-object dataset)
parser.add_argument('--use_gpu', default=1, type=int)
parser.add_argument('--render_num_samples', default=512, type=int)
parser.add_argument('--denoise', default=1, type=int)
parser.add_argument('--cast_shadows', default=1, type=int,
    help="Setting --cast_shadows 0 stops the objects from casting shadows.")
parser.add_argument('--colors', nargs='+', default=['red', 'blue'],
    help="Colors of the dataset (each object takes every color).")
parser.add_argument('--skip_existing', default=1, type=int,
    help="Do not re-render images that already exist.")
parser.add_argument('--scene_ids', nargs='+', type=int, default=None,
    help="Only render these scene IDs (for previews).")
parser.add_argument('--preview_tiles', default=12, type=int,
    help="Number of scenes on the contact sheet.")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

Progress = single.Progress
say = single.say


def write_json(path, data, indent=1):
  with open(path, 'w') as f:
    json.dump(single.to_jsonable(data), f, indent=indent)


def load_json(path):
  with open(path, 'r') as f:
    return json.load(f)


def write_rgb_png(path, rgb):
  """ Write an (H, W, 3) array of values in [0, 1] (top row first) as a PNG """
  h, w, _ = rgb.shape
  img = bpy.data.images.new('png_out', w, h, alpha=False)
  img.colorspace_settings.name = 'Non-Color'
  rgba = np.ones((h, w, 4), dtype=np.float32)
  rgba[..., :3] = rgb[::-1]
  img.pixels.foreach_set(rgba.ravel())
  img.filepath_raw = path
  img.file_format = 'PNG'
  img.save()
  bpy.data.images.remove(img)


def remove_object(obj):
  """ Delete an object together with its mesh and materials """
  mesh = obj.data
  materials = [m for m in mesh.materials if m is not None]
  utils.delete_object(obj)
  if mesh.users == 0:
    bpy.data.meshes.remove(mesh)
  for m in materials:
    if m.users == 0:
      bpy.data.materials.remove(m)


def footprint_radius(properties, shape, size):
  """ Radius of the circle containing the object's footprint on the ground """
  r = single.object_scale(properties, shape, size)
  return r * math.sqrt(2) if shape == 'cube' else r


def outline(mask):
  """ Pixels of a boolean mask that touch a pixel outside the mask """
  inner = mask.copy()
  inner[1:, :] &= mask[:-1, :]
  inner[:-1, :] &= mask[1:, :]
  inner[:, 1:] &= mask[:, :-1]
  inner[:, :-1] &= mask[:, 1:]
  return mask & ~inner


def silhouette_gap_px(mask1, mask2):
  """
  Smallest distance (in output pixels) between the pixel centers of two
  non-overlapping object masks, minus one pixel: the width of the empty strip
  between the silhouettes. Returns -1 if the masks overlap.
  """
  if (mask1 & mask2).any():
    return -1.0
  p1 = np.argwhere(outline(mask1)).astype(np.float64)
  p2 = np.argwhere(outline(mask2)).astype(np.float64)
  d = np.sqrt(((p1[:, None, :] - p2[None, :, :]) ** 2).sum(-1)).min()
  return float(d - 1.0)


class Context(object):
  """ Base scene, camera and assets shared by all stages that use Blender """
  def __init__(self, args):
    self.single = load_json(args.single_placements_json)
    single.apply_geometry_from_placements(self.single)
    missing = [s for s in SHAPES if s not in single.SHAPES]
    if missing:
      raise ValueError('Shapes %s are not in %s' % (missing,
                                                    args.single_placements_json))
    args.size = self.single['object']['size']
    self.properties = single.load_properties(args)
    self.scene = single.load_base_scene(args)
    self.camera = single.configure_camera(self.single['camera'])
    self.rotation = self.single['object']['rotation_deg']
    self.colors = single.color_table(self.properties)
    self.args = args

  def add_object(self, shape, location, color=None):
    obj = single.place_shape(self.args, self.properties, shape, location,
                             self.rotation)
    obj.visible_shadow = bool(self.args.cast_shadows)
    if color is not None:
      utils.add_material(obj, self.properties['materials'][MATERIAL],
                         Color=self.colors[color]['material_color_input'])
    return obj

  def shape_masks(self, positions, tmp_path):
    """
    Supersampled mask of every shape at every position, rendered with the
    object alone in the scene. Keys are (shape, position key).
    """
    masks = {}
    with single.MaskRenderSettings(self.scene):
      for shape in SHAPES:
        for key, world in positions:
          obj = self.add_object(shape, world[shape])
          masks[(shape, key)] = single.render_mask(self.scene, tmp_path)
          remove_object(obj)
    return masks


# ---------------------------------------------------------------------------
# Layout: choose the nine positions and check all position/shape pairs
# ---------------------------------------------------------------------------

def check_pair(masks_hi, world, properties, size, args, p1, s1, p2, s2):
  """ Check that two objects at two positions neither overlap nor touch """
  m1, m2 = masks_hi[(s1, p1)], masks_hi[(s2, p2)]
  overlap = int((m1 & m2).sum())
  px1 = single.pixel_coverage(m1) > 0
  px2 = single.pixel_coverage(m2) > 0
  gap = silhouette_gap_px(px1, px2)
  c1, c2 = world[p1][s1], world[p2][s2]
  world_gap = (math.hypot(c1[0] - c2[0], c1[1] - c2[1])
               - footprint_radius(properties, s1, size)
               - footprint_radius(properties, s2, size))
  shared = int(((single.patch_coverage(m1) > 0) &
                (single.patch_coverage(m2) > 0)).sum())
  return {
    'silhouette_overlap_subpixels': overlap,
    'silhouette_gap_px': gap,
    'world_gap': world_gap,
    'shared_patches': shared,
    'valid': bool(overlap == 0 and gap >= args.min_gap_px and
                  world_gap >= args.min_world_gap),
  }


def compute_layout(args):
  """
  Evaluate every grid_size x grid_size subset of the single-object grid made
  of every grid_step-th row and column, check all pairs of positions for all
  shape pairs, and keep the subset with the fewest invalid pairs (then fewest
  patches shared by both objects, then largest border margin and largest
  silhouette gap).
  """
  ctx = Context(args)
  os.makedirs(args.output_dir, exist_ok=True)
  tmp_path = os.path.join(os.path.abspath(args.output_dir), '_tmp_mask.png')
  single_positions = ctx.single['positions']
  say('Layout: rendering masks of all shapes at all single-object positions')
  masks_hi = ctx.shape_masks([(p['index'], p['world'])
                              for p in single_positions], tmp_path)
  world = {p['index']: p['world'] for p in single_positions}

  n, step = args.grid_size, args.grid_step
  starts = range(single.GRID_N - step * (n - 1))
  if not starts:
    raise ValueError('A %dx%d grid with step %d does not fit into the %dx%d '
                     'single-object grid' % (n, n, step, single.GRID_N,
                                             single.GRID_N))
  index_sets = [tuple(start + step * i for i in range(n)) for start in starts]
  candidates = []
  for rows, cols in itertools.product(index_sets, repeat=2):
    indices = [r * single.GRID_N + c for r in rows for c in cols]
    pairs = []
    for i, j in itertools.combinations(range(len(indices)), 2):
      for s1, s2 in itertools.product(SHAPES, SHAPES):
        check = check_pair(masks_hi, world, ctx.properties, args.size, args,
                           indices[i], s1, indices[j], s2)
        check.update({'positions': [i, j], 'shapes': [s1, s2]})
        pairs.append(check)
    margins = [single_positions[k]['measured'][s]['margin_px']
               for k in indices for s in SHAPES]
    candidate = {
      'single_object_rows': list(rows), 'single_object_cols': list(cols),
      'single_object_indices': indices,
      'invalid_pairs': sum(not p['valid'] for p in pairs),
      'pairs_sharing_patches': sum(p['shared_patches'] > 0 for p in pairs),
      'min_margin_px': min(margins),
      'min_silhouette_gap_px': min(p['silhouette_gap_px'] for p in pairs),
      'min_world_gap': min(p['world_gap'] for p in pairs),
    }
    say('  rows %s cols %s: %d invalid pairs, %d sharing patches, min margin '
        '%.1f px, min gap %.1f px' % (rows, cols, candidate['invalid_pairs'],
        candidate['pairs_sharing_patches'], candidate['min_margin_px'],
        candidate['min_silhouette_gap_px']))
    candidates.append((candidate, pairs))

  best, pairs = min(candidates, key=lambda c: (
      c[0]['invalid_pairs'], c[0]['pairs_sharing_patches'],
      -c[0]['min_margin_px'], -c[0]['min_silhouette_gap_px']))
  say('Chosen single-object rows %s, cols %s' % (
      best['single_object_rows'], best['single_object_cols']))

  positions = []
  for pid, k in enumerate(best['single_object_indices']):
    sp = single_positions[k]
    positions.append({
      'position_id': pid, 'row': pid // n, 'col': pid % n,
      'single_object_index': k,
      'target_px': sp['target_px'],
      'center_patch': sp['center_patch'],
      'world': sp['world'],
    })

  layout = {
    'description': 'Positions of the two-object VLM dataset (a %dx%d '
                   'subset of the single-object grid) and the check of every '
                   'position pair for every shape pair. Generated by '
                   'render_two_object_dataset.py.' % (n, n),
    'created': datetime.now().isoformat(timespec='seconds'),
    'layout_settings': {k: getattr(args, k) for k in LAYOUT_SETTINGS},
    'single_placements_created': ctx.single['created'],
    'grid': {
      'rows': n, 'cols': n,
      'single_object_rows': best['single_object_rows'],
      'single_object_cols': best['single_object_cols'],
      'spacing_patches': step * single.GRID_SPACING_PATCHES,
      'spacing_px': step * single.GRID_SPACING_PATCHES * single.PATCH_SIZE,
      'center_offset_in_patch_px':
          ctx.single['grid']['center_offset_in_patch_px'],
      'row_order': 'position_id = row * %d + col; row 0 is the top (farthest) '
                   'row, col 0 the left column' % n,
    },
    'positions': positions,
    'criteria': {
      'min_gap_px': args.min_gap_px,
      'min_world_gap': args.min_world_gap,
      'checks': 'silhouettes (supersampled masks of each object rendered '
                'alone) must not overlap and must be at least min_gap_px '
                'apart; footprints on the ground at least min_world_gap '
                'apart',
    },
    'candidates': [c for c, _ in candidates],
    'pair_checks': pairs,
    'excluded_layouts': [p for p in pairs if not p['valid']],
  }
  os.makedirs(os.path.dirname(os.path.abspath(args.layout_json)),
              exist_ok=True)
  write_json(args.layout_json, layout)
  # Masks at the chosen positions, for previews of the layout
  keys = [(shape, p['position_id'], p['single_object_index'])
          for shape in SHAPES for p in positions]
  np.savez_compressed(
      os.path.join(os.path.abspath(args.output_dir), 'layout_masks.npz'),
      shapes=np.array([k[0] for k in keys]),
      position_ids=np.array([k[1] for k in keys]),
      pixel_coverage=np.stack([single.pixel_coverage(masks_hi[(k[0], k[2])])
                               for k in keys]).astype(np.float32))
  say('Wrote layout to %s (%d excluded position/shape pairs)'
      % (args.layout_json, len(layout['excluded_layouts'])))
  return layout


# ---------------------------------------------------------------------------
# Enumeration of scenes, assignments, questions and groupings
# ---------------------------------------------------------------------------

def scene_image_name(scene_id, objs):
  (p0, s0, c0), (p1, s1, c1) = objs
  return 'scene%04d_%s-%s-pos%d_%s-%s-pos%d.png' % (
      scene_id, s0, c0, p0, s1, c1, p1)


def enumerate_dataset(layout):
  """
  Enumerate all unique scenes and all labeled assignments.

  A scene is the unordered pair of objects (position, shape, color). Its two
  objects are stored in "slots" ordered by position ID. An assignment labels
  one slot A and the other B; since the positions differ, every scene has
  exactly two assignments (A in slot 0, or A in slot 1).

  Scenes and assignments whose position/shape pair failed the layout check
  are returned separately as excluded. IDs are counted over the full
  enumeration so they stay the same whatever is excluded.
  """
  excluded_pairs = {(tuple(p['positions']), tuple(p['shapes']))
                    for p in layout['excluded_layouts']}

  def excluded(objs):
    (p0, s0, _), (p1, s1, _) = objs
    return ((p0, p1), (s0, s1)) in excluded_pairs

  positions = range(len(layout['positions']))
  scenes, excluded_scenes, scene_by_key = [], [], {}
  scene_id = 0
  for p0, p1 in itertools.combinations(positions, 2):
    for s0, s1 in itertools.product(SHAPES, SHAPES):
      for c0, c1 in itertools.product(COLORS, COLORS):
        objs = ((p0, s0, c0), (p1, s1, c1))
        record = {'scene_id': scene_id, 'objects': objs,
                  'image': 'images/' + scene_image_name(scene_id, objs)}
        if excluded(objs):
          excluded_scenes.append(record)
        else:
          scenes.append(record)
          scene_by_key[objs] = record
        scene_id += 1

  assignments, excluded_assignments = [], []
  assignment_id = 0
  for pA, pB in itertools.permutations(positions, 2):
    for sA, sB in itertools.product(SHAPES, SHAPES):
      for cA, cB in itertools.product(COLORS, COLORS):
        a, b = (pA, sA, cA), (pB, sB, cB)
        objs = (a, b) if pA < pB else (b, a)
        record = {'assignment_id': assignment_id,
                  'A': a, 'B': b, 'A_slot': 0 if pA < pB else 1}
        if objs in scene_by_key:
          record['scene_id'] = scene_by_key[objs]['scene_id']
          assignments.append(record)
        else:
          excluded_assignments.append(record)
        assignment_id += 1
  return scenes, excluded_scenes, assignments, excluded_assignments


def make_questions(objs):
  """
  Questions whose reference picks out exactly one object. objs maps object
  IDs to (position, shape, color). Ambiguous references (two objects or no
  object with the queried shape/color) are left out.
  """
  questions = []
  for shape in SHAPES:
    ids = [o for o in OBJECT_IDS if objs[o][1] == shape]
    if len(ids) == 1:
      questions.append({
        'question': 'What color is the %s?' % shape, 'type': 'color',
        'reference': {'shape': shape}, 'queried_object': ids[0],
        'answer': objs[ids[0]][2],
      })
  for color in COLORS:
    ids = [o for o in OBJECT_IDS if objs[o][2] == color]
    if len(ids) == 1:
      questions.append({
        'question': 'What shape is the %s object?' % color, 'type': 'shape',
        'reference': {'color': color}, 'queried_object': ids[0],
        'answer': objs[ids[0]][1],
      })
  return questions


FACTORS = ['position', 'shape', 'color']


def build_groupings(assignments):
  """
  Groups of assignments for controlled comparisons:

  - changed_factor groups: one object's position, shape or color varies, while
    the other five attributes (including everything about the other object)
    are fixed. Members are ordered by the varying value.
  - distractor groups: one object (the target) is fixed and the other object
    (the distractor) takes every position, shape and color.
  """
  def attrs(a):
    return {'A.position': a['A'][0], 'A.shape': a['A'][1], 'A.color': a['A'][2],
            'B.position': a['B'][0], 'B.shape': a['B'][1], 'B.color': a['B'][2]}

  groups = []
  for obj in OBJECT_IDS:
    for factor in FACTORS:
      varying = '%s.%s' % (obj, factor)
      by_key = defaultdict(list)
      for a in assignments:
        at = attrs(a)
        key = tuple(sorted((k, v) for k, v in at.items() if k != varying))
        by_key[key].append((at[varying], a['assignment_id']))
      for key, members in sorted(by_key.items()):
        if len(members) < 2:
          continue
        members.sort()
        groups.append({
          'group_id': 'change_%s_%s_%04d' % (obj, factor, len(groups)),
          'comparison': 'change_%s' % factor,
          'changed_object': obj,
          'fixed': dict(key),
          'varying': varying,
          'members': [{'assignment_id': aid, 'value': v} for v, aid in members],
        })
  for target in OBJECT_IDS:
    distractor = 'B' if target == 'A' else 'A'
    by_key = defaultdict(list)
    for a in assignments:
      by_key[a[target]].append(a['assignment_id'])
    for (p, s, c), members in sorted(by_key.items()):
      groups.append({
        'group_id': 'distractor_%s_%04d' % (target, len(groups)),
        'comparison': 'fixed_target_changing_distractor',
        'target_object': target,
        'distractor_object': distractor,
        'fixed': {'%s.position' % target: p, '%s.shape' % target: s,
                  '%s.color' % target: c},
        'members': [{'assignment_id': aid} for aid in sorted(members)],
      })
  return groups


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_object_masks(ctx, objs, tmp_path):
  """
  Render the scene with each object in its own flat color and return the
  visible (occlusion-aware) supersampled mask of each object, plus the number
  of covered subpixels that belong to neither object.
  """
  scene = ctx.scene
  with single.MaskRenderSettings(scene):
    scene.display.shading.color_type = 'OBJECT'
    objs[0].color = (1.0, 0.0, 0.0, 1.0)
    objs[1].color = (0.0, 1.0, 0.0, 1.0)
    scene.render.filepath = tmp_path
    bpy.ops.render.render(write_still=True)
  img = single.read_png(tmp_path)
  os.remove(tmp_path)
  covered = img[..., 3] > 0.5
  red = img[..., 0] > 0.5
  green = img[..., 1] > 0.5
  visible = [covered & red & ~green, covered & green & ~red]
  other = int((covered & ~visible[0] & ~visible[1]).sum())
  return visible, other


def object_record(layout, ctx, measured, pos, shape, color):
  """ Metadata of one object, shared by scene and assignment records """
  p = layout['positions'][pos]
  m = measured[(shape, pos)]
  return {
    'position_id': pos, 'row': p['row'], 'col': p['col'],
    'single_object_position_index': p['single_object_index'],
    'shape': shape, 'color': color,
    'color_rgb_255': ctx.colors[color]['rgb_255'],
    'material': MATERIAL, 'size': ctx.args.size,
    'world_location': p['world'][shape],
    'rotation_deg': ctx.rotation,
    'projected_center_px': p['target_px'],
    'center_patch': p['center_patch'],
    'mask': 'masks/%s_pos%d.png' % (shape, pos),
    'mask_area_px': m['area_px'],
    'mask_bbox_px': m['bbox_px'],
    'mask_centroid_px': m['centroid_px'],
    'margin_px': m['margin_px'],
    'patch_coverage': m['coverage'],
  }


def render_settings(ctx):
  scene = ctx.scene
  return {
    'blender_version': bpy.app.version_string,
    'engine': 'CYCLES',
    'resolution': [single.IMAGE_SIZE, single.IMAGE_SIZE],
    'samples': scene.cycles.samples,
    'adaptive_sampling': scene.cycles.use_adaptive_sampling,
    'seed': scene.cycles.seed,
    'denoising': scene.cycles.use_denoising,
    'view_transform': scene.view_settings.view_transform,
    'cast_shadows': bool(ctx.args.cast_shadows),
    'shapes': list(SHAPES),
    'colors': list(COLORS),
    'material': MATERIAL,
    'size': ctx.args.size,
    'rotation_deg': ctx.rotation,
    'camera': ctx.single['camera'],
    'lights': ctx.single['lights'],
    'single_placements_created': ctx.single['created'],
  }


def render_dataset(args, layout):
  ctx = Context(args)
  out_dir = os.path.abspath(args.output_dir)
  single.check_existing_render(out_dir, args)
  for sub in ['images', 'masks']:
    os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
  tmp_path = os.path.join(out_dir, '_tmp_mask.png')
  positions = layout['positions']

  # Masks of each shape at each position, rendered with the object alone.
  # The objects of a scene never overlap, so these are also the visible
  # masks in every scene; this is checked for each rendered scene below.
  say('Rendering masks of each shape at each position')
  masks_hi = ctx.shape_masks([(p['position_id'], p['world'])
                              for p in positions], tmp_path)
  measured = {}
  for (shape, pos), mask in masks_hi.items():
    m = single.mask_measurements(mask)
    m['coverage'] = single.coverage_stats(
        single.patch_coverage(mask), ctx.single['calibration']['fragment_threshold'])
    measured[(shape, pos)] = m
    single.write_gray_png(os.path.join(out_dir, 'masks', '%s_pos%d.png'
                                       % (shape, pos)),
                          single.pixel_coverage(mask))

  scenes, excluded_scenes, assignments, excluded_assignments = \
      enumerate_dataset(layout)
  if args.scene_ids is not None:
    wanted = set(args.scene_ids)
    scenes = [s for s in scenes if s['scene_id'] in wanted]
    ids = {s['scene_id'] for s in scenes}
    assignments = [a for a in assignments if a['scene_id'] in ids]

  scene_records = []
  progress = Progress(len(scenes), 'Rendering scenes')
  for done, sc in enumerate(scenes):
    objs = [ctx.add_object(shape, positions[pos]['world'][shape], color)
            for (pos, shape, color) in sc['objects']]

    # Both objects must be fully visible: their visible masks in the scene
    # must equal the masks rendered alone
    visible, other = render_object_masks(ctx, objs, tmp_path)
    mismatch = [int((v != masks_hi[(shape, pos)]).sum())
                for v, (pos, shape, _) in zip(visible, sc['objects'])]

    image_path = os.path.join(out_dir, sc['image'])
    if not (args.skip_existing and os.path.exists(image_path)):
      ctx.scene.render.filepath = image_path
      bpy.ops.render.render(write_still=True)
    for obj in objs:
      remove_object(obj)

    (p0, s0, _), (p1, s1, _) = sc['objects']
    pair = next(p for p in layout['pair_checks']
                if p['positions'] == [p0, p1] and p['shapes'] == [s0, s1])
    record = {
      'scene_id': sc['scene_id'],
      'image': sc['image'],
      'objects': [dict(object_record(layout, ctx, measured, *o), slot=k)
                  for k, o in enumerate(sc['objects'])],
      'assignment_ids': [],
      'silhouette_gap_px': pair['silhouette_gap_px'],
      'world_gap': pair['world_gap'],
      'shared_patches': pair['shared_patches'],
      'visible_mask_mismatch_subpixels': mismatch,
      'unexplained_mask_subpixels': other,
    }
    scene_records.append(record)
    progress.update(done + 1, os.path.basename(sc['image']))
  progress.close()

  # Assignment records with questions
  scene_by_id = {r['scene_id']: r for r in scene_records}
  assignment_records, questions = [], []
  for a in assignments:
    sc = scene_by_id[a['scene_id']]
    sc['assignment_ids'].append(a['assignment_id'])
    objs = {'A': a['A'], 'B': a['B']}
    record = {
      'assignment_id': a['assignment_id'],
      'scene_id': a['scene_id'],
      'image': sc['image'],
      'A': dict(object_record(layout, ctx, measured, *a['A']),
                slot=a['A_slot']),
      'B': dict(object_record(layout, ctx, measured, *a['B']),
                slot=1 - a['A_slot']),
      'questions': [],
    }
    for k, q in enumerate(make_questions(objs)):
      q = dict(q, question_id='a%04d_q%d' % (a['assignment_id'], k),
               assignment_id=a['assignment_id'], scene_id=a['scene_id'],
               image=sc['image'])
      record['questions'].append(q)
      questions.append(q)
    assignment_records.append(record)

  # Per-object patch coverage, in slot order, for every rendered scene
  keys = sorted(masks_hi, key=lambda k: (SHAPES.index(k[0]), k[1]))
  patch_cov = {k: single.patch_coverage(masks_hi[k]).astype(np.float32)
               for k in keys}
  np.savez_compressed(
      os.path.join(out_dir, 'coverage.npz'),
      scene_ids=np.array([r['scene_id'] for r in scene_records]),
      scene_patch_coverage=np.stack(
          [np.stack([patch_cov[(o['shape'], o['position_id'])]
                     for o in r['objects']]) for r in scene_records])
          if scene_records else np.zeros((0, 2, single.NUM_PATCHES, single.NUM_PATCHES)),
      mask_shapes=np.array([k[0] for k in keys]),
      mask_position_ids=np.array([k[1] for k in keys]),
      mask_pixel_coverage=np.stack([single.pixel_coverage(masks_hi[k])
                                    for k in keys]).astype(np.float32),
      mask_patch_coverage=np.stack([patch_cov[k] for k in keys]),
  )

  write_json(os.path.join(out_dir, 'scenes.json'), scene_records)
  write_json(os.path.join(out_dir, 'assignments.json'), assignment_records)
  write_json(os.path.join(out_dir, 'questions.json'), questions)
  groupings = build_groupings(assignments)
  write_json(os.path.join(out_dir, 'groupings.json'), groupings)
  write_json(os.path.join(out_dir, 'excluded.json'), {
    'excluded_layouts': layout['excluded_layouts'],
    'excluded_scenes': excluded_scenes,
    'excluded_assignments': excluded_assignments,
  })
  write_json(os.path.join(out_dir, 'render_settings.json'), render_settings(ctx))
  write_json(os.path.join(out_dir, 'layout.json'), layout)
  write_assignments_csv(os.path.join(out_dir, 'assignments.csv'),
                        assignment_records)
  say('Rendered %d scenes (%d assignments, %d questions) to %s'
      % (len(scene_records), len(assignment_records), len(questions), out_dir))


def write_assignments_csv(path, records):
  columns = ['assignment_id', 'scene_id', 'image']
  object_fields = ['slot', 'position_id', 'shape', 'color', 'mask',
                   'projected_center_px', 'mask_bbox_px']
  with open(path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(columns + ['%s_%s' % (o, k) for o in OBJECT_IDS
                               for k in object_fields] + ['questions'])
    for r in records:
      row = [r[c] for c in columns]
      for o in OBJECT_IDS:
        for k in object_fields:
          v = r[o][k]
          row.append(json.dumps(v) if isinstance(v, list) else v)
      row.append(json.dumps([[q['question'], q['queried_object'], q['answer']]
                             for q in r['questions']]))
      writer.writerow(row)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify(args, layout):
  """
  Check the rendered dataset and measure how much the controlled comparisons
  change pixels outside the intended object. Writes verification.json.
  """
  out_dir = os.path.abspath(args.output_dir)
  scenes = load_json(os.path.join(out_dir, 'scenes.json'))
  assignments = load_json(os.path.join(out_dir, 'assignments.json'))
  questions = load_json(os.path.join(out_dir, 'questions.json'))
  groupings = load_json(os.path.join(out_dir, 'groupings.json'))
  excluded = load_json(os.path.join(out_dir, 'excluded.json'))
  coverage = np.load(os.path.join(out_dir, 'coverage.npz'))
  single_pl = load_json(args.single_placements_json)
  min_margin = single_pl['calibration']['min_margin_px']
  checks = []

  def check(name, ok, detail=''):
    checks.append({'check': name, 'ok': bool(ok), 'detail': detail})
    say('  [%s] %s %s' % ('ok' if ok else 'FAIL', name, detail))

  say('Verifying %s' % out_dir)
  n_pos = len(layout['positions'])
  full_assignments = n_pos * (n_pos - 1) * len(SHAPES) ** 2 * len(COLORS) ** 2
  n_excl_a = len(excluded['excluded_assignments'])
  n_excl_s = len(excluded['excluded_scenes'])
  check('assignments accounted for',
        len(assignments) + n_excl_a == full_assignments,
        '%d valid + %d excluded = %d of %d' % (len(assignments), n_excl_a,
        len(assignments) + n_excl_a, full_assignments))
  check('scenes accounted for',
        len(scenes) + n_excl_s == full_assignments // 2,
        '%d unique + %d excluded = %d of %d' % (len(scenes), n_excl_s,
        len(scenes) + n_excl_s, full_assignments // 2))
  per_scene = [len(s['assignment_ids']) for s in scenes]
  check('each unique scene is rendered once for its two assignments',
        all(n == 2 for n in per_scene) and
        len({s['image'] for s in scenes}) == len(scenes),
        '%d assignments share %d images (%d duplicate renders avoided)'
        % (len(assignments), len(scenes), len(assignments) - len(scenes)))

  # Images, visibility and masks
  images = {}
  bad_size = []
  for s in scenes:
    img = single.read_png(os.path.join(out_dir, s['image']))
    if img.shape[:2] != (single.IMAGE_SIZE, single.IMAGE_SIZE):
      bad_size.append(s['image'])
    images[s['image']] = np.round(img[..., :3] * 255).astype(np.uint8)
  check('images are %dx%d' % (single.IMAGE_SIZE, single.IMAGE_SIZE),
        not bad_size, '%d wrong' % len(bad_size))
  hidden = [s['scene_id'] for s in scenes
            if any(s['visible_mask_mismatch_subpixels']) or
            s['unexplained_mask_subpixels']]
  check('both objects fully visible, no other objects', not hidden,
        '%d scenes with occlusion or extra pixels' % len(hidden))
  margins = [o['margin_px'] for s in scenes for o in s['objects']]
  check('objects keep the border margin', min(margins) >= min_margin,
        'min margin %.2f px (required %.1f)' % (min(margins), min_margin))
  gaps = [s['silhouette_gap_px'] for s in scenes]
  check('silhouettes are separated', min(gaps) >= args.min_gap_px,
        'min gap %.2f px' % min(gaps))
  shared = sum(s['shared_patches'] > 0 for s in scenes)
  check('no patch contains both objects', shared == 0,
        '%d scenes with shared patches' % shared)
  # Stored masks must be the per-pixel coverage that the metadata refers to
  mask_index = {(str(s), int(p)): k for k, (s, p) in enumerate(
      zip(coverage['mask_shapes'], coverage['mask_position_ids']))}
  worst = 0.0
  for (shape, pos), k in mask_index.items():
    png = single.read_png(os.path.join(out_dir, 'masks', '%s_pos%d.png'
                                       % (shape, pos)))[..., 0]
    worst = max(worst, float(np.abs(png - coverage['mask_pixel_coverage'][k]).max()))
  check('mask PNGs match coverage arrays', worst <= 1.0 / 255 + 1e-6,
        'max difference %.4f' % worst)

  # Groupings change only the intended attribute
  by_id = {a['assignment_id']: a for a in assignments}

  def attrs(a):
    return {'%s.%s' % (o, f): a[o]['position_id' if f == 'position' else f]
            for o in OBJECT_IDS for f in FACTORS}

  bad_groups = 0
  for g in groupings:
    members = [attrs(by_id[m['assignment_id']]) for m in g['members']]
    for m in members:
      if any(m[k] != v for k, v in g['fixed'].items()):
        bad_groups += 1
        break
    if g['comparison'].startswith('change_'):
      values = [m[g['varying']] for m in members]
      if len(set(values)) != len(values):
        bad_groups += 1
  counts = defaultdict(int)
  for g in groupings:
    counts[g['comparison']] += 1
  check('matched groups change only the intended attribute', bad_groups == 0,
        '%d groups: %s' % (len(groupings), dict(counts)))

  # Questions: unique references and correct answers
  bad_q = 0
  for q in questions:
    a = by_id[q['assignment_id']]
    key = 'shape' if q['type'] == 'color' else 'color'
    matches = [o for o in OBJECT_IDS if a[o][key] == q['reference'][key]]
    target = a[q['queried_object']]
    if matches != [q['queried_object']] or target[q['type']] != q['answer']:
      bad_q += 1
  expected = sum(len(make_questions({o: (a[o]['position_id'], a[o]['shape'],
                                         a[o]['color']) for o in OBJECT_IDS}))
                 for a in assignments)
  check('question references are unique and answers match', bad_q == 0 and
        expected == len(questions), '%d questions (%d per image on average), '
        '%d wrong' % (len(questions), len(questions) / max(len(assignments), 1),
                      bad_q))

  # Pixel-level effect of controlled comparisons on the unchanged object
  pixel_cov = {k: coverage['mask_pixel_coverage'][i]
               for k, i in mask_index.items()}
  diffs = measure_comparisons(groupings, by_id, images, pixel_cov)
  for name, d in diffs.items():
    say('  %s: unchanged object mean |diff| %.2f (p95 %.2f, max pixel %d), '
        'background mean |diff| %.2f (max pixel %d), %d image pairs'
        % (name, d['unchanged_object_mean_abs_diff']['mean'],
           d['unchanged_object_mean_abs_diff']['p95'],
           d['unchanged_object_max_abs_diff']['max'],
           d['background_mean_abs_diff']['mean'],
           d['background_max_abs_diff']['max'], d['image_pairs']))

  report = {
    'created': datetime.now().isoformat(timespec='seconds'),
    'counts': {
      'assignments_enumerated': full_assignments,
      'assignments_valid': len(assignments),
      'assignments_excluded': n_excl_a,
      'unique_scenes': len(scenes),
      'scenes_excluded': n_excl_s,
      'duplicate_renders_avoided': len(assignments) - len(scenes),
      'questions_assignment_level': len(questions),
      'questions_image_level': len({(q['scene_id'], q['question'])
                                    for q in questions}),
      'groups': dict(counts),
    },
    'checks': checks,
    'controlled_comparison_pixel_changes': diffs,
    'notes': 'Pixel differences are in 0-255 units. "unchanged object" is '
             'the interior of the object that is not changed by the '
             'comparison; changes there come from shadows and reflections '
             'of the changed object and from rendering noise.',
  }
  write_json(os.path.join(out_dir, 'verification.json'), report)
  failed = [c for c in checks if not c['ok']]
  say('Verification: %d checks, %d failed' % (len(checks), len(failed)))
  if failed:
    raise RuntimeError('Verification failed: %s'
                       % ', '.join(c['check'] for c in failed))
  return report


def background_mask(objects):
  """ Pixels further than BACKGROUND_DISTANCE_PX from all objects' boxes """
  near = np.zeros((single.IMAGE_SIZE, single.IMAGE_SIZE), dtype=bool)
  d = BACKGROUND_DISTANCE_PX
  for o in objects:
    x0, y0, x1, y1 = o['mask_bbox_px']
    near[max(int(y0) - d, 0):int(math.ceil(y1)) + d,
         max(int(x0) - d, 0):int(math.ceil(x1)) + d] = True
  return ~near


def measure_comparisons(groupings, by_id, images, pixel_cov):
  """
  For every pair of images in a change_* group, measure the change inside the
  unchanged object (pixels fully covered by it) and in the background far
  from both objects.
  """
  stats = defaultdict(lambda: defaultdict(list))
  seen = set()
  for g in groupings:
    if not g['comparison'].startswith('change_'):
      continue
    other = 'B' if g['changed_object'] == 'A' else 'A'
    members = [by_id[m['assignment_id']] for m in g['members']]
    for a1, a2 in itertools.combinations(members, 2):
      o = a1[other]
      key = (g['comparison'],) + tuple(sorted([a1['image'], a2['image']]))
      if key in seen:
        continue
      seen.add(key)
      img1 = images[a1['image']].astype(np.int16)
      img2 = images[a2['image']].astype(np.int16)
      diff = np.abs(img1 - img2).max(axis=-1)
      inside = pixel_cov[(o['shape'], o['position_id'])] >= 0.999
      background = (background_mask([a1['A'], a1['B']]) &
                    background_mask([a2['A'], a2['B']]))
      s = stats[g['comparison']]
      s['unchanged_object_mean_abs_diff'].append(float(diff[inside].mean()))
      s['unchanged_object_max_abs_diff'].append(int(diff[inside].max()))
      s['background_mean_abs_diff'].append(float(diff[background].mean()))
      s['background_max_abs_diff'].append(int(diff[background].max()))
  summary = {}
  for name, s in stats.items():
    summary[name] = {'image_pairs': len(s['unchanged_object_mean_abs_diff'])}
    for k, v in s.items():
      v = np.array(v)
      summary[name][k] = {'mean': float(v.mean()),
                          'p95': float(np.percentile(v, 95)),
                          'max': float(v.max())}
  return summary


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------

def overlay(image, masks):
  """
  Image (H, W, 3 in [0, 1]) with the 14 px patch grid and the outlines and a
  light tint of the object masks (yellow: slot 0, magenta: slot 1).
  """
  out = image.copy()
  out[::single.PATCH_SIZE, :] = 0.5 * out[::single.PATCH_SIZE, :] + 0.5
  out[:, ::single.PATCH_SIZE] = 0.5 * out[:, ::single.PATCH_SIZE] + 0.5
  for mask, color in zip(masks, [(1.0, 1.0, 0.0), (1.0, 0.0, 1.0)]):
    m = mask > 0.5
    out[m] = 0.8 * out[m] + 0.2 * np.array(color)
    out[outline(m)] = color
  return out


def preview(args):
  """
  Contact sheet: each tile shows a rendered image (left) and the same image
  with patch grid and mask overlays (right). The overlays are separate files
  and never written into the model input images.
  """
  out_dir = os.path.abspath(args.output_dir)
  scenes = load_json(os.path.join(out_dir, 'scenes.json'))
  n = min(args.preview_tiles, len(scenes))
  chosen = [scenes[int(i)] for i in np.linspace(0, len(scenes) - 1, n)]
  tiles = []
  for s in chosen:
    img = single.read_png(os.path.join(out_dir, s['image']))[..., :3]
    masks = [single.read_png(os.path.join(out_dir, o['mask']))[..., 0]
             for o in s['objects']]
    tiles.append(np.concatenate([img, overlay(img, masks)], axis=1))
  cols = 2
  while len(tiles) % cols:
    tiles.append(np.ones_like(tiles[0]))
  rows = [np.concatenate(tiles[i:i + cols], axis=1)
          for i in range(0, len(tiles), cols)]
  sheet = np.concatenate(rows, axis=0)
  preview_dir = os.path.join(out_dir, 'preview')
  os.makedirs(preview_dir, exist_ok=True)
  write_rgb_png(os.path.join(preview_dir, 'contact_sheet.png'), sheet)
  for s in chosen:
    img = single.read_png(os.path.join(out_dir, s['image']))[..., :3]
    masks = [single.read_png(os.path.join(out_dir, o['mask']))[..., 0]
             for o in s['objects']]
    write_rgb_png(os.path.join(preview_dir, 'overlay_' +
                               os.path.basename(s['image'])),
                  overlay(img, masks))
  write_json(os.path.join(preview_dir, 'contact_sheet.json'), {
    'layout': 'tiles in reading order, %d per row; each tile is the image '
              'and its overlay (yellow: slot 0, magenta: slot 1)' % cols,
    'scenes': [{'scene_id': s['scene_id'], 'image': s['image']}
               for s in chosen],
  })
  say('Wrote contact sheet with %d scenes to %s' % (n, preview_dir))


# ---------------------------------------------------------------------------

def main(args):
  global SHAPES, COLORS
  stages = set(args.stage)
  if 'all' in stages:
    stages = {'layout', 'render', 'verify', 'preview'}
  # Image and patch geometry of the single-object placements (also needed by
  # the verify and preview stages, which do not load the scene)
  single_placements = load_json(args.single_placements_json)
  single.apply_geometry_from_placements(single_placements)

  layout = None
  if os.path.exists(args.layout_json):
    layout = load_json(args.layout_json)
  if 'layout' in stages:
    if layout is None:
      reason = 'no layout file yet'
    elif args.relayout:
      reason = '--relayout 1'
    else:
      changed = layout_changes(layout, args, single_placements)
      reason = 'changed settings: %s' % ', '.join(changed) if changed else None
    if reason:
      say('Computing layout (%s)' % reason)
      SHAPES = list(args.shapes)
      layout = compute_layout(args)
    else:
      say('Layout is up to date: %s' % args.layout_json)
  if stages & {'render', 'verify', 'preview'}:
    if layout is None:
      raise RuntimeError('No layout file at %s; run the layout stage first'
                         % args.layout_json)
    changed = layout_changes(layout, args, single_placements)
    if changed:
      raise RuntimeError('The layout was computed with different settings '
                         '(%s); run the layout stage first' % ', '.join(changed))

  # The layout fixes the shapes; colors are a rendering setting, and verify
  # and preview without render use the colors that were rendered
  SHAPES = list(layout['layout_settings']['shapes']
                if layout and 'layout_settings' in layout else SHAPES)
  COLORS = list(args.colors)
  settings_file = os.path.join(os.path.abspath(args.output_dir),
                               'render_settings.json')
  if 'render' not in stages and os.path.exists(settings_file):
    COLORS = load_json(settings_file).get('colors', COLORS)
  if 'render' in stages:
    render_dataset(args, layout)
  if 'verify' in stages:
    verify(args, layout)
  if 'preview' in stages:
    preview(args)


def layout_changes(layout, args, single_placements):
  """
  Layout settings that differ from those stored with the layout, plus the
  single-object placements file if the layout was made from another one.
  """
  stored = layout.get('layout_settings')
  if stored is None:
    say('WARNING: the layout file does not record its layout settings')
    return []
  changed = [k for k in LAYOUT_SETTINGS
             if single.comparable(stored.get(k)) !=
             single.comparable(getattr(args, k))]
  if layout.get('single_placements_created') != single_placements['created']:
    changed.append('single_placements_json')
  return changed


if __name__ == '__main__':
  main(single.parse_args(parser, utils.extract_args(), PATH_SETTINGS))
