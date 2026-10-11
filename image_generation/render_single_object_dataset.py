"""
Renders a controlled single-object CLEVR dataset for VLM (LLaVA-1.5) analysis.

Every image contains exactly one object. The dataset is the full combination of
36 positions x 3 shapes x 8 colors = 864 images, rendered directly at 336x336
(the LLaVA-1.5 input size; 14x14-pixel patches give a 24x24 patch grid).

Everything except shape, color and position is fixed: camera, lights,
background, material (CLEVR "rubber", i.e. matte), size (CLEVR "small" by
default) and object orientation. For a given shape and position the object and its material
are created once and only the material's Color input is changed between
colors, so the color images of one shape-position differ in nothing else.

Positions are chosen in projected image coordinates:
- The 36 projected object centers (projection of each object's 3D center) form
  a 6x6 grid with a spacing of exactly 3 patches (42 px), so every object has
  the same offset relative to the patch boundaries.
- That shared sub-patch offset and the projected object size (camera zoom) are
  chosen by measuring, on supersampled object masks, how many patches contain
  only a tiny fragment of the object, summed over all three shapes.
- The world placement for each projected center is found by casting a ray
  through that pixel and intersecting it with the plane at the height of the
  object's center.
- The camera keeps the CLEVR viewing direction and look-at point but is moved
  far back with a long lens, so that moving an object across the grid changes
  its projected size only by a few percent. With CLEVR "small" objects (and a
  correspondingly longer lens) all placements stay within the region CLEVR
  places objects in, where the ground is flat and the lighting was designed
  for; "large" objects would push the grid onto the curved backdrop and
  behind the key light.

The calibrated placements are saved to a JSON file and reused for rendering,
so the dataset can be regenerated exactly.

Settings can be given on the command line or in a settings file (--config,
e.g. output/CLEVR_SINGLE_OBJECT/settings.json), which has a "layout"
section (anything that changes the placements: image and patch size, grid,
object size, shapes, camera, margins), a "render" section (colors, shadows,
quality, subsets) and a "paths" section. Command-line arguments override the
file. Usage (from the image_generation directory):

blender --background --python render_single_object_dataset.py -- [args]

Stages: "layout" calibrates only if the placements file is missing or was
made with other layout settings (or --recalibrate 1 is given); "calibrate"
always calibrates; "render" renders from the placements file and stops if it
was made with other layout settings; "all" (default) is layout + render.
"""

import argparse, csv, json, math, os, sys
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import bpy, bpy_extras
import numpy as np
import time
from mathutils import Vector, Quaternion
import utils
from render_images import enable_gpu


# Print progress immediately, also when the output is piped (e.g. to a
# notebook)
sys.stdout.reconfigure(line_buffering=True)

ALL_SHAPES = ['sphere', 'cube', 'cylinder']
ALL_COLORS = ['gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow']
# Shapes of the current dataset; set from the settings / placements in main()
SHAPES = list(ALL_SHAPES)
COLORS = ALL_COLORS
MATERIAL = 'rubber'

# Image, patch and grid geometry. These are the defaults (LLaVA-1.5); they are
# replaced by apply_geometry() from the settings or the placements file.
IMAGE_SIZE = 336
PATCH_SIZE = 14
NUM_PATCHES = IMAGE_SIZE // PATCH_SIZE
GRID_N = 6
GRID_SPACING_PATCHES = 3
# Index of the patch containing the first grid center; with 3-patch spacing
# this centers the grid in the image (63..273 px for an offset of 7 px)
GRID_FIRST_PATCH = 4
# Masks are rendered at this many samples per pixel along each axis
MASK_SUPERSAMPLING = 4

# Settings that determine the placements. They are stored in the placements
# file, and rendering refuses to use placements made with other values.
LAYOUT_SETTINGS = ['image_size', 'patch_size', 'grid_size',
                   'grid_spacing_patches', 'size', 'shapes', 'camera_distance',
                   'look_at', 'sphere_diameters_px', 'fragment_threshold',
                   'min_margin_px', 'max_ground_tilt_px']
# Settings that are file paths; relative paths in a settings file are
# relative to that file
PATH_SETTINGS = ['placements_json', 'output_dir', 'base_scene_blendfile',
                 'properties_json', 'shape_dir', 'material_dir']


def apply_geometry(image_size, patch_size, grid_n, spacing_patches,
                   first_patch=None):
  """
  Set the image, patch and grid constants used throughout this module. The
  grid is centered in the image unless first_patch is given.
  """
  global IMAGE_SIZE, PATCH_SIZE, NUM_PATCHES, GRID_N, GRID_SPACING_PATCHES
  global GRID_FIRST_PATCH
  if image_size % patch_size:
    raise ValueError('image_size must be a multiple of patch_size')
  num_patches = image_size // patch_size
  span = spacing_patches * (grid_n - 1)
  if span >= num_patches:
    raise ValueError('a %dx%d grid with a spacing of %d patches does not fit '
                     'into %d patches' % (grid_n, grid_n, spacing_patches,
                                          num_patches))
  IMAGE_SIZE, PATCH_SIZE, NUM_PATCHES = image_size, patch_size, num_patches
  GRID_N, GRID_SPACING_PATCHES = grid_n, spacing_patches
  GRID_FIRST_PATCH = ((num_patches - span) // 2 if first_patch is None
                      else first_patch)


def apply_geometry_from_placements(placements):
  """ Use the geometry and shapes that a placements file was made with """
  global SHAPES
  apply_geometry(placements['image']['width'], placements['image']['patch_size'],
                 placements['grid']['rows'], placements['grid']['spacing_patches'],
                 placements['grid']['first_patch_index'])
  SHAPES = list(placements['object']['scale'])


def read_config(path, parser, path_settings):
  """
  Read a settings file into a dict of argument defaults. Only the "layout",
  "render" and "paths" sections are used; other top-level keys (name,
  description, ...) are for documentation.
  """
  with open(path, 'r') as f:
    config = json.load(f)
  base = os.path.dirname(os.path.abspath(path))
  known = {a.dest for a in parser._actions}
  values = {}
  for section in ('layout', 'render', 'paths'):
    for key, value in config.get(section, {}).items():
      if key not in known:
        raise ValueError('Unknown setting "%s" in %s' % (key, path))
      if (key in path_settings and isinstance(value, str)
          and not os.path.isabs(value)):
        value = os.path.normpath(os.path.join(base, value))
      values[key] = value
  return values


def parse_args(parser, argv, path_settings):
  """ Parse arguments, taking defaults from --config if it is given """
  pre = argparse.ArgumentParser(add_help=False)
  pre.add_argument('--config')
  known, _ = pre.parse_known_args(argv)
  if known.config:
    parser.set_defaults(**read_config(known.config, parser, path_settings))
  return parser.parse_args(argv)


def comparable(value):
  """ JSON-normalized value, so that settings read back from JSON compare """
  return json.loads(json.dumps(to_jsonable(value)))


class Progress(object):
  """
  Minimal progress bar, printed as one full line per step so that it stays
  readable between Blender's own log lines.
  """
  def __init__(self, total, label):
    self.total = max(total, 1)
    self.label = label
    self.start = time.time()

  def update(self, done, detail=''):
    elapsed = time.time() - self.start
    eta = elapsed / done * (self.total - done) if done else 0
    width = 30
    filled = int(width * done / self.total)
    say('%s [%s%s] %d/%d %5.1f%% | %s elapsed | ETA %s %s' % (
        self.label, '#' * filled, '-' * (width - filled), done, self.total,
        100.0 * done / self.total, format_seconds(elapsed),
        format_seconds(eta), detail))

  def close(self):
    pass


def format_seconds(s):
  s = int(s)
  return '%d:%02d:%02d' % (s // 3600, s % 3600 // 60, s % 60)


def say(msg):
  """ Print a status line immediately, in order with Blender's own output """
  print(msg, flush=True)


def path_arg(p):
  return os.path.join(SCRIPT_DIR, p)


parser = argparse.ArgumentParser()
parser.add_argument('--config', default=None,
    help="Settings file (JSON) with layout, render and paths sections.")
parser.add_argument('--stage', default='all',
    choices=['all', 'layout', 'calibrate', 'render'],
    help="'layout' calibrates if needed, 'calibrate' always calibrates, " +
         "'render' renders the dataset from the placements file, 'all' is " +
         "layout + render.")
parser.add_argument('--placements_json',
    default=path_arg('../output/CLEVR_SINGLE_OBJECT/placements.json'),
    help="Where the fixed camera and object placements are stored.")
parser.add_argument('--recalibrate', default=0, type=int,
    help="With --stage layout or all, recompute the placements even if they " +
         "are up to date.")
parser.add_argument('--output_dir',
    default=path_arg('../output/CLEVR_SINGLE_OBJECT/generation'),
    help="Directory for the rendered images, masks and metadata.")

# Input assets
parser.add_argument('--base_scene_blendfile',
    default=path_arg('data/base_scene.blend'))
parser.add_argument('--properties_json',
    default=path_arg('data/properties.json'))
parser.add_argument('--shape_dir', default=path_arg('data/shapes'))
parser.add_argument('--material_dir', default=path_arg('data/materials'))

# Layout options (changing any of them requires a new calibration)
parser.add_argument('--image_size', default=336, type=int,
    help="Width and height of the rendered images in pixels.")
parser.add_argument('--patch_size', default=14, type=int,
    help="Patch size of the vision encoder in pixels.")
parser.add_argument('--grid_size', default=6, type=int,
    help="Number of grid rows and columns.")
parser.add_argument('--grid_spacing_patches', default=3, type=int,
    help="Distance between neighboring grid positions, in patches.")
parser.add_argument('--camera_distance', default=200.0, type=float,
    help="Distance of the camera from the look-at point. The CLEVR camera " +
         "is 11.3 units away; a larger distance with a correspondingly " +
         "longer lens reduces the change in projected size across the grid.")
parser.add_argument('--size', default='small',
    help="CLEVR size used for all objects (a key of the sizes field of the " +
         "properties file). The camera zoom is adapted, so this mostly " +
         "controls how much of the scene the grid covers.")
parser.add_argument('--look_at', default=[0.0, 0.0], type=float, nargs=2,
    help="Point on the ground plane (x y) that appears at the image center; " +
         "the CLEVR camera looks at the origin.")
parser.add_argument('--max_ground_tilt_px', default=0.5, type=float,
    help="Maximum height difference of the ground across an object's " +
         "footprint, in pixels at the look-at point.")
parser.add_argument('--sphere_diameters_px', type=float, nargs='+',
    default=None,
    help="Candidate projected diameters of the sphere at the look-at point; " +
         "this sets the camera zoom. Default: every whole pixel from 3.55 " +
         "to 4.3 patches (50-60 px for 14 px patches).")
parser.add_argument('--fragment_threshold', default=0.10, type=float,
    help="A patch whose object coverage is above 0 but below this fraction " +
         "counts as a fragment patch. Calibration minimizes the number of " +
         "fragment patches plus half the number of patches with coverage " +
         "between this threshold and 0.25.")
parser.add_argument('--min_margin_px', default=None, type=float,
    help="Minimum distance between any object pixel and the image border " +
         "(default: one patch).")

# Rendering options
parser.add_argument('--use_gpu', default=1, type=int)
parser.add_argument('--render_num_samples', default=512, type=int)
parser.add_argument('--denoise', default=1, type=int)
parser.add_argument('--cast_shadows', default=1, type=int,
    help="Setting --cast_shadows 0 stops the object from casting shadows " +
         "(onto the ground or itself). Indirect light, such as color " +
         "bleeding onto the floor, is unaffected.")
parser.add_argument('--skip_existing', default=1, type=int,
    help="Do not re-render images that already exist in the output folder.")
parser.add_argument('--shapes', nargs='+', default=ALL_SHAPES,
    help="Shapes of the dataset (layout setting); rendering can use a subset.")
parser.add_argument('--colors', nargs='+', default=ALL_COLORS,
    help="Colors to render.")
parser.add_argument('--positions', nargs='+', type=int, default=None,
    help="Only render these position indices (for previews).")


def resolve_layout_settings(args):
  """ Fill in layout defaults that depend on other settings """
  if args.sphere_diameters_px is None:
    p = args.patch_size
    args.sphere_diameters_px = [float(d) for d in
                                range(int(round(3.55 * p)), int(round(4.3 * p)) + 1)]
  if args.min_margin_px is None:
    args.min_margin_px = float(args.patch_size)
  return {k: getattr(args, k) for k in LAYOUT_SETTINGS}


# ---------------------------------------------------------------------------
# Scene setup
# ---------------------------------------------------------------------------

def load_properties(args):
  with open(args.properties_json, 'r') as f:
    properties = json.load(f)
  for kind, names in [('shapes', SHAPES), ('colors', COLORS),
                      ('materials', [MATERIAL]), ('sizes', [args.size])]:
    missing = [n for n in names if n not in properties[kind]]
    if missing:
      raise ValueError('%s missing from %s: %s'
                       % (kind, args.properties_json, missing))
  return properties


def object_scale(properties, shape, size):
  """ Scale of the unit-size shape model, following the CLEVR conventions """
  r = properties['sizes'][size]
  if shape == 'cube':
    r /= math.sqrt(2)
  return r


def load_base_scene(args):
  bpy.ops.wm.open_mainfile(filepath=args.base_scene_blendfile)
  utils.load_materials(args.material_dir)

  scene = bpy.context.scene
  render_args = scene.render
  render_args.engine = 'CYCLES'
  render_args.resolution_x = IMAGE_SIZE
  render_args.resolution_y = IMAGE_SIZE
  render_args.resolution_percentage = 100
  render_args.pixel_aspect_x = 1
  render_args.pixel_aspect_y = 1
  render_args.film_transparent = False
  render_args.dither_intensity = 0
  render_args.image_settings.file_format = 'PNG'
  render_args.image_settings.color_mode = 'RGB'
  render_args.image_settings.color_depth = '8'

  # Match the plain sRGB look of the original CLEVR renders
  scene.display_settings.display_device = 'sRGB'
  scene.view_settings.view_transform = 'Standard'
  scene.view_settings.look = 'None'
  scene.view_settings.exposure = 0
  scene.view_settings.gamma = 1

  # Same Cycles settings as render_images.py, plus a fixed sampling pattern
  cycles = scene.cycles
  cycles.samples = args.render_num_samples
  cycles.use_adaptive_sampling = False
  cycles.seed = 0
  cycles.use_animated_seed = False
  cycles.use_denoising = bool(args.denoise)
  cycles.blur_glossy = 2.0
  cycles.min_transparent_bounces = 8
  cycles.transparent_max_bounces = 8
  if args.use_gpu == 1 and enable_gpu():
    cycles.device = 'GPU'
  return scene


def design_camera(look_at, distance, pixels_per_unit):
  """
  Camera with the viewing direction of the CLEVR base scene camera (which
  tracks the origin from (7.48, -6.51, 5.34) without roll), placed `distance`
  units from the look-at point and zoomed so that one world unit at the
  look-at point spans `pixels_per_unit` pixels.
  """
  base_cam = bpy.data.objects['Camera']
  forward = base_cam.matrix_world.to_quaternion() @ Vector((0, 0, -1))
  forward.normalize()
  rotation = forward.to_track_quat('-Z', 'Y')
  target = Vector((look_at[0], look_at[1], 0.0))
  location = target - forward * distance
  sensor_width = base_cam.data.sensor_width
  lens = pixels_per_unit * distance * sensor_width / IMAGE_SIZE
  return {
    'location': list(location),
    'rotation_quaternion': list(rotation),
    'rotation_euler_xyz_deg': [math.degrees(a) for a in rotation.to_euler('XYZ')],
    'lens_mm': lens,
    'sensor_width_mm': sensor_width,
    'sensor_fit': 'AUTO',
    'clip_start': 1.0,
    'clip_end': 10.0 * distance,
    'look_at': list(target),
    'distance': distance,
    'pixels_per_unit_at_look_at': pixels_per_unit,
    'elevation_deg': math.degrees(math.asin(-forward.z)),
    'azimuth_deg': math.degrees(math.atan2(forward.y, forward.x)),
  }


def configure_camera(cam_params):
  camera = bpy.data.objects['Camera']
  # The base scene camera tracks the Empty at the origin (as do the lights,
  # which are left unchanged); use the stored rotation instead
  camera.constraints.clear()
  camera.location = cam_params['location']
  camera.rotation_mode = 'QUATERNION'
  camera.rotation_quaternion = cam_params['rotation_quaternion']
  camera.data.type = 'PERSP'
  camera.data.lens_unit = 'MILLIMETERS'
  camera.data.lens = cam_params['lens_mm']
  camera.data.sensor_width = cam_params['sensor_width_mm']
  camera.data.sensor_fit = cam_params['sensor_fit']
  camera.data.shift_x = 0
  camera.data.shift_y = 0
  camera.data.clip_start = cam_params['clip_start']
  camera.data.clip_end = cam_params['clip_end']
  bpy.context.view_layer.update()
  return camera


def object_rotation_deg(cam_params):
  """
  Rotation shared by all objects: the cube shows one face square-on to the
  camera, so its silhouette edges run along the image (and patch) axes.
  """
  return cam_params['azimuth_deg'] % 90.0


def light_info():
  info = {}
  for obj in bpy.data.objects:
    if obj.type == 'LIGHT':
      info[obj.name] = {
        'type': obj.data.type,
        'energy': obj.data.energy,
        'location': list(obj.location),
        'rotation_euler': list(obj.rotation_euler),
      }
  return info


# ---------------------------------------------------------------------------
# Projection helpers
# ---------------------------------------------------------------------------

def pixel_to_world(scene, camera, u, v, z):
  """
  World point at height z that projects to pixel coordinates (u, v), with
  (0, 0) the top-left image corner and (336, 336) the bottom-right corner.
  """
  # view_frame returns the frame corners as top-right, bottom-right,
  # bottom-left, top-left in camera space
  tr, br, bl, tl = camera.data.view_frame(scene=scene)
  p_local = tl + (tr - tl) * (u / IMAGE_SIZE) + (bl - tl) * (v / IMAGE_SIZE)
  origin = camera.matrix_world.translation
  direction = camera.matrix_world @ p_local - origin
  t = (z - origin.z) / direction.z
  return origin + direction * t


def world_to_pixel(scene, camera, pos):
  x, y, _ = bpy_extras.object_utils.world_to_camera_view(scene, camera, pos)
  return (x * IMAGE_SIZE, (1.0 - y) * IMAGE_SIZE)


def ground_height(scene, x, y):
  """ Height of the (subdivided, gently curved) ground surface at (x, y) """
  depsgraph = bpy.context.evaluated_depsgraph_get()
  hit, loc, _, _, obj, _ = scene.ray_cast(
      depsgraph, Vector((x, y, 50.0)), Vector((0, 0, -1)))
  if not hit or obj.name != 'Ground':
    raise RuntimeError('No ground below (%.2f, %.2f)' % (x, y))
  return loc.z


def ground_tilt(scene, x, y, radius):
  """
  Largest height difference between the ground below the center of an
  object's footprint and the ground below its rim. The ground of the base
  scene is not exactly planar and curves up into a backdrop far from the
  center, so objects must stand where it is locally flat.
  """
  z0 = ground_height(scene, x, y)
  return max(abs(ground_height(scene, x + radius * math.cos(a),
                               y + radius * math.sin(a)) - z0)
             for a in np.linspace(0, 2 * math.pi, 16, endpoint=False))


def solve_world_location(scene, camera, u, v, scale):
  """
  Location of an object's center such that it rests on the ground and
  projects to pixel (u, v).
  """
  z = scale
  for _ in range(10):
    w = pixel_to_world(scene, camera, u, v, z)
    z_new = ground_height(scene, w.x, w.y) + scale
    if abs(z_new - z) < 1e-7:
      break
    z = z_new
  return pixel_to_world(scene, camera, u, v, z)


# ---------------------------------------------------------------------------
# Masks and patch coverage
# ---------------------------------------------------------------------------

def read_png(path):
  """ Read a PNG as a float array of shape (H, W, C), top row first """
  img = bpy.data.images.load(path, check_existing=False)
  w, h = img.size
  c = img.channels
  pixels = np.empty(w * h * c, dtype=np.float32)
  img.pixels.foreach_get(pixels)
  bpy.data.images.remove(img)
  return pixels.reshape(h, w, c)[::-1]


def write_gray_png(path, values):
  """ Write a 2D array of values in [0, 1] (top row first) as a PNG """
  h, w = values.shape
  img = bpy.data.images.new('mask_out', w, h, alpha=False)
  img.colorspace_settings.name = 'Non-Color'
  rgba = np.ones((h, w, 4), dtype=np.float32)
  rgba[..., :3] = values[::-1, :, None]
  img.pixels.foreach_set(rgba.ravel())
  img.filepath_raw = path
  img.file_format = 'PNG'
  img.save()
  bpy.data.images.remove(img)


class MaskRenderSettings(object):
  """
  Context manager that switches the scene to flat, unlit, non-antialiased
  rendering of all objects except the ground at MASK_SUPERSAMPLING times the
  output resolution, so that the alpha channel is a supersampled object mask.
  """
  def __init__(self, scene):
    self.scene = scene

  def __enter__(self):
    scene = self.scene
    r = scene.render
    shading = scene.display.shading
    self.saved = {
      'engine': r.engine, 'percentage': r.resolution_percentage,
      'film_transparent': r.film_transparent, 'filepath': r.filepath,
      'color_mode': r.image_settings.color_mode,
      'render_aa': scene.display.render_aa,
      'shading': (shading.light, shading.color_type, tuple(shading.single_color),
                  shading.show_shadows, shading.show_cavity,
                  shading.show_object_outline, shading.show_specular_highlight),
    }
    r.engine = 'BLENDER_WORKBENCH'
    r.resolution_percentage = 100 * MASK_SUPERSAMPLING
    r.film_transparent = True
    r.image_settings.color_mode = 'RGBA'
    scene.display.render_aa = 'OFF'
    shading.light = 'FLAT'
    shading.color_type = 'SINGLE'
    shading.single_color = (1.0, 1.0, 1.0)
    shading.show_shadows = False
    shading.show_cavity = False
    shading.show_object_outline = False
    shading.show_specular_highlight = False
    bpy.data.objects['Ground'].hide_render = True
    return self

  def __exit__(self, *exc):
    scene = self.scene
    r = scene.render
    shading = scene.display.shading
    s = self.saved
    r.engine = s['engine']
    r.resolution_percentage = s['percentage']
    r.film_transparent = s['film_transparent']
    r.filepath = s['filepath']
    r.image_settings.color_mode = s['color_mode']
    scene.display.render_aa = s['render_aa']
    (shading.light, shading.color_type, shading.single_color,
     shading.show_shadows, shading.show_cavity, shading.show_object_outline,
     shading.show_specular_highlight) = s['shading']
    bpy.data.objects['Ground'].hide_render = False
    return False


def render_mask(scene, tmp_path):
  """
  Render the supersampled binary object mask (must be called inside
  MaskRenderSettings). Returns a boolean array of shape
  (336 * MASK_SUPERSAMPLING, 336 * MASK_SUPERSAMPLING), top row first.
  """
  scene.render.filepath = tmp_path
  bpy.ops.render.render(write_still=True)
  mask = read_png(tmp_path)[..., 3] > 0.5
  os.remove(tmp_path)
  return mask


def pixel_coverage(mask_hi):
  """ Fraction of each output pixel covered by the object """
  s = MASK_SUPERSAMPLING
  return mask_hi.reshape(IMAGE_SIZE, s, IMAGE_SIZE, s).mean(axis=(1, 3))


def patch_coverage(mask_hi):
  """ Fraction of each 14x14 patch covered by the object (24x24 array) """
  p = PATCH_SIZE * MASK_SUPERSAMPLING
  return mask_hi.reshape(NUM_PATCHES, p, NUM_PATCHES, p).mean(axis=(1, 3))


def coverage_cost(cov, threshold):
  """
  Masks of the patches entering the calibration objective: fragment patches
  (0 < coverage < threshold) count fully, small partial patches
  (threshold <= coverage < 0.25) count half.
  """
  frag = (cov > 0) & (cov < threshold)
  small = (cov >= threshold) & (cov < 0.25)
  return frag, small


def coverage_stats(cov, threshold):
  frag, small = coverage_cost(cov, threshold)
  return {
    'patches_touched': int((cov > 0).sum()),
    'patches_fragment': int(frag.sum()),
    'patches_below_0.05': int(((cov > 0) & (cov < 0.05)).sum()),
    'patches_below_0.25': int(((cov > 0) & (cov < 0.25)).sum()),
    'patches_at_least_0.5': int((cov >= 0.5).sum()),
    'patches_at_least_0.9': int((cov >= 0.9).sum()),
    'patch_area_equivalent': float(cov.sum()),
    'cost': float(frag.sum() + 0.5 * small.sum()),
  }


def offset_cost_table(mask_hi, center_hi, threshold):
  """
  For a mask whose object's projected center is at hi-res coordinates
  center_hi = (cx, cy), compute the calibration cost for every sub-patch
  offset of that center relative to the patch grid, assuming that moving the
  object by less than a patch only translates its silhouette.

  Returns an array of shape (P, P) indexed by [offset_y, offset_x] in hi-res
  pixels, where P is the patch size in hi-res pixels.
  """
  P = PATCH_SIZE * MASK_SUPERSAMPLING
  H, W = mask_hi.shape
  ys, xs = np.nonzero(mask_hi)
  integral = np.zeros((H + 1, W + 1), dtype=np.int64)
  integral[1:, 1:] = mask_hi.astype(np.int64).cumsum(0).cumsum(1)
  offsets = np.arange(P)

  def boundaries(c, lo, hi, n):
    # Patch boundaries (in mask coordinates) such that the object center lies
    # `offset` hi-res pixels after a boundary
    base = np.round(c - offsets).astype(np.int64)
    kmin = int(math.floor((lo - c) / P)) - 1
    kmax = int(math.ceil((hi + 1 - c) / P)) + 2
    ks = np.arange(kmin, kmax + 1)
    return np.clip(base[:, None] + P * ks[None, :], 0, n)

  by = boundaries(center_hi[1], ys.min(), ys.max(), H)
  bx = boundaries(center_hi[0], xs.min(), xs.max(), W)
  g = integral[by[:, :, None, None], bx[None, None, :, :]]
  counts = g[:, 1:, :, 1:] - g[:, :-1, :, 1:] - g[:, 1:, :, :-1] + g[:, :-1, :, :-1]
  cov = counts / float(P * P)
  frag, small = coverage_cost(cov, threshold)
  return frag.sum(axis=(1, 3)) + 0.5 * small.sum(axis=(1, 3))


def mask_measurements(mask_hi):
  s = float(MASK_SUPERSAMPLING)
  ys, xs = np.nonzero(mask_hi)
  bbox = [xs.min() / s, ys.min() / s, (xs.max() + 1) / s, (ys.max() + 1) / s]
  return {
    'area_px': float(mask_hi.sum() / s ** 2),
    'centroid_px': [float((xs.mean() + 0.5) / s), float((ys.mean() + 0.5) / s)],
    'bbox_px': [float(b) for b in bbox],
    'bbox_size_px': [float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1])],
    'margin_px': float(min(bbox[0], bbox[1],
                           IMAGE_SIZE - bbox[2], IMAGE_SIZE - bbox[3])),
  }


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------

def grid_centers(offset_px):
  """ Projected centers (u or v) of the grid rows/columns for an offset """
  return [PATCH_SIZE * (GRID_FIRST_PATCH + GRID_SPACING_PATCHES * i) + offset_px
          for i in range(GRID_N)]


def grid_positions(offset_x_px, offset_y_px):
  us = grid_centers(offset_x_px)
  vs = grid_centers(offset_y_px)
  positions = []
  for row in range(GRID_N):
    for col in range(GRID_N):
      positions.append({
        'index': row * GRID_N + col, 'row': row, 'col': col,
        'target_px': [us[col], vs[row]],
        'center_patch': [int(vs[row] // PATCH_SIZE), int(us[col] // PATCH_SIZE)],
      })
  return positions


def choose_offset(cost, feasible):
  """
  Pick the feasible offset with the lowest cost; ties are broken by the mean
  cost in a +-0.5 px neighborhood (robustness), then by closeness to a
  centered grid.
  """
  P = cost.shape[0]
  neighborhood = np.zeros_like(cost, dtype=np.float64)
  r = MASK_SUPERSAMPLING // 2
  n = 0
  for dy in range(-r, r + 1):
    for dx in range(-r, r + 1):
      neighborhood += np.roll(np.roll(cost, dy, axis=0), dx, axis=1)
      n += 1
  neighborhood /= n
  center = (PATCH_SIZE // 2) * MASK_SUPERSAMPLING
  oy, ox = np.meshgrid(np.arange(P), np.arange(P), indexing='ij')
  dist = np.abs(oy - center) + np.abs(ox - center)
  best = None
  for y, x in zip(*np.nonzero(feasible)):
    key = (cost[y, x], neighborhood[y, x], dist[y, x])
    if best is None or key < best[0]:
      best = (key, int(y), int(x))
  if best is None:
    raise RuntimeError('No grid offset satisfies the margin constraint')
  return best


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def place_shape(args, properties, shape, location, rotation_deg):
  blend_name = properties['shapes'][shape]
  x, y, z = location
  obj = utils.add_object(args.shape_dir, blend_name,
                         object_scale(properties, shape, args.size), (x, y),
                         theta=rotation_deg)
  obj.location.z = z
  bpy.context.view_layer.update()
  return obj


def render_shape_masks(args, scene, camera, properties, shape, targets,
                       rotation_deg, tmp_path):
  """ Render hi-res masks for one shape at a list of projected centers """
  scale = object_scale(properties, shape, args.size)
  masks = []
  with MaskRenderSettings(scene):
    for (u, v) in targets:
      w = solve_world_location(scene, camera, u, v, scale)
      obj = place_shape(args, properties, shape, w, rotation_deg)
      masks.append(render_mask(scene, tmp_path))
      utils.delete_object(obj)
  return masks


def evaluate_placements(args, scene, camera, properties, rotation, offset_px,
                        tmp_path):
  """
  Render masks of every shape at all grid positions for the given offset and
  compute, for every alternative offset, the summed calibration cost and
  whether all objects would keep the required margin.
  """
  s = MASK_SUPERSAMPLING
  P = PATCH_SIZE * s
  positions = grid_positions(*offset_px)
  targets = [tuple(p['target_px']) for p in positions]
  masks = {}
  total = np.zeros((P, P))
  feasible = np.ones((P, P), dtype=bool)
  o = np.arange(P) / float(s)
  for shape in SHAPES:
    masks[shape] = render_shape_masks(args, scene, camera, properties, shape,
                                      targets, rotation, tmp_path)
    for t, mask in zip(targets, masks[shape]):
      total += offset_cost_table(mask, (t[0] * s, t[1] * s),
                                 args.fragment_threshold)
      # Moving to another offset shifts the mask's bounding box
      x0, y0, x1, y1 = mask_measurements(mask)['bbox_px']
      dx = o - offset_px[0]
      dy = o - offset_px[1]
      feasible &= ((x0 + dx >= args.min_margin_px) &
                   (IMAGE_SIZE - (x1 + dx) >= args.min_margin_px))[None, :]
      feasible &= ((y0 + dy >= args.min_margin_px) &
                   (IMAGE_SIZE - (y1 + dy) >= args.min_margin_px))[:, None]
  return positions, masks, total, feasible


def calibrate(args):
  properties = load_properties(args)
  scene = load_base_scene(args)
  os.makedirs(args.output_dir, exist_ok=True)
  tmp_path = os.path.join(os.path.abspath(args.output_dir), '_tmp_mask.png')
  s = MASK_SUPERSAMPLING
  num_masks = GRID_N * GRID_N * len(SHAPES)
  sphere_world_diameter = 2.0 * object_scale(properties, 'sphere', args.size)
  centered = [PATCH_SIZE // 2, PATCH_SIZE // 2]

  # Stage 1: for each candidate zoom, render all shapes at all positions and
  # find the best shared offset (predicted by translating the masks)
  print('Calibration stage 1: zoom and grid offset')
  candidates = []
  for diameter in args.sphere_diameters_px:
    cam_params = design_camera(args.look_at, args.camera_distance,
                               diameter / sphere_world_diameter)
    camera = configure_camera(cam_params)
    rotation = object_rotation_deg(cam_params)
    _, _, total, feasible = evaluate_placements(
        args, scene, camera, properties, rotation, centered, tmp_path)
    if not feasible.any():
      print('  sphere diameter %.1f px: objects do not fit' % diameter)
      continue
    key, oy, ox = choose_offset(total, feasible)
    print('  sphere diameter %.1f px: cost %.2f per image at offset '
          '(%.2f, %.2f) px' % (diameter, key[0] / num_masks, ox / float(s),
                               oy / float(s)))
    candidates.append({
      'sphere_diameter_px': diameter,
      'cost_per_image': float(key[0]) / num_masks,
      'neighborhood_cost_per_image': float(key[1]) / num_masks,
      'offset_px': [ox / float(s), oy / float(s)],
    })
  if not candidates:
    raise RuntimeError('No candidate zoom fits; try smaller diameters')
  best = min(candidates, key=lambda c: (c['cost_per_image'],
                                        c['neighborhood_cost_per_image'],
                                        -c['sphere_diameter_px']))
  diameter = best['sphere_diameter_px']
  print('Chosen sphere diameter: %.1f px' % diameter)

  pixels_per_unit = diameter / sphere_world_diameter
  cam_params = design_camera(args.look_at, args.camera_distance,
                             pixels_per_unit)
  camera = configure_camera(cam_params)
  rotation = object_rotation_deg(cam_params)

  # Stage 2: render at the chosen offset and re-check the choice with the
  # actual masks until it is stable
  print('Calibration stage 2: verifying offset')
  # The cost table entry for the rendered offset itself is exact (no shift)
  offset_px = best['offset_px']
  evaluated = {}
  for _ in range(4):
    positions, masks, total, feasible = evaluate_placements(
        args, scene, camera, properties, rotation, offset_px, tmp_path)
    oy0, ox0 = int(round(offset_px[1] * s)), int(round(offset_px[0] * s))
    evaluated[tuple(offset_px)] = (total[oy0, ox0], positions, masks)
    key, oy, ox = choose_offset(total, feasible)
    new_offset = [ox / float(s), oy / float(s)]
    print('  offset (%.2f, %.2f) px: cost %.3f per image; predicted best '
          '(%.2f, %.2f) px with %.3f' % (offset_px[0], offset_px[1],
          total[oy0, ox0] / num_masks, new_offset[0], new_offset[1],
          key[0] / num_masks))
    if tuple(new_offset) in evaluated:
      break
    offset_px = new_offset
  offset_px = list(min(evaluated, key=lambda k: evaluated[k][0]))
  _, positions, masks = evaluated[tuple(offset_px)]
  print('Chosen offset (%.2f, %.2f) px' % tuple(offset_px))

  # Final measurements at the chosen placements
  print('Calibration stage 3: measuring final placements')
  for p in positions:
    p['world'] = {}
    p['measured'] = {}
  for shape in SHAPES:
    scale = object_scale(properties, shape, args.size)
    for p, mask in zip(positions, masks[shape]):
      u, v = p['target_px']
      w = solve_world_location(scene, camera, u, v, scale)
      reproj = world_to_pixel(scene, camera, w)
      m = mask_measurements(mask)
      m.update(coverage_stats(patch_coverage(mask), args.fragment_threshold))
      m['reprojection_error_px'] = math.hypot(reproj[0] - u, reproj[1] - v)
      m['ground_height'] = w.z - scale
      m['ground_tilt'] = ground_tilt(scene, w.x, w.y,
                                     properties['sizes'][args.size])
      m['ground_tilt_px'] = m['ground_tilt'] * pixels_per_unit
      if m['ground_tilt_px'] > args.max_ground_tilt_px:
        raise RuntimeError('%s at position %d (x=%.2f, y=%.2f) is not on flat '
                           'ground; move --look_at towards the camera'
                           % (shape, p['index'], w.x, w.y))
      p['world'][shape] = [w.x, w.y, w.z]
      p['measured'][shape] = m

  summary = summarize_positions(positions, args)
  for shape, sm in summary.items():
    print('  %-8s fragments/img mean %.2f max %d | area %.0f-%.0f px '
          '(max/min %.3f) | min margin %.1f px'
          % (shape, sm['fragments_mean'], sm['fragments_max'],
             sm['area_px_min'], sm['area_px_max'], sm['area_ratio_max_min'],
             sm['margin_px_min']))
    if sm['margin_px_min'] < args.min_margin_px:
      raise RuntimeError('Some %s placements violate the margin' % shape)

  placements = {
    'description': 'Fixed camera and object placements for the single-object '
                   'VLM dataset (%d positions x %d shapes). Generated by '
                   'render_single_object_dataset.py.' % (GRID_N * GRID_N,
                                                         len(SHAPES)),
    'layout_settings': resolve_layout_settings(args),
    'created': datetime.now().isoformat(timespec='seconds'),
    'blender_version': bpy.app.version_string,
    'image': {
      'width': IMAGE_SIZE, 'height': IMAGE_SIZE, 'patch_size': PATCH_SIZE,
      'patch_grid': [NUM_PATCHES, NUM_PATCHES],
      'pixel_coordinates': 'x right, y down, (0, 0) = top-left image corner; '
                           'pixel (i, j) covers [i, i+1) x [j, j+1)',
    },
    'grid': {
      'rows': GRID_N, 'cols': GRID_N,
      'spacing_patches': GRID_SPACING_PATCHES,
      'spacing_px': GRID_SPACING_PATCHES * PATCH_SIZE,
      'first_patch_index': GRID_FIRST_PATCH,
      'center_offset_in_patch_px': offset_px,
      'centers_x_px': grid_centers(offset_px[0]),
      'centers_y_px': grid_centers(offset_px[1]),
      'row_order': 'row 0 is the top (farthest) row, col 0 the left column; '
                   'index = row * %d + col' % GRID_N,
      'projected_center': 'projection of the object\'s 3D center (bounding '
                          'box center) through the camera',
    },
    'camera': cam_params,
    'lights': light_info(),
    'object': {
      'size': args.size,
      'scale': {shape: object_scale(properties, shape, args.size)
                for shape in SHAPES},
      'material': MATERIAL,
      'material_node_group': properties['materials'][MATERIAL],
      'material_inputs': material_defaults(properties),
      'rotation_deg': rotation,
      'rotation_note': 'same yaw for every shape and position; the cube shows '
                       'one face square-on to the camera',
    },
    'colors': color_table(properties),
    'positions': positions,
    'calibration': {
      'mask_supersampling': MASK_SUPERSAMPLING,
      'fragment_threshold': args.fragment_threshold,
      'cost': 'per mask: #patches with 0 < coverage < fragment_threshold + '
              '0.5 * #patches with fragment_threshold <= coverage < 0.25',
      'min_margin_px': args.min_margin_px,
      'zoom_candidates': candidates,
      'chosen_sphere_diameter_px': diameter,
      'summary': summary,
    },
  }
  os.makedirs(os.path.dirname(os.path.abspath(args.placements_json)),
              exist_ok=True)
  with open(args.placements_json, 'w') as f:
    json.dump(to_jsonable(placements), f, indent=2)
  # Masks of the final placements, for previews of the layout
  keys = [(shape, p['index']) for shape in SHAPES for p in positions]
  np.savez_compressed(
      os.path.join(os.path.abspath(args.output_dir), 'layout_masks.npz'),
      shapes=np.array([k[0] for k in keys]),
      position_index=np.array([k[1] for k in keys]),
      pixel_coverage=np.stack([pixel_coverage(masks[shape][i])
                               for shape in SHAPES
                               for i in range(len(positions))]).astype(np.float32))
  print('Wrote placements to %s' % args.placements_json)
  return placements


def summarize_positions(positions, args):
  summary = {}
  for shape in SHAPES:
    ms = [p['measured'][shape] for p in positions]
    areas = [m['area_px'] for m in ms]
    widths = [m['bbox_size_px'][0] for m in ms]
    heights = [m['bbox_size_px'][1] for m in ms]
    frags = [m['patches_fragment'] for m in ms]
    summary[shape] = {
      'fragments_mean': float(np.mean(frags)),
      'fragments_max': int(max(frags)),
      'patches_touched_mean': float(np.mean([m['patches_touched'] for m in ms])),
      'patches_below_0.25_mean':
          float(np.mean([m['patches_below_0.25'] for m in ms])),
      'area_px_min': min(areas), 'area_px_max': max(areas),
      'area_ratio_max_min': max(areas) / min(areas),
      'bbox_width_px_range': [min(widths), max(widths)],
      'bbox_height_px_range': [min(heights), max(heights)],
      'margin_px_min': min(m['margin_px'] for m in ms),
      'reprojection_error_px_max': max(m['reprojection_error_px'] for m in ms),
      'ground_tilt_px_max': max(m['ground_tilt_px'] for m in ms),
      'ground_height_range': [min(m['ground_height'] for m in ms),
                              max(m['ground_height'] for m in ms)],
    }
  return summary


def material_defaults(properties):
  group = bpy.data.node_groups[properties['materials'][MATERIAL]]
  values = {}
  for item in group.interface.items_tree:
    if getattr(item, 'in_out', None) != 'INPUT':
      continue
    if item.name in ('Color', 'Normal'):
      continue
    values[item.name] = item.default_value
  return values


def color_table(properties):
  table = {}
  for name in COLORS:
    rgb = properties['colors'][name]
    table[name] = {
      'rgb_255': rgb,
      # CLEVR convention: the 0-255 values divided by 255 are used directly
      # as the (linear) Color input of the material node group
      'material_color_input': [c / 255.0 for c in rgb] + [1.0],
    }
  return table


def to_jsonable(x):
  if isinstance(x, dict):
    return {k: to_jsonable(v) for k, v in x.items()}
  if isinstance(x, (list, tuple)):
    return [to_jsonable(v) for v in x]
  if isinstance(x, (np.floating, float)):
    return float(x)
  if isinstance(x, (np.integer,)):
    return int(x)
  if hasattr(x, '__len__') and not isinstance(x, str):
    return [to_jsonable(v) for v in x]
  return x


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def check_existing_render(out_dir, args):
  """
  With --skip_existing 1, existing images are kept. Stop if they were
  rendered with settings that change how an image looks, so that a dataset
  never mixes images rendered differently.
  """
  path = os.path.join(out_dir, 'render_settings.json')
  if not args.skip_existing or not os.path.exists(path):
    return
  with open(path, 'r') as f:
    previous = json.load(f)
  current = {'samples': args.render_num_samples,
             'denoising': bool(args.denoise),
             'cast_shadows': bool(args.cast_shadows)}
  changed = ['%s: %s -> %s' % (k, previous.get(k), v)
             for k, v in current.items()
             if k in previous and previous[k] != v]
  if changed:
    raise RuntimeError('Existing images in %s were rendered with other '
                       'settings (%s). Use --skip_existing 0 to re-render '
                       'them, or a new output folder.'
                       % (out_dir, ', '.join(changed)))


def set_material_color(obj, rgba):
  for node in obj.data.materials[0].node_tree.nodes:
    if node.bl_idname == 'ShaderNodeGroup':
      node.inputs['Color'].default_value = rgba
      return
  raise RuntimeError('Material group node not found')


def render_dataset(args, placements):
  apply_geometry_from_placements(placements)
  missing = [s for s in args.shapes if s not in SHAPES]
  if missing:
    raise ValueError('Shapes %s are not in the placements file' % missing)
  args.size = placements['object']['size']
  properties = load_properties(args)
  scene = load_base_scene(args)
  camera = configure_camera(placements['camera'])
  rotation = placements['object']['rotation_deg']
  size = args.size

  # Make sure the assets still match what the placements were made with
  colors = color_table(properties)
  for name in COLORS:
    if colors[name]['rgb_255'] != placements['colors'][name]['rgb_255']:
      raise RuntimeError('Color %s differs from the placements file' % name)
  for name, info in light_info().items():
    stored = placements['lights'][name]['location']
    if max(abs(a - b) for a, b in zip(info['location'], stored)) > 1e-4:
      raise RuntimeError('Light %s differs from the placements file' % name)

  out_dir = os.path.abspath(args.output_dir)
  check_existing_render(out_dir, args)
  image_dir = os.path.join(out_dir, 'images')
  mask_dir = os.path.join(out_dir, 'masks')
  os.makedirs(image_dir, exist_ok=True)
  os.makedirs(mask_dir, exist_ok=True)
  tmp_path = os.path.join(out_dir, '_tmp_mask.png')

  positions = placements['positions']
  if args.positions is not None:
    positions = [p for p in positions if p['index'] in args.positions]

  records = []
  pixel_masks = {}
  patch_covs = {}
  progress = Progress(len(args.shapes) * len(positions) * len(args.colors),
                      'Rendering images')
  for shape in args.shapes:
    for p in positions:
      x, y, z = p['world'][shape]
      obj = place_shape(args, properties, shape, (x, y, z), rotation)
      obj.visible_shadow = bool(args.cast_shadows)
      u, v = world_to_pixel(scene, camera, obj.location)
      tu, tv = p['target_px']
      if math.hypot(u - tu, v - tv) > 0.01:
        raise RuntimeError('Object does not project to its target pixel')

      # The mask is the same for all colors of this shape and position
      with MaskRenderSettings(scene):
        mask_hi = render_mask(scene, tmp_path)
      pix_cov = pixel_coverage(mask_hi)
      cov = patch_coverage(mask_hi)
      pixel_masks[(shape, p['index'])] = pix_cov
      patch_covs[(shape, p['index'])] = cov
      mask_name = '%s_pos%02d.png' % (shape, p['index'])
      write_gray_png(os.path.join(mask_dir, mask_name), pix_cov)
      measured = mask_measurements(mask_hi)
      stats = coverage_stats(cov, placements['calibration']['fragment_threshold'])

      utils.add_material(obj, properties['materials'][MATERIAL],
                         Color=colors[args.colors[0]]['material_color_input'])
      for color in args.colors:
        set_material_color(obj, colors[color]['material_color_input'])
        image_name = '%s_%s_pos%02d.png' % (shape, color, p['index'])
        image_path = os.path.join(image_dir, image_name)
        if not (args.skip_existing and os.path.exists(image_path)):
          scene.render.filepath = image_path
          bpy.ops.render.render(write_still=True)
        image = read_png(image_path)
        inside = pix_cov >= 0.999
        mean_rgb = image[..., :3][inside].mean(axis=0) * 255.0
        records.append({
          'image': 'images/' + image_name,
          'mask': 'masks/' + mask_name,
          'shape': shape,
          'color': color,
          'color_rgb_255': colors[color]['rgb_255'],
          'material': MATERIAL,
          'size': size,
          'position_index': p['index'],
          'row': p['row'],
          'col': p['col'],
          'projected_center_px': [tu, tv],
          'center_patch': p['center_patch'],
          'world_location': [x, y, z],
          'rotation_deg': rotation,
          'cast_shadows': bool(args.cast_shadows),
          'mask_area_px': measured['area_px'],
          'mask_centroid_px': measured['centroid_px'],
          'mask_bbox_px': measured['bbox_px'],
          'margin_px': measured['margin_px'],
          'patches_touched': stats['patches_touched'],
          'patches_fragment': stats['patches_fragment'],
          'patches_at_least_0.5': stats['patches_at_least_0.5'],
          'patches_at_least_0.9': stats['patches_at_least_0.9'],
          'object_mean_rgb_rendered': [float(c) for c in mean_rgb],
        })
        progress.update(len(records), image_name)
      utils.delete_object(obj)

  write_outputs(args, placements, records, pixel_masks, patch_covs)


def write_outputs(args, placements, records, pixel_masks, patch_covs):
  out_dir = os.path.abspath(args.output_dir)
  with open(os.path.join(out_dir, 'metadata.json'), 'w') as f:
    json.dump(to_jsonable(records), f, indent=1)
  with open(os.path.join(out_dir, 'metadata.csv'), 'w', newline='') as f:
    writer = csv.writer(f)
    columns = list(records[0].keys())
    writer.writerow(columns)
    for r in records:
      writer.writerow([json.dumps(to_jsonable(r[c])) if isinstance(r[c], list)
                       else r[c] for c in columns])
  keys = sorted(pixel_masks.keys(), key=lambda k: (SHAPES.index(k[0]), k[1]))
  np.savez_compressed(
      os.path.join(out_dir, 'coverage.npz'),
      shapes=np.array([k[0] for k in keys]),
      position_index=np.array([k[1] for k in keys]),
      pixel_coverage=np.stack([pixel_masks[k] for k in keys]).astype(np.float32),
      patch_coverage=np.stack([patch_covs[k] for k in keys]).astype(np.float32),
  )
  with open(os.path.join(out_dir, 'placements.json'), 'w') as f:
    json.dump(to_jsonable(placements), f, indent=2)
  scene = bpy.context.scene
  with open(os.path.join(out_dir, 'render_settings.json'), 'w') as f:
    json.dump(to_jsonable({
      'blender_version': bpy.app.version_string,
      'engine': 'CYCLES',
      'resolution': [IMAGE_SIZE, IMAGE_SIZE],
      'samples': scene.cycles.samples,
      'adaptive_sampling': scene.cycles.use_adaptive_sampling,
      'seed': scene.cycles.seed,
      'denoising': scene.cycles.use_denoising,
      'view_transform': scene.view_settings.view_transform,
      'cast_shadows': bool(args.cast_shadows),
      'shapes': list(args.shapes),
      'colors': list(args.colors),
      'positions': args.positions,
    }), f, indent=2)

  # Every requested combination must have exactly one image
  n_positions = (len(args.positions) if args.positions is not None
                 else len(placements['positions']))
  expected = len(args.shapes) * n_positions * len(args.colors)
  combos = {(r['shape'], r['color'], r['position_index']) for r in records}
  missing = [r['image'] for r in records
             if not os.path.exists(os.path.join(out_dir, r['image']))]
  if len(records) != expected or len(combos) != expected or missing:
    raise RuntimeError('Expected %d images, have %d records (%d unique), '
                       '%d files missing' % (expected, len(records),
                                             len(combos), len(missing)))
  say('Rendered %d images (%d shapes x %d positions x %d colors) to %s'
      % (len(records), len(args.shapes), n_positions, len(args.colors),
         out_dir))


def main(args):
  global SHAPES
  settings = resolve_layout_settings(args)
  for shape in args.shapes:
    if shape not in ALL_SHAPES:
      raise ValueError('Unknown shape %s' % shape)
  for color in args.colors:
    if color not in ALL_COLORS:
      raise ValueError('Unknown color %s' % color)
  apply_geometry(args.image_size, args.patch_size, args.grid_size,
                 args.grid_spacing_patches)
  SHAPES = list(args.shapes)

  placements = None
  if os.path.exists(args.placements_json):
    with open(args.placements_json, 'r') as f:
      placements = json.load(f)
  if args.stage in ('all', 'layout'):
    if placements is None:
      reason = 'no placements file yet'
    elif args.recalibrate:
      reason = '--recalibrate 1'
    else:
      changed = layout_changes(placements, settings)
      reason = 'changed settings: %s' % ', '.join(changed) if changed else None
    if reason:
      say('Calibrating (%s)' % reason)
      placements = calibrate(args)
    else:
      say('Placements are up to date: %s' % args.placements_json)
  elif args.stage == 'calibrate':
    placements = calibrate(args)
  if args.stage in ('all', 'render'):
    if placements is None:
      raise RuntimeError('No placements file at %s; run the layout stage '
                         'first' % args.placements_json)
    changed = layout_changes(placements, settings)
    if changed:
      raise RuntimeError('The placements were calibrated with different '
                         'layout settings (%s); run the layout stage first'
                         % ', '.join(changed))
    render_dataset(args, placements)


def layout_changes(placements, settings):
  """ Layout settings that differ from those stored with the placements """
  stored = placements.get('layout_settings')
  if stored is None:
    say('WARNING: the placements file does not record its layout settings')
    return []
  return [k for k in LAYOUT_SETTINGS
          if comparable(stored.get(k)) != comparable(settings[k])]


if __name__ == '__main__':
  main(parse_args(parser, utils.extract_args(), PATH_SETTINGS))
