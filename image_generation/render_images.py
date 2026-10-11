# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from __future__ import print_function
import math, sys, random, argparse, json, os, tempfile
from datetime import datetime as dt
from collections import Counter

"""
Renders random scenes using Blender, each with with a random number of objects;
each object has a random size, position, color, and shape. Objects will be
nonintersecting but may partially occlude each other. Output images will be
written to disk as PNGs, and we will also write a JSON file for each image with
ground-truth scene information.

This file expects to be run from Blender like this:

blender --background --python render_images.py -- [arguments to this script]
"""

INSIDE_BLENDER = True
try:
  import bpy, bpy_extras
  from mathutils import Vector
except ImportError as e:
  INSIDE_BLENDER = False
if INSIDE_BLENDER:
  # Make utils.py importable without having to touch Blender's site-packages
  sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
  try:
    import utils
  except ImportError as e:
    print("\nERROR")
    print("Running render_images.py from Blender and cannot import utils.py.") 
    print("You may need to add a .pth file to the site-packages of Blender's")
    print("bundled python with a command like this:\n")
    print("echo $PWD >> $BLENDER/$VERSION/python/lib/python3.5/site-packages/clevr.pth")
    print("\nWhere $BLENDER is the directory where Blender is installed, and")
    print("$VERSION is your Blender version (such as 2.78).")
    sys.exit(1)

parser = argparse.ArgumentParser()

# Input options
parser.add_argument('--base_scene_blendfile', default='data/base_scene.blend',
    help="Base blender file on which all scenes are based; includes " +
          "ground plane, lights, and camera.")
parser.add_argument('--properties_json', default='data/properties.json',
    help="JSON file defining objects, materials, sizes, and colors. " +
         "The \"colors\" field maps from CLEVR color names to RGB values; " +
         "The \"sizes\" field maps from CLEVR size names to scalars used to " +
         "rescale object models; the \"materials\" and \"shapes\" fields map " +
         "from CLEVR material and shape names to .blend files in the " +
         "--object_material_dir and --shape_dir directories respectively.")
parser.add_argument('--shape_dir', default='data/shapes',
    help="Directory where .blend files for object models are stored")
parser.add_argument('--material_dir', default='data/materials',
    help="Directory where .blend files for materials are stored")
parser.add_argument('--scene_spec', default=None,
    help="Optional path to a JSON file describing the scenes to render. It " +
         "should contain {\"scenes\": [{\"objects\": [...]}, ...]}, where " +
         "each object may give \"shape\", \"color\", \"material\", " +
         "\"size\", \"x\", \"y\" and \"rotation\" (degrees). Any " +
         "field that is left out is chosen randomly. One image is rendered " +
         "per scene and --num_images is ignored.")
parser.add_argument('--shape_color_combos_json', default=None,
    help="Optional path to a JSON file mapping shape names to a list of " +
         "allowed color names for that shape. This allows rendering images " +
         "for CLEVR-CoGenT.")

# Settings for objects
parser.add_argument('--min_objects', default=3, type=int,
    help="The minimum number of objects to place in each scene")
parser.add_argument('--max_objects', default=10, type=int,
    help="The maximum number of objects to place in each scene")
parser.add_argument('--min_dist', default=0.25, type=float,
    help="The minimum allowed distance between object centers")
parser.add_argument('--margin', default=0.4, type=float,
    help="Along all cardinal directions (left, right, front, back), all " +
         "objects will be at least this distance apart. This makes resolving " +
         "spatial relationships slightly less ambiguous.")
parser.add_argument('--min_pixels_per_object', default=200, type=int,
    help="All objects will have at least this many visible pixels in the " +
         "final rendered images; this ensures that no objects are fully " +
         "occluded by other objects.")
parser.add_argument('--max_retries', default=50, type=int,
    help="The number of times to try placing an object before giving up and " +
         "re-placing all objects in the scene.")

# Output settings
parser.add_argument('--start_idx', default=0, type=int,
    help="The index at which to start for numbering rendered images. Setting " +
         "this to non-zero values allows you to distribute rendering across " +
         "multiple machines and recombine the results later.")
parser.add_argument('--num_images', default=5, type=int,
    help="The number of images to render")
parser.add_argument('--filename_prefix', default='CLEVR',
    help="This prefix will be prepended to the rendered images and JSON scenes")
parser.add_argument('--split', default='new',
    help="Name of the split for which we are rendering. This will be added to " +
         "the names of rendered images, and will also be stored in the JSON " +
         "scene structure for each image.")
parser.add_argument('--output_image_dir', default='../output/images/',
    help="The directory where output images will be stored. It will be " +
         "created if it does not exist.")
parser.add_argument('--output_scene_dir', default='../output/scenes/',
    help="The directory where output JSON scene structures will be stored. " +
         "It will be created if it does not exist.")
parser.add_argument('--output_scene_file', default='../output/CLEVR_scenes.json',
    help="Path to write a single JSON file containing all scene information")
parser.add_argument('--output_blend_dir', default='output/blendfiles',
    help="The directory where blender scene files will be stored, if the " +
         "user requested that these files be saved using the " +
         "--save_blendfiles flag; in this case it will be created if it does " +
         "not already exist.")
parser.add_argument('--save_blendfiles', type=int, default=0,
    help="Setting --save_blendfiles 1 will cause the blender scene file for " +
         "each generated image to be stored in the directory specified by " +
         "the --output_blend_dir flag. These files are not saved by default " +
         "because they take up ~5-10MB each.")
parser.add_argument('--version', default='1.0',
    help="String to store in the \"version\" field of the generated JSON file")
parser.add_argument('--license',
    default="Creative Commons Attribution (CC-BY 4.0)",
    help="String to store in the \"license\" field of the generated JSON file")
parser.add_argument('--date', default=dt.today().strftime("%m/%d/%Y"),
    help="String to store in the \"date\" field of the generated JSON file; " +
         "defaults to today's date")

# Rendering options
parser.add_argument('--use_gpu', default=0, type=int,
    help="Setting --use_gpu 1 enables GPU-accelerated rendering using " +
         "Metal (Apple Silicon), OptiX/CUDA (NVIDIA), HIP (AMD) or oneAPI " +
         "(Intel), whichever is available.")
parser.add_argument('--width', default=320, type=int,
    help="The width (in pixels) for the rendered images")
parser.add_argument('--height', default=240, type=int,
    help="The height (in pixels) for the rendered images")
parser.add_argument('--key_light_jitter', default=1.0, type=float,
    help="The magnitude of random jitter to add to the key light position.")
parser.add_argument('--fill_light_jitter', default=1.0, type=float,
    help="The magnitude of random jitter to add to the fill light position.")
parser.add_argument('--back_light_jitter', default=1.0, type=float,
    help="The magnitude of random jitter to add to the back light position.")
parser.add_argument('--camera_jitter', default=0.5, type=float,
    help="The magnitude of random jitter to add to the camera position")
parser.add_argument('--render_num_samples', default=512, type=int,
    help="The number of samples to use when rendering. Larger values will " +
         "result in nicer images but will cause rendering to take longer.")
parser.add_argument('--render_min_bounces', default=8, type=int,
    help="The minimum number of bounces to use for rendering.")
parser.add_argument('--render_max_bounces', default=8, type=int,
    help="The maximum number of bounces to use for rendering.")
parser.add_argument('--render_tile_size', default=256, type=int,
    help="The tile size to use for rendering. This should not affect the " +
         "quality of the rendered image but may affect the speed; CPU-based " +
         "rendering may achieve better performance using smaller tile sizes " +
         "while larger tile sizes may be optimal for GPU-based rendering.")

def main(args):
  num_digits = 6
  prefix = '%s_%s_' % (args.filename_prefix, args.split)
  img_template = '%s%%0%dd.png' % (prefix, num_digits)
  scene_template = '%s%%0%dd.json' % (prefix, num_digits)
  blend_template = '%s%%0%dd.blend' % (prefix, num_digits)
  img_template = os.path.join(args.output_image_dir, img_template)
  scene_template = os.path.join(args.output_scene_dir, scene_template)
  blend_template = os.path.join(args.output_blend_dir, blend_template)

  if not os.path.isdir(args.output_image_dir):
    os.makedirs(args.output_image_dir)
  if not os.path.isdir(args.output_scene_dir):
    os.makedirs(args.output_scene_dir)
  if args.save_blendfiles == 1 and not os.path.isdir(args.output_blend_dir):
    os.makedirs(args.output_blend_dir)
  
  scene_specs = None
  num_images = args.num_images
  if args.scene_spec is not None:
    with open(args.scene_spec, 'r') as f:
      scene_specs = json.load(f)['scenes']
    num_images = len(scene_specs)

  all_scene_paths = []
  for i in range(num_images):
    img_path = img_template % (i + args.start_idx)
    scene_path = scene_template % (i + args.start_idx)
    all_scene_paths.append(scene_path)
    blend_path = None
    if args.save_blendfiles == 1:
      blend_path = blend_template % (i + args.start_idx)
    if scene_specs is not None:
      object_specs = scene_specs[i]['objects']
    else:
      num_objects = random.randint(args.min_objects, args.max_objects)
      object_specs = [{} for _ in range(num_objects)]
    render_scene(args,
      object_specs=object_specs,
      output_index=(i + args.start_idx),
      output_split=args.split,
      output_image=img_path,
      output_scene=scene_path,
      output_blendfile=blend_path,
    )

  # After rendering all images, combine the JSON files for each scene into a
  # single JSON file.
  all_scenes = []
  for scene_path in all_scene_paths:
    with open(scene_path, 'r') as f:
      all_scenes.append(json.load(f))
  output = {
    'info': {
      'date': args.date,
      'version': args.version,
      'split': args.split,
      'license': args.license,
    },
    'scenes': all_scenes
  }
  with open(args.output_scene_file, 'w') as f:
    json.dump(output, f)



def render_scene(args,
    object_specs=(),
    output_index=0,
    output_split='none',
    output_image='render.png',
    output_scene='render_json',
    output_blendfile=None,
  ):

  # Load the main blendfile
  bpy.ops.wm.open_mainfile(filepath=args.base_scene_blendfile)

  # Load materials
  utils.load_materials(args.material_dir)

  # Set render arguments so we can get pixel coordinates later.
  # We use functionality specific to the CYCLES renderer so BLENDER_RENDER
  # cannot be used.
  render_args = bpy.context.scene.render
  render_args.engine = "CYCLES"
  render_args.filepath = output_image
  render_args.resolution_x = args.width
  render_args.resolution_y = args.height
  render_args.resolution_percentage = 100
  bpy.context.scene.cycles.tile_size = args.render_tile_size
  # Match the plain sRGB look of the original CLEVR renders
  bpy.context.scene.view_settings.view_transform = 'Standard'
  use_gpu = args.use_gpu == 1 and enable_gpu()

  # Some CYCLES-specific stuff
  bpy.context.scene.cycles.blur_glossy = 2.0
  bpy.context.scene.cycles.samples = args.render_num_samples
  bpy.context.scene.cycles.min_transparent_bounces = args.render_min_bounces
  bpy.context.scene.cycles.transparent_max_bounces = args.render_max_bounces
  if use_gpu:
    bpy.context.scene.cycles.device = 'GPU'

  # This will give ground-truth information about the scene and its objects
  scene_struct = {
      'split': output_split,
      'image_index': output_index,
      'image_filename': os.path.basename(output_image),
      'objects': [],
      'directions': {},
  }

  def rand(L):
    return 2.0 * L * (random.random() - 0.5)

  # Add random jitter to camera position
  if args.camera_jitter > 0:
    for i in range(3):
      bpy.data.objects['Camera'].location[i] += rand(args.camera_jitter)

  # Figure out the left, up, and behind directions along the plane and record
  # them in the scene structure
  camera = bpy.data.objects['Camera']
  plane_normal = Vector((0, 0, 1))
  cam_behind = camera.matrix_world.to_quaternion() @ Vector((0, 0, -1))
  cam_left = camera.matrix_world.to_quaternion() @ Vector((-1, 0, 0))
  cam_up = camera.matrix_world.to_quaternion() @ Vector((0, 1, 0))
  plane_behind = (cam_behind - cam_behind.project(plane_normal)).normalized()
  plane_left = (cam_left - cam_left.project(plane_normal)).normalized()
  plane_up = cam_up.project(plane_normal).normalized()

  # Save all six axis-aligned directions in the scene struct
  scene_struct['directions']['behind'] = tuple(plane_behind)
  scene_struct['directions']['front'] = tuple(-plane_behind)
  scene_struct['directions']['left'] = tuple(plane_left)
  scene_struct['directions']['right'] = tuple(-plane_left)
  scene_struct['directions']['above'] = tuple(plane_up)
  scene_struct['directions']['below'] = tuple(-plane_up)

  # Add random jitter to lamp positions
  if args.key_light_jitter > 0:
    for i in range(3):
      bpy.data.objects['Lamp_Key'].location[i] += rand(args.key_light_jitter)
  if args.back_light_jitter > 0:
    for i in range(3):
      bpy.data.objects['Lamp_Back'].location[i] += rand(args.back_light_jitter)
  if args.fill_light_jitter > 0:
    for i in range(3):
      bpy.data.objects['Lamp_Fill'].location[i] += rand(args.fill_light_jitter)

  # Now add the objects
  objects, blender_objects = add_objects(scene_struct, object_specs, args, camera)

  # Render the scene and dump the scene data structure
  scene_struct['objects'] = objects
  scene_struct['relationships'] = compute_all_relationships(scene_struct)
  while True:
    try:
      bpy.ops.render.render(write_still=True)
      break
    except Exception as e:
      print(e)

  with open(output_scene, 'w') as f:
    json.dump(scene_struct, f, indent=2)

  if output_blendfile is not None:
    bpy.ops.wm.save_as_mainfile(filepath=output_blendfile)


def enable_gpu():
  """
  Enable whichever GPU backend Cycles can find. Returns True on success.
  """
  cycles_prefs = bpy.context.preferences.addons['cycles'].preferences
  for device_type in ['METAL', 'OPTIX', 'CUDA', 'HIP', 'ONEAPI']:
    try:
      cycles_prefs.compute_device_type = device_type
    except TypeError:
      continue
    cycles_prefs.refresh_devices()
    gpus = [d for d in cycles_prefs.devices if d.type == device_type]
    if gpus:
      for d in gpus:
        d.use = True
      print('Rendering on GPU with %s' % device_type)
      return True
  print('No GPU found; rendering on CPU')
  return False


def resolve_object_spec(spec, properties, shape_color_combos):
  """
  Turn a (possibly partial) object description from a scene spec into a full
  set of object attributes, choosing anything that is missing at random.
  """
  def check(kind, value):
    if value not in properties[kind]:
      raise ValueError('Unknown %s "%s"; choose from %s'
                       % (kind, value, sorted(properties[kind])))
    return value

  shape_name = spec.get('shape')
  color = spec.get('color')
  if shape_color_combos is not None and shape_name is None and color is None:
    shape_name, color_choices = random.choice(shape_color_combos)
    color = random.choice(color_choices)
  if shape_name is None:
    shape_name = random.choice(list(properties['shapes']))
  if color is None:
    color = random.choice(list(properties['colors']))
  check('shapes', shape_name)

  # Colors can be given by name or as [r, g, b] in the range 0-255
  if isinstance(color, str):
    color_name = check('colors', color)
    rgb = properties['colors'][color]
  else:
    color_name = spec.get('color_name', 'custom')
    rgb = color
  rgba = [float(c) / 255.0 for c in rgb] + [1.0]

  size_name = check('sizes', spec.get('size') or
                    random.choice(list(properties['sizes'])))
  material_name = check('materials', spec.get('material') or
                        random.choice(list(properties['materials'])))

  return {
    'shape': shape_name,
    'blend_name': properties['shapes'][shape_name],
    'color': color_name,
    'rgba': rgba,
    'size': size_name,
    'r': properties['sizes'][size_name],
    'material': material_name,
    'blend_material': properties['materials'][material_name],
    'rotation': spec.get('rotation', 360.0 * random.random()),
    'x': spec.get('x'),
    'y': spec.get('y'),
  }


def position_ok(x, y, r, positions, scene_struct, args):
  """
  Check that a new object is further than min_dist from all other objects, and
  further than margin along the four cardinal directions.
  """
  for (xx, yy, rr) in positions:
    dx, dy = x - xx, y - yy
    dist = math.sqrt(dx * dx + dy * dy)
    if dist - r - rr < args.min_dist:
      return False
    for direction_name in ['left', 'right', 'front', 'behind']:
      direction_vec = scene_struct['directions'][direction_name]
      assert direction_vec[2] == 0
      margin = dx * direction_vec[0] + dy * direction_vec[1]
      if 0 < margin < args.margin:
        return False
  return True


def place_objects(resolved, scene_struct, args):
  """
  Choose (x, y) positions for all objects that do not have a fixed position.
  Objects with a fixed position are used as-is. Returns a list of (x, y) or
  None if placement failed too many times.
  """
  fixed = [(o['x'], o['y'], o['r']) for o in resolved if o['x'] is not None]
  positions = list(fixed)
  out = []
  for o in resolved:
    if o['x'] is not None:
      out.append((o['x'], o['y']))
      continue
    for _ in range(args.max_retries):
      x = random.uniform(-3, 3) if o['x'] is None else o['x']
      y = random.uniform(-3, 3) if o['y'] is None else o['y']
      if position_ok(x, y, o['r'], positions, scene_struct, args):
        break
    else:
      return None
    positions.append((x, y, o['r']))
    out.append((x, y))
  return out


def add_objects(scene_struct, object_specs, args, camera):
  """
  Add objects to the current blender scene. Each entry of object_specs is a
  dict that may fix some attributes of the object; the rest are random.
  """

  # Load the property file
  with open(args.properties_json, 'r') as f:
    properties = json.load(f)

  shape_color_combos = None
  if args.shape_color_combos_json is not None:
    with open(args.shape_color_combos_json, 'r') as f:
      shape_color_combos = list(json.load(f).items())

  all_fixed = all(spec.get('x') is not None and spec.get('y') is not None
                  for spec in object_specs)
  while True:
    resolved = [resolve_object_spec(spec, properties, shape_color_combos)
                for spec in object_specs]
    # A missing x or y is treated as "choose this position randomly"
    for o in resolved:
      if o['x'] is None or o['y'] is None:
        o['x'] = o['y'] = None
    xys = place_objects(resolved, scene_struct, args)
    if xys is None:
      print('Could not place objects; retrying')
      continue

    objects = []
    blender_objects = []
    for o, (x, y) in zip(resolved, xys):
      r = o['r']
      # For cube, adjust the size a bit
      if o['blend_name'] == 'SmoothCube_v2':
        r /= math.sqrt(2)

      # Actually add the object to the scene
      obj = utils.add_object(args.shape_dir, o['blend_name'], r, (x, y),
                             theta=o['rotation'])
      blender_objects.append(obj)
      utils.add_material(obj, o['blend_material'], Color=o['rgba'])

      # Record data about the object in the scene data structure
      pixel_coords = utils.get_camera_coords(camera, obj.location)
      objects.append({
        'shape': o['shape'],
        'size': o['size'],
        'material': o['material'],
        '3d_coords': tuple(obj.location),
        'rotation': o['rotation'],
        'pixel_coords': pixel_coords,
        'color': o['color'],
      })

    # Check that all objects are at least partially visible in the rendered image
    if check_visibility(blender_objects, args.min_pixels_per_object):
      return objects, blender_objects
    if all_fixed:
      print('WARNING: some objects are (almost) fully occluded or out of view')
      return objects, blender_objects

    # If any of the objects are fully occluded then start over; delete all
    # objects from the scene and place them all again.
    print('Some objects are occluded; replacing objects')
    for obj in blender_objects:
      utils.delete_object(obj)


def compute_all_relationships(scene_struct, eps=0.2):
  """
  Computes relationships between all pairs of objects in the scene.
  
  Returns a dictionary mapping string relationship names to lists of lists of
  integers, where output[rel][i] gives a list of object indices that have the
  relationship rel with object i. For example if j is in output['left'][i] then
  object j is left of object i.
  """
  all_relationships = {}
  for name, direction_vec in scene_struct['directions'].items():
    if name == 'above' or name == 'below': continue
    all_relationships[name] = []
    for i, obj1 in enumerate(scene_struct['objects']):
      coords1 = obj1['3d_coords']
      related = set()
      for j, obj2 in enumerate(scene_struct['objects']):
        if obj1 == obj2: continue
        coords2 = obj2['3d_coords']
        diff = [coords2[k] - coords1[k] for k in [0, 1, 2]]
        dot = sum(diff[k] * direction_vec[k] for k in [0, 1, 2])
        if dot > eps:
          related.add(j)
      all_relationships[name].append(sorted(list(related)))
  return all_relationships


def check_visibility(blender_objects, min_pixels_per_object):
  """
  Check whether all objects in the scene have some minimum number of visible
  pixels; to accomplish this we assign random (but distinct) colors to all
  objects, and render using no lighting or shading or antialiasing; this
  ensures that each object is just a solid uniform color. We can then count
  the number of pixels of each color in the output image to check the visibility
  of each object.

  Returns True if all objects are visible and False otherwise.
  """
  f, path = tempfile.mkstemp(suffix='.png')
  object_colors = render_shadeless(blender_objects, path=path)
  img = bpy.data.images.load(path)
  p = list(img.pixels)
  bpy.data.images.remove(img)
  color_count = Counter((p[i], p[i+1], p[i+2], p[i+3])
                        for i in range(0, len(p), 4))
  os.remove(path)
  if len(color_count) != len(blender_objects) + 1:
    return False
  for _, count in color_count.most_common():
    if count < min_pixels_per_object:
      return False
  return True


def render_shadeless(blender_objects, path='flat.png'):
  """
  Render a version of the scene with shading disabled and unique colors
  assigned to all objects, and return a set of all colors that should be in the
  rendered image. The image itself is written to path. This is used to ensure
  that all objects will be visible in the final rendered scene.
  """
  scene = bpy.context.scene
  render_args = scene.render
  shading = scene.display.shading

  # Cache the render args we are about to clobber
  old_filepath = render_args.filepath
  old_engine = render_args.engine
  old_aa = scene.display.render_aa
  old_dither = render_args.dither_intensity
  old_shading = (shading.light, shading.color_type, shading.show_shadows,
                 shading.show_cavity, shading.show_object_outline)

  # Override some render settings to have flat shading
  render_args.filepath = path
  render_args.engine = 'BLENDER_WORKBENCH'
  scene.display.render_aa = 'OFF'
  render_args.dither_intensity = 0
  shading.light = 'FLAT'
  shading.color_type = 'OBJECT'
  shading.show_shadows = False
  shading.show_cavity = False
  shading.show_object_outline = False

  # Hide the ground so only the objects render
  bpy.data.objects['Ground'].hide_render = True

  # Give all objects random flat colors
  object_colors = set()
  old_colors = []
  for obj in blender_objects:
    old_colors.append(tuple(obj.color))
    while True:
      r, g, b = [random.random() for _ in range(3)]
      if (r, g, b) not in object_colors: break
    object_colors.add((r, g, b))
    obj.color = (r, g, b, 1.0)

  # Render the scene
  bpy.ops.render.render(write_still=True)

  # Undo the above
  for color, obj in zip(old_colors, blender_objects):
    obj.color = color
  bpy.data.objects['Ground'].hide_render = False
  render_args.filepath = old_filepath
  render_args.engine = old_engine
  scene.display.render_aa = old_aa
  render_args.dither_intensity = old_dither
  (shading.light, shading.color_type, shading.show_shadows,
   shading.show_cavity, shading.show_object_outline) = old_shading

  return object_colors


if __name__ == '__main__':
  if INSIDE_BLENDER:
    # Run normally
    argv = utils.extract_args()
    args = parser.parse_args(argv)
    main(args)
  elif '--help' in sys.argv or '-h' in sys.argv:
    parser.print_help()
  else:
    print('This script is intended to be called from blender like this:')
    print()
    print('blender --background --python render_images.py -- [args]')
    print()
    print('You can also run as a standalone python script to view all')
    print('arguments like this:')
    print()
    print('python render_images.py --help')

