"""
Exports a single- or two-object VLM dataset in the layout and scene format of
CLEVR v1.0:

  <target_dir>/                     e.g. output/CLEVR_SINGLE_OBJECT
  ├── scenes/CLEVR_<split>_scenes.json
  └── images/<split>/CLEVR_<split>_000000.png, ...

Only these two paths are written; anything else in the target folder (for
example the generator output in generation/) is left alone.

All images go into one split (default "train"); the datasets are meant for
inference, and task-specific splits can be drawn from it later.

Each scene has the fields that CLEVR's render_images.py writes (image_index,
image_filename, split, objects with shape/size/material/color/3d_coords/
rotation/pixel_coords, directions, relationships) plus "source_image", the
file name in the generator output, which links back to the masks, patch
coverage, A/B assignments and questions there. Images are hard-linked to
the rendered files (no extra disk space; copied where hard links are not
possible), not re-rendered.

This is a plain Python script (no Blender needed). From image_generation/:

python export_clevr_format.py --dataset single \
    --source_dir ../output/CLEVR_SINGLE_OBJECT/generation \
    --target_dir ../output/CLEVR_SINGLE_OBJECT
"""

import argparse, hashlib, json, os, shutil
from datetime import datetime

from render_images import compute_all_relationships


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, choices=['single', 'dual'],
    help="Which generator wrote --source_dir.")
parser.add_argument('--source_dir', required=True,
    help="Output folder of render_single_object_dataset.py or "
         "render_two_object_dataset.py.")
parser.add_argument('--target_dir', required=True,
    help="Dataset folder that receives images/<split> and scenes/.")
parser.add_argument('--split', default='train')
parser.add_argument('--version', default='1.0')
parser.add_argument('--license', default='Creative Commons Attribution (CC-BY 4.0)')
parser.add_argument('--overwrite', default=0, type=int,
    help="Replace an existing export of the same split.")
parser.add_argument('--hardlink', default=1, type=int,
    help="Hard-link images instead of copying them.")


# ---------------------------------------------------------------------------
# Camera geometry (same conventions as Blender and CLEVR's render_images.py)
# ---------------------------------------------------------------------------

def quat_rotate(q, v):
  """ Rotate vector v by the unit quaternion q = (w, x, y, z) """
  w, x, y, z = q
  vx, vy, vz = v
  # t = 2 * cross(q.xyz, v); v' = v + w * t + cross(q.xyz, t)
  tx = 2 * (y * vz - z * vy)
  ty = 2 * (z * vx - x * vz)
  tz = 2 * (x * vy - y * vx)
  return (vx + w * tx + (y * tz - z * ty),
          vy + w * ty + (z * tx - x * tz),
          vz + w * tz + (x * ty - y * tx))


def normalized(v):
  n = sum(c * c for c in v) ** 0.5
  return tuple(c / n for c in v)


class Camera(object):
  """ Perspective camera stored in the generator's placements """
  def __init__(self, params, image_size):
    self.location = params['location']
    self.q = params['rotation_quaternion']
    self.right = quat_rotate(self.q, (1, 0, 0))
    self.up = quat_rotate(self.q, (0, 1, 0))
    self.forward = quat_rotate(self.q, (0, 0, -1))
    # Square image with sensor fit AUTO: the sensor width spans the image
    self.focal_px = params['lens_mm'] / params['sensor_width_mm'] * image_size
    self.size = image_size

  def pixel_coords(self, pos):
    """
    CLEVR's pixel_coords: (px, py) image coordinates with the origin at the
    top-left, rounded to integers, and the depth along the viewing direction
    (bpy_extras.object_utils.world_to_camera_view's z).
    """
    d = [p - c for p, c in zip(pos, self.location)]
    depth = sum(a * b for a, b in zip(d, self.forward))
    x = sum(a * b for a, b in zip(d, self.right)) / depth
    y = sum(a * b for a, b in zip(d, self.up)) / depth
    px = self.size / 2.0 + self.focal_px * x
    py = self.size / 2.0 - self.focal_px * y
    return [int(round(px)), int(round(py)), depth]

  def directions(self):
    """ The six directions as computed in render_images.py """
    plane_normal = (0.0, 0.0, 1.0)

    def on_plane(v):
      dot = sum(a * b for a, b in zip(v, plane_normal))
      return normalized(tuple(a - dot * n for a, n in zip(v, plane_normal)))

    behind = on_plane(self.forward)
    left = on_plane(tuple(-c for c in self.right))
    up = plane_normal
    return {
      'behind': list(behind), 'front': [-c for c in behind],
      'left': list(left), 'right': [-c for c in left],
      'above': list(up), 'below': [-c for c in up],
    }


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def clevr_object(camera, shape, color, size, material, location, rotation):
  return {
    'shape': shape,
    'size': size,
    'material': material,
    '3d_coords': list(location),
    'rotation': rotation,
    'pixel_coords': camera.pixel_coords(location),
    'color': color,
  }


def single_object_scenes(single_dir):
  """ (source image path, objects) for every single-object image """
  placements = load_json(os.path.join(single_dir, 'placements.json'))
  camera = Camera(placements['camera'], placements['image']['width'])
  records = load_json(os.path.join(single_dir, 'metadata.json'))
  scenes = []
  for r in records:
    obj = clevr_object(camera, r['shape'], r['color'], r['size'],
                       r['material'], r['world_location'], r['rotation_deg'])
    scenes.append((r['image'], [obj], [r['projected_center_px']]))
  return camera, scenes


def dual_object_scenes(dual_dir):
  """ (source image path, objects) for every unique two-object image """
  settings = load_json(os.path.join(dual_dir, 'render_settings.json'))
  camera = Camera(settings['camera'], settings['resolution'][0])
  records = sorted(load_json(os.path.join(dual_dir, 'scenes.json')),
                   key=lambda r: r['scene_id'])
  scenes = []
  for r in records:
    objs = [clevr_object(camera, o['shape'], o['color'], o['size'],
                         o['material'], o['world_location'], o['rotation_deg'])
            for o in r['objects']]
    scenes.append((r['image'], objs,
                   [o['projected_center_px'] for o in r['objects']]))
  return camera, scenes


def link_or_copy(source, target, hardlink):
  if hardlink:
    try:
      os.link(source, target)
      return
    except OSError:
      pass
  shutil.copyfile(source, target)


def export(source_dir, out_dir, camera, scenes, args):
  image_dir = os.path.join(out_dir, 'images', args.split)
  scene_file = os.path.join(out_dir, 'scenes', 'CLEVR_%s_scenes.json' % args.split)
  if os.path.exists(image_dir) or os.path.exists(scene_file):
    if not args.overwrite:
      raise RuntimeError('%s already has a "%s" export; use --overwrite 1 to '
                         'replace it' % (out_dir, args.split))
    shutil.rmtree(image_dir, ignore_errors=True)
    if os.path.exists(scene_file):
      os.remove(scene_file)
  os.makedirs(image_dir)
  os.makedirs(os.path.join(out_dir, 'scenes'), exist_ok=True)

  directions = camera.directions()
  clevr_scenes = []
  for index, (source, objects, _) in enumerate(scenes):
    filename = 'CLEVR_%s_%06d.png' % (args.split, index)
    link_or_copy(os.path.join(source_dir, source),
                 os.path.join(image_dir, filename), args.hardlink)
    scene = {
      'split': args.split,
      'image_index': index,
      'image_filename': filename,
      'objects': objects,
      'directions': directions,
      'source_image': os.path.basename(source),
    }
    scene['relationships'] = compute_all_relationships(scene)
    clevr_scenes.append(scene)

  output = {
    'info': {
      'date': datetime.today().strftime('%m/%d/%Y'),
      'version': args.version,
      'split': args.split,
      'license': args.license,
    },
    'scenes': clevr_scenes,
  }
  with open(scene_file, 'w') as f:
    json.dump(output, f)
  verify(out_dir, source_dir, scene_file, scenes, args)


def verify(out_dir, source_dir, scene_file, scenes, args):
  """ Check counts, image contents and pixel_coords after exporting """
  data = load_json(scene_file)
  exported = data['scenes']
  assert len(exported) == len(scenes), 'scene count'
  files = sorted(os.listdir(os.path.join(out_dir, 'images', args.split)))
  assert files == sorted(s['image_filename'] for s in exported), 'image files'
  worst = 0.0
  for s, (source, _, centers) in zip(exported, scenes):
    copied = os.path.join(out_dir, 'images', args.split, s['image_filename'])
    assert file_hash(copied) == file_hash(os.path.join(source_dir, source)), \
        'image content of %s' % s['image_filename']
    for obj, center in zip(s['objects'], centers):
      worst = max(worst, abs(obj['pixel_coords'][0] - center[0]),
                  abs(obj['pixel_coords'][1] - center[1]))
  assert worst <= 0.5, 'pixel_coords differ from projected centers'
  n_objects = sorted({len(s['objects']) for s in exported})
  print('%s: %d scenes with %s objects, %d images, pixel_coords match the '
        'projected centers (max deviation %.2f px)'
        % (out_dir, len(exported), n_objects, len(files), worst))


def file_hash(path):
  with open(path, 'rb') as f:
    return hashlib.sha256(f.read()).hexdigest()


def load_json(path):
  with open(path, 'r') as f:
    return json.load(f)


def main(args):
  source_dir = os.path.abspath(args.source_dir)
  if args.dataset == 'single':
    camera, scenes = single_object_scenes(source_dir)
  else:
    camera, scenes = dual_object_scenes(source_dir)
  export(source_dir, os.path.abspath(args.target_dir), camera, scenes, args)


if __name__ == '__main__':
  main(parser.parse_args())
