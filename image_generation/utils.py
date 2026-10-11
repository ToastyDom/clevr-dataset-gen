# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import sys, random, os, math
import bpy, bpy_extras


"""
Some utility functions for interacting with Blender
"""


def extract_args(input_argv=None):
  """
  Pull out command-line arguments after "--". Blender ignores command-line flags
  after --, so this lets us forward command line arguments from the blender
  invocation to our own script.
  """
  if input_argv is None:
    input_argv = sys.argv
  output_argv = []
  if '--' in input_argv:
    idx = input_argv.index('--')
    output_argv = input_argv[(idx + 1):]
  return output_argv


def parse_args(parser, argv=None):
  return parser.parse_args(extract_args(argv))


def delete_object(obj):
  """ Delete a specified blender object """
  bpy.data.objects.remove(obj, do_unlink=True)


def get_camera_coords(cam, pos):
  """
  For a specified point, get both the 3D coordinates and 2D pixel-space
  coordinates of the point from the perspective of the camera.

  Inputs:
  - cam: Camera object
  - pos: Vector giving 3D world-space position

  Returns a tuple of:
  - (px, py, pz): px and py give 2D image-space coordinates; pz gives depth
    in the range [-1, 1]
  """
  scene = bpy.context.scene
  x, y, z = bpy_extras.object_utils.world_to_camera_view(scene, cam, pos)
  scale = scene.render.resolution_percentage / 100.0
  w = int(scale * scene.render.resolution_x)
  h = int(scale * scene.render.resolution_y)
  px = int(round(x * w))
  py = int(round(h - y * h))
  return (px, py, z)


def add_object(object_dir, name, scale, loc, theta=0):
  """
  Load an object from a file. We assume that in the directory object_dir, there
  is a file named "$name.blend" which contains a single object named "$name"
  that has unit size and is centered at the origin.

  - scale: scalar giving the size that the object should be in the scene
  - loc: tuple (x, y) giving the coordinates on the ground plane where the
    object should be placed.
  - theta: rotation around the vertical axis, in degrees

  Returns the new blender object.
  """
  # First figure out how many of this object are already in the scene so we can
  # give the new object a unique name
  count = 0
  for obj in bpy.data.objects:
    if obj.name.startswith(name):
      count += 1

  filepath = os.path.join(object_dir, '%s.blend' % name)
  with bpy.data.libraries.load(filepath) as (data_from, data_to):
    data_to.objects = [name]
  obj = data_to.objects[0]
  bpy.context.scene.collection.objects.link(obj)
  obj.name = '%s_%d' % (name, count)

  # Rotate, scale, and translate it
  x, y = loc
  obj.rotation_euler[2] = math.radians(theta)
  obj.scale = [s * scale for s in obj.scale]
  obj.location = (x, y, scale)
  bpy.context.view_layer.objects.active = obj
  bpy.context.view_layer.update()
  return obj


def load_materials(material_dir):
  """
  Load materials from a directory. We assume that the directory contains .blend
  files with one material each. The file X.blend has a single NodeTree item named
  X; this NodeTree item must have a "Color" input that accepts an RGBA value.
  """
  for fn in os.listdir(material_dir):
    if not fn.endswith('.blend'): continue
    name = os.path.splitext(fn)[0]
    filepath = os.path.join(material_dir, fn)
    with bpy.data.libraries.load(filepath) as (data_from, data_to):
      data_to.node_groups = [name]


def add_material(obj, name, **properties):
  """
  Create a new material and assign it to obj. "name" should be the
  name of a material that has been previously loaded using load_materials.
  """
  mat = bpy.data.materials.new('Material_%d' % len(bpy.data.materials))
  nodes = mat.node_tree.nodes

  # Make sure it doesn't already have materials
  assert len(obj.data.materials) == 0
  obj.data.materials.append(mat)

  # Replace the default Principled BSDF with a group node that uses the
  # preloaded node group
  output_node = nodes['Material Output']
  for n in list(nodes):
    if n != output_node:
      nodes.remove(n)
  group_node = nodes.new('ShaderNodeGroup')
  group_node.node_tree = bpy.data.node_groups[name]

  # Find and set the "Color" input of the new group node
  for inp in group_node.inputs:
    if inp.name in properties:
      inp.default_value = properties[inp.name]

  # Wire the output of the new group node to the input of
  # the MaterialOutput node
  mat.node_tree.links.new(
      group_node.outputs['Shader'],
      output_node.inputs['Surface'],
  )
